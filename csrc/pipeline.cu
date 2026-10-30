#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdio>
#include <cuda/atomic>
#include <limits>
#include <nano_ep/ops.hpp>
#include <nano_ep/pipeline.hpp>
#include <utility>

#include "layout.hpp"

namespace nano_ep {

namespace {

static_assert(WindowLayout::kHiddenAlignment == 32 * sizeof(int4) / sizeof(at::BFloat16));

void check_payload_inputs(const torch::Tensor& window,
                          const torch::Tensor& x,
                          const torch::Tensor& indices,
                          const c10::optional<torch::Tensor>& weights,
                          int world_size,
                          int64_t num_experts,
                          int64_t max_tokens) {
    TORCH_CHECK(x.device() == window.device() && x.dim() == 2 && x.is_contiguous(),
                "x must be contiguous CUDA [T,H] on the runtime device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "dispatch x must be bfloat16");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % WindowLayout::kHiddenAlignment == 0, "dispatch hidden must be a positive multiple of 256");
    TORCH_CHECK(indices.device() == x.device() && indices.scalar_type() == torch::kInt64 && indices.dim() == 2 &&
                    indices.size(0) == x.size(0) && indices.is_contiguous(),
                "indices must be contiguous CUDA int64 [T,K] on the x device");
    TORCH_CHECK(num_experts > 0 && num_experts % world_size == 0, "num_experts must be positive and divisible by world_size");
    TORCH_CHECK(x.size(1) > 0 && indices.size(1) > 0 && indices.size(1) <= 32 && indices.size(1) <= num_experts,
                "hidden must be positive and top-k must be in [1,min(32,num_experts)]");
    TORCH_CHECK(max_tokens > 0 && x.size(0) <= max_tokens, "source token count exceeds max_tokens");
    if (weights.has_value()) {
        TORCH_CHECK(weights->device() == x.device() && weights->scalar_type() == torch::kFloat32 && weights->sizes() == indices.sizes() &&
                        weights->is_contiguous(),
                    "weights must be contiguous CUDA float32 [T,K] on the x device");
    }
}

void check_cached_tensor(const torch::Tensor& tensor, const torch::Tensor& reference, at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(
        tensor.device() == reference.device() && tensor.scalar_type() == torch::kInt64 && tensor.sizes() == shape && tensor.is_contiguous(),
        "cached ",
        name,
        " has invalid CUDA int64 storage or shape");
}

__global__ void notify_counts_kernel(const int64_t* peer_addresses,
                                     int64_t* local_counts,
                                     const int64_t* send_counts,
                                     const int64_t* send_expert_counts,
                                     int64_t* recv_counts,
                                     int rank,
                                     int world_size,
                                     int64_t num_experts,
                                     int64_t count_offset,
                                     unsigned long long timeout_cycles) {
    using Count = cuda::atomic_ref<int64_t, cuda::thread_scope_system>;
    const int64_t experts_per_rank = num_experts / world_size;
    const int64_t num_counts = world_size + num_experts;

    // Publish every outgoing entry before entering any receive spin loop.
    for (int64_t i = threadIdx.x; i < num_counts; i += blockDim.x) {
        const bool rank_count = i < world_size;
        const int64_t expert = i - world_size;
        const int destination = rank_count ? static_cast<int>(i) : static_cast<int>(expert / experts_per_rank);
        const int64_t dst_index = rank_count ? rank : world_size + rank * experts_per_rank + expert % experts_per_rank;
        const int64_t value = rank_count ? send_counts[i] : send_expert_counts[expert];
        auto* peer_window = reinterpret_cast<uint8_t*>(peer_addresses[destination]);
        auto* peer_counts = reinterpret_cast<int64_t*>(peer_window + count_offset);
        Count outgoing(peer_counts[dst_index]);
        outgoing.store(value, cuda::memory_order_release);
    }
    __syncthreads();

    const auto start = clock64();
    for (int64_t i = threadIdx.x; i < num_counts; i += blockDim.x) {
        Count incoming(local_counts[i]);
        while (true) {
            const auto count = incoming.load(cuda::memory_order_acquire);
            if (count >= 0) {
                recv_counts[i] = count;
                incoming.store(WindowLayout::kCountNotReady, cuda::memory_order_relaxed);
                break;
            }
            if (clock64() - start > timeout_cycles) {
                printf("NanoEP notify timeout: rank=%d index=%lld count=%lld\n",
                       rank,
                       static_cast<long long>(i),
                       static_cast<long long>(count));
                asm volatile("trap;");
            }
            __nanosleep(64);
        }
    }
}

template <typename data_t, bool kReuseSlots>
__global__ void reserve_and_put_kernel(const int64_t* peer_addresses,
                                       const data_t* x,         // [T, H]
                                       const int64_t* indices,  // [T, K]
                                       const float* weights,    // [T, K]，allow nullptr
                                       int64_t* dst_slots,      // [T, R], fresh output or read-only cached mapping
                                       int rank,
                                       int world_size,
                                       int64_t num_tokens,  // T
                                       int64_t hidden,      // H
                                       int64_t num_topk,    // K
                                       int64_t experts_per_rank,
                                       int64_t max_tokens,  // max tokens per source rank
                                       int64_t recv_next_offset,
                                       int64_t payload_offset,
                                       int64_t payload_row_bytes) {
    // Each warp is responsible for one token
    const int lane = threadIdx.x % 32;
    const int64_t token = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (token >= num_tokens)
        return;

    // Fist K lane handle one routing
    // Expert[0, 1, 2, 3, 4, 5, 6, 7] -> Rank[0, 0, 1, 1, 2, 2, 3, 3]
    int destination = -1;
    if (lane < num_topk) {
        const int64_t expert_id = indices[token * num_topk + lane];
        if (expert_id < -1 || expert_id >= world_size * experts_per_rank) {
            printf("NanoEP reserve_and_put invalid expert: token=%lld expert=%lld\n",
                   static_cast<long long>(token),
                   static_cast<long long>(expert_id));
            asm volatile("trap;");
            return;
        }
        if (expert_id >= 0) {
            destination = expert_id / experts_per_rank;
        }
    }

    // e.g. for lane 0, same_destination = 0x00000003
    const unsigned same_destination = __match_any_sync(0xffffffffu, destination);

    const int leader = 31 - __clz(same_destination);
    const bool is_sender = (destination >= 0 && lane == leader);

    unsigned long long slot = 0;
    if (is_sender) {
        if constexpr (kReuseSlots) {
            slot = static_cast<unsigned long long>(dst_slots[token * world_size + destination]);

        } else {
            auto* target_window = reinterpret_cast<uint8_t*>(peer_addresses[destination]);
            auto* recv_next = reinterpret_cast<unsigned long long*>(target_window + recv_next_offset);
            using Counter = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
            Counter counter(*recv_next);
            slot = counter.fetch_add(1ULL, cuda::memory_order_relaxed);
            dst_slots[token * world_size + destination] = static_cast<int64_t>(slot);
        }

        if (!(slot < static_cast<unsigned long long>(world_size) * max_tokens)) {
            printf("Invalid destination slot\n");
            asm volatile("trap;");
        }
    }

    // e.g. senders = 0b00000000000000000000000010101010
    unsigned senders = __ballot_sync(0xffffffffu, is_sender);
    while (senders != 0) {
        // find the first sender in the warp
        const int sender = __ffs(senders) - 1;

        // broadcast the destination
        const int64_t dst = __shfl_sync(0xffffffffu, destination, sender);
        const int64_t dst_slot = __shfl_sync(0xffffffffu, slot, sender);
        auto* peer_window = reinterpret_cast<uint8_t*>(peer_addresses[dst]);

        // write payload together
        auto* row = peer_window + payload_offset + dst_slot * payload_row_bytes;
        auto* row_src_rank = reinterpret_cast<int64_t*>(row);
        auto* row_src_token = reinterpret_cast<int64_t*>(row + 8);
        auto* row_topk = reinterpret_cast<int64_t*>(row + WindowLayout::kPayloadTopkOffset);
        auto* row_weights = reinterpret_cast<float*>(row + WindowLayout::payload_weights_offset(num_topk));
        auto* row_x = reinterpret_cast<data_t*>(row + WindowLayout::payload_x_offset(num_topk));

        if (lane == 0) {
            row_src_rank[0] = rank;
            row_src_token[0] = token;
        }
        for (int64_t k = lane; k < num_topk; k += 32) {
            row_topk[k] = indices[token * num_topk + k];
            if (weights != nullptr) {
                row_weights[k] = weights[token * num_topk + k];
            }
        }
        for (int64_t h = lane; h < hidden; h += 32) {
            row_x[h] = x[token * hidden + h];
        }

        // clear the first sender
        senders &= senders - 1;
    }
}

__global__ void build_receive_metadata_kernel(const uint8_t* stage_buffer,
                                              int64_t* src_ranks,
                                              int64_t* src_tokens,
                                              int64_t* local_topk_idx,
                                              int64_t* expanded_slots,
                                              unsigned long long* expert_cursor,
                                              int64_t recv_count,
                                              int64_t row_bytes,
                                              int64_t rank,
                                              int64_t experts_per_rank,
                                              int64_t num_topk) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    if (row >= recv_count)
        return;

    // ┌──────────┬───────────┬───────────┬────────────┬─────────┬─────────┐
    // │ src_rank │ src_token │ topk IDs  │ weights    │ x       │ padding │
    // │ int64    │ int64     │ int64[K]  │ fp32[K]    │ bf16[H] │         │
    // └──────────┴───────────┴───────────┴────────────┴─────────┴─────────┘
    const auto* payload = stage_buffer + row * row_bytes;
    const auto* provenance = reinterpret_cast<const int64_t*>(payload);
    const auto* topk = reinterpret_cast<const int64_t*>(payload + WindowLayout::kPayloadTopkOffset);
    if (lane == 0) {
        src_ranks[row] = provenance[0];
        src_tokens[row] = provenance[1];
    }

    const int64_t expert = lane < num_topk ? topk[lane] : -1;
    const int64_t expert_start = rank * experts_per_rank;
    const bool in_range = expert >= expert_start && expert - expert_start < experts_per_rank;
    const long long local_expert = in_range ? expert - expert_start : -1;
    const unsigned same_expert = __match_any_sync(0xffffffffu, local_expert);
    if (in_range && __popc(same_expert) != 1) {
        printf("NanoEP receive metadata duplicate expert: row=%lld expert=%lld\n",
               static_cast<long long>(row),
               static_cast<long long>(expert));
        asm volatile("trap;");
        return;
    }
    if (lane < num_topk) {
        const int64_t index = row * num_topk + lane;
        if (in_range) {
            local_topk_idx[index] = local_expert;
            // Each expert owns a disjoint segment; only its cursor needs an atomic update.
            expanded_slots[index] = static_cast<int64_t>(atomicAdd(expert_cursor + local_expert, 1ULL));
        } else {
            local_topk_idx[index] = -1;
            expanded_slots[index] = -1;
        }
    }
}

template <typename output_t, bool kExpand, bool kWithWeights>
__global__ void dispatch_copy_epilogue_kernel(const uint8_t* stage_buffer,    // [N, row_bytes]
                                              const int64_t* local_topk_idx,  // [N,K]
                                              const int64_t* expanded_slots,  // [N,K], invalid = -1
                                              output_t* recv_x,               // compact [N,H], expanded [C,H]
                                              int64_t* recv_topk_idx,         // compact [N,K], expanded nullptr
                                              float* recv_weights,            // compact [N,K], expanded [C], optional
                                              int64_t recv_count,             // N
                                              int64_t row_bytes,
                                              int64_t hidden,   // H
                                              int64_t num_topk  // K
) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    if (row >= recv_count)
        return;

    const auto* payload = stage_buffer + row * row_bytes;
    const auto* weights = reinterpret_cast<const float*>(payload + WindowLayout::payload_weights_offset(num_topk));
    const auto* x = reinterpret_cast<const output_t*>(payload + WindowLayout::payload_x_offset(num_topk));

    if constexpr (kExpand) {
        for (int64_t k = 0; k < num_topk; ++k) {
            const int64_t slot = expanded_slots[row * num_topk + k];
            if (slot < 0) {
                continue;
            }

            if constexpr (kWithWeights) {
                if (lane == 0) {
                    recv_weights[slot] = weights[k];
                }
            }

            for (int64_t h = lane; h < hidden; h += warp_size) {
                recv_x[slot * hidden + h] = x[h];
            }
        }
    } else {
        for (int64_t k = lane; k < num_topk; k += warp_size) {
            recv_topk_idx[row * num_topk + k] = local_topk_idx[row * num_topk + k];
            if constexpr (kWithWeights) {
                recv_weights[row * num_topk + k] = weights[k];
            }
        }
        for (int64_t h = lane; h < hidden; h += warp_size) {
            recv_x[row * hidden + h] = x[h];
        }
    }
}

}  // namespace

std::vector<torch::Tensor> notify_counts(PeerRuntime& runtime,
                                         const torch::Tensor& send_counts,
                                         const torch::Tensor& send_expert_counts,
                                         int64_t count_offset) {
    const auto window = runtime.local_window();
    const int world_size = runtime.world_size();
    constexpr int threads = 256;

    TORCH_CHECK(send_counts.device() == window.device() && send_counts.scalar_type() == torch::kInt64 && send_counts.dim() == 1 &&
                    send_counts.numel() == world_size && send_counts.is_contiguous(),
                "send_counts must be contiguous CUDA int64 [world_size] on the runtime device");
    TORCH_CHECK(send_expert_counts.device() == window.device() && send_expert_counts.scalar_type() == torch::kInt64 &&
                    send_expert_counts.dim() == 1 && send_expert_counts.numel() > 0 && send_expert_counts.numel() % world_size == 0 &&
                    send_expert_counts.is_contiguous(),
                "send_expert_counts must be contiguous CUDA int64 [E], with positive E divisible by world_size");
    const int64_t num_experts = send_expert_counts.numel();
    const int64_t num_counts = world_size + num_experts;
    TORCH_CHECK(count_offset >= world_size * 8 && count_offset % 8 == 0 && count_offset <= window.numel() &&
                    num_counts <= (window.numel() - count_offset) / 8,
                "count slots must be aligned, inside the window and separate from barrier signals");

    c10::cuda::CUDAGuard guard(window.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    auto recv_counts = torch::empty({num_counts}, send_counts.options());
    const auto addresses = runtime.peer_addresses();

    constexpr unsigned long long timeout_cycles = 20'000'000'000ULL;
    notify_counts_kernel<<<1, threads, 0, stream.stream()>>>(addresses.data_ptr<int64_t>(),
                                                             reinterpret_cast<int64_t*>(window.data_ptr<uint8_t>() + count_offset),
                                                             send_counts.data_ptr<int64_t>(),
                                                             send_expert_counts.data_ptr<int64_t>(),
                                                             recv_counts.data_ptr<int64_t>(),
                                                             runtime.rank(),
                                                             world_size,
                                                             num_experts,
                                                             count_offset,
                                                             timeout_cycles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto recv_rank_counts = recv_counts.narrow(0, 0, world_size);
    auto recv_expert_counts = recv_counts.narrow(0, world_size, num_experts).view({world_size, num_experts / world_size}).sum(0);
    return {recv_rank_counts, recv_expert_counts};
}

torch::Tensor reserve_and_put(PeerRuntime& runtime,
                              const torch::Tensor& x,
                              const torch::Tensor& indices,
                              const c10::optional<torch::Tensor>& weights,
                              int64_t num_experts,
                              int64_t max_tokens,
                              const WindowLayout& layout,
                              const c10::optional<torch::Tensor>& cached_dst_slots) {
    const auto window = runtime.local_window();
    const int rank = runtime.rank();
    const int world_size = runtime.world_size();
    check_payload_inputs(window, x, indices, weights, world_size, num_experts, max_tokens);

    const int64_t num_tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t num_topk = indices.size(1);
    TORCH_CHECK(layout.num_bytes <= window.numel(), "runtime window is smaller than the payload layout");
    if (cached_dst_slots.has_value()) {
        const auto& slots = *cached_dst_slots;
        TORCH_CHECK(slots.device() == x.device() && slots.scalar_type() == torch::kInt64 && slots.dim() == 2 &&
                        slots.size(0) == num_tokens && slots.size(1) == world_size && slots.is_contiguous(),
                    "cached_dst_slots must be contiguous CUDA int64 [T,R] on the x device");
    }

    c10::cuda::CUDAGuard guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto addresses = runtime.peer_addresses();
    const bool reuse_slots = cached_dst_slots.has_value();
    auto dst_slots = reuse_slots ? *cached_dst_slots : torch::full({num_tokens, world_size}, -1, indices.options());

    if (num_tokens == 0)
        return dst_slots;

    const int64_t experts_per_rank = num_experts / world_size;
    const int64_t recv_next_offset = layout.recv_next_offset;
    const int64_t row_bytes = WindowLayout::payload_row_bytes(hidden, num_topk);
    const auto* weight_ptr = weights.has_value() ? weights->data_ptr<float>() : nullptr;
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (num_tokens + warps_per_block - 1) / warps_per_block;

    if (reuse_slots) {
        using scalar_t = at::BFloat16;

        reserve_and_put_kernel<scalar_t, true><<<blocks, threads, 0, stream.stream()>>>(addresses.data_ptr<int64_t>(),
                                                                                        x.data_ptr<scalar_t>(),
                                                                                        indices.data_ptr<int64_t>(),
                                                                                        weight_ptr,
                                                                                        dst_slots.data_ptr<int64_t>(),
                                                                                        rank,
                                                                                        world_size,
                                                                                        num_tokens,
                                                                                        hidden,
                                                                                        num_topk,
                                                                                        experts_per_rank,
                                                                                        max_tokens,
                                                                                        recv_next_offset,
                                                                                        layout.payload_offset,
                                                                                        row_bytes);
    } else {
        using scalar_t = at::BFloat16;

        reserve_and_put_kernel<scalar_t, false><<<blocks, threads, 0, stream.stream()>>>(addresses.data_ptr<int64_t>(),
                                                                                         x.data_ptr<scalar_t>(),
                                                                                         indices.data_ptr<int64_t>(),
                                                                                         weight_ptr,
                                                                                         dst_slots.data_ptr<int64_t>(),
                                                                                         rank,
                                                                                         world_size,
                                                                                         num_tokens,
                                                                                         hidden,
                                                                                         num_topk,
                                                                                         experts_per_rank,
                                                                                         max_tokens,
                                                                                         recv_next_offset,
                                                                                         layout.payload_offset,
                                                                                         row_bytes);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst_slots;
}

std::vector<torch::Tensor> build_receive_metadata(
    const torch::Tensor& stage_buffer, const torch::Tensor& expert_offsets, int64_t rank, int64_t experts_per_rank, int64_t num_topk) {
    TORCH_CHECK(
        stage_buffer.is_cuda() && stage_buffer.scalar_type() == torch::kUInt8 && stage_buffer.dim() == 2 && stage_buffer.is_contiguous(),
        "stage_buffer must be contiguous CUDA uint8 [N,row_bytes]");
    TORCH_CHECK(rank >= 0 && experts_per_rank > 0 && num_topk > 0 && num_topk <= 32,
                "rank must be nonnegative, experts_per_rank positive and num_topk in [1,32]");
    TORCH_CHECK(expert_offsets.device() == stage_buffer.device() && expert_offsets.scalar_type() == torch::kInt64 &&
                    expert_offsets.dim() == 1 && expert_offsets.numel() == experts_per_rank + 1 && expert_offsets.is_contiguous(),
                "expert_offsets must be contiguous CUDA int64 [experts_per_rank+1] on the stage_buffer device");
    const int64_t recv_count = stage_buffer.size(0);
    const int64_t row_bytes = stage_buffer.size(1);
    TORCH_CHECK(row_bytes >= WindowLayout::payload_row_bytes(1, num_topk) && row_bytes % WindowLayout::kRowAlignment == 0 &&
                    reinterpret_cast<uintptr_t>(stage_buffer.data_ptr<uint8_t>()) % WindowLayout::kAlignment == 0,
                "stage_buffer must contain aligned full dispatch payload rows");

    c10::cuda::CUDAGuard guard(stage_buffer.device());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto options = expert_offsets.options();
    auto src_ranks = torch::empty({recv_count}, options);
    auto src_tokens = torch::empty({recv_count}, options);
    auto local_topk_idx = torch::empty({recv_count, num_topk}, options);
    auto expanded_slots = torch::empty({recv_count, num_topk}, options);
    if (recv_count == 0)
        return {src_ranks, src_tokens, local_topk_idx, expanded_slots};

    // Preserve the padded offsets for replay and give this launch private cursors.
    auto expert_cursor = expert_offsets.narrow(0, 0, experts_per_rank).clone();
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (recv_count + warps_per_block - 1) / warps_per_block;
    build_receive_metadata_kernel<<<blocks, threads, 0, stream.stream()>>>(
        stage_buffer.data_ptr<uint8_t>(),
        src_ranks.data_ptr<int64_t>(),
        src_tokens.data_ptr<int64_t>(),
        local_topk_idx.data_ptr<int64_t>(),
        expanded_slots.data_ptr<int64_t>(),
        reinterpret_cast<unsigned long long*>(expert_cursor.data_ptr<int64_t>()),
        recv_count,
        row_bytes,
        rank,
        experts_per_rank,
        num_topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {src_ranks, src_tokens, local_topk_idx, expanded_slots};
}

DispatchState dispatch_main(PeerRuntime& runtime,
                            const torch::Tensor& x,                       // [T, H]
                            const torch::Tensor& indices,                 // [T, K] routing ids, -1 means masked
                            const c10::optional<torch::Tensor>& weights,  // [T, K] routing weights
                            int64_t num_experts,
                            int64_t max_tokens,
                            int64_t alignment,
                            const std::vector<torch::Tensor>& cached_meta_data,
                            bool allow_multiple_reduction) {
    TORCH_CHECK(alignment > 0, "alignment must be positive");
    const auto window = runtime.local_window();
    const int world_size = runtime.world_size();
    const bool fresh = cached_meta_data.empty();
    TORCH_CHECK(fresh || cached_meta_data.size() == 9, "cached metadata must contain nine prepared tensors from dispatch_epilogue");
    TORCH_CHECK(world_size <= 32 && (world_size & (world_size - 1)) == 0, "dispatch world_size must be one of {1,2,4,8,16,32}");
    const auto& routing_input = fresh ? indices : cached_meta_data[8];
    check_payload_inputs(window, x, routing_input, weights, world_size, num_experts, max_tokens);
    const int64_t num_topk = routing_input.size(1);
    const auto window_layout = WindowLayout::make(world_size, max_tokens, x.size(1), num_topk, num_experts, allow_multiple_reduction);
    TORCH_CHECK(window_layout.num_bytes <= window.numel(), "runtime window is smaller than the payload layout");
    runtime.peer_addresses();

    c10::cuda::CUDAGuard guard(window.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto stream_handle = reinterpret_cast<int64_t>(stream.stream());
    const auto route_indices = fresh ? routing_input.clone() : routing_input;

    if (!fresh) {
        const int64_t recv_count = cached_meta_data[0].numel();
        const int64_t local_experts = num_experts / world_size;
        TORCH_CHECK(recv_count <= world_size * max_tokens, "cached receive count exceeds capacity");
        check_cached_tensor(cached_meta_data[0], x, {recv_count}, "src_ranks");
        check_cached_tensor(cached_meta_data[1], x, {recv_count}, "src_tokens");
        check_cached_tensor(cached_meta_data[2], x, {recv_count, num_topk}, "local_topk_idx");
        check_cached_tensor(cached_meta_data[3], x, {local_experts}, "expert_counts");
        check_cached_tensor(cached_meta_data[4], x, {local_experts + 1}, "expert_offsets");
        check_cached_tensor(cached_meta_data[5], x, {recv_count, num_topk}, "expanded_slots");
        check_cached_tensor(cached_meta_data[6], x, {x.size(0), world_size}, "dst_slots");
        check_cached_tensor(cached_meta_data[7], x, {world_size}, "send_counts");
    }

    // Clear the recv_next counters
    if (fresh) {
        C10_CUDA_CHECK(
            cudaMemsetAsync(window.data_ptr<uint8_t>() + window_layout.recv_next_offset, 0, sizeof(unsigned long long), stream.stream()));
    }

    // Every rank finishes the previous operation before counts/slots are reused.
    runtime.barrier(stream_handle);

    torch::Tensor send_counts;
    torch::Tensor dst_slots;
    torch::Tensor recv_counts;
    torch::Tensor recv_expert_counts;
    torch::Tensor expert_offsets;

    if (fresh) {
        auto counts = route_counts(route_indices, num_experts, world_size);
        send_counts = counts[0];         // GPU int64 [R]
        auto expert_counts = counts[1];  // GPU int64 [E]
        auto notified_counts = notify_counts(runtime, send_counts, expert_counts, window_layout.count_offset);
        recv_counts = notified_counts[0];         // GPU int64 [R]
        recv_expert_counts = notified_counts[1];  // GPU int64 [E/R]
        dst_slots = reserve_and_put(runtime, x, route_indices, weights, num_experts, max_tokens, window_layout);

        const int64_t local_experts = recv_expert_counts.numel();
        // padded counts for each local expert
        auto padded_counts = torch::floor_divide(recv_expert_counts + alignment - 1, alignment) * alignment;
        // [L+1] L is the number of local experts
        expert_offsets = torch::zeros({local_experts + 1}, recv_expert_counts.options());
        expert_offsets.narrow(0, 1, local_experts).copy_(padded_counts.cumsum(0));

    } else {
        send_counts = cached_meta_data[7];
        dst_slots = cached_meta_data[6];
        recv_expert_counts = cached_meta_data[3];
        expert_offsets = cached_meta_data[4];
        reserve_and_put(runtime, x, route_indices, weights, num_experts, max_tokens, window_layout, cached_meta_data[6]);
    }

    // Keep all payload writes above this barrier; the epilogue reads completed data.
    runtime.barrier(stream_handle);

    const int64_t recv_count = fresh ? recv_counts.sum().item<int64_t>() : cached_meta_data[0].numel();
    const int64_t local_experts = num_experts / world_size;
    const int64_t expanded_capacity = expert_offsets[local_experts].item<int64_t>();
    const int64_t row_bytes = WindowLayout::payload_row_bytes(x.size(1), num_topk);

    auto stage_buffer =
        runtime.local_window().narrow(0, window_layout.payload_offset, recv_count * row_bytes).view({recv_count, row_bytes});

    std::vector<torch::Tensor> metadata;
    if (fresh) {
        metadata = {
            recv_expert_counts,  // 0: expert_counts      [L]
            expert_offsets,      // 1: expert_offsets     [L+1]
            dst_slots,           // 2: dst_slots          [T,R]
            send_counts,         // 3: send_counts        [R]
            route_indices,       // 4: source snapshot    [T,K]
        };
    } else {
        metadata = cached_meta_data;
    }

    auto counts = torch::tensor({recv_count, expanded_capacity}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    return DispatchState{
        std::move(metadata),
        counts,
        stage_buffer,
        x.size(1),
        x.scalar_type(),
        runtime.rank(),
        !fresh,
    };
}

std::vector<torch::Tensor> dispatch_epilogue(DispatchState& state, bool expand, bool with_weights, bool zero_padding) {
    TORCH_CHECK(state.metadata.size() == (state.metadata_prepared ? 9 : 5) && state.stage_buffer.is_cuda(),
                "dispatch_epilogue requires a native dispatch state");
    TORCH_CHECK(state.x_dtype == torch::kBFloat16, "dispatch output dtype must be bfloat16");
    TORCH_CHECK(state.hidden > 0 && state.hidden % WindowLayout::kHiddenAlignment == 0,
                "dispatch hidden must be a positive multiple of 256");
    c10::cuda::CUDAGuard guard(state.stage_buffer.device());
    if (!state.metadata_prepared) {
        const int64_t local_experts = state.metadata[0].numel();
        const int64_t num_topk = state.metadata[4].size(1);
        auto receive = build_receive_metadata(state.stage_buffer, state.metadata[1], state.rank, local_experts, num_topk);
        state.metadata = {
            receive[0],         // src_ranks
            receive[1],         // src_tokens
            receive[2],         // local_topk_idx
            state.metadata[0],  // expert_counts
            state.metadata[1],  // expert_offsets
            receive[3],         // expanded_slots
            state.metadata[2],  // dst_slots
            state.metadata[3],  // send_counts
            state.metadata[4],  // source_topk_snapshot
        };
        state.metadata_prepared = true;
    }

    const auto* counts = state.counts.data_ptr<int64_t>();  // CPU
    const int64_t N = counts[0];
    const int64_t C = counts[1];
    const int64_t H = state.hidden;
    const int64_t K = state.metadata[8].size(1);

    torch::Tensor x;
    const auto x_options = state.stage_buffer.options().dtype(state.x_dtype);
    const auto weight_options = state.stage_buffer.options().dtype(torch::kFloat32);
    auto local_indices = torch::empty({0}, state.metadata[2].options());
    auto weights = torch::empty({0}, weight_options);

    if (expand) {
        x = zero_padding ? torch::zeros({C, H}, x_options) : torch::empty({C, H}, x_options);
        if (with_weights) {
            weights = zero_padding ? torch::zeros({C}, weight_options) : torch::empty({C}, weight_options);
        }
    } else {
        x = torch::empty({N, H}, x_options);
        local_indices = torch::empty({N, K}, state.metadata[2].options());
        if (with_weights) {
            weights = torch::empty({N, K}, weight_options);
        }
    }

    if (N == 0)
        return {x, local_indices, weights};

    const auto stream = c10::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (N + warps_per_block - 1) / warps_per_block;

    using scalar_t = at::BFloat16;

    const auto launch = [&]<bool kExpand, bool kWithWeights>() {
        dispatch_copy_epilogue_kernel<scalar_t, kExpand, kWithWeights>
            <<<blocks, threads, 0, stream.stream()>>>(state.stage_buffer.data_ptr<uint8_t>(),
                                                      state.metadata[2].data_ptr<int64_t>(),
                                                      state.metadata[5].data_ptr<int64_t>(),
                                                      x.data_ptr<scalar_t>(),
                                                      kExpand ? nullptr : local_indices.data_ptr<int64_t>(),
                                                      kWithWeights ? weights.data_ptr<float>() : nullptr,
                                                      N,
                                                      state.stage_buffer.size(1),
                                                      H,
                                                      K);
    };
    if (expand) {
        if (with_weights)
            launch.template operator()<true, true>();
        else
            launch.template operator()<true, false>();
    } else {
        if (with_weights)
            launch.template operator()<false, true>();
        else
            launch.template operator()<false, false>();
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {x, local_indices, weights};
}

torch::Tensor combine_main(PeerRuntime& runtime,
                           const torch::Tensor& x,  // compact [N,H] or expanded [C,H]
                           const std::vector<torch::Tensor>& metadata,
                           const c10::optional<torch::Tensor>& weights,  // compact [N,K] or expanded [C], optional
                           int64_t max_tokens,
                           bool expand,
                           bool allow_multiple_reduction,
                           const CombineMainSpecialization& specialization) {
    // metadata[0]  src_ranks           [N]
    // metadata[1]  src_tokens          [N]
    // metadata[2]  local_topk_idx      [N,K]
    // metadata[3]  expert_counts       [L]
    // metadata[4]  expert_offsets      [L+1]
    // metadata[5]  expanded_slots      [N,K]
    // metadata[6]  dst_slots           [T,R]
    // metadata[7]  send_counts         [R]
    // metadata[8]  source_topk_snapshot [T,K]

    const auto window = runtime.local_window();
    const int world_size = runtime.world_size();
    TORCH_CHECK(metadata.size() == 9, "combine requires nine prepared metadata tensors from dispatch_epilogue");
    TORCH_CHECK(x.device() == window.device() && x.dim() == 2 && x.is_contiguous() && x.size(1) > 0,
                "combine x must be contiguous CUDA [N,H] or [C,H] on the runtime device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "combine x must be bfloat16");
    TORCH_CHECK(x.size(1) % WindowLayout::kHiddenAlignment == 0, "combine hidden must be a positive multiple of 256");
    TORCH_CHECK(metadata[8].dim() == 2 && metadata[3].dim() == 1 && metadata[3].numel() > 0,
                "combine metadata requires a [T,K] source snapshot and [L] expert counts");

    const int64_t recv_count = metadata[0].numel();
    const int64_t source_tokens = metadata[8].size(0);
    const int64_t hidden = x.size(1);
    const int64_t num_topk = metadata[8].size(1);
    const int64_t local_experts = metadata[3].numel();
    const int64_t num_experts = local_experts * world_size;
    TORCH_CHECK(hidden == specialization.hidden && num_topk == specialization.topk && world_size == specialization.world_size &&
                    local_experts == specialization.experts_per_rank && max_tokens == specialization.max_tokens,
                "combine main inputs do not match the compiled geometry");
    const auto layout = WindowLayout::make(world_size, max_tokens, hidden, num_topk, num_experts, allow_multiple_reduction);
    TORCH_CHECK(num_topk <= num_experts && source_tokens <= max_tokens && recv_count <= world_size * max_tokens,
                "combine routing geometry exceeds expert/token capacity");
    TORCH_CHECK(layout.num_bytes <= window.numel(), "runtime window is smaller than the combine layout");
    check_cached_tensor(metadata[0], x, {recv_count}, "src_ranks");
    check_cached_tensor(metadata[1], x, {recv_count}, "src_tokens");
    check_cached_tensor(metadata[2], x, {recv_count, num_topk}, "local_topk_idx");
    check_cached_tensor(metadata[3], x, {local_experts}, "expert_counts");
    check_cached_tensor(metadata[4], x, {local_experts + 1}, "expert_offsets");
    check_cached_tensor(metadata[5], x, {recv_count, num_topk}, "expanded_slots");
    check_cached_tensor(metadata[6], x, {source_tokens, world_size}, "dst_slots");
    check_cached_tensor(metadata[7], x, {world_size}, "send_counts");
    check_cached_tensor(metadata[8], x, {source_tokens, num_topk}, "source_topk_snapshot");
    TORCH_CHECK(expand ? (x.size(0) >= recv_count && (recv_count != 0 || x.size(0) == 0)) : x.size(0) == recv_count,
                "combine x rows do not match the receive layout");
    if (weights.has_value()) {
        const bool shape_matches = expand ? (weights->dim() == 1 && weights->size(0) == x.size(0))
                                          : (weights->dim() == 2 && weights->size(0) == recv_count && weights->size(1) == num_topk);
        TORCH_CHECK(
            weights->device() == x.device() && weights->scalar_type() == torch::kFloat32 && weights->is_contiguous() && shape_matches,
            "combine weights must be contiguous CUDA float32 [N,K] or expanded [C] on the x device");
    }

    c10::cuda::CUDAGuard guard(window.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto stream_handle = reinterpret_cast<int64_t>(stream.stream());
    const auto addresses = runtime.peer_addresses();
    const int64_t row_bytes = WindowLayout::return_row_bytes(hidden, num_topk);
    auto returned =
        window.narrow(0, layout.return_offset, layout.return_capacity * row_bytes).view({layout.return_partitions, max_tokens, row_bytes});

    runtime.barrier(stream_handle);

    specialization.launch(addresses, x, metadata, weights, runtime.rank(), layout, expand, allow_multiple_reduction, stream.stream());

    runtime.barrier(stream_handle);

    return returned;
}

}  // namespace nano_ep
