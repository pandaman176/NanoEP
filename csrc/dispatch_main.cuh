#pragma once

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdio>
#include <cuda/atomic>
#include <limits>
#include <nano_ep/barrier.cuh>
#include <nano_ep/math.cuh>

#include "layout.hpp"

namespace nano_ep {

// All notify threads publish before reusing the shared buffer for received counts.
// Return only after every received count is visible to the notify group.
template <int kNumThreads, int kWorldSize, int kExpertsPerRank, uint64_t kTimeoutCycles>
__device__ __forceinline__ void notify_peer_counts(const int64_t* peer_addresses,
                                                   int64_t* window_counts,
                                                   uint32_t* counts,  // [R+E], outgoing then incoming
                                                   int rank,
                                                   int64_t count_offset) {
    using Count = cuda::atomic_ref<int64_t, cuda::thread_scope_system>;
    constexpr int kNumExperts = kWorldSize * kExpertsPerRank;
    constexpr int64_t kNumCounts = kWorldSize + kNumExperts;

    // Publish every outgoing entry before entering any receive spin loop.
    for (int64_t i = threadIdx.x; i < kNumCounts; i += kNumThreads) {
        const bool rank_count = i < kWorldSize;
        const int64_t expert = i - kWorldSize;
        const int destination = rank_count ? static_cast<int>(i) : static_cast<int>(expert / kExpertsPerRank);
        const int64_t dst_index = rank_count ? rank : kWorldSize + rank * kExpertsPerRank + expert % kExpertsPerRank;
        auto* peer_window = reinterpret_cast<uint8_t*>(peer_addresses[destination]);
        auto* peer_counts = reinterpret_cast<int64_t*>(peer_window + count_offset);
        Count outgoing(peer_counts[dst_index]);
        outgoing.store(static_cast<int64_t>(counts[i]), cuda::memory_order_release);
    }
    notify_group_barrier<kNumThreads>();

    const auto start = clock64();
    for (int64_t i = threadIdx.x; i < kNumCounts; i += kNumThreads) {
        Count incoming(window_counts[i]);
        while (true) {
            const auto count = incoming.load(cuda::memory_order_acquire);
            if (count >= 0) {
                counts[i] = static_cast<uint32_t>(count);
                incoming.store(WindowLayout::kCountNotReady, cuda::memory_order_relaxed);
                break;
            }
            if (clock64() - start > kTimeoutCycles) {
                printf("NanoEP notify timeout: rank=%d index=%lld count=%lld\n",
                       rank,
                       static_cast<long long>(i),
                       static_cast<long long>(count));
                asm volatile("trap;");
            }
            __nanosleep(64);
        }
    }
    notify_group_barrier<kNumThreads>();
}

// Counts are source-major after notification. Keep them read-only while the
// first two notify warps independently build rank and expert prefix sums.
template <int kWorldSize, int kExpertsPerRank, int64_t kExpertAlignment>
__device__ __forceinline__ void finalize_receive_counts(const uint32_t* counts,
                                                        int64_t* recv_rank_prefix,  // [R], inclusive
                                                        int64_t* expert_counts,     // [L], actual assignments
                                                        int64_t* expert_offsets     // [L+1], padded exclusive
) {
    static_assert(kExpertAlignment > 0);
    constexpr int kWarpSize = 32;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    if (warp == 0) {
        const int64_t count = lane < kWorldSize ? counts[lane] : 0;
        const int64_t prefix = warp_inclusive_sum(count, lane);
        if (lane < kWorldSize)
            recv_rank_prefix[lane] = prefix;
    } else if (warp == 1) {
        int64_t carry = 0;
        if (lane == 0)
            expert_offsets[0] = 0;
        for (int64_t base = 0; base < kExpertsPerRank; base += kWarpSize) {
            const int64_t expert = base + lane;
            int64_t actual = 0;
            if (expert < kExpertsPerRank) {
#pragma unroll
                for (int source = 0; source < kWorldSize; ++source)
                    actual += counts[kWorldSize + source * kExpertsPerRank + expert];
                expert_counts[expert] = actual;
            }
            const int64_t padded = (actual + kExpertAlignment - 1) / kExpertAlignment * kExpertAlignment;
            const int64_t prefix = carry + warp_inclusive_sum(padded, lane);
            if (expert < kExpertsPerRank)
                expert_offsets[expert + 1] = prefix;
            carry = __shfl_sync(0xffffffffu, prefix, kWarpSize - 1);
        }
    }
}

template <int kNumNotifyWarps, int kWorldSize, int kExpertsPerRank, int kTopk, int64_t kExpertAlignment, uint64_t kTimeoutCycles>
__device__ __forceinline__ void count_and_notify(const int64_t* peer_addresses,
                                                 const int64_t* indices,
                                                 uint32_t* local_counts,
                                                 uint64_t* reduction,
                                                 int64_t* send_counts,
                                                 int64_t* recv_rank_prefix,
                                                 int64_t* recv_expert_counts,
                                                 int64_t* expert_offsets,
                                                 int rank,
                                                 int64_t num_tokens,
                                                 int64_t count_offset) {
    constexpr int kWarpSize = 32;
    constexpr int kNumNotifyThreads = kNumNotifyWarps * kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_idx = threadIdx.x / kWarpSize;
    constexpr int kNumExperts = kWorldSize * kExpertsPerRank;
    constexpr int64_t kNumCounts = kWorldSize + kNumExperts;
    for (int64_t i = threadIdx.x; i < kNumCounts; i += kNumNotifyThreads)
        local_counts[i] = 0;
    notify_group_barrier<kNumNotifyThreads>();

    // calculate the expert counts and send counts for each destination rank.
    const int64_t token_start = static_cast<int64_t>(warp_idx) * gridDim.x + blockIdx.x;
    const int64_t token_stride = static_cast<int64_t>(kNumNotifyWarps) * gridDim.x;
    for (int64_t token = token_start; token < num_tokens; token += token_stride) {
        int destination = -1;
        if (lane < kTopk) {
            const int64_t expert = indices[token * kTopk + lane];
            if (expert < -1 || expert >= kNumExperts) {
                printf("NanoEP notify invalid expert: token=%lld expert=%lld\n",
                       static_cast<long long>(token),
                       static_cast<long long>(expert));
                asm volatile("trap;");
                return;
            }
            if (expert >= 0) {
                atomicAdd_block(local_counts + kWorldSize + expert, 1U);
                destination = expert / kExpertsPerRank;
            }
        }
        const unsigned same_destination = __match_any_sync(0xffffffffu, destination);
        // leader is the greatest lane with the same destination
        const int leader = 31 - __clz(same_destination);
        if (destination >= 0 && lane == leader)
            atomicAdd_block(local_counts + destination, 1U);
    }
    notify_group_barrier<kNumNotifyThreads>();

    using Reduction = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>;
    // Every CTA contributes once per entry, including zero-count CTAs.
    for (int64_t i = threadIdx.x; i < kNumCounts; i += kNumNotifyThreads) {
        Reduction status(reduction[i]);
        const uint64_t contribution = (uint64_t{1} << 32) | local_counts[i];
        status.fetch_add(contribution, cuda::memory_order_release);
    }
    if (blockIdx.x == 0) {
        // CTA 0 is responsible for waiting for all contributions and exchanging counts with peers.
        const auto start = clock64();
        for (int64_t i = threadIdx.x; i < kNumCounts; i += kNumNotifyThreads) {
            Reduction status(reduction[i]);
            while (true) {
                const auto observed = status.load(cuda::memory_order_acquire);
                if ((observed >> 32) == gridDim.x) {
                    local_counts[i] = static_cast<uint32_t>(observed);
                    if (i < kWorldSize)
                        send_counts[i] = local_counts[i];
                    // All contributors finished this entry; reset it for the next dispatch.
                    status.store(0, cuda::memory_order_relaxed);
                    break;
                }
                if (clock64() - start > kTimeoutCycles) {
                    printf("NanoEP notify reduction timeout: rank=%d index=%lld arrived=%llu expected=%u\n",
                           rank,
                           static_cast<long long>(i),
                           static_cast<unsigned long long>(observed >> 32),
                           gridDim.x);
                    asm volatile("trap;");
                }
                __nanosleep(64);
            }
        }
        notify_group_barrier<kNumNotifyThreads>();

        auto* local_window = reinterpret_cast<uint8_t*>(peer_addresses[rank]);
        notify_peer_counts<kNumNotifyThreads, kWorldSize, kExpertsPerRank, kTimeoutCycles>(
            peer_addresses, reinterpret_cast<int64_t*>(local_window + count_offset), local_counts, rank, count_offset);

        // calculate expert_offsets and recv_expert_counts
        finalize_receive_counts<kWorldSize, kExpertsPerRank, kExpertAlignment>(
            local_counts, recv_rank_prefix, recv_expert_counts, expert_offsets);
    }
}

template <typename data_t, int kWorldSize, int kExpertsPerRank, bool kReuseSlots, int kNumNotifyWarps>
__device__ __forceinline__ void dispatch_payload_rows(const int64_t* peer_addresses,
                                                      const data_t* x,
                                                      const int64_t* indices,
                                                      const float* weights,
                                                      int64_t* dst_slots,
                                                      int rank,
                                                      int64_t num_tokens,
                                                      int64_t hidden,
                                                      int64_t num_topk,
                                                      int64_t max_tokens,
                                                      int64_t recv_next_offset,
                                                      int64_t payload_offset,
                                                      int64_t payload_row_bytes) {
    constexpr int kWarpSize = 32;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_idx = threadIdx.x / kWarpSize;
    // reserve slots and write payloads to the peer windows.
    const int64_t token_start = static_cast<int64_t>(warp_idx - kNumNotifyWarps) * gridDim.x + blockIdx.x;
    const int64_t token_stride = static_cast<int64_t>(blockDim.x / kWarpSize - kNumNotifyWarps) * gridDim.x;

    // Warps process disjoint token sequences across the SM-sized grid.
    for (int64_t token = token_start; token < num_tokens; token += token_stride) {
        // Fist K lane handle one routing
        // Expert[0, 1, 2, 3, 4, 5, 6, 7] -> Rank[0, 0, 1, 1, 2, 2, 3, 3]
        int destination = -1;
        if (lane < num_topk) {
            const int64_t expert_id = indices[token * num_topk + lane];
            if (expert_id < -1 || expert_id >= kWorldSize * kExpertsPerRank) {
                printf("NanoEP dispatch invalid expert: token=%lld expert=%lld\n",
                       static_cast<long long>(token),
                       static_cast<long long>(expert_id));
                asm volatile("trap;");
                return;
            }
            if (expert_id >= 0) {
                destination = expert_id / kExpertsPerRank;
            }
        }

        // e.g. for lane 0, same_destination = 0x00000003
        const unsigned same_destination = __match_any_sync(0xffffffffu, destination);

        const int leader = 31 - __clz(same_destination);
        const bool is_sender = (destination >= 0 && lane == leader);

        unsigned long long slot = 0;
        if (is_sender) {
            if constexpr (kReuseSlots) {
                slot = static_cast<unsigned long long>(dst_slots[token * kWorldSize + destination]);

            } else {
                auto* target_window = reinterpret_cast<uint8_t*>(peer_addresses[destination]);
                auto* recv_next = reinterpret_cast<unsigned long long*>(target_window + recv_next_offset);
                using Counter = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;
                Counter counter(*recv_next);
                slot = counter.fetch_add(1ULL, cuda::memory_order_relaxed);
                dst_slots[token * kWorldSize + destination] = static_cast<int64_t>(slot);
            }

            if (!(slot < static_cast<unsigned long long>(kWorldSize) * max_tokens)) {
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
            for (int64_t k = lane; k < num_topk; k += kWarpSize) {
                row_topk[k] = indices[token * num_topk + k];
                if (weights != nullptr) {
                    row_weights[k] = weights[token * num_topk + k];
                }
            }
            for (int64_t h = lane; h < hidden; h += kWarpSize) {
                row_x[h] = x[token * hidden + h];
            }

            // clear the first sender
            senders &= senders - 1;
        }
    }
}

template <int kHidden,
          int kTopk,
          int kWorldSize,
          int kExpertsPerRank,
          int64_t kMaxTokens,
          int64_t kExpertAlignment,
          bool kReuseSlots,
          uint64_t kTimeoutCycles = kBarrierTimeoutCycles>
__global__ void dispatch_main_kernel(const int64_t* peer_addresses,
                                     const at::BFloat16* x,        // [T,H]
                                     const int64_t* indices,       // [T,K]
                                     const float* weights,         // [T,K], or nullptr
                                     int64_t* dst_slots,           // [T,R], fresh output or cached mapping
                                     int64_t* send_counts,         // [R], fresh only
                                     uint64_t* reduction,          // [R+E], fresh only
                                     int64_t* recv_rank_prefix,    // [R], fresh inclusive prefix
                                     int64_t* recv_expert_counts,  // [L], fresh actual counts
                                     int64_t* expert_offsets,      // [L+1], fresh padded exclusive prefix
                                     int rank,
                                     int64_t num_tokens) {
    constexpr int kNumNotifyWarps = kReuseSlots ? 0 : 4;
    constexpr int kNumExperts = kWorldSize * kExpertsPerRank;
    constexpr auto kLayout = WindowLayout::make(kWorldSize, kMaxTokens, kHidden, kTopk, kNumExperts);
    constexpr int64_t kRowBytes = WindowLayout::payload_row_bytes(kHidden, kTopk);
    static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0);
    static_assert(kTopk > 0 && kTopk <= 32 && kTopk <= kNumExperts);
    static_assert(kWorldSize > 0 && kWorldSize <= 32 && (kWorldSize & (kWorldSize - 1)) == 0);
    static_assert(kExpertsPerRank > 0 && kMaxTokens > 0 && kExpertAlignment > 0);

    const int warp_idx = threadIdx.x / 32;
    if constexpr (!kReuseSlots) {
        __shared__ uint32_t local_counts[kWorldSize + kNumExperts];
        if (warp_idx < kNumNotifyWarps) {
            count_and_notify<kNumNotifyWarps, kWorldSize, kExpertsPerRank, kTopk, kExpertAlignment, kTimeoutCycles>(peer_addresses,
                                                                                                                    indices,
                                                                                                                    local_counts,
                                                                                                                    reduction,
                                                                                                                    send_counts,
                                                                                                                    recv_rank_prefix,
                                                                                                                    recv_expert_counts,
                                                                                                                    expert_offsets,
                                                                                                                    rank,
                                                                                                                    num_tokens,
                                                                                                                    kLayout.count_offset);
        }
    }
    if (warp_idx >= kNumNotifyWarps) {
        dispatch_payload_rows<at::BFloat16, kWorldSize, kExpertsPerRank, kReuseSlots, kNumNotifyWarps>(peer_addresses,
                                                                                                       x,
                                                                                                       indices,
                                                                                                       weights,
                                                                                                       dst_slots,
                                                                                                       rank,
                                                                                                       num_tokens,
                                                                                                       kHidden,
                                                                                                       kTopk,
                                                                                                       kMaxTokens,
                                                                                                       kLayout.recv_next_offset,
                                                                                                       kLayout.payload_offset,
                                                                                                       kRowBytes);
    }
}

template <int kHidden, int kTopk, int kWorldSize, int kExpertsPerRank, int64_t kMaxTokens, int64_t kExpertAlignment>
void prepare_dispatch_launch(const torch::Tensor& x) {
    TORCH_CHECK(x.size(0) <= std::numeric_limits<uint32_t>::max(), "fresh dispatch token count exceeds packed notify count capacity");
    // Each specialization has fixed shared-memory and register requirements.
    static thread_local int checked_device = -1;
    if (checked_device != x.get_device()) {
        int supported = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, x.get_device()));
        TORCH_CHECK(supported, "fresh dispatch requires CUDA cooperative launch support");
        const auto kernel = dispatch_main_kernel<kHidden, kTopk, kWorldSize, kExpertsPerRank, kMaxTokens, kExpertAlignment, false>;
        int active_blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, 256, 0));
        TORCH_CHECK(active_blocks_per_sm >= 1, "fresh dispatch cannot fit a 256-thread cooperative block on an SM");
        checked_device = x.get_device();
    }
}

template <int kHidden, int kTopk, int kWorldSize, int kExpertsPerRank, int64_t kMaxTokens, int64_t kExpertAlignment>
void launch_dispatch(const torch::Tensor& addresses,
                     const torch::Tensor& window,
                     const torch::Tensor& x,
                     const torch::Tensor& indices,
                     const c10::optional<torch::Tensor>& weights,
                     const torch::Tensor& dst_slots,
                     const torch::Tensor& send_counts,
                     const torch::Tensor& recv_rank_prefix,
                     const torch::Tensor& recv_expert_counts,
                     const torch::Tensor& expert_offsets,
                     int rank,
                     bool cached_mode,
                     cudaStream_t stream) {
    const auto launch = [&]<bool kReuseSlots>() {
        const auto kernel = dispatch_main_kernel<kHidden, kTopk, kWorldSize, kExpertsPerRank, kMaxTokens, kExpertAlignment, kReuseSlots>;
        const auto* address_ptr = addresses.data_ptr<int64_t>();
        const auto* x_ptr = x.data_ptr<at::BFloat16>();
        const auto* indices_ptr = indices.data_ptr<int64_t>();
        const auto* weight_ptr = weights.has_value() ? weights->data_ptr<float>() : nullptr;
        auto* slots_ptr = dst_slots.data_ptr<int64_t>();
        auto* send_ptr = kReuseSlots ? nullptr : send_counts.data_ptr<int64_t>();
        constexpr auto kLayout = WindowLayout::make(kWorldSize, kMaxTokens, kHidden, kTopk, kWorldSize * kExpertsPerRank);
        auto* reduction_ptr =
            kReuseSlots ? nullptr : reinterpret_cast<uint64_t*>(window.data_ptr<uint8_t>() + kLayout.notify_reduction_offset);
        auto* recv_ptr = kReuseSlots ? nullptr : recv_rank_prefix.data_ptr<int64_t>();
        auto* expert_ptr = kReuseSlots ? nullptr : recv_expert_counts.data_ptr<int64_t>();
        auto* offsets_ptr = kReuseSlots ? nullptr : expert_offsets.data_ptr<int64_t>();
        int64_t num_tokens = x.size(0);
        void* args[] = {&address_ptr,
                        &x_ptr,
                        &indices_ptr,
                        &weight_ptr,
                        &slots_ptr,
                        &send_ptr,
                        &reduction_ptr,
                        &recv_ptr,
                        &expert_ptr,
                        &offsets_ptr,
                        &rank,
                        &num_tokens};
        constexpr int threads = 256;
        const int blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        if constexpr (kReuseSlots) {
            C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(blocks), dim3(threads), args, 0, stream));
        } else {
            C10_CUDA_CHECK(
                cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), dim3(blocks), dim3(threads), args, 0, stream));
        }
    };
    if (cached_mode)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace nano_ep
