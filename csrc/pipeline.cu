#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdio>
#include <cuda/atomic>
#include <limits>
#include <nano_ep/barrier.cuh>
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

struct DispatchMainResult {
    torch::Tensor dst_slots;
    torch::Tensor send_counts;
    torch::Tensor recv_rank_prefix;
    torch::Tensor recv_expert_counts;
    torch::Tensor expert_offsets;
};

DispatchMainResult launch_dispatch_main(PeerRuntime& runtime,
                                        const torch::Tensor& x,
                                        const torch::Tensor& indices,
                                        const c10::optional<torch::Tensor>& weights,
                                        int64_t num_experts,
                                        const c10::optional<torch::Tensor>& cached_dst_slots,
                                        int64_t entry_epoch,
                                        const DispatchMainSpecialization& specialization) {
    const auto window = runtime.local_window();
    int rank = runtime.rank();
    int world_size = runtime.world_size();

    int64_t num_tokens = x.size(0);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto addresses = runtime.peer_addresses();
    const bool cached_mode = cached_dst_slots.has_value();
    auto dst_slots = cached_mode ? *cached_dst_slots : torch::full({num_tokens, world_size}, -1, indices.options());
    auto send_counts = cached_mode ? torch::Tensor() : torch::empty({world_size}, indices.options());
    const int64_t local_experts = num_experts / world_size;
    auto recv_rank_prefix = cached_mode ? torch::Tensor() : torch::empty({world_size}, indices.options());
    auto recv_expert_counts = cached_mode ? torch::Tensor() : torch::empty({local_experts}, indices.options());
    auto expert_offsets = cached_mode ? torch::Tensor() : torch::empty({local_experts + 1}, indices.options());

    specialization.launch(addresses,
                          window,
                          x,
                          indices,
                          weights,
                          dst_slots,
                          send_counts,
                          recv_rank_prefix,
                          recv_expert_counts,
                          expert_offsets,
                          rank,
                          cached_mode,
                          entry_epoch,
                          stream.stream());
    return {dst_slots, send_counts, recv_rank_prefix, recv_expert_counts, expert_offsets};
}

}  // namespace

DispatchState dispatch_main(PeerRuntime& runtime,
                            const torch::Tensor& x,                       // [T, H]
                            const torch::Tensor& indices,                 // [T, K] routing ids, -1 means masked
                            const c10::optional<torch::Tensor>& weights,  // [T, K] routing weights
                            int64_t num_experts,
                            int64_t max_tokens,
                            int64_t alignment,
                            const std::vector<torch::Tensor>& cached_meta_data,
                            int64_t cached_expanded_capacity,
                            bool allow_multiple_reduction,
                            const DispatchMainSpecialization& specialization) {
    TORCH_CHECK(alignment > 0, "alignment must be positive");
    const auto window = runtime.local_window();
    const int world_size = runtime.world_size();
    const bool fresh = cached_meta_data.empty();
    TORCH_CHECK(fresh || cached_meta_data.size() == 9, "cached metadata must contain nine prepared tensors from dispatch_epilogue");
    TORCH_CHECK(world_size <= 32 && (world_size & (world_size - 1)) == 0, "dispatch world_size must be one of {1,2,4,8,16,32}");
    const auto& routing_input = fresh ? indices : cached_meta_data[8];
    check_payload_inputs(window, x, routing_input, weights, world_size, num_experts, max_tokens);
    const int64_t num_topk = routing_input.size(1);
    TORCH_CHECK(specialization.hidden == x.size(1) && specialization.topk == num_topk && specialization.world_size == world_size &&
                    specialization.experts_per_rank == num_experts / world_size && specialization.max_tokens == max_tokens &&
                    specialization.expert_alignment == alignment,
                "dispatch main inputs do not match the compiled geometry");
    const auto window_layout = WindowLayout::make(world_size, max_tokens, x.size(1), num_topk, num_experts, allow_multiple_reduction);
    TORCH_CHECK(window_layout.num_bytes <= window.numel(), "runtime window is smaller than the payload layout");
    runtime.peer_addresses();

    c10::cuda::CUDAGuard guard(window.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto stream_handle = reinterpret_cast<int64_t>(stream.stream());
    const auto route_indices = fresh ? routing_input.clone() : routing_input;
    // Validate specialized cooperative resources before mutating the window.
    specialization.prepare(x, !fresh);

    if (!fresh) {
        TORCH_CHECK(cached_expanded_capacity >= 0, "cached expanded capacity must be nonnegative");
        const int64_t recv_count = cached_meta_data[0].numel();
        const int64_t local_experts = num_experts / world_size;
        TORCH_CHECK(recv_count <= world_size * max_tokens, "cached receive count exceeds capacity");
        check_cached_tensor(cached_meta_data[0], x, {recv_count}, "src_ranks");
        check_cached_tensor(cached_meta_data[1], x, {recv_count}, "src_tokens");
        check_cached_tensor(cached_meta_data[2], x, {recv_count, num_topk}, "local_topk_idx");
        check_cached_tensor(cached_meta_data[3], x, {local_experts}, "expert_counts");
        check_cached_tensor(cached_meta_data[4], x, {local_experts + 1}, "expert_offsets");
        if (cached_meta_data[5].dim() == 1)
            check_cached_tensor(cached_meta_data[5], x, {0}, "expanded_slots");
        else
            check_cached_tensor(cached_meta_data[5], x, {recv_count, num_topk}, "expanded_slots");
        check_cached_tensor(cached_meta_data[6], x, {x.size(0), world_size}, "dst_slots");
        check_cached_tensor(cached_meta_data[7], x, {world_size}, "send_counts");
    }

    const int64_t entry_epoch = runtime.reserve_barrier_epochs(stream_handle, 2);

    torch::Tensor send_counts;
    torch::Tensor dst_slots;
    torch::Tensor recv_rank_prefix;
    torch::Tensor recv_expert_counts;
    torch::Tensor expert_offsets;

    if (fresh) {
        auto result = launch_dispatch_main(runtime, x, route_indices, weights, num_experts, c10::nullopt, entry_epoch, specialization);
        send_counts = std::move(result.send_counts);                // GPU int64 [R]
        dst_slots = std::move(result.dst_slots);                    // GPU int64 [T,R]
        recv_rank_prefix = std::move(result.recv_rank_prefix);      // GPU int64 [R], inclusive by source
        recv_expert_counts = std::move(result.recv_expert_counts);  // GPU int64 [L], actual
        expert_offsets = std::move(result.expert_offsets);          // GPU int64 [L+1], padded exclusive

    } else {
        send_counts = cached_meta_data[7];
        dst_slots = cached_meta_data[6];
        recv_expert_counts = cached_meta_data[3];
        expert_offsets = cached_meta_data[4];
        launch_dispatch_main(runtime, x, route_indices, weights, num_experts, cached_meta_data[6], entry_epoch, specialization);
    }

    const int64_t recv_count = fresh ? recv_rank_prefix[world_size - 1].item<int64_t>() : cached_meta_data[0].numel();
    const int64_t local_experts = num_experts / world_size;
    const int64_t expanded_capacity = fresh ? expert_offsets[local_experts].item<int64_t>() : cached_expanded_capacity;
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
    // metadata[5]  expanded_slots      expanded [N,K], compact [0]
    // metadata[6]  dst_slots           [T,R]
    // metadata[7]  send_counts         [R]
    // metadata[8]  source_topk_snapshot [T,K]

    const auto window = runtime.local_window();
    const int world_size = runtime.world_size();
    TORCH_CHECK(metadata.size() == 9, "combine requires nine prepared metadata tensors from dispatch_epilogue");
    TORCH_CHECK(x.device() == window.device() && x.dim() == 2 && x.is_contiguous() && x.size(1) > 0,
                "combine x must be contiguous CUDA [N,H] or [C,H] on the runtime device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "combine x must be bfloat16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % sizeof(int4) == 0, "combine x data pointer must be 16-byte aligned");
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
    if (expand)
        check_cached_tensor(metadata[5], x, {recv_count, num_topk}, "expanded_slots");
    else
        check_cached_tensor(metadata[5], x, {0}, "expanded_slots");
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

    const int64_t entry_epoch = runtime.reserve_barrier_epochs(stream_handle, 2);

    specialization.launch(
        addresses, x, metadata, weights, runtime.rank(), layout, expand, allow_multiple_reduction, entry_epoch, stream.stream());

    return returned;
}

}  // namespace nano_ep
