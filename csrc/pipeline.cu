#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <limits>
#include <nano_ep/jit.hpp>
#include <nano_ep/pipeline.hpp>
#include <utility>

#include "layout.hpp"

namespace nano_ep {

namespace {

static_assert(WindowLayout::kHiddenAlignment == 32 * sizeof(int4) / sizeof(at::BFloat16));

int resolve_main_sms(int num_sms) {
    const int device_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    TORCH_CHECK(num_sms >= 0 && num_sms <= device_sms, "num_sms must be between 0 and the device SM count (", device_sms, ")");
    return num_sms == 0 ? device_sms : num_sms;
}

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
    TORCH_CHECK(x.numel() == 0 || reinterpret_cast<uintptr_t>(x.data_ptr()) % sizeof(int4) == 0,
                "dispatch x data pointer must be 16-byte aligned");
    TORCH_CHECK(indices.device() == x.device() && indices.scalar_type() == torch::kInt64 && indices.dim() == 2 &&
                    indices.size(0) == x.size(0) && indices.is_contiguous(),
                "indices must be contiguous CUDA int64 [T,K] on the x device");
    TORCH_CHECK(num_experts > 0 && num_experts % world_size == 0, "num_experts must be positive and divisible by world_size");
    TORCH_CHECK(max_tokens <= std::numeric_limits<int32_t>::max(), "max_tokens exceeds int32 metadata range");
    TORCH_CHECK(max_tokens > 0 && x.size(0) <= max_tokens, "source token count exceeds max_tokens");
    if (weights.has_value()) {
        TORCH_CHECK(weights->device() == x.device() && weights->scalar_type() == torch::kFloat32 && weights->sizes() == indices.sizes() &&
                        weights->is_contiguous(),
                    "weights must be contiguous CUDA float32 [T,K] on the x device");
    }
}

void check_cached_tensor(const torch::Tensor& tensor,
                         const torch::Tensor& reference,
                         at::IntArrayRef shape,
                         const char* name,
                         at::ScalarType dtype = torch::kInt64) {
    TORCH_CHECK(tensor.device() == reference.device() && tensor.scalar_type() == dtype && tensor.sizes() == shape && tensor.is_contiguous(),
                "cached ",
                name,
                " has invalid CUDA storage or shape");
}

// All ranks use BF16 x, matching geometry and either fresh or cached mode.
// Specialize H/K/R/experts-per-rank/Tmax/alignment; actual T and rank remain runtime inputs.
// Validate inputs before window mutation or barriers.
// Keep the specialization module alive while its callbacks are in use.
// Fresh cooperative sends split each CTA into four notify and four payload
// warps. Notify uses shared-memory counts and persistent packed CTA reductions;
// CTA 0 consumes/resets the reductions and exchanges counts while payload work
// proceeds. Incoming counts reuse the same shared buffer after all sends.
// CTA 0's warp 0 builds the rank prefix; warp 1 sums experts, aligns and scans
// them directly into actual counts and padded offsets, using int64 accumulators.
// Fresh expanded calls also initialize private expert cursors in the same scan;
// expand must match the following epilogue and the cached metadata layout.
// Both variants launch cooperatively, including T=0 ranks. Cached sends use
// eight payload warps. CTA 0 exchanges entry/exit epochs inside the kernel;
// grid synchronization protects workspace reuse; system release/acquire publishes payloads.
// Fresh counts fit uint32; rank/expert counts require per-CTA shared memory.
// Launch resource checks precede window mutation and the entry barrier.
// Local storage checks precede window mutation and the entry barrier. Routing
// values are a caller precondition: -1 or distinct valid experts per token.
// Existing kernels assert bounds/uniqueness on device, without a validation
// launch or status readback. Device failures are asynchronous and can invalidate
// the CUDA context. Fresh N/C are published to mapped host scalars by notify
// warps; CPU polls them with the GIL released before allocating exact shapes.
// Count readiness does not imply payload or GPU metadata completion; enqueue
// the epilogue on the same stream. Cached calls do not touch the host mailbox.
// Cached metadata is the unmodified vector prepared by dispatch_epilogue on this
// runtime with the same H/K/E/capacity/alignment; the Python handle checks this
// ownership/configuration contract. cached_expanded_capacity is the host capacity
// saved with that metadata; fresh calls ignore it. Cached N comes from tensor shape.
// A local failure is not a collective error
// exchange: the distributed launcher must terminate peers on rank failure.
// Pass the allocation's allow_multiple_reduction mode even though dispatch
// payloads are unchanged: the total window capacity check includes returns.
DispatchState dispatch_main(PeerRuntime& runtime,
                            const torch::Tensor& x,                       // [T, H]
                            const torch::Tensor& indices,                 // [T, K] routing ids, -1 means masked
                            const c10::optional<torch::Tensor>& weights,  // [T, K] routing weights
                            int64_t num_experts,
                            int64_t max_tokens,
                            const std::vector<torch::Tensor>& cached_meta_data,
                            int64_t cached_expanded_capacity,
                            bool allow_multiple_reduction,
                            bool expand,
                            int num_sms,
                            const JitKernel& kernel,
                            const torch::Tensor& window,
                            int64_t consumer_stream) {
    const int world_size = runtime.world_size();
    const bool fresh = cached_meta_data.empty();
    const auto& routing_input = fresh ? indices : cached_meta_data[5];
    const int64_t num_topk = routing_input.size(1);
    const int64_t local_experts = num_experts / world_size;
    const auto window_layout = WindowLayout::make(world_size, max_tokens, x.size(1), num_topk, num_experts, allow_multiple_reduction);
    TORCH_CHECK(window_layout.num_bytes <= window.numel(), "runtime window is smaller than the payload layout");
    const auto addresses = runtime.peer_addresses();

    c10::cuda::CUDAGuard guard(window.device());
    const int main_sms = resolve_main_sms(num_sms);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto stream_handle = reinterpret_cast<int64_t>(stream.stream());
    if (fresh) {
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "Fresh dispatch with host count polling cannot be CUDA Graph captured");
    }
    torch::Tensor topk_snapshot;
    {
        c10::cuda::OptionalCUDAStreamGuard allocation_guard;
        if (consumer_stream != -1)
            allocation_guard.reset_stream(
                c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(consumer_stream), x.get_device()));
        topk_snapshot = fresh ? torch::empty_like(routing_input) : routing_input;
    }

    if (!fresh) {
        const int64_t recv_count = cached_meta_data[0].size(0);
        check_cached_tensor(cached_meta_data[0], x, {recv_count, num_topk + 2}, "recv_metadata", torch::kInt32);
        check_cached_tensor(cached_meta_data[1], x, {local_experts}, "expert_counts");
        check_cached_tensor(cached_meta_data[2], x, {local_experts + 1}, "expert_offsets");
        check_cached_tensor(cached_meta_data[3], x, {x.size(0), world_size}, "dst_slots");
        check_cached_tensor(cached_meta_data[4], x, {world_size}, "send_counts");
        check_cached_tensor(cached_meta_data[6], x, {world_size}, "recv_rank_prefix");
    }

    const int64_t entry_epoch = runtime.reserve_barrier_epochs(stream_handle, 2);

    torch::Tensor send_counts;
    torch::Tensor dst_slots;
    torch::Tensor recv_rank_prefix;
    torch::Tensor recv_expert_counts;
    torch::Tensor expert_offsets;
    torch::Tensor expert_cursor;

    if (fresh) {
        c10::cuda::OptionalCUDAStreamGuard allocation_guard;
        if (consumer_stream != -1)
            allocation_guard.reset_stream(
                c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(consumer_stream), x.get_device()));
        const auto options = routing_input.options();
        dst_slots = torch::empty({x.size(0), world_size}, options);   // GPU int64 [T,R]
        send_counts = torch::empty({world_size}, options);            // GPU int64 [R], by destination
        recv_rank_prefix = torch::empty({world_size}, options);       // GPU int64 [R], inclusive by source
        recv_expert_counts = torch::empty({local_experts}, options);  // GPU int64 [L], actual
        expert_offsets = torch::empty({local_experts + 1}, options);  // GPU int64 [L+1], padded exclusive
        if (expand)
            expert_cursor = torch::empty({local_experts}, options);  // GPU int64 [L], mutable reservation cursor
    } else {
        send_counts = cached_meta_data[4];
        dst_slots = cached_meta_data[3];
        recv_expert_counts = cached_meta_data[1];
        expert_offsets = cached_meta_data[2];
        recv_rank_prefix = cached_meta_data[6];
    }

    auto* mapped_counts = fresh ? runtime.prepare_dispatch_counts() : nullptr;
    try {
        const auto* addresses_ptr = addresses.data_ptr<int64_t>();
        const auto* x_ptr = x.data_ptr<at::BFloat16>();
        const auto* indices_ptr = routing_input.data_ptr<int64_t>();
        auto* snapshot_ptr = fresh ? topk_snapshot.data_ptr<int64_t>() : nullptr;
        const auto* weights_ptr = weights ? weights->data_ptr<float>() : nullptr;
        auto* slots_ptr = dst_slots.data_ptr<int64_t>();
        auto* send_ptr = fresh ? send_counts.data_ptr<int64_t>() : nullptr;
        auto* reduction_ptr =
            fresh ? reinterpret_cast<uint64_t*>(window.data_ptr<uint8_t>() + window_layout.notify_reduction_offset) : nullptr;
        auto* recv_ptr = fresh ? recv_rank_prefix.data_ptr<int64_t>() : nullptr;
        auto* expert_ptr = fresh ? recv_expert_counts.data_ptr<int64_t>() : nullptr;
        auto* offsets_ptr = fresh ? expert_offsets.data_ptr<int64_t>() : nullptr;
        auto* cursor_ptr = expert_cursor.defined() ? expert_cursor.data_ptr<int64_t>() : nullptr;
        const int rank = runtime.rank();
        const int64_t tokens = x.size(0);
        launch_kernel(kernel,
                      stream.stream(),
                      main_sms,
                      false,
                      addresses_ptr,
                      x_ptr,
                      indices_ptr,
                      snapshot_ptr,
                      weights_ptr,
                      slots_ptr,
                      send_ptr,
                      reduction_ptr,
                      recv_ptr,
                      expert_ptr,
                      offsets_ptr,
                      cursor_ptr,
                      mapped_counts,
                      rank,
                      tokens,
                      entry_epoch);
    } catch (...) {
        if (fresh)
            runtime.fail_dispatch_counts();
        throw;
    }

    const auto receive_counts = fresh ? runtime.wait_dispatch_counts(stream.stream())
                                      : std::array<int64_t, 2>{cached_meta_data[0].size(0), cached_expanded_capacity};
    const int64_t recv_count = receive_counts[0];
    const int64_t expanded_capacity = receive_counts[1];
    std::vector<torch::Tensor> metadata;
    if (fresh) {
        metadata = {
            recv_expert_counts,  // 0: expert_counts      [L]
            expert_offsets,      // 1: expert_offsets     [L+1]
            dst_slots,           // 2: dst_slots          [T,R]
            send_counts,         // 3: send_counts        [R]
            topk_snapshot,       // 4: source snapshot    [T,K]
            recv_rank_prefix,    // 5: inclusive source counts [R]
        };
    } else {
        metadata = cached_meta_data;
    }

    return DispatchState{
        std::move(metadata),
        {recv_count, expanded_capacity},
        window,
        window_layout.payload_offset,
        max_tokens,
        x.size(1),
        x.scalar_type(),
        runtime.rank(),
        std::move(expert_cursor),
    };
}

// Collective on the current stream. Use the same reduction mode and capacity
// as the owning allocation. Return the offset of uint8 [P,Tmax,row_bytes] storage,
// where P=K without local reduction, else min(R,K). Rows contain BF16 x[H],
// padding to a 32-byte boundary, then FP32 weights[K]. Absent/masked weights
// must not be read.
// Rank layout is selected iff allow_multiple_reduction && R<=K. Otherwise a
// compact/local-reduced row uses its first valid top-k slot as representative;
// expanded single-reduction sends each assignment to its own k partition.
// One cooperative kernel contains entry/exit barriers and payload work.
// Entry/exit GPU barriers protect reuse and publication. Expanded padding is
// never sent; local reduction uses FP32 and rounds partial sums to BF16.
// Storage validation is host-only; metadata values/ownership and exact expanded
// capacity are handle preconditions. No routing scan or GPU scalar readback is
// added. The device kernel asserts provenance and expanded-slot bounds. Empty
// ranks still participate in both barriers.
int64_t combine_main(PeerRuntime& runtime,
                     const torch::Tensor& x,  // compact [N,H] or expanded [C,H]
                     const std::vector<torch::Tensor>& metadata,
                     const c10::optional<torch::Tensor>& weights,  // compact [N,K] or expanded [C], optional
                     int64_t max_tokens,
                     bool expand,
                     bool allow_multiple_reduction,
                     int num_sms,
                     const JitKernel& kernel,
                     const torch::Tensor& window) {
    // Metadata: receive rows, expert counts and offsets, destination slots, send counts, source routing, rank prefix.

    const int world_size = runtime.world_size();
    TORCH_CHECK(metadata.size() == 7, "combine requires seven prepared metadata tensors from dispatch_epilogue");
    TORCH_CHECK(x.device() == window.device() && x.dim() == 2 && x.is_contiguous() && x.size(1) > 0,
                "combine x must be contiguous CUDA [N,H] or [C,H] on the runtime device");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "combine x must be bfloat16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % sizeof(int4) == 0, "combine x data pointer must be 16-byte aligned");
    TORCH_CHECK(x.size(1) % WindowLayout::kHiddenAlignment == 0, "combine hidden must be a positive multiple of 256");
    TORCH_CHECK(metadata[5].dim() == 2 && metadata[1].dim() == 1 && metadata[1].numel() > 0,
                "combine metadata requires a [T,K] source snapshot and [L] expert counts");

    const int64_t recv_count = metadata[0].size(0);
    const int64_t source_tokens = metadata[5].size(0);
    const int64_t hidden = x.size(1);
    const int64_t num_topk = metadata[5].size(1);
    const int64_t local_experts = metadata[1].numel();
    const int64_t num_experts = local_experts * world_size;
    const auto layout = WindowLayout::make(world_size, max_tokens, hidden, num_topk, num_experts, allow_multiple_reduction);
    TORCH_CHECK(num_topk <= num_experts && source_tokens <= max_tokens && recv_count <= world_size * max_tokens,
                "combine routing geometry exceeds expert/token capacity");
    TORCH_CHECK(layout.num_bytes <= window.numel(), "runtime window is smaller than the combine layout");
    check_cached_tensor(metadata[0], x, {recv_count, num_topk + 2}, "recv_metadata", torch::kInt32);
    check_cached_tensor(metadata[1], x, {local_experts}, "expert_counts");
    check_cached_tensor(metadata[2], x, {local_experts + 1}, "expert_offsets");
    check_cached_tensor(metadata[3], x, {source_tokens, world_size}, "dst_slots");
    check_cached_tensor(metadata[4], x, {world_size}, "send_counts");
    check_cached_tensor(metadata[5], x, {source_tokens, num_topk}, "source_topk_snapshot");
    check_cached_tensor(metadata[6], x, {world_size}, "recv_rank_prefix");
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
    const int main_sms = resolve_main_sms(num_sms);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto stream_handle = reinterpret_cast<int64_t>(stream.stream());
    const auto addresses = runtime.peer_addresses();
    const int64_t entry_epoch = runtime.reserve_barrier_epochs(stream_handle, 2);

    launch_kernel(kernel,
                  stream.stream(),
                  main_sms,
                  false,
                  addresses.data_ptr<int64_t>(),
                  x.data_ptr<at::BFloat16>(),
                  weights ? weights->data_ptr<float>() : nullptr,
                  metadata[0].data_ptr<int32_t>(),
                  runtime.rank(),
                  recv_count,
                  x.size(0),
                  layout.return_offset,
                  entry_epoch);

    return layout.return_offset;
}

}  // namespace

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::array<int64_t, 2>, std::shared_ptr<CompletionEvent>> dispatch(
    PeerRuntime& runtime,
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const c10::optional<torch::Tensor>& weights,
    int64_t num_experts,
    int64_t max_tokens,
    int64_t alignment,
    const std::vector<torch::Tensor>& cached_metadata,
    int64_t cached_expanded_capacity,
    bool allow_multiple_reduction,
    bool expand,
    bool zero_padding,
    int num_sms,
    int64_t comm_stream,
    int64_t previous_event,
    bool async_op) {
    const bool cached = !cached_metadata.empty();
    TORCH_CHECK(!cached || cached_metadata.size() == 7, "cached metadata must contain seven prepared tensors from dispatch_epilogue");
    const auto& routing = cached ? cached_metadata[5] : indices;
    const auto window = runtime.local_window();
    const int ranks = runtime.world_size();
    check_payload_inputs(window, x, routing, weights, ranks, num_experts, max_tokens);
    c10::cuda::CUDAGuard guard(window.device());
    const auto compute_stream = c10::cuda::getCurrentCUDAStream();
    const auto stream = c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(comm_stream), window.get_device());
    if (previous_event != -1) {
        if (!async_op && stream != compute_stream) {
            c10::Event ready(c10::DeviceType::CUDA);
            ready.record(compute_stream);
            stream.unwrap().wait(ready);
        }
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), reinterpret_cast<cudaEvent_t>(previous_event), 0));
    } else if (stream != compute_stream) {
        c10::Event ready(c10::DeviceType::CUDA);
        ready.record(compute_stream);
        stream.unwrap().wait(ready);
    }

    c10::cuda::CUDAStreamGuard stream_guard(stream);
    const int64_t consumer_stream = async_op ? -1 : reinterpret_cast<int64_t>(compute_stream.stream());
    const auto& main = dispatch_main_kernel(x.size(1), routing.size(1), ranks, num_experts / ranks, max_tokens, alignment, cached);
    const auto record_inputs = [&] {
        x.record_stream(stream);
        if (!cached)
            indices.record_stream(stream);
        if (weights)
            weights->record_stream(stream);
        for (const auto& tensor : cached_metadata)
            tensor.record_stream(stream);
    };
    auto state = dispatch_main(runtime,
                               x,
                               indices,
                               weights,
                               num_experts,
                               max_tokens,
                               cached_metadata,
                               cached_expanded_capacity,
                               allow_multiple_reduction,
                               expand,
                               num_sms,
                               main,
                               window,
                               consumer_stream);
    const auto& epilogue =
        dispatch_epilogue_kernel(x.size(1), routing.size(1), num_experts / ranks, cached, expand, weights.has_value(), zero_padding);
    auto outputs = dispatch_epilogue(epilogue, state, cached, expand, weights.has_value(), consumer_stream);
    std::shared_ptr<CompletionEvent> event;
    if (consumer_stream == -1) {
        record_inputs();
        auto protected_tensors = outputs;
        protected_tensors.insert(protected_tensors.end(), state.metadata.begin(), state.metadata.end());
        event = std::make_shared<CompletionEvent>(stream, std::move(protected_tensors));
    } else {
        const auto consumer = c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(consumer_stream), x.get_device());
        c10::Event completion(c10::DeviceType::CUDA);
        completion.record(stream);
        consumer.unwrap().wait(completion);
    }
    return std::make_tuple(std::move(outputs), std::move(state.metadata), std::move(state.counts), std::move(event));
}

std::tuple<std::vector<torch::Tensor>, std::shared_ptr<CompletionEvent>> combine(PeerRuntime& runtime,
                                                                                 const torch::Tensor& x,
                                                                                 const std::vector<torch::Tensor>& metadata,
                                                                                 const c10::optional<torch::Tensor>& weights,
                                                                                 const c10::optional<torch::Tensor>& bias,
                                                                                 int64_t max_tokens,
                                                                                 bool expand,
                                                                                 bool allow_multiple_reduction,
                                                                                 int num_sms,
                                                                                 int64_t comm_stream,
                                                                                 int64_t previous_event,
                                                                                 bool async_op) {
    TORCH_CHECK(metadata.size() == 7 && x.dim() == 2 && metadata[5].dim() == 2,
                "combine requires a 2D payload and seven prepared metadata tensors");
    const auto window = runtime.local_window();
    const int ranks = runtime.world_size();
    const int64_t local_experts = metadata[1].numel();
    c10::cuda::CUDAGuard guard(window.device());
    if (bias) {
        TORCH_CHECK(bias->device() == window.device() && bias->scalar_type() == torch::kBFloat16 && bias->is_contiguous() &&
                        bias->sizes() == at::IntArrayRef({metadata[5].size(0), x.size(1)}),
                    "combine bias must be contiguous [T,H], with the output dtype on the stage_buffer device");
    }
    const auto compute_stream = c10::cuda::getCurrentCUDAStream();
    const auto stream = c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(comm_stream), window.get_device());
    if (previous_event != -1) {
        if (!async_op && stream != compute_stream) {
            c10::Event ready(c10::DeviceType::CUDA);
            ready.record(compute_stream);
            stream.unwrap().wait(ready);
        }
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), reinterpret_cast<cudaEvent_t>(previous_event), 0));
    } else if (stream != compute_stream) {
        c10::Event ready(c10::DeviceType::CUDA);
        ready.record(compute_stream);
        stream.unwrap().wait(ready);
    }
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    const int64_t consumer_stream = async_op ? -1 : reinterpret_cast<int64_t>(compute_stream.stream());
    const auto& main = combine_main_kernel(
        x.size(1), metadata[5].size(1), ranks, local_experts, max_tokens, expand, allow_multiple_reduction, weights.has_value());
    const auto record_inputs = [&] {
        x.record_stream(stream);
        if (weights)
            weights->record_stream(stream);
        if (bias)
            bias->record_stream(stream);
        for (const auto& tensor : metadata)
            tensor.record_stream(stream);
    };
    const auto return_offset =
        combine_main(runtime, x, metadata, weights, max_tokens, expand, allow_multiple_reduction, num_sms, main, window);
    const auto& epilogue = combine_epilogue_kernel(
        x.size(1), metadata[5].size(1), ranks, local_experts, max_tokens, expand, allow_multiple_reduction, weights.has_value());
    auto outputs = combine_epilogue(epilogue, window, return_offset, metadata[5], x.size(1), bias, weights.has_value(), consumer_stream);
    std::shared_ptr<CompletionEvent> event;
    if (consumer_stream == -1) {
        record_inputs();
        event = std::make_shared<CompletionEvent>(stream, outputs);
    } else {
        const auto consumer = c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(consumer_stream), x.get_device());
        c10::Event completion(c10::DeviceType::CUDA);
        completion.record(stream);
        consumer.unwrap().wait(completion);
    }
    return std::make_tuple(std::move(outputs), std::move(event));
}

}  // namespace nano_ep
