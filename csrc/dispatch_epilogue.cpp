#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <limits>
#include <nano_ep/jit.hpp>
#include <nano_ep/pipeline.hpp>

namespace nano_ep {

// Return [recv_x, recv_topk_idx, recv_weights]. Absent optional tensors are
// zero-length tensors; Python converts them using explicit mode/presence flags.
// Fresh receive metadata and payload copying share one kernel; register-held
// assignments feed the copy directly. Cached dispatch retains the handle mappings.
// Use the same communication stream as dispatch_main; construct the Python
// handle after this call and expose completion via its event.
std::vector<torch::Tensor> dispatch_epilogue(
    const JitKernel& kernel, DispatchState& state, bool cached, bool expand, bool with_weights, int64_t consumer_stream) {
    const auto& source_topk = state.metadata[cached ? 5 : 4];
    const int64_t topk = source_topk.size(1);
    const int world_size = state.metadata[cached ? 6 : 5].numel();
    const int64_t max_tokens = state.max_tokens;
    const auto [N, C] = state.counts;
    TORCH_CHECK(max_tokens > 0 && N >= 0 && N <= world_size * max_tokens, "dispatch receive count exceeds stage capacity");

    c10::cuda::CUDAGuard guard(state.window.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::OptionalCUDAStreamGuard allocation_guard;
    if (consumer_stream != -1)
        allocation_guard.reset_stream(
            c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(consumer_stream), state.window.get_device()));

    TORCH_CHECK(C >= 0 && (!expand || C <= std::numeric_limits<int32_t>::max()), "expanded capacity exceeds int32 metadata range");
    if (!cached) {
        auto receive = torch::empty({N, topk + 2}, state.metadata[0].options().dtype(torch::kInt32));
        state.metadata = {
            receive, state.metadata[0], state.metadata[1], state.metadata[2], state.metadata[3], state.metadata[4], state.metadata[5]};
    }

    const auto x_options = state.window.options().dtype(state.x_dtype);
    const auto weight_options = state.window.options().dtype(torch::kFloat32);
    auto weights = torch::empty({0}, weight_options);

    auto local_indices = expand ? torch::empty({0}, state.metadata[1].options()) : torch::empty({N, topk}, state.metadata[1].options());
    torch::Tensor x = expand ? torch::empty({C, state.hidden}, x_options) : torch::empty({N, state.hidden}, x_options);
    if (with_weights) {
        weights = expand ? torch::empty({C}, weight_options) : torch::empty({N, topk}, weight_options);
    }
    allocation_guard.reset();

    if (N == 0)
        return {x, local_indices, weights};

    const bool pdl = use_pdl();
    launch_kernel(kernel,
                  stream.stream(),
                  at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
                  pdl,
                  state.window.data_ptr<uint8_t>() + state.stage_offset,
                  state.metadata[0].data_ptr<int32_t>(),
                  (cached || !expand) ? nullptr : reinterpret_cast<unsigned long long*>(state.expert_cursor.data_ptr<int64_t>()),
                  state.metadata[1].data_ptr<int64_t>(),
                  state.metadata[2].data_ptr<int64_t>(),
                  x.data_ptr<at::BFloat16>(),
                  expand ? nullptr : local_indices.data_ptr<int64_t>(),
                  with_weights ? weights.data_ptr<float>() : nullptr,
                  N,
                  state.metadata[6].data_ptr<int64_t>(),
                  world_size,
                  max_tokens,
                  state.rank,
                  pdl);
    return {x, local_indices, weights};
}

}  // namespace nano_ep
