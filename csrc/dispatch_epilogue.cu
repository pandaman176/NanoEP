#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <nano_ep/pipeline.hpp>

#include "dispatch_epilogue.cuh"

namespace nano_ep {

constexpr int kHidden = NANO_EP_HIDDEN;
constexpr int kTopk = NANO_EP_TOPK;
constexpr int kExpertsPerRank = NANO_EP_EXPERTS_PER_RANK;
constexpr bool kCachedMode = NANO_EP_CACHED_MODE;
constexpr int64_t kRowBytes = WindowLayout::payload_x_offset(kTopk) + int64_t{kHidden} * sizeof(at::BFloat16);
static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0);
static_assert(kTopk > 0 && kTopk <= 32 && kExpertsPerRank > 0);

// Return [recv_x, recv_topk_idx, recv_weights]. Absent optional tensors are
// zero-length tensors; Python converts them using explicit mode/presence flags.
// Fresh receive metadata and payload copying share one kernel; register-held
// assignments feed the copy directly. Cached/prepared states retain their mappings.
// Use the same communication stream as dispatch_main; construct the Python
// handle after this call and expose completion via its event.
static std::vector<torch::Tensor> dispatch_epilogue(DispatchState& state, bool expand, bool with_weights, bool zero_padding) {
    TORCH_CHECK(state.metadata.size() == (state.metadata_prepared ? 10 : 6) && state.stage_buffer.is_cuda(),
                "dispatch_epilogue requires a native dispatch state");
    TORCH_CHECK(state.metadata_prepared == kCachedMode, "dispatch epilogue cached_mode does not match the state");
    TORCH_CHECK(state.x_dtype == torch::kBFloat16, "dispatch output dtype must be bfloat16");
    TORCH_CHECK(state.hidden > 0 && state.hidden % WindowLayout::kHiddenAlignment == 0,
                "dispatch hidden must be a positive multiple of 256");
    const auto& expert_counts = state.metadata[kCachedMode ? 3 : 0];
    const auto& source_topk = state.metadata[kCachedMode ? 8 : 4];
    const auto& recv_rank_prefix = state.metadata[kCachedMode ? 9 : 5];
    const int world_size = recv_rank_prefix.numel();
    TORCH_CHECK(state.hidden == kHidden && source_topk.dim() == 2 && source_topk.size(1) == kTopk &&
                    expert_counts.numel() == kExpertsPerRank && state.stage_buffer.dim() == 2 && state.stage_buffer.size(1) == kRowBytes,
                "dispatch epilogue inputs do not match the compiled geometry");
    TORCH_CHECK(world_size > 0 && world_size <= 32 && state.stage_buffer.size(0) % world_size == 0,
                "dispatch stage must contain equally sized source partitions");
    const int64_t max_tokens = state.stage_buffer.size(0) / world_size;
    const auto* counts = state.counts.data_ptr<int64_t>();  // CPU
    const int64_t N = counts[0];
    const int64_t C = counts[1];
    TORCH_CHECK(max_tokens > 0 && N >= 0 && N <= state.stage_buffer.size(0), "dispatch receive count exceeds stage capacity");

    c10::cuda::CUDAGuard guard(state.stage_buffer.device());
    if constexpr (kCachedMode) {
        TORCH_CHECK(state.metadata[5].dim() == (expand ? 2 : 1), "cached dispatch must retain the handle layout");
    }
    if constexpr (!kCachedMode) {
        const auto options = state.metadata[0].options();
        std::vector<torch::Tensor> receive = {torch::empty({N}, options),
                                              torch::empty({N}, options),
                                              torch::empty({N, kTopk}, options),
                                              expand ? torch::empty({N, kTopk}, options) : torch::empty({0}, options)};
        state.metadata = {
            receive[0],         // src_ranks
            receive[1],         // src_tokens
            receive[2],         // local_topk_idx
            state.metadata[0],  // expert_counts
            state.metadata[1],  // expert_offsets
            receive[3],         // expanded_slots [N,K], compact [0]
            state.metadata[2],  // dst_slots
            state.metadata[3],  // send_counts
            state.metadata[4],  // source_topk_snapshot
            state.metadata[5],  // recv_rank_prefix
        };
    }

    const auto x_options = state.stage_buffer.options().dtype(state.x_dtype);
    const auto weight_options = state.stage_buffer.options().dtype(torch::kFloat32);
    auto weights = torch::empty({0}, weight_options);

    auto local_indices = expand ? torch::empty({0}, state.metadata[2].options()) : torch::empty({N, kTopk}, state.metadata[2].options());
    torch::Tensor x = expand ? torch::empty({C, kHidden}, x_options) : torch::empty({N, kHidden}, x_options);
    if (with_weights) {
        weights = expand ? torch::empty({C}, weight_options) : torch::empty({N, kTopk}, weight_options);
    }

    if (N == 0) {
        state.metadata_prepared = true;
        return {x, local_indices, weights};
    }

    const auto stream = c10::cuda::getCurrentCUDAStream();
    // Expanded reservation uses private cursors, preserving immutable padded offsets.
    torch::Tensor expert_cursor;
    if constexpr (!kCachedMode) {
        if (expand)
            expert_cursor = state.metadata[4].narrow(0, 0, kExpertsPerRank).clone();
    }

    constexpr int threads = 256;
    const int blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    const auto launch = [&]<bool kExpand, bool kWithWeights, bool kZeroPadding>() {
        dispatch_copy_epilogue_kernel<kHidden, kTopk, kExpertsPerRank, kCachedMode, kExpand, kWithWeights, kZeroPadding>
            <<<blocks, threads, 0, stream.stream()>>>(
                state.stage_buffer.data_ptr<uint8_t>(),
                kCachedMode ? nullptr : state.metadata[0].data_ptr<int64_t>(),
                kCachedMode ? nullptr : state.metadata[1].data_ptr<int64_t>(),
                state.metadata[2].data_ptr<int64_t>(),
                kExpand ? state.metadata[5].data_ptr<int64_t>() : nullptr,
                (kCachedMode || !kExpand) ? nullptr : reinterpret_cast<unsigned long long*>(expert_cursor.data_ptr<int64_t>()),
                state.metadata[3].data_ptr<int64_t>(),
                state.metadata[4].data_ptr<int64_t>(),
                x.data_ptr<at::BFloat16>(),
                kExpand ? nullptr : local_indices.data_ptr<int64_t>(),
                kWithWeights ? weights.data_ptr<float>() : nullptr,
                N,
                state.metadata[9].data_ptr<int64_t>(),
                world_size,
                max_tokens,
                state.rank);
    };
    const auto weighted = [&]<bool kExpand, bool kZeroPadding>() {
        if (with_weights)
            launch.template operator()<kExpand, true, kZeroPadding>();
        else
            launch.template operator()<kExpand, false, kZeroPadding>();
    };
    if (expand) {
        if (zero_padding)
            weighted.template operator()<true, true>();
        else
            weighted.template operator()<true, false>();
    } else {
        weighted.template operator()<false, false>();
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    state.metadata_prepared = true;

    return {x, local_indices, weights};
}

}  // namespace nano_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("dispatch_epilogue",
               &nano_ep::dispatch_epilogue,
               pybind11::arg("state"),
               pybind11::arg("expand"),
               pybind11::arg("with_weights"),
               pybind11::arg("zero_padding"));
    module.attr("geometry") = pybind11::make_tuple(NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_EXPERTS_PER_RANK);
    module.attr("cached_mode") = pybind11::bool_(nano_ep::kCachedMode);
}
