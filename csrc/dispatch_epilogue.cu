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
constexpr int64_t kRowBytes = WindowLayout::payload_x_offset(kTopk) + int64_t{kHidden} * sizeof(at::BFloat16);
static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0);
static_assert(kTopk > 0 && kTopk <= 32 && kExpertsPerRank > 0);

static std::vector<torch::Tensor> build_receive_metadata(const torch::Tensor& stage_buffer,
                                                         const torch::Tensor& expert_offsets,
                                                         int64_t rank) {
    TORCH_CHECK(
        stage_buffer.is_cuda() && stage_buffer.scalar_type() == torch::kUInt8 && stage_buffer.dim() == 2 && stage_buffer.is_contiguous(),
        "stage_buffer must be contiguous CUDA uint8 [N,row_bytes]");
    TORCH_CHECK(rank >= 0, "rank must be nonnegative");
    TORCH_CHECK(expert_offsets.device() == stage_buffer.device() && expert_offsets.scalar_type() == torch::kInt64 &&
                    expert_offsets.dim() == 1 && expert_offsets.numel() == kExpertsPerRank + 1 && expert_offsets.is_contiguous(),
                "expert_offsets must be contiguous CUDA int64 [kExpertsPerRank+1] on the stage_buffer device");
    const int64_t recv_count = stage_buffer.size(0);
    TORCH_CHECK(
        stage_buffer.size(1) == kRowBytes && reinterpret_cast<uintptr_t>(stage_buffer.data_ptr<uint8_t>()) % WindowLayout::kAlignment == 0,
        "stage_buffer must contain aligned full dispatch payload rows");

    c10::cuda::CUDAGuard guard(stage_buffer.device());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const auto options = expert_offsets.options();
    auto src_ranks = torch::empty({recv_count}, options);
    auto src_tokens = torch::empty({recv_count}, options);
    auto local_topk_idx = torch::empty({recv_count, kTopk}, options);
    auto expanded_slots = torch::empty({recv_count, kTopk}, options);
    if (recv_count == 0)
        return {src_ranks, src_tokens, local_topk_idx, expanded_slots};

    // Preserve the padded offsets for replay and give this launch private cursors.
    auto expert_cursor = expert_offsets.narrow(0, 0, kExpertsPerRank).clone();
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (recv_count + warps_per_block - 1) / warps_per_block;
    build_receive_metadata_kernel<kHidden, kTopk, kExpertsPerRank>
        <<<blocks, threads, 0, stream.stream()>>>(stage_buffer.data_ptr<uint8_t>(),
                                                  src_ranks.data_ptr<int64_t>(),
                                                  src_tokens.data_ptr<int64_t>(),
                                                  local_topk_idx.data_ptr<int64_t>(),
                                                  expanded_slots.data_ptr<int64_t>(),
                                                  reinterpret_cast<unsigned long long*>(expert_cursor.data_ptr<int64_t>()),
                                                  recv_count,
                                                  rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {src_ranks, src_tokens, local_topk_idx, expanded_slots};
}

// Return [recv_x, recv_topk_idx, recv_weights]. Absent optional tensors are
// zero-length tensors; Python converts them using explicit mode/presence flags.
// Fresh receive metadata is built once, before the independent copy kernel.
// Cached/prepared states retain their mappings. Use the same communication
// stream as dispatch_main; construct the Python handle after this call and
// expose completion via its event.
static std::vector<torch::Tensor> dispatch_epilogue(DispatchState& state, bool expand, bool with_weights, bool zero_padding) {
    TORCH_CHECK(state.metadata.size() == (state.metadata_prepared ? 9 : 5) && state.stage_buffer.is_cuda(),
                "dispatch_epilogue requires a native dispatch state");
    TORCH_CHECK(state.x_dtype == torch::kBFloat16, "dispatch output dtype must be bfloat16");
    TORCH_CHECK(state.hidden > 0 && state.hidden % WindowLayout::kHiddenAlignment == 0,
                "dispatch hidden must be a positive multiple of 256");
    const auto& expert_counts = state.metadata[state.metadata_prepared ? 3 : 0];
    const auto& source_topk = state.metadata[state.metadata_prepared ? 8 : 4];
    TORCH_CHECK(state.hidden == kHidden && source_topk.dim() == 2 && source_topk.size(1) == kTopk &&
                    expert_counts.numel() == kExpertsPerRank && state.stage_buffer.dim() == 2 && state.stage_buffer.size(1) == kRowBytes,
                "dispatch epilogue inputs do not match the compiled geometry");
    c10::cuda::CUDAGuard guard(state.stage_buffer.device());
    if (!state.metadata_prepared) {
        auto receive = build_receive_metadata(state.stage_buffer, state.metadata[1], state.rank);
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

    torch::Tensor x;
    const auto x_options = state.stage_buffer.options().dtype(state.x_dtype);
    const auto weight_options = state.stage_buffer.options().dtype(torch::kFloat32);
    auto local_indices = torch::empty({0}, state.metadata[2].options());
    auto weights = torch::empty({0}, weight_options);

    if (expand) {
        x = zero_padding ? torch::zeros({C, kHidden}, x_options) : torch::empty({C, kHidden}, x_options);
        if (with_weights) {
            weights = zero_padding ? torch::zeros({C}, weight_options) : torch::empty({C}, weight_options);
        }
    } else {
        x = torch::empty({N, kHidden}, x_options);
        local_indices = torch::empty({N, kTopk}, state.metadata[2].options());
        if (with_weights) {
            weights = torch::empty({N, kTopk}, weight_options);
        }
    }

    if (N == 0)
        return {x, local_indices, weights};

    const auto stream = c10::cuda::getCurrentCUDAStream();

    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (N + warps_per_block - 1) / warps_per_block;

    const auto launch = [&]<bool kExpand, bool kWithWeights>() {
        dispatch_copy_epilogue_kernel<kHidden, kTopk, kExpand, kWithWeights>
            <<<blocks, threads, 0, stream.stream()>>>(state.stage_buffer.data_ptr<uint8_t>(),
                                                      state.metadata[2].data_ptr<int64_t>(),
                                                      state.metadata[5].data_ptr<int64_t>(),
                                                      x.data_ptr<at::BFloat16>(),
                                                      kExpand ? nullptr : local_indices.data_ptr<int64_t>(),
                                                      kWithWeights ? weights.data_ptr<float>() : nullptr,
                                                      N);
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

}  // namespace nano_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("dispatch_epilogue",
               &nano_ep::dispatch_epilogue,
               pybind11::arg("state"),
               pybind11::arg("expand"),
               pybind11::arg("with_weights"),
               pybind11::arg("zero_padding"));
    module.def("build_receive_metadata",
               &nano_ep::build_receive_metadata,
               pybind11::arg("stage_buffer"),
               pybind11::arg("expert_offsets"),
               pybind11::arg("rank"));
    module.attr("geometry") = pybind11::make_tuple(NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_EXPERTS_PER_RANK);
}
