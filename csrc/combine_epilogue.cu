#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <limits>

#include "combine_epilogue.cuh"

namespace nano_ep {

// Return [combined_x, combined_weights]. Source T and K come from
// source_topk_snapshot[T,K]; combined_x is BF16 [T,H], and requested weights
// are FP32 [T,K]. (H,K,R,E/R,Tmax) must match the compiled geometry.
// H must be a positive multiple of 256; actual T may vary up to Tmax.
// The source routing and mode determine which return partitions were written.
// Deduplicate destination ranks for compact/local-reduced x; expanded without
// local reduction reads each valid k. Weights select their original k from the
// corresponding writer's row, independently of x reduction. Masked weight
// slots are zero; tokens with no valid assignments still receive optional bias.
// Read BF16 contributions or partial sums, accumulate FP32, add source bias
// once, then cast to BF16. x_dtype must be BF16.
// Bias is optional contiguous [T,H] with x_dtype. Without weights the second
// result is an empty CUDA FP32 [0] tensor. Outputs own their storage; the caller
// orders this read after combine_main and before the next dispatch or combine.
static std::vector<torch::Tensor> combine_epilogue(const torch::Tensor& stage_buffer,
                                                   const torch::Tensor& source_topk_snapshot,
                                                   int64_t hidden,
                                                   at::ScalarType x_dtype,
                                                   int64_t num_experts,
                                                   int64_t world_size,
                                                   int64_t max_tokens,
                                                   const c10::optional<torch::Tensor>& bias,
                                                   bool with_weights,
                                                   bool expand,
                                                   bool allow_multiple_reduction) {
    // Input Check
    TORCH_CHECK(
        stage_buffer.is_cuda() && stage_buffer.scalar_type() == torch::kUInt8 && stage_buffer.dim() == 3 && stage_buffer.is_contiguous(),
        "combine stage_buffer must be contiguous CUDA uint8 [P,Tmax,row_bytes]");
    TORCH_CHECK(source_topk_snapshot.device() == stage_buffer.device() && source_topk_snapshot.scalar_type() == torch::kInt64 &&
                    source_topk_snapshot.dim() == 2 && source_topk_snapshot.is_contiguous(),
                "source_topk_snapshot must be contiguous CUDA int64 [T,K] on the stage_buffer device");
    TORCH_CHECK(x_dtype == torch::kBFloat16, "combine output dtype must be bfloat16");
    TORCH_CHECK(hidden > 0 && hidden % WindowLayout::kHiddenAlignment == 0, "combine hidden must be a positive multiple of 256");
    TORCH_CHECK(world_size > 0 && world_size <= 32 && num_experts > 0 && num_experts % world_size == 0,
                "combine requires world_size in [1,32] and positive num_experts divisible by world_size");
    const int64_t num_tokens = source_topk_snapshot.size(0);
    const int64_t num_topk = source_topk_snapshot.size(1);
    TORCH_CHECK(hidden > 0 && hidden <= std::numeric_limits<int>::max() && num_topk > 0 && num_topk <= 32 && num_topk <= num_experts &&
                    max_tokens > 0 && num_tokens <= max_tokens,
                "combine hidden/top-k or source token capacity is invalid");
    const int partitions = WindowLayout::num_return_partitions(world_size, num_topk, allow_multiple_reduction);
    const int64_t row_bytes = WindowLayout::return_row_bytes(hidden, num_topk);
    TORCH_CHECK(stage_buffer.size(0) == partitions && stage_buffer.size(1) == max_tokens && stage_buffer.size(2) == row_bytes,
                "combine stage_buffer shape does not match reduction mode, capacity and row layout");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(stage_buffer.data_ptr<uint8_t>()) % WindowLayout::kAlignment == 0,
                "combine stage_buffer must be aligned for FP32 loads");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == stage_buffer.device() && bias->scalar_type() == x_dtype && bias->dim() == 2 &&
                        bias->size(0) == num_tokens && bias->size(1) == hidden && bias->is_contiguous(),
                    "combine bias must be contiguous [T,H], with the output dtype on the stage_buffer device");
    }

    TORCH_CHECK(hidden == NANO_EP_HIDDEN && num_topk == NANO_EP_TOPK && world_size == NANO_EP_WORLD_SIZE &&
                    num_experts / world_size == NANO_EP_EXPERTS_PER_RANK && max_tokens == NANO_EP_MAX_TOKENS,
                "combine epilogue inputs do not match the compiled geometry");
    c10::cuda::CUDAGuard guard(stage_buffer.device());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    auto combined_x = torch::empty({num_tokens, hidden}, stage_buffer.options().dtype(x_dtype));
    auto combined_weights = torch::empty(with_weights ? std::vector<int64_t>{num_tokens, num_topk} : std::vector<int64_t>{0},
                                         stage_buffer.options().dtype(torch::kFloat32));

    if (num_tokens != 0) {
        constexpr int threads = 256;
        const int blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const auto launch = [&]<bool kExpand, bool kMultiple, bool kWeights>() {
            combine_reduce_epilogue_kernel<NANO_EP_HIDDEN,
                                           NANO_EP_TOPK,
                                           NANO_EP_WORLD_SIZE,
                                           NANO_EP_EXPERTS_PER_RANK,
                                           NANO_EP_MAX_TOKENS,
                                           kExpand,
                                           kMultiple,
                                           kWeights><<<blocks, threads, 0, stream.stream()>>>(
                stage_buffer.data_ptr<uint8_t>(),
                source_topk_snapshot.data_ptr<int64_t>(),
                bias.has_value() ? reinterpret_cast<const nv_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr,
                reinterpret_cast<nv_bfloat16*>(combined_x.data_ptr<at::BFloat16>()),
                kWeights ? combined_weights.data_ptr<float>() : nullptr,
                num_tokens);
        };
        const auto weighted = [&]<bool kExpand, bool kMultiple>() {
            if (with_weights)
                launch.template operator()<kExpand, kMultiple, true>();
            else
                launch.template operator()<kExpand, kMultiple, false>();
        };
        if (expand) {
            if (allow_multiple_reduction)
                weighted.template operator()<true, true>();
            else
                weighted.template operator()<true, false>();
        } else {
            if (allow_multiple_reduction)
                weighted.template operator()<false, true>();
            else
                weighted.template operator()<false, false>();
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {combined_x, combined_weights};
}

}  // namespace nano_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("combine_epilogue",
          &nano_ep::combine_epilogue,
          pybind11::arg("stage_buffer"),
          pybind11::arg("source_topk_snapshot"),
          pybind11::arg("hidden"),
          pybind11::arg("x_dtype"),
          pybind11::arg("num_experts"),
          pybind11::arg("world_size"),
          pybind11::arg("max_tokens"),
          pybind11::arg("bias"),
          pybind11::arg("with_weights"),
          pybind11::arg("expand"),
          pybind11::arg("allow_multiple_reduction"));
    m.attr("geometry") =
        pybind11::make_tuple(NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_WORLD_SIZE, NANO_EP_EXPERTS_PER_RANK, NANO_EP_MAX_TOKENS);
}
