#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <nano_ep/jit.hpp>
#include <nano_ep/pipeline.hpp>

#include "layout.hpp"

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
// once, then cast to BF16.
// Bias is optional contiguous [T,H] with BF16 dtype. Without weights the second
// result is an empty CUDA FP32 [0] tensor. Outputs own their storage; the caller
// orders this read after combine_main and before the next dispatch or combine.
std::vector<torch::Tensor> combine_epilogue(const JitKernel& kernel,
                                            const torch::Tensor& window,
                                            int64_t return_offset,
                                            const torch::Tensor& source_topk_snapshot,
                                            int64_t hidden,
                                            const c10::optional<torch::Tensor>& bias,
                                            bool with_weights,
                                            int64_t consumer_stream) {
    const int64_t num_tokens = source_topk_snapshot.size(0);
    const int64_t num_topk = source_topk_snapshot.size(1);
    c10::cuda::CUDAGuard guard(window.device());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::OptionalCUDAStreamGuard allocation_guard;
    if (consumer_stream != -1)
        allocation_guard.reset_stream(
            c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(consumer_stream), window.get_device()));
    auto combined_x = torch::empty({num_tokens, hidden}, window.options().dtype(torch::kBFloat16));
    auto combined_weights = torch::empty(with_weights ? std::vector<int64_t>{num_tokens, num_topk} : std::vector<int64_t>{0},
                                         window.options().dtype(torch::kFloat32));
    allocation_guard.reset();

    if (num_tokens != 0) {
        const bool pdl = use_pdl();
        launch_kernel(kernel,
                      stream.stream(),
                      at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
                      pdl,
                      window.data_ptr<uint8_t>() + return_offset,
                      source_topk_snapshot.data_ptr<int64_t>(),
                      bias ? bias->data_ptr<at::BFloat16>() : nullptr,
                      combined_x.data_ptr<at::BFloat16>(),
                      with_weights ? combined_weights.data_ptr<float>() : nullptr,
                      num_tokens,
                      pdl);
    }
    return {combined_x, combined_weights};
}

}  // namespace nano_ep
