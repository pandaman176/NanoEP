#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdio>

#include "combine_kernel_under_test.cuh"
#include "layout.hpp"

// Logical peers use distinct allocations on one physical GPU. This probe tests
// the actual kernel/launcher's addressing; real IPC/barriers use peer_worker.
void run_return_kernel(const std::vector<torch::Tensor>& windows,
                       const torch::Tensor& x,
                       const std::vector<torch::Tensor>& metadata,
                       const c10::optional<torch::Tensor>& weights,
                       int rank,
                       int64_t capacity,
                       bool expand,
                       bool mode) {
    c10::cuda::CUDAGuard guard(x.device());
    const int ranks = windows.size();
    const auto layout = nano_ep::WindowLayout::make(ranks, capacity, x.size(1), metadata[8].size(1), ranks * 3, mode);
    auto addresses = torch::empty({ranks}, torch::TensorOptions().dtype(torch::kInt64));
    for (int peer = 0; peer < ranks; ++peer)
        addresses.data_ptr<int64_t>()[peer] = reinterpret_cast<int64_t>(windows[peer].data_ptr<uint8_t>());
    addresses = addresses.to(x.device());
    nano_ep::launch_combine_return(
        addresses, x, metadata, weights, rank, ranks, capacity, layout, expand, mode, c10::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_return_kernel);
    module.def("layout", [](int ranks, int64_t capacity, int hidden, int topk, bool mode) {
        const auto layout = nano_ep::WindowLayout::make(ranks, capacity, hidden, topk, ranks * 3, mode);
        return std::vector<int64_t>{layout.return_offset, layout.num_bytes};
    });
}
