#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdio>

#include "combine_main.cuh"
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
    TORCH_CHECK(
        x.size(1) == NANO_EP_HIDDEN && metadata[8].size(1) == NANO_EP_TOPK && ranks == NANO_EP_WORLD_SIZE && capacity == NANO_EP_MAX_TOKENS,
        "probe inputs do not match the compiled geometry");
    nano_ep::launch_combine_return<NANO_EP_HIDDEN, NANO_EP_TOPK, NANO_EP_WORLD_SIZE, NANO_EP_MAX_TOKENS>(
        addresses, x, metadata, weights, rank, layout, expand, mode, c10::cuda::getCurrentCUDAStream().stream());
}

// Each warp exposes the production helper's result from every participating lane.
__global__ void compact_slots_kernel(const int64_t* input, int64_t* output, int32_t* counts, int64_t rows) {
    const int lane = threadIdx.x % 32;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= rows)
        return;
    const int64_t lane_slot = lane < NANO_EP_TOPK ? input[row * NANO_EP_TOPK + lane] : -1;
    const unsigned valid_mask = __ballot_sync(0xffffffffu, lane_slot >= 0);
    int64_t slots[NANO_EP_TOPK];
    const int count = nano_ep::compute_expanded_slots(lane_slot, valid_mask, slots);
    counts[row * 32 + lane] = count;
    for (int k = 0; k < NANO_EP_TOPK; ++k)
        output[(row * 32 + lane) * NANO_EP_TOPK + k] = k < count ? slots[k] : -1;
}

std::vector<torch::Tensor> compact_slots(const torch::Tensor& input) {
    TORCH_CHECK(input.dim() == 2 && input.size(1) == NANO_EP_TOPK && input.is_contiguous(), "expected contiguous [N,K] slot input");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({input.size(0), 32, NANO_EP_TOPK}, input.options());
    auto counts = torch::empty({input.size(0), 32}, input.options().dtype(torch::kInt32));
    if (input.size(0) != 0) {
        constexpr int threads = 128;
        const int64_t blocks = (input.size(0) + threads / 32 - 1) / (threads / 32);
        compact_slots_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            input.data_ptr<int64_t>(), output.data_ptr<int64_t>(), counts.data_ptr<int32_t>(), input.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run_return_kernel);
    module.def("compact_slots", &compact_slots);
    module.def("layout", [](int ranks, int64_t capacity, int hidden, int topk, bool mode) {
        const auto layout = nano_ep::WindowLayout::make(ranks, capacity, hidden, topk, ranks * 3, mode);
        return std::vector<int64_t>{layout.return_offset, layout.num_bytes};
    });
}
