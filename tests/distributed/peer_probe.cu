#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda/atomic>
#include <cstdio>

// Test-only reads through imported pointers; no production API is added.
__global__ void read_words_kernel(const int64_t* addresses, int64_t* output, int64_t ranks, int64_t offset, int64_t words) {
    const int64_t i = threadIdx.x;
    if (i < ranks * words) {
        const auto* peer = reinterpret_cast<const int64_t*>(addresses[i / words]);
        output[i] = peer[offset + i % words];
    }
}

torch::Tensor read_peer_words(const torch::Tensor& addresses, int64_t offset, int64_t words) {
    c10::cuda::CUDAGuard guard(addresses.device());
    const auto ranks = addresses.numel();
    TORCH_CHECK(ranks * words <= 256, "Probe must fit in one block");
    auto output = torch::empty({ranks, words}, addresses.options());
    read_words_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        addresses.data_ptr<int64_t>(), output.data_ptr<int64_t>(), ranks, offset, words);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Delay an old arena reader until a fast peer has entered the next collective.
__global__ void wait_for_arrival_kernel(int64_t* signals, int peer, int64_t epoch, unsigned long long timeout_cycles) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_system> signal(signals[peer]);
    const auto start = clock64();
    while (signal.load(cuda::memory_order_acquire) < epoch) {
        if (clock64() - start > timeout_cycles) {
            printf("NanoEP arena probe timeout: peer=%d epoch=%lld\n", peer, static_cast<long long>(epoch));
            asm volatile("trap;");
        }
        __nanosleep(64);
    }
}

void wait_for_arrival(const torch::Tensor& window, int peer, int64_t epoch) {
    c10::cuda::CUDAGuard guard(window.device());
    constexpr unsigned long long timeout_cycles = 20'000'000'000ULL;
    wait_for_arrival_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<int64_t*>(window.data_ptr<uint8_t>()), peer, epoch, timeout_cycles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("read_peer_words", &read_peer_words);
    module.def("wait_for_arrival", &wait_for_arrival);
}
