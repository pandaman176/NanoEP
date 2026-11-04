#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <nano_ep/pipeline.hpp>

#include <cstdio>
#include <cuda/atomic>

#include "dispatch_main.cuh"

// Run the production exchange helper with synthetic counts for delayed-peer tests.
template <int kWorldSize, int kExpertsPerRank>
__global__ void probe_notify_peer_counts_kernel(const int64_t* addresses,
                                                int64_t* local_counts,
                                                const int64_t* send_counts,
                                                const int64_t* expert_counts,
                                                int64_t* received,
                                                int rank,
                                                int64_t count_offset) {
    constexpr int threads = 128;
    constexpr int num_counts = kWorldSize + kWorldSize * kExpertsPerRank;
    __shared__ uint32_t counts[num_counts];
    for (int i = threadIdx.x; i < num_counts; i += threads)
        counts[i] = static_cast<uint32_t>(i < kWorldSize ? send_counts[i] : expert_counts[i - kWorldSize]);
    nano_ep::notify_group_barrier<threads>();
    nano_ep::notify_peer_counts<threads, kWorldSize, kExpertsPerRank, nano_ep::kBarrierTimeoutCycles>(
        addresses, local_counts, counts, rank, count_offset);
    for (int i = threadIdx.x; i < num_counts; i += threads)
        received[i] = counts[i];
}

torch::Tensor run_notify_peer_counts(const torch::Tensor& addresses,
                                     const torch::Tensor& window,
                                     const torch::Tensor& send_counts,
                                     const torch::Tensor& expert_counts,
                                     int rank,
                                     int64_t count_offset) {
    c10::cuda::CUDAGuard guard(window.device());
    const int ranks = addresses.numel();
    const int64_t experts = expert_counts.numel();
    auto received = torch::empty({ranks + experts}, send_counts.options());
    const auto launch = [&]<int kWorldSize, int kExpertsPerRank>() {
        probe_notify_peer_counts_kernel<kWorldSize, kExpertsPerRank><<<1, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            addresses.data_ptr<int64_t>(),
            reinterpret_cast<int64_t*>(window.data_ptr<uint8_t>() + count_offset),
            send_counts.data_ptr<int64_t>(),
            expert_counts.data_ptr<int64_t>(),
            received.data_ptr<int64_t>(),
            rank,
            count_offset);
    };
    if (ranks == 1 && experts == 4)
        launch.template operator()<1, 4>();
    else if (ranks == 1 && experts == 260)
        launch.template operator()<1, 260>();
    else if (ranks == 2 && experts == 4)
        launch.template operator()<2, 2>();
    else if (ranks == 2 && experts == 260)
        launch.template operator()<2, 130>();
    else
        TORCH_CHECK(false, "unsupported notify probe geometry");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return received;
}

template <int kWorldSize, int kExpertsPerRank, int64_t kExpertAlignment>
__global__ void probe_finalize_receive_counts_kernel(const int64_t* incoming,
                                                     int64_t* rank_prefix,
                                                     int64_t* expert_counts,
                                                     int64_t* expert_offsets,
                                                     int64_t* totals) {
    constexpr int threads = 128;
    constexpr int num_counts = kWorldSize + kWorldSize * kExpertsPerRank;
    __shared__ uint32_t counts[num_counts];
    for (int i = threadIdx.x; i < num_counts; i += threads)
        counts[i] = static_cast<uint32_t>(incoming[i]);
    nano_ep::notify_group_barrier<threads>();
    nano_ep::finalize_receive_counts<kWorldSize, kExpertsPerRank, kExpertAlignment>(counts, rank_prefix, expert_counts, expert_offsets, totals);
}

template <int kWorldSize, typename Launch>
void select_count_probe_experts(int64_t experts, const Launch& launch) {
    switch (experts) {
        case 1:
            launch.template operator()<kWorldSize, 1>();
            break;
        case 31:
            launch.template operator()<kWorldSize, 31>();
            break;
        case 32:
            launch.template operator()<kWorldSize, 32>();
            break;
        case 33:
            launch.template operator()<kWorldSize, 33>();
            break;
        case 129:
            launch.template operator()<kWorldSize, 129>();
            break;
        case 260:
            launch.template operator()<kWorldSize, 260>();
            break;
        default:
            TORCH_CHECK(false, "unsupported count probe experts");
    }
}

std::vector<torch::Tensor> run_finalize_receive_counts(const torch::Tensor& incoming, int ranks, int64_t alignment) {
    c10::cuda::CUDAGuard guard(incoming.device());
    const int64_t experts = (incoming.numel() - ranks) / ranks;
    auto rank_prefix = torch::full({ranks}, -777, incoming.options());
    auto expert_counts = torch::full({experts}, -777, incoming.options());
    auto expert_offsets = torch::full({experts + 1}, -777, incoming.options());
    // Numerical probes retain totals on-device; runtime uses mapped host storage.
    auto totals = torch::full({2}, -777, incoming.options());
    const auto select_alignment = [&]<int64_t kExpertAlignment>() {
        const auto launch = [&]<int kWorldSize, int kExpertsPerRank>() {
            probe_finalize_receive_counts_kernel<kWorldSize, kExpertsPerRank, kExpertAlignment>
                <<<1, 128, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(incoming.data_ptr<int64_t>(),
                                                                            rank_prefix.data_ptr<int64_t>(),
                                                                            expert_counts.data_ptr<int64_t>(),
                                                                            expert_offsets.data_ptr<int64_t>(),
                                                                            totals.data_ptr<int64_t>());
        };
        switch (ranks) {
            case 1:
                select_count_probe_experts<1>(experts, launch);
                break;
            case 2:
                select_count_probe_experts<2>(experts, launch);
                break;
            case 8:
                select_count_probe_experts<8>(experts, launch);
                break;
            case 32:
                select_count_probe_experts<32>(experts, launch);
                break;
            default:
                TORCH_CHECK(false, "unsupported count probe ranks");
        }
    };
    switch (alignment) {
        case 1:
            select_alignment.template operator()<1>();
            break;
        case 3:
            select_alignment.template operator()<3>();
            break;
        case 4:
            select_alignment.template operator()<4>();
            break;
        case 16:
            select_alignment.template operator()<16>();
            break;
        default:
            TORCH_CHECK(false, "unsupported test alignment");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rank_prefix, expert_counts, expert_offsets, totals};
}

template <int kWorldSize, int kExpertsPerRank, bool kReuseSlots>
__global__ void probe_payload_kernel(const int64_t* addresses,
                                     const at::BFloat16* x,
                                     const int64_t* indices,
                                     const float* weights,
                                     int64_t* slots,
                                     int rank,
                                     int64_t tokens,
                                     int64_t hidden,
                                     int64_t topk,
                                     int64_t capacity,
                                     int64_t send_next_offset,
                                     int64_t payload_offset,
                                     int64_t row_bytes) {
    nano_ep::dispatch_payload_rows<at::BFloat16, kWorldSize, kExpertsPerRank, kReuseSlots, 0>(
        addresses, x, indices, weights, slots, rank, tokens, hidden, topk, capacity, send_next_offset, payload_offset, row_bytes);
}

// Exercise the production payload device helper with caller-owned slots.
// Notify/exchange behavior is tested through dispatch_main.
void run_dispatch_payload(const torch::Tensor& addresses,
                          const torch::Tensor& x,
                          const torch::Tensor& indices,
                          const c10::optional<torch::Tensor>& weights,
                          const torch::Tensor& slots,
                          int rank,
                          int64_t num_experts,
                          int64_t max_tokens,
                          bool reuse_slots) {
    c10::cuda::CUDAGuard guard(x.device());
    const int world_size = addresses.numel();
    const auto layout = nano_ep::WindowLayout::make(world_size, max_tokens, x.size(1), indices.size(1), num_experts);
    if (x.size(0) == 0)
        return;
    constexpr int threads = 256;
    const int blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const auto geometry = [&]<int kWorldSize, int kExpertsPerRank>() {
        const auto launch = [&]<bool kReuseSlots>() {
            probe_payload_kernel<kWorldSize, kExpertsPerRank, kReuseSlots>
                <<<blocks, threads, 0, stream>>>(addresses.data_ptr<int64_t>(),
                                                 x.data_ptr<at::BFloat16>(),
                                                 indices.data_ptr<int64_t>(),
                                                 weights.has_value() ? weights->data_ptr<float>() : nullptr,
                                                 slots.data_ptr<int64_t>(),
                                                 rank,
                                                 x.size(0),
                                                 x.size(1),
                                                 indices.size(1),
                                                 max_tokens,
                                                 layout.send_next_offset,
                                                 layout.payload_offset,
                                                 nano_ep::WindowLayout::payload_row_bytes(x.size(1), indices.size(1)));
        };
        if (reuse_slots)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    };
    if (world_size == 1 && num_experts == 3)
        geometry.template operator()<1, 3>();
    else if (world_size == 1 && num_experts == 4)
        geometry.template operator()<1, 4>();
    else if (world_size == 2 && num_experts == 4)
        geometry.template operator()<2, 2>();
    else if (world_size == 8 && num_experts == 16)
        geometry.template operator()<8, 2>();
    else if (world_size == 32 && num_experts == 32)
        geometry.template operator()<32, 1>();
    else
        TORCH_CHECK(false, "unsupported payload probe geometry");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    module.def("make_dispatch_state", [](const torch::Tensor& stage,
                                          const std::vector<torch::Tensor>& metadata,
                                          const torch::Tensor& counts,
                                          int64_t hidden,
                                          int64_t rank,
                                          bool prepared) {
        return nano_ep::DispatchState{metadata, counts, stage, hidden, torch::kBFloat16, rank, prepared};
    });
    module.def("run_finalize_receive_counts", &run_finalize_receive_counts);
    module.def("run_notify_peer_counts", &run_notify_peer_counts);
    module.def("run_dispatch_payload", &run_dispatch_payload);
    module.def("read_peer_words", &read_peer_words);
    module.def("wait_for_arrival", &wait_for_arrival);
}
