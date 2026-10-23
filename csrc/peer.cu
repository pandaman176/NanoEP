#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdio>
#include <cstring>
#include <cuda/atomic>
#include <exception>
#include <limits>
#include <nano_ep/peer.hpp>
#include <utility>

namespace nano_ep {

PeerRuntime::PeerRuntime(int rank, int world_size, int64_t num_bytes) : rank_(rank), world_size_(world_size), num_bytes_(num_bytes) {
    // Input Checks
    TORCH_CHECK(world_size > 0, "Invalid world size");
    TORCH_CHECK(rank >= 0 && rank < world_size, "Invalid rank");
    TORCH_CHECK(num_bytes > 0, "Invalid number of bytes");

    // Get Device
    C10_CUDA_CHECK(cudaGetDevice(&device_));

    auto stream = c10::cuda::getCurrentCUDAStream(device_);
    peer_ptrs_.resize(world_size_, nullptr);

    C10_CUDA_CHECK(cudaMalloc(&local_ptr_, static_cast<size_t>(num_bytes_)));

    try {
        C10_CUDA_CHECK(cudaMemsetAsync(local_ptr_, 0, static_cast<size_t>(num_bytes_), stream.stream()));

        C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    } catch (...) {
        // Cleanup on failure
        cudaFree(local_ptr_);
        local_ptr_ = nullptr;
        throw;
    }

    peer_ptrs_[rank_] = local_ptr_;
}

PeerRuntime::~PeerRuntime() noexcept {
    if (closed_)
        return;

    try {
        c10::cuda::CUDAGuard device_guard(device_);
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        close_peer_mappings();
        if (!exported_) {
            close();
        } else {
            // Importers' progress is unknown without collective close.
            std::fprintf(stderr, "NanoEP rank %d: exported window retained; explicit collective close is required\n", rank_);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NanoEP rank %d: local destructor cleanup failed: %s\n", rank_, error.what());
    } catch (...) {
        std::fprintf(stderr, "NanoEP rank %d: local destructor cleanup failed\n", rank_);
    }
}

void PeerRuntime::barrier(int64_t stream_handle) {
    TORCH_CHECK(last_barrier_epoch_ < std::numeric_limits<int64_t>::max(), "Barrier epoch exhausted");
    barrier(stream_handle, last_barrier_epoch_ + 1);
}

IpcHandleBytes PeerRuntime::export_local_handle() const {
    TORCH_CHECK(!closing_ && !closed_, "PeerRuntime is closing or closed");

    c10::cuda::CUDAGuard device_guard(device_);

    cudaIpcMemHandle_t handle{};
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, local_ptr_));
    exported_ = true;

    IpcHandleBytes bytes{};
    std::memcpy(bytes.data(), &handle, sizeof(handle));
    return bytes;
}

void PeerRuntime::import_peer_handles(const std::vector<IpcHandleBytes>& handles) {
    TORCH_CHECK(!closing_ && !closed_, "PeerRuntime is closing or closed");
    TORCH_CHECK(handles.size() == static_cast<size_t>(world_size_), "number of handles != world_size");
    TORCH_CHECK(!peer_addresses_.defined(), "peer_addresses already set");

    c10::cuda::CUDAGuard device_guard(device_);

    // Commit the mappings only after all peers and the device table are ready.
    std::vector<void*> imported(world_size_, nullptr);
    imported[rank_] = local_ptr_;

    try {
        for (int peer = 0; peer < world_size_; ++peer) {
            if (peer == rank_)
                continue;

            cudaIpcMemHandle_t handle{};
            std::memcpy(&handle, handles[peer].data(), sizeof(handle));
            C10_CUDA_CHECK(cudaIpcOpenMemHandle(&imported[peer], handle, cudaIpcMemLazyEnablePeerAccess));

            cudaPointerAttributes attributes{};
            C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, imported[peer]));
            TORCH_CHECK(attributes.device != device_, "Ranks must use distinct GPUs");

            int can_access = 0;
            C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_, attributes.device));
            TORCH_CHECK(can_access, "GPU ", device_, " cannot directly access rank ", peer, "'s GPU ", attributes.device);

            int native_atomics = 0;
            C10_CUDA_CHECK(cudaDeviceGetP2PAttribute(&native_atomics, cudaDevP2PAttrNativeAtomicSupported, device_, attributes.device));
            TORCH_CHECK(native_atomics, "GPU ", device_, " lacks native peer atomics to rank ", peer, "'s GPU ", attributes.device);
        }

        auto host_addresses = torch::empty({world_size_}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* addresses = host_addresses.data_ptr<int64_t>();
        for (int peer = 0; peer < world_size_; ++peer) {
            addresses[peer] = reinterpret_cast<int64_t>(imported[peer]);
        }

        // A blocking upload keeps the host table alive until the transfer completes.
        auto device_addresses = host_addresses.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_),
                                                  /*non_blocking=*/false);

        peer_ptrs_ = std::move(imported);
        peer_addresses_ = std::move(device_addresses);
    } catch (...) {
        for (int peer = 0; peer < world_size_; ++peer) {
            if (peer != rank_ && imported[peer] != nullptr) {
                // Retain failed mappings for the explicit collective teardown.
                if (cudaIpcCloseMemHandle(imported[peer]) != cudaSuccess) {
                    peer_ptrs_[peer] = imported[peer];
                    closing_ = true;
                }
            }
        }
        throw;
    }
}

torch::Tensor PeerRuntime::peer_addresses() const {
    TORCH_CHECK(!closing_ && !closed_, "PeerRuntime is closing or closed");
    TORCH_CHECK(peer_addresses_.defined(), "peer_addresses not imported");

    return peer_addresses_;
}

torch::Tensor PeerRuntime::local_window() const {
    TORCH_CHECK(!closing_ && !closed_, "PeerRuntime is closing or closed");

    return torch::from_blob(local_ptr_, {num_bytes_}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_));
}

__global__ void barrier_kernel(const int64_t* peer_addresses,
                               uint64_t* local_window,
                               int64_t rank,
                               int64_t world_size,
                               uint64_t epoch,
                               unsigned long long timeout_cycles) {
    const int64_t peer_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (peer_id >= world_size)
        return;

    using Signal = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>;

    // Send
    uint64_t* peer_window = reinterpret_cast<uint64_t*>(peer_addresses[peer_id]);
    Signal outgoing(peer_window[rank]);
    outgoing.store(epoch, cuda::memory_order_release);

    // Receive
    Signal incoming(local_window[peer_id]);
    const auto start = clock64();
    while (true) {
        uint64_t observed = incoming.load(cuda::memory_order_acquire);
        if (observed >= epoch) {
            break;
        }

        if (clock64() - start > timeout_cycles) {
            printf("Rank %lld timed out waiting for rank %lld at epoch %llu (observed %llu)\n",
                   static_cast<long long>(rank),
                   static_cast<long long>(peer_id),
                   static_cast<unsigned long long>(epoch),
                   static_cast<unsigned long long>(observed));

            asm volatile("trap;");
        }

        __nanosleep(64);
    }
}

void PeerRuntime::barrier(int64_t stream_handle, int64_t epoch) {
    TORCH_CHECK(!closing_ && !closed_, "PeerRuntime is closing or closed");
    TORCH_CHECK(peer_addresses_.defined(), "Peer handles are not imported");
    TORCH_CHECK(epoch > last_barrier_epoch_, "Barrier epoch must strictly increase");

    constexpr int threads = 32;

    TORCH_CHECK(world_size_ <= threads, "Too many ranks");
    TORCH_CHECK(num_bytes_ >= world_size_ * static_cast<int64_t>(sizeof(uint64_t)), "Window is too small for barrier signals");

    c10::cuda::CUDAGuard device_guard(device_);

    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    constexpr unsigned long long timeout_cycles = 20'000'000'000ULL;

    TORCH_CHECK(last_barrier_epoch_ == 0 || stream == barrier_stream_, "Barriers must use the same stream");

    barrier_kernel<<<1, threads, 0, stream>>>(peer_addresses_.data_ptr<int64_t>(),
                                              reinterpret_cast<uint64_t*>(local_ptr_),
                                              rank_,
                                              world_size_,
                                              static_cast<uint64_t>(epoch),
                                              timeout_cycles);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    barrier_stream_ = stream;
    last_barrier_epoch_ = epoch;
}

void PeerRuntime::close_peer_mappings() {
    if (closed_ || peer_mappings_closed_)
        return;

    c10::cuda::CUDAGuard device_guard(device_);
    closing_ = true;

    for (int peer = 0; peer < world_size_; ++peer) {
        if (peer != rank_ && peer_ptrs_[peer] != nullptr) {
            C10_CUDA_CHECK(cudaIpcCloseMemHandle(peer_ptrs_[peer]));
            peer_ptrs_[peer] = nullptr;
        }
    }

    peer_addresses_ = torch::Tensor();
    peer_mappings_closed_ = true;
}

void PeerRuntime::close() {
    if (closed_)
        return;

    TORCH_CHECK(peer_mappings_closed_, "Call close_peer_mappings() and complete a host group barrier before close()");
    c10::cuda::CUDAGuard device_guard(device_);

    C10_CUDA_CHECK(cudaFree(local_ptr_));
    local_ptr_ = nullptr;
    peer_ptrs_[rank_] = nullptr;
    closed_ = true;
}

}  // namespace nano_ep
