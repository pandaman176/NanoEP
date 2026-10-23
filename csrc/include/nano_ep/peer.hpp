#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nano_ep {

using IpcHandleBytes = std::array<std::byte, sizeof(cudaIpcMemHandle_t)>;

// Owns a fixed-capacity CUDA IPC allocation and imported peer mappings.
// No CPU operation in this interface transports token or routing payloads.
// Window offsets have identical meanings on each rank; virtual addresses
// need not match. A later NCCL window backend can preserve this contract.
class PeerRuntime {
public:
    PeerRuntime(int rank, int world_size, int64_t num_bytes);
    ~PeerRuntime() noexcept;
    PeerRuntime(const PeerRuntime&) = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    IpcHandleBytes export_local_handle() const;
    void import_peer_handles(const std::vector<IpcHandleBytes>& handles);
    torch::Tensor peer_addresses() const;
    torch::Tensor local_window() const;
    int rank() const noexcept { return rank_; }
    int world_size() const noexcept { return world_size_; }
    // Advance the barrier epoch when entering each collective communication phase.
    void barrier(int64_t stream_handle);
    void barrier(int64_t stream_handle, int64_t epoch);
    // Caller drains all window users before phase 1, then completes a host
    // group barrier across all importers before phase 2 (close).
    void close_peer_mappings();
    void close();

private:
    int rank_;
    int world_size_;
    int64_t num_bytes_;
    int device_ = -1;
    void* local_ptr_ = nullptr;
    std::vector<void*> peer_ptrs_;
    torch::Tensor peer_addresses_;
    mutable bool exported_ = false;
    bool closing_ = false;
    bool peer_mappings_closed_ = false;
    bool closed_ = false;
    int64_t last_barrier_epoch_ = 0;
    cudaStream_t barrier_stream_ = nullptr;
};

// Future backends consume offsets, never pointers from another process.
struct WindowOffset {
    uint64_t bytes;
};

struct NativeRouteView {
    const int64_t* topk_idx;
    const int64_t* destination_slots;
    const int64_t* recv_source_ranks;
    const int64_t* recv_source_tokens;
    const int64_t* expanded_slots;
    int64_t num_source_tokens;
    int64_t num_recv_tokens;
    int64_t num_topk;
};

}  // namespace nano_ep
