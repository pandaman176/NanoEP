#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cuda/atomic>

namespace nano_ep {

inline constexpr unsigned long long kBarrierTimeoutCycles = 20'000'000'000ULL;

// Only the named group participates; other warp roles can keep working.
template <int kNumThreads>
__device__ __forceinline__ void notify_group_barrier() {
    static_assert(kNumThreads > 0 && kNumThreads % 32 == 0);
    asm volatile("bar.sync 1, %0;" : : "n"(kNumThreads) : "memory");
}

// One CTA exchanges rank epochs with system-scope release/acquire.
// The caller orders participating writers before the publishing CTA.
__device__ __forceinline__ void peer_barrier(
    const int64_t* peer_addresses, uint64_t* local_signals, int rank, int world_size, uint64_t epoch) {
    const int peer = threadIdx.x;
    if (peer >= world_size)
        return;

    using Signal = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>;
    auto* remote_signals = reinterpret_cast<uint64_t*>(peer_addresses[peer]);
    Signal outgoing(remote_signals[rank]);
    outgoing.store(epoch, cuda::memory_order_release);

    Signal incoming(local_signals[peer]);
    const auto start = clock64();
    while (true) {
        const uint64_t observed = incoming.load(cuda::memory_order_acquire);
        if (observed >= epoch)
            break;
        if (clock64() - start > kBarrierTimeoutCycles) {
            printf(
                "Rank %d timed out waiting for rank %d at epoch %llu (observed "
                "%llu)\n",
                rank,
                peer,
                static_cast<unsigned long long>(epoch),
                static_cast<unsigned long long>(observed));
            asm volatile("trap;");
        }
        __nanosleep(64);
    }
}

}  // namespace nano_ep
