#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nano_ep {

// Bulk copies move 16-byte-multiple payloads between 16-byte-aligned global addresses and
// 16-byte-aligned shared addresses. Staged buffers follow the 32-byte granularity of the wire
// layout, so every field keeps the alignment it has at its destination. Issue each load, wait,
// and store from the appropriate lane, then reconverge before reusing the shared buffer.

// generic -> shared address
__device__ __forceinline__ uint32_t tma_shared_address(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// L2 policy for the copied range: dead once consumed, or reused right away.
enum TmaCacheHint : uint64_t {
    kTmaEvictFirst = 0x12f0000000000000ULL,
    kTmaEvictNormal = 0x1000000000000000ULL,
};

// All lanes elect at the issue site. Only the elected lane returns true.
__device__ __forceinline__ int elect_one_sync() {
    int pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffffu));
    return pred;
}

// The init is published to the async proxy so the first bulk copy observes the barrier state.
__device__ __forceinline__ void tma_mbarrier_init(uint64_t* mbar, int arrive_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(tma_shared_address(mbar)), "r"(arrive_count));
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

// Reserves the byte count that the pending bulk load must deliver before the phase flips.
__device__ __forceinline__ void tma_mbarrier_expect_tx(uint64_t* mbar, int num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(tma_shared_address(mbar)), "r"(num_bytes));
}

// Wait for the load. The caller advances the expected parity before the next load.
__device__ __forceinline__ void tma_mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred ready;\n\t"
        "WAIT_TMA:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1, %2;\n\t"
        "@ready bra TMA_READY;\n\t"
        "bra WAIT_TMA;\n\t"
        "TMA_READY:\n\t"
        "}" ::"r"(tma_shared_address(mbar)),
        "r"(phase),
        "r"(0x989680u));
}

// Publishes ordinary shared stores to the async proxy, so a following bulk store reads them.
__device__ __forceinline__ void tma_fence_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Orders a thread's completed bulk stores before a system-scope signal that peers observe.
__device__ __forceinline__ void tma_fence_system() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void tma_load_1d(
    void* smem, const void* global, uint64_t* mbar, int num_bytes, TmaCacheHint hint = kTmaEvictFirst) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
                     tma_shared_address(smem)),
                 "l"(global),
                 "r"(num_bytes),
                 "r"(tma_shared_address(mbar)),
                 "l"(static_cast<uint64_t>(hint))
                 : "memory");
}

__device__ __forceinline__ void tma_store_1d(void* global, const void* smem, int num_bytes, TmaCacheHint hint = kTmaEvictNormal) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(global),
                 "r"(tma_shared_address(smem)),
                 "r"(num_bytes),
                 "l"(static_cast<uint64_t>(hint))
                 : "memory");
}

__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Waits until at most kPending committed groups are still in flight.
template <int kPending = 0>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kPending) : "memory");
}

// Waits only until the engine has read the shared source, so the buffer is reusable while the
// global write is still travelling.
template <int kPending = 0>
__device__ __forceinline__ void tma_store_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(kPending) : "memory");
}

}  // namespace nano_ep
