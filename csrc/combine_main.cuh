#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cuda/std/bit>
#include <layout.hpp>
#include <nano_ep/barrier.cuh>
#include <nano_ep/tma.cuh>

namespace nano_ep {

struct PackedBf16 {
    uint16_t bits[8];
};
static_assert(sizeof(PackedBf16) == sizeof(int4));

template <int kTopk>
__device__ __forceinline__ int compute_expanded_slots(int64_t lane_slot, unsigned valid_mask, int64_t (&slots)[kTopk]) {
    // All lanes participate; compact valid slots in ascending original-k order.
    // Only slots[0,count) are initialized and consumed.
    const int count = __popc(valid_mask);
    unsigned remaining = valid_mask;
    for (int i = 0; i < count; ++i) {
        const int k = __ffs(remaining) - 1;
        slots[i] = __shfl_sync(0xffffffffu, lane_slot, k);
        remaining &= remaining - 1;
    }
    return count;
}

// One activation row staged per warp, so the peer write leaves the warp as a single bulk copy.
template <int kHidden>
inline constexpr auto kCombineReturnPlan = HiddenRowPlan::make(kHidden, true);

template <int kHidden,
          int kTopk,
          int kWorldSize,
          int64_t kMaxTokens,
          typename data_t,
          bool kExpand,
          bool kAllowMultipleReduction,
          bool kWithWeights>
__device__ __forceinline__ void combine_return_rows(const int64_t* peer_addresses,  // [R], includes self
                                                    const data_t* x,                // compact [N,H], expanded [C,H]
                                                    const float* weights,           // compact [N,K], expanded [C], or nullptr
                                                    const int32_t* recv_metadata,   // [N,K+2]
                                                    int rank,
                                                    int64_t recv_count,    // N, one warp per row
                                                    int64_t input_rows,    // N or C; bounds for expanded x/weights reads
                                                    int64_t return_offset  // bytes from each peer window base
) {
    constexpr int kWarpSize = 32;
    constexpr int64_t hidden = kHidden;
    constexpr int64_t return_row_bytes = WindowLayout::return_weights_offset(kHidden) + ((kTopk * sizeof(float) + 31) / 32) * 32;
    constexpr int kElemsPerVector = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int kVectorsPerLane = kHidden / (kWarpSize * kElemsPerVector);
    constexpr int kUnrollFactor = [] {
        for (int factor = 4; factor >= 1; --factor)
            if (kVectorsPerLane % factor == 0)
                return factor;
        return 1;
    }();
    static_assert(kHidden > 0 && kHidden % (kWarpSize * kElemsPerVector) == 0);
    static_assert(kTopk > 0 && kTopk <= 32 && kWorldSize > 0 && kWorldSize <= 32 && kMaxTokens > 0);

    // Each warp processes one source token per iteration
    const int lane = threadIdx.x % kWarpSize;
    const int warp_idx = threadIdx.x / kWarpSize;
    const int64_t global_warp_idx = static_cast<int64_t>(warp_idx) * gridDim.x + blockIdx.x;
    const int64_t token_stride = static_cast<int64_t>(blockDim.x / kWarpSize) * gridDim.x;

    constexpr auto kPlan = kCombineReturnPlan<kHidden>;
    static_assert(kPlan.stage_bytes <= kMaxSmemBytes, "one staged row must fit in shared memory");
    constexpr int kXBytes = static_cast<int>(kPlan.row_bytes);
    // Bulk copies move a 16-byte-multiple size between 16-byte-aligned addresses.
    static_assert(kXBytes % 16 == 0 && kPlan.stage_bytes % 16 == 0);

    extern __shared__ __align__(32) uint8_t stage_smem[];
    uint8_t* const stage = stage_smem + kPlan.stage_offset(warp_idx);
    auto* mbar = reinterpret_cast<uint64_t*>(stage + kPlan.row_bytes);
    if (elect_one_sync())
        tma_mbarrier_init(mbar);
    __syncwarp();
    // Full-warp elections select the same lane, which owns phase and the bulk stores.
    uint32_t phase = 0;

    for (int64_t row = global_warp_idx; row < recv_count; row += token_stride) {
        const int64_t src_rank = recv_metadata[row * (kTopk + 2)];
        const int64_t src_token = recv_metadata[row * (kTopk + 2) + 1];
        if (src_rank < 0 || src_rank >= kWorldSize || src_token < 0 || src_token >= kMaxTokens) {
            if (lane == 0)
                printf("NanoEP combine_return invalid source: row=%lld rank=%lld token=%lld\n",
                       static_cast<long long>(row),
                       static_cast<long long>(src_rank),
                       static_cast<long long>(src_token));
            asm volatile("trap;");
            return;
        }

        const bool valid_k = lane < kTopk && recv_metadata[row * (kTopk + 2) + 2 + lane] >= 0;
        const unsigned valid_mask = __ballot_sync(0xffffffffu, valid_k);
        int64_t lane_slot = -1;
        if constexpr (kExpand) {
            if (valid_k) {
                lane_slot = recv_metadata[row * (kTopk + 2) + 2 + lane];
            }
        }

        if (valid_mask == 0) {
            if (lane == 0)
                printf("NanoEP combine_return row has no local assignment: row=%lld\n", static_cast<long long>(row));
            asm volatile("trap;");
            return;
        }
        if constexpr (kExpand) {
            if (valid_k) {
                const int64_t slot = lane_slot;
                if (slot < 0 || slot >= input_rows) {
                    printf("NanoEP combine_return invalid expanded slot: row=%lld k=%d slot=%lld\n",
                           static_cast<long long>(row),
                           lane,
                           static_cast<long long>(slot));
                    asm volatile("trap;");
                    return;
                }
            }
        }
        __syncwarp();

        auto* peer_window = reinterpret_cast<uint8_t*>(peer_addresses[src_rank]);
        if constexpr (kExpand && !kAllowMultipleReduction) {
            // Keep assignments separate until the source's final reduction.
#pragma unroll
            for (int64_t k = 0; k < kTopk; ++k) {
                // check if current k is valid
                if ((valid_mask & (1u << k)) == 0)
                    continue;
                const int64_t slot = __shfl_sync(0xffffffffu, lane_slot, k);
                const int64_t peer_idx = k * kMaxTokens + src_token;
                auto* peer_row = peer_window + return_offset + peer_idx * return_row_bytes;
                if (elect_one_sync()) {
                    tma_store_wait<0>();
                    tma_mbarrier_expect_tx(mbar, kXBytes);
                    tma_load_1d(stage, x + slot * kHidden, mbar, kXBytes);
                    tma_mbarrier_wait(mbar, phase);
                    phase ^= 1;
                    tma_store_1d(peer_row, stage, kXBytes);
                    tma_store_commit();
                }
                __syncwarp();
                if constexpr (kWithWeights) {
                    if (lane == 0) {
                        auto* peer_weights = reinterpret_cast<float*>(peer_row + WindowLayout::return_weights_offset(hidden));
                        peer_weights[k] = weights[slot];
                    }
                }
            }
        } else {
            // Compact and locally reduced expanded x each return one vector per row.
            const int first_valid_k = __ffs(valid_mask) - 1;
            const bool use_rank_layout = kAllowMultipleReduction && kWorldSize <= kTopk;
            const int partition = use_rank_layout ? rank : first_valid_k;
            const int64_t peer_idx = partition * kMaxTokens + src_token;
            auto* peer_row = peer_window + return_offset + peer_idx * return_row_bytes;

            const bool no_local_reduce = !kExpand || (kAllowMultipleReduction && __popc(valid_mask) == 1);
            if (no_local_reduce) {
                int64_t input_row = row;
                if constexpr (kExpand) {
                    // Broadcast the only valid assignment's slot to the whole warp.
                    input_row = __shfl_sync(0xffffffffu, lane_slot, first_valid_k);
                }
                if (elect_one_sync()) {
                    tma_store_wait<0>();
                    tma_mbarrier_expect_tx(mbar, kXBytes);
                    tma_load_1d(stage, x + input_row * kHidden, mbar, kXBytes);
                    tma_mbarrier_wait(mbar, phase);
                    phase ^= 1;
                    tma_store_1d(peer_row, stage, kXBytes);
                    tma_store_commit();
                }
                __syncwarp();
            } else if constexpr (kExpand) {
                int64_t slots[kTopk];
                // number of valid k's in this row
                const int count = compute_expanded_slots(lane_slot, valid_mask, slots);

                auto* output_vectors = reinterpret_cast<int4*>(stage);
                for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
                    float2 accum[kUnrollFactor][kElemsPerVector / 2] = {};
                    for (int k = 0; k < count; ++k) {
                        const int64_t slot = slots[k];
                        const auto* input_vectors = reinterpret_cast<const int4*>(x + slot * kHidden);
                        for (int u = 0; u < kUnrollFactor; ++u) {
                            const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                            // Load 16bytes, treated as 8 2byte bits: value.bits[0..7]
                            const auto value = cuda::std::bit_cast<PackedBf16>(input_vectors[index]);
                            for (int j = 0; j < kElemsPerVector / 2; ++j) {
                                accum[u][j].x += __bfloat162float(__ushort_as_bfloat16(value.bits[2 * j]));
                                accum[u][j].y += __bfloat162float(__ushort_as_bfloat16(value.bits[2 * j + 1]));
                            }
                        }
                    }
                    // Overlap the first input loads with the previous store before reusing the stage.
                    if (step == 0) {
                        tma_store_wait<0>();
                        __syncwarp();
                    }
                    for (int u = 0; u < kUnrollFactor; ++u) {
                        const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                        PackedBf16 value;
                        for (int j = 0; j < kElemsPerVector / 2; ++j) {
                            value.bits[2 * j] = __bfloat16_as_ushort(__float2bfloat16_rn(accum[u][j].x));
                            value.bits[2 * j + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(accum[u][j].y));
                        }
                        output_vectors[index] = cuda::std::bit_cast<int4>(value);
                    }
                }
                tma_fence_shared();
                __syncwarp();
                if (elect_one_sync()) {
                    tma_store_1d(peer_row, stage, kXBytes);
                    tma_store_commit();
                }
                __syncwarp();
            }
            if constexpr (kWithWeights) {
                // One lane writes each original top-k position in the same return row.
                if (valid_k) {
                    auto* peer_weights = reinterpret_cast<float*>(peer_row + WindowLayout::return_weights_offset(hidden));
                    const int64_t weight_idx = kExpand ? lane_slot : row * kTopk + lane;
                    peer_weights[lane] = weights[weight_idx];
                }
            }
        }
        __syncwarp();
    }

    // Flush every issuing lane before the cooperative exit barrier.
    tma_store_commit();
    tma_store_wait<0>();
    __syncwarp();
}

template <int kHidden,
          int kTopk,
          int kWorldSize,
          int64_t kMaxTokens,
          typename data_t,
          bool kExpand,
          bool kAllowMultipleReduction,
          bool kWithWeights>
// One warp owns one staged row for the whole kernel, so the block shape is whatever the
// shared-memory budget holds up to the 32-warp cap.
__global__ void __launch_bounds__(kCombineReturnPlan<kHidden>.max_warps() * 32, 1) combine_main_kernel(const int64_t* peer_addresses,
                                                                                                       const data_t* x,
                                                                                                       const float* weights,
                                                                                                       const int32_t* recv_metadata,
                                                                                                       int rank,
                                                                                                       int64_t recv_count,
                                                                                                       int64_t input_rows,
                                                                                                       int64_t return_offset,
                                                                                                       int64_t entry_epoch) {
    auto grid = cooperative_groups::this_grid();
    auto* local_signals = reinterpret_cast<uint64_t*>(peer_addresses[rank]);

    // Previous stream work has finished before this kernel. Only CTA 0 exchanges
    // readiness; the grid waits before any CTA overwrites the shared arena.
    if (blockIdx.x == 0)
        peer_barrier(peer_addresses, local_signals, rank, kWorldSize, static_cast<uint64_t>(entry_epoch));

    // All CTAs wait for CTA 0
    grid.sync();

    combine_return_rows<kHidden, kTopk, kWorldSize, kMaxTokens, data_t, kExpand, kAllowMultipleReduction, kWithWeights>(
        peer_addresses, x, weights, recv_metadata, rank, recv_count, input_rows, return_offset);

    // Order all writers before CTA 0's cumulative system-scope release.
    // Empty ranks participate too. Stream ordering delays the epilogue until
    // the whole kernel ends, so no final grid sync is needed after the exchange.
    grid.sync();
    if (blockIdx.x == 0)
        peer_barrier(peer_addresses, local_signals, rank, kWorldSize, static_cast<uint64_t>(entry_epoch + 1));
}

}  // namespace nano_ep
