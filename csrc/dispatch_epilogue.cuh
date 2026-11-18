#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <layout.hpp>
#include <nano_ep/math.cuh>
#include <nano_ep/tma.cuh>

namespace nano_ep {

struct ReceiveAssignment {
    int64_t local_expert;
    int64_t slot;
};

// Each lane retains its assignment for immediate payload copying. The stored
// metadata owns the mapping used by combine and subsequent cached dispatches.
template <int kTopk, int kExpertsPerRank, bool kExpand>
__device__ __forceinline__ ReceiveAssignment build_receive_metadata_row(
    const int64_t* topk, int32_t* recv_metadata, unsigned long long* expert_cursor, int64_t row, int64_t rank, int lane) {
    const int64_t expert = lane < kTopk ? topk[lane] : -1;
    const int64_t expert_start = rank * kExpertsPerRank;
    const bool in_range = expert >= expert_start && expert - expert_start < kExpertsPerRank;
    const long long local_expert = in_range ? expert - expert_start : -1;
    const unsigned same_expert = __match_any_sync(0xffffffffu, local_expert);
    if (in_range && __popc(same_expert) != 1) {
        printf("NanoEP receive metadata duplicate expert: row=%lld expert=%lld\n",
               static_cast<long long>(row),
               static_cast<long long>(expert));
        asm volatile("trap;");
        return {-1, -1};
    }
    int64_t slot = -1;
    if (lane < kTopk) {
        const int64_t index = row * (kTopk + 2) + 2 + lane;
        if constexpr (kExpand) {
            if (in_range) {
                // Each expert owns a disjoint segment; only its cursor needs an atomic update.
                slot = static_cast<int64_t>(atomicAdd(expert_cursor + local_expert, 1ULL));
            }
        }
        recv_metadata[index] = static_cast<int32_t>(kExpand ? slot : local_expert);
    }
    return {local_expert, slot};
}

// All 32 lanes participate, including warps without a receive row.
template <int kHidden, int kExpertsPerRank, bool kWithWeights>
__device__ __forceinline__ void do_zero_padding(const int64_t* expert_counts,
                                                const int64_t* expert_offsets,
                                                nv_bfloat16* recv_x,
                                                float* recv_weights,
                                                uint8_t* zero_stage,
                                                int64_t warp_idx,
                                                int64_t total_warps,
                                                int lane) {
    constexpr int kWarpSize = 32;
    constexpr int kXBytes = kHidden * static_cast<int>(sizeof(nv_bfloat16));
    constexpr int kVectorsPerLane = kHidden / (sizeof(int4) / sizeof(nv_bfloat16)) / kWarpSize;
    constexpr int kUnrollFactor = [] {
        for (int factor = 4; factor >= 1; --factor)
            if (kVectorsPerLane % factor == 0)
                return factor;
        return 1;
    }();
    // Distribute padding ordinals across all warps, including those without a receive row.
    int64_t padding_idx = warp_idx;
    bool stage_zeroed = false;
    for (int base = 0; base < kExpertsPerRank; base += kWarpSize) {
        const int expert = base + lane;
        int64_t gap_start = 0, gap_size = 0;
        if (expert < kExpertsPerRank) {
            gap_start = expert_offsets[expert] + expert_counts[expert];
            gap_size = expert_offsets[expert + 1] - gap_start;
        }
        const int64_t gap_prefix = warp_inclusive_sum(gap_size, lane);
        // total number of padding rows in current wave
        const int64_t wave_gaps = __shfl_sync(0xffffffffu, gap_prefix, kWarpSize - 1);
        if (!stage_zeroed && padding_idx < wave_gaps) {
            // All live-row store issuers must finish before the warp overwrites its stage.
            __syncwarp();
            auto* zero_vectors = reinterpret_cast<int4*>(zero_stage);
            for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                for (int u = 0; u < kUnrollFactor; ++u) {
                    const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                    zero_vectors[index] = int4{0, 0, 0, 0};
                }
            }
            tma_fence_shared();
            __syncwarp();
            stage_zeroed = true;
        }
        for (; padding_idx < wave_gaps; padding_idx += total_warps) {
            // the first expert with gap_prefix > padding_idx is the owner
            const unsigned owners = __ballot_sync(0xffffffffu, gap_prefix > padding_idx);
            const int owner = __ffs(owners) - 1;

            const int64_t owner_start = __shfl_sync(0xffffffffu, gap_start, owner);
            // Exclusive sum of gap sizes.
            const int64_t owner_prefix = __shfl_sync(0xffffffffu, gap_prefix - gap_size, owner);
            const int64_t slot = owner_start + padding_idx - owner_prefix;
            if (elect_one_sync()) {
                tma_store_1d(recv_x + slot * kHidden, zero_stage, kXBytes);
                tma_store_commit();
                if constexpr (kWithWeights)
                    recv_weights[slot] = 0.0f;
            }
            __syncwarp();
        }
        padding_idx -= wave_gaps;
    }
    if (stage_zeroed)
        tma_store_wait<0>();
}

// Each warp stages the complete wire row with one bulk load.
template <int kHidden, int kTopk>
inline constexpr auto kDispatchEpiloguePlan = PayloadRowPlan::make(kHidden, kTopk);

template <int kHidden, int kTopk, int kExpertsPerRank, bool kCachedMode, bool kExpand, bool kWithWeights, bool kZeroPadding>
__global__ void __launch_bounds__(kDispatchEpiloguePlan<kHidden, kTopk>.total_warps * 32, 1)
    dispatch_epilogue_kernel(const uint8_t* stage_buffer,        // [R*Tmax, kRowBytes], source-major
                             int32_t* recv_metadata,             // [N,K+2], source and assignment mapping
                             unsigned long long* expert_cursor,  // [L], fresh expanded mode only
                             const int64_t* expert_counts,       // [L], actual assignment counts
                             const int64_t* expert_offsets,      // [L+1], padded offsets
                             nv_bfloat16* recv_x,                // compact [N,H], expanded [C,H]
                             int64_t* recv_topk_idx,             // compact [N,K], expanded nullptr
                             float* recv_weights,                // compact [N,K], expanded [C], optional
                             int64_t recv_count,                 // N
                             const int64_t* recv_rank_prefix,    // [R], inclusive counts by source
                             int world_size,
                             int64_t max_tokens,
                             int64_t rank,
                             bool use_pdl) {
    static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0 && kTopk > 0 && kTopk <= 32);
    constexpr auto kPlan = kDispatchEpiloguePlan<kHidden, kTopk>;
    static_assert(kPlan.payload_warps >= 1, "payload row staging does not fit in shared memory");
    constexpr int kRowBytes = static_cast<int>(kPlan.row_bytes);
    constexpr int kWarpSize = 32;
    const int lane = threadIdx.x % kWarpSize;
    const int warp_idx = threadIdx.x / kWarpSize;
    extern __shared__ __align__(32) uint8_t stage_smem[];
    constexpr int64_t kXOffset = WindowLayout::payload_x_offset(kTopk);
    constexpr int kXBytes = kHidden * static_cast<int>(sizeof(nv_bfloat16));
    static_assert(kRowBytes % 16 == 0 && kXOffset % 16 == 0 && kXBytes % 16 == 0 && kPlan.stage_bytes % 16 == 0);
    auto* stage = stage_smem + kPlan.stage_offset(warp_idx);
    auto* mbar = reinterpret_cast<uint64_t*>(stage + kPlan.row_bytes);
    if (elect_one_sync())
        tma_mbarrier_init(mbar);
    __syncwarp();

    // Wait for the main kernel before reading receive metadata or the peer payload.
    if (use_pdl)
        cudaGridDependencySynchronize();

    const int64_t source_end = lane < world_size ? recv_rank_prefix[lane] : 0;
    const int64_t shifted_end = __shfl_up_sync(0xffffffffu, source_end, 1);
    const int64_t source_begin = lane == 0 ? 0 : shifted_end;

    // Distribute the first rows across blocks before assigning another warp to each block.
    const int64_t global_warp_idx = static_cast<int64_t>(warp_idx) * gridDim.x + blockIdx.x;
    const int64_t total_warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarpSize);
    uint32_t phase = 0;

    for (int64_t row = global_warp_idx; row < recv_count; row += total_warps) {
        // each lane represent a source rank, and the first lane with row < source_end is the owner of this row
        const unsigned sources = __ballot_sync(0xffffffffu, lane < world_size && row < source_end);
        const int source = __ffs(sources) - 1;
        const int64_t begin = __shfl_sync(0xffffffffu, source_begin, source);
        const int64_t physical_slot = source * max_tokens + row - begin;
        // ┌──────────┬───────────┬───────────┬────────────┬─────────┬─────────┐
        // │ src_rank │ src_token │ topk IDs  │ weights    │ x       │ padding │
        // │ int64    │ int64     │ int64[K]  │ fp32[K]    │ bf16[H] │         │
        // └──────────┴───────────┴───────────┴────────────┴─────────┴─────────┘
        const auto* payload = stage_buffer + physical_slot * kRowBytes;
        // Drain each lane's previous stores before reusing the stage.
        tma_store_wait<0>();
        __syncwarp();
        if (elect_one_sync()) {
            tma_mbarrier_expect_tx(mbar, kRowBytes);
            tma_load_1d(stage, payload, mbar, kRowBytes);
        }

        // Use global top-k indices or cached mappings while the bulk load is in flight.
        ReceiveAssignment assignment{-1, -1};
        if constexpr (kCachedMode) {
            if (lane < kTopk) {
                if constexpr (kExpand)
                    assignment.slot = recv_metadata[row * (kTopk + 2) + 2 + lane];
                else
                    assignment.local_expert = recv_metadata[row * (kTopk + 2) + 2 + lane];
            }
        } else {
            const auto* topk = reinterpret_cast<const int64_t*>(payload + WindowLayout::kPayloadTopkOffset);
            assignment = build_receive_metadata_row<kTopk, kExpertsPerRank, kExpand>(topk, recv_metadata, expert_cursor, row, rank, lane);
        }
        if constexpr (!kExpand) {
            if (lane < kTopk)
                recv_topk_idx[row * kTopk + lane] = assignment.local_expert;
        }

        if (elect_one_sync()) {
            tma_mbarrier_wait(mbar, phase);
            phase ^= 1;
        }
        __syncwarp();

        if constexpr (kExpand) {
            // One owned slot per lane, so each sender broadcasts the staged row itself.
            if (assignment.slot >= 0) {
                auto* dst_row = reinterpret_cast<uint8_t*>(recv_x + assignment.slot * kHidden);
                tma_store_1d(dst_row, stage + kXOffset, kXBytes);
                tma_store_commit();
            }
        } else if (elect_one_sync()) {
            // Every received row lands in one output row, so one bulk store moves it.
            tma_store_1d(recv_x + row * kHidden, stage + kXOffset, kXBytes);
            tma_store_commit();
        }

        if constexpr (kWithWeights) {
            const auto* weights = reinterpret_cast<const float*>(stage + WindowLayout::payload_weights_offset(kTopk));
            if constexpr (kExpand) {
                if (assignment.slot >= 0)
                    recv_weights[assignment.slot] = weights[lane];
            } else if (lane < kTopk) {
                recv_weights[row * kTopk + lane] = weights[lane];
            }
        }
        if constexpr (!kCachedMode) {
            if (lane == 0) {
                const auto* provenance = reinterpret_cast<const int64_t*>(stage);
                recv_metadata[row * (kTopk + 2)] = static_cast<int32_t>(provenance[0]);
                recv_metadata[row * (kTopk + 2) + 1] = static_cast<int32_t>(provenance[1]);
            }
        }
        __syncwarp();
    }
    tma_store_wait<0>();

    if constexpr (kExpand && kZeroPadding) {
        do_zero_padding<kHidden, kExpertsPerRank, kWithWeights>(
            expert_counts, expert_offsets, recv_x, recv_weights, stage + kXOffset, global_warp_idx, total_warps, lane);
    }
    tma_fence_system();
}

}  // namespace nano_ep
