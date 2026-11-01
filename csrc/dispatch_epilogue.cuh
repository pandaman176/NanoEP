#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <cstdio>

#include "layout.hpp"

namespace nano_ep {

// All 32 lanes participate; each lane receives the sum of lanes [0, lane].
__device__ __forceinline__ int64_t warp_inclusive_sum(int64_t value, int lane) {
    constexpr int kWarpSize = 32;
#pragma unroll
    for (int delta = 1; delta < kWarpSize; delta *= 2) {
        const int64_t previous = __shfl_up_sync(0xffffffffu, value, delta);
        if (lane >= delta)
            value += previous;
    }
    return value;
}

struct ReceiveAssignment {
    int64_t local_expert;
    int64_t slot;
};

// Each lane retains its assignment for immediate payload copying. The stored
// metadata owns the mapping used by combine and subsequent cached dispatches.
template <int kTopk, int kExpertsPerRank>
__device__ __forceinline__ ReceiveAssignment build_receive_metadata_row(const uint8_t* payload,
                                                                        int64_t* src_ranks,
                                                                        int64_t* src_tokens,
                                                                        int64_t* local_topk_idx,
                                                                        int64_t* expanded_slots,
                                                                        unsigned long long* expert_cursor,
                                                                        int64_t row,
                                                                        int64_t rank,
                                                                        int lane) {
    // ┌──────────┬───────────┬───────────┬────────────┬─────────┬─────────┐
    // │ src_rank │ src_token │ topk IDs  │ weights    │ x       │ padding │
    // │ int64    │ int64     │ int64[K]  │ fp32[K]    │ bf16[H] │         │
    // └──────────┴───────────┴───────────┴────────────┴─────────┴─────────┘
    const auto* provenance = reinterpret_cast<const int64_t*>(payload);
    const auto* topk = reinterpret_cast<const int64_t*>(payload + WindowLayout::kPayloadTopkOffset);
    if (lane == 0) {
        src_ranks[row] = provenance[0];
        src_tokens[row] = provenance[1];
    }

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
        const int64_t index = row * kTopk + lane;
        if (in_range) {
            // Each expert owns a disjoint segment; only its cursor needs an atomic update.
            slot = static_cast<int64_t>(atomicAdd(expert_cursor + local_expert, 1ULL));
        }
        local_topk_idx[index] = local_expert;
        expanded_slots[index] = slot;
    }
    return {local_expert, slot};
}

// All 32 lanes participate, including warps without a receive row.
template <int kHidden, int kExpertsPerRank, int kUnrollFactor, bool kWithWeights>
__device__ __forceinline__ void do_zero_padding(const int64_t* expert_counts,
                                                const int64_t* expert_offsets,
                                                at::BFloat16* recv_x,
                                                float* recv_weights,
                                                int64_t warp_idx,
                                                int64_t total_warps,
                                                int lane) {
    constexpr int kWarpSize = 32;
    constexpr int kVectorsPerLane = kHidden / (sizeof(int4) / sizeof(at::BFloat16)) / kWarpSize;
    // Distribute padding ordinals across all warps, including those without a receive row.
    int64_t padding_idx = warp_idx;
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
        for (; padding_idx < wave_gaps; padding_idx += total_warps) {
            // the first expert with gap_prefix > padding_idx is the owner
            const unsigned owners = __ballot_sync(0xffffffffu, gap_prefix > padding_idx);
            const int owner = __ffs(owners) - 1;

            const int64_t owner_start = __shfl_sync(0xffffffffu, gap_start, owner);
            // Exclusive sum of gap sizes.
            const int64_t owner_prefix = __shfl_sync(0xffffffffu, gap_prefix - gap_size, owner);
            const int64_t slot = owner_start + padding_idx - owner_prefix;
            auto* output_vectors = reinterpret_cast<int4*>(recv_x + slot * kHidden);
            for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                for (int u = 0; u < kUnrollFactor; ++u) {
                    const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                    output_vectors[index] = int4{0, 0, 0, 0};
                }
            }
            if constexpr (kWithWeights) {
                if (lane == 0)
                    recv_weights[slot] = 0.0f;
            }
        }
        padding_idx -= wave_gaps;
    }
}

template <int kHidden, int kTopk, int kExpertsPerRank, bool kCachedMode, bool kExpand, bool kWithWeights, bool kZeroPadding>
__global__ void dispatch_copy_epilogue_kernel(const uint8_t* stage_buffer,        // [N, kRowBytes]
                                              int64_t* src_ranks,                 // [N], written only in fresh mode
                                              int64_t* src_tokens,                // [N], written only in fresh mode
                                              int64_t* local_topk_idx,            // [N,K]
                                              int64_t* expanded_slots,            // [N,K], invalid = -1
                                              unsigned long long* expert_cursor,  // [L], fresh mode only
                                              const int64_t* expert_counts,       // [L], actual assignment counts
                                              const int64_t* expert_offsets,      // [L+1], padded offsets
                                              at::BFloat16* recv_x,               // compact [N,H], expanded [C,H]
                                              int64_t* recv_topk_idx,             // compact [N,K], expanded nullptr
                                              float* recv_weights,                // compact [N,K], expanded [C], optional
                                              int64_t recv_count,                 // N
                                              int64_t rank) {
    static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0 && kTopk > 0 && kTopk <= 32);
    constexpr int64_t kRowBytes = WindowLayout::payload_x_offset(kTopk) + int64_t{kHidden} * sizeof(at::BFloat16);
    constexpr int kWarpSize = 32;
    constexpr int kElemsPerVector = sizeof(int4) / sizeof(at::BFloat16);
    constexpr int kVectorsPerRow = kHidden / kElemsPerVector;
    static_assert(kVectorsPerRow % kWarpSize == 0);
    constexpr int kVectorsPerLane = kVectorsPerRow / kWarpSize;
    constexpr int kUnrollFactor = [] {
        for (int factor = 4; factor >= 1; --factor)
            if (kVectorsPerLane % factor == 0)
                return factor;
        return 1;
    }();
    const int lane = threadIdx.x % kWarpSize;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const int64_t total_warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarpSize);

    if (row < recv_count) {
        const auto* payload = stage_buffer + row * kRowBytes;
        const auto* weights = reinterpret_cast<const float*>(payload + WindowLayout::payload_weights_offset(kTopk));
        const auto* x_vectors = reinterpret_cast<const int4*>(payload + WindowLayout::payload_x_offset(kTopk));
        ReceiveAssignment assignment{-1, -1};
        if constexpr (kCachedMode) {
            if (lane < kTopk) {
                if constexpr (kExpand)
                    assignment.slot = expanded_slots[row * kTopk + lane];
                else
                    assignment.local_expert = local_topk_idx[row * kTopk + lane];
            }
        } else {
            assignment = build_receive_metadata_row<kTopk, kExpertsPerRank>(
                payload, src_ranks, src_tokens, local_topk_idx, expanded_slots, expert_cursor, row, rank, lane);
        }

        if constexpr (kExpand) {
            // All lanes exchange register-held slots; no fresh metadata is reread.
            unsigned remaining = __ballot_sync(0xffffffffu, assignment.slot >= 0);
            while (remaining != 0) {
                const int k = __ffs(remaining) - 1;
                const int64_t slot = __shfl_sync(0xffffffffu, assignment.slot, k);

                if constexpr (kWithWeights) {
                    if (lane == 0) {
                        recv_weights[slot] = weights[k];
                    }
                }

                auto* output_vectors = reinterpret_cast<int4*>(recv_x + slot * kHidden);
                for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                    for (int u = 0; u < kUnrollFactor; ++u) {
                        const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                        output_vectors[index] = x_vectors[index];
                    }
                }
                remaining &= remaining - 1;
            }
        } else {
            for (int64_t k = lane; k < kTopk; k += kWarpSize) {
                recv_topk_idx[row * kTopk + k] = assignment.local_expert;
                if constexpr (kWithWeights) {
                    recv_weights[row * kTopk + k] = weights[k];
                }
            }
            auto* output_vectors = reinterpret_cast<int4*>(recv_x + row * kHidden);
            for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                for (int u = 0; u < kUnrollFactor; ++u) {
                    const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                    output_vectors[index] = x_vectors[index];
                }
            }
        }
    }

    if constexpr (kExpand && kZeroPadding) {
        do_zero_padding<kHidden, kExpertsPerRank, kUnrollFactor, kWithWeights>(
            expert_counts, expert_offsets, recv_x, recv_weights, row, total_warps, lane);
    }
}

}  // namespace nano_ep
