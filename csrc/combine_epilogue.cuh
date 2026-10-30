#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cuda/std/bit>

#include "layout.hpp"

namespace nano_ep {

struct PackedBf16 {
    uint16_t bits[8];
};
static_assert(sizeof(PackedBf16) == sizeof(int4));

template <int kNumPartitionsInLayout, typename FetchPartition>
__device__ __forceinline__ void compute_partition_slots(int (&partition_slot_idx)[kNumPartitionsInLayout],
                                                        unsigned representatives,
                                                        const FetchPartition& fetch_partition) {
    // all lanes scan set bits in ascending order and fetch each partition
    // once. Compact valid partitions to the front and fill remaining entries
    // with -1. fetch_partition may contain a full-warp shuffle.
#pragma unroll
    for (int i = 0; i < kNumPartitionsInLayout; ++i)
        if (representatives != 0) {
            const int representative = __ffs(representatives) - 1;
            partition_slot_idx[i] = fetch_partition(representative);
            representatives &= representatives - 1;  // clear the first representative

        } else {
            partition_slot_idx[i] = -1;
        }
}

__device__ __forceinline__ void restore_weights(const uint8_t* stage_buffer,  // [P,Tmax,row_bytes]
                                                float* combined_weights,      // [T,K], including masked zeros
                                                int lane,
                                                int partition,  // this lane's return partition, or -1
                                                int64_t row,
                                                int64_t hidden,
                                                int64_t num_topk,
                                                int64_t max_tokens,
                                                int64_t return_row_bytes) {
    // Each top-k lane reads its weight from the selected partition; masked
    // lanes write zero without accessing the return buffer.
    if (lane < num_topk) {
        float weight = 0.0f;
        if (partition >= 0) {
            const auto* record = stage_buffer + (partition * max_tokens + row) * return_row_bytes;
            weight = reinterpret_cast<const float*>(record + WindowLayout::return_weights_offset(hidden))[lane];
        }
        combined_weights[row * num_topk + lane] = weight;
    }
}

template <int kHidden,  // H
          int kTopk,    // K
          int kWorldSize,
          int64_t kExpertsPerRank,
          int64_t kMaxTokens,  // partition stride is Tmax, not actual T
          bool kExpand,
          bool kAllowMultipleReduction,
          bool kWithWeights>
__global__ void combine_reduce_epilogue_kernel(const uint8_t* stage_buffer,  // [P,Tmax,row_bytes], BF16 x[H], aligned FP32 weights[K]
                                               const int64_t* source_topk,   // [T,K], global expert IDs, masked = -1
                                               const nv_bfloat16* bias,      // [T,H], or nullptr
                                               nv_bfloat16* combined_x,      // [T,H], write every element
                                               float* combined_weights,      // [T,K], or nullptr; masked slots must be zero
                                               int64_t num_tokens) {         // actual source T
    constexpr int kWarpSize = 32;
    constexpr int kElemsPerVector = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr bool kRankLayout = kAllowMultipleReduction && kWorldSize <= kTopk;
    constexpr int kNumPartitionsInLayout = kAllowMultipleReduction ? (kWorldSize < kTopk ? kWorldSize : kTopk) : kTopk;
    constexpr int64_t kWeightOffset = kHidden * sizeof(nv_bfloat16);
    constexpr int64_t kReturnRowBytes = kWeightOffset + ((kTopk * sizeof(float) + 31) / 32) * 32;
    constexpr int kVectorsPerLane = kHidden / (kWarpSize * kElemsPerVector);
    constexpr int kUnrollFactor = [] {
        for (int factor = 4; factor >= 1; --factor)
            if (kVectorsPerLane % factor == 0)
                return factor;
        return 1;
    }();
    static_assert(kHidden > 0 && kHidden % (kWarpSize * kElemsPerVector) == 0);
    static_assert(kTopk > 0 && kTopk <= 32 && kWorldSize > 0 && kWorldSize <= 32);
    static_assert(kExpertsPerRank > 0 && kMaxTokens > 0);

    // one warp owns one source token; lanes stride over hidden coordinates in int4 vectors.
    const int lane = threadIdx.x % kWarpSize;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    if (row >= num_tokens)
        return;

    // Expanded without local reduction reads each valid k partition. Otherwise
    // deduplicate destination ranks, selecting rank or first-k partitions exactly
    // as combine_return_kernel does. Never read unpublished rows or weight fields.
    // Accumulate x in FP32, add optional bias once, then cast once to BF16.
    // Restore valid weights independently at their original k; write masked zeros.
    const int64_t expert = lane < kTopk ? source_topk[row * kTopk + lane] : -1;
    const int destination = expert >= 0 ? static_cast<int>(expert / kExpertsPerRank) : -1;
    int partitions[kNumPartitionsInLayout];
    int weight_partition = -1;
    if constexpr (kAllowMultipleReduction || !kExpand) {
        const unsigned same_destination = __match_any_sync(0xffffffffu, destination);
        const int first_k = __ffs(same_destination) - 1;
        // finding all first valid k for each rank
        const unsigned representatives = __ballot_sync(0xffffffffu, destination >= 0 && lane == first_k);

        // Compute the slot indices for each partition.
        compute_partition_slots(partitions, representatives, [=](int representative) {
            const int owner = __shfl_sync(0xffffffffu, destination, representative);
            return kRankLayout ? owner : representative;
        });
        weight_partition = destination >= 0 ? (kRankLayout ? destination : first_k) : -1;
    } else {
        // kExpand && !kAllowMultipleReduction: [K,Tmax] layout.
        // finding all valid k
        const unsigned valid_ks = __ballot_sync(0xffffffffu, expert >= 0);
        compute_partition_slots(partitions, valid_ks, [](int k) { return k; });
        weight_partition = expert >= 0 ? lane : -1;
    }

    // Restore weights once, outside the hidden-coordinate loop.
    if constexpr (kWithWeights) {
        restore_weights(stage_buffer, combined_weights, lane, weight_partition, row, kHidden, kTopk, kMaxTokens, kReturnRowBytes);
    }

    // Each int4 covers eight BF16 coordinates; H % 256 == 0 leaves no hidden tail.
    auto* output_vectors = reinterpret_cast<int4*>(combined_x + row * kHidden);
    for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
        float2 accum[kUnrollFactor][kElemsPerVector / 2] = {};
#pragma unroll
        for (int p = 0; p < kNumPartitionsInLayout; ++p) {
            const int partition = partitions[p];
            if (partition < 0)
                break;
            const auto* record = stage_buffer + (partition * kMaxTokens + row) * kReturnRowBytes;
            const auto* input_vectors = reinterpret_cast<const int4*>(record);
#pragma unroll
            for (int u = 0; u < kUnrollFactor; ++u) {
                const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                // Load 16bytes, treated as 8 2byte bits: value.bits[0..7]
                const auto value = cuda::std::bit_cast<PackedBf16>(input_vectors[index]);
#pragma unroll
                for (int j = 0; j < kElemsPerVector / 2; ++j) {
                    accum[u][j].x += __bfloat162float(__ushort_as_bfloat16(value.bits[2 * j]));
                    accum[u][j].y += __bfloat162float(__ushort_as_bfloat16(value.bits[2 * j + 1]));
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kUnrollFactor; ++u) {
            const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
            PackedBf16 value;
#pragma unroll
            for (int j = 0; j < kElemsPerVector / 2; ++j) {
                if (bias != nullptr) {
                    const int64_t h = row * kHidden + index * kElemsPerVector + j * 2;
                    accum[u][j].x += __bfloat162float(bias[h]);
                    accum[u][j].y += __bfloat162float(bias[h + 1]);
                }
                const auto rounded = __float22bfloat162_rn(accum[u][j]);
                value.bits[2 * j] = __bfloat16_as_ushort(__low2bfloat16(rounded));
                value.bits[2 * j + 1] = __bfloat16_as_ushort(__high2bfloat16(rounded));
            }
            output_vectors[index] = cuda::std::bit_cast<int4>(value);
        }
    }
}

}  // namespace nano_ep
