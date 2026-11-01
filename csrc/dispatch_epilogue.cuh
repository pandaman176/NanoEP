#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <cstdio>

#include "layout.hpp"

namespace nano_ep {

template <int kHidden, int kTopk, int kExpertsPerRank>
__global__ void build_receive_metadata_kernel(const uint8_t* stage_buffer,
                                              int64_t* src_ranks,
                                              int64_t* src_tokens,
                                              int64_t* local_topk_idx,
                                              int64_t* expanded_slots,
                                              unsigned long long* expert_cursor,
                                              int64_t recv_count,
                                              int64_t rank) {
    static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0 && kTopk > 0 && kTopk <= 32);
    constexpr int64_t kRowBytes = WindowLayout::payload_x_offset(kTopk) + int64_t{kHidden} * sizeof(at::BFloat16);
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    if (row >= recv_count)
        return;

    // ┌──────────┬───────────┬───────────┬────────────┬─────────┬─────────┐
    // │ src_rank │ src_token │ topk IDs  │ weights    │ x       │ padding │
    // │ int64    │ int64     │ int64[K]  │ fp32[K]    │ bf16[H] │         │
    // └──────────┴───────────┴───────────┴────────────┴─────────┴─────────┘
    const auto* payload = stage_buffer + row * kRowBytes;
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
        return;
    }
    if (lane < kTopk) {
        const int64_t index = row * kTopk + lane;
        if (in_range) {
            local_topk_idx[index] = local_expert;
            // Each expert owns a disjoint segment; only its cursor needs an atomic update.
            expanded_slots[index] = static_cast<int64_t>(atomicAdd(expert_cursor + local_expert, 1ULL));
        } else {
            local_topk_idx[index] = -1;
            expanded_slots[index] = -1;
        }
    }
}

template <int kHidden, int kTopk, bool kExpand, bool kWithWeights>
__global__ void dispatch_copy_epilogue_kernel(const uint8_t* stage_buffer,    // [N, kRowBytes]
                                              const int64_t* local_topk_idx,  // [N,K]
                                              const int64_t* expanded_slots,  // [N,K], invalid = -1
                                              at::BFloat16* recv_x,           // compact [N,H], expanded [C,H]
                                              int64_t* recv_topk_idx,         // compact [N,K], expanded nullptr
                                              float* recv_weights,            // compact [N,K], expanded [C], optional
                                              int64_t recv_count              // N
) {
    static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0 && kTopk > 0 && kTopk <= 32);
    constexpr int64_t kRowBytes = WindowLayout::payload_x_offset(kTopk) + int64_t{kHidden} * sizeof(at::BFloat16);
    constexpr int warp_size = 32;
    constexpr int kElemsPerVector = sizeof(int4) / sizeof(at::BFloat16);
    constexpr int kVectorsPerRow = kHidden / kElemsPerVector;
    static_assert(kVectorsPerRow % warp_size == 0);
    constexpr int kVectorsPerLane = kVectorsPerRow / warp_size;
    constexpr int kUnrollFactor = [] {
        for (int factor = 4; factor >= 1; --factor)
            if (kVectorsPerLane % factor == 0)
                return factor;
        return 1;
    }();
    const int lane = threadIdx.x % warp_size;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    if (row >= recv_count)
        return;

    const auto* payload = stage_buffer + row * kRowBytes;
    const auto* weights = reinterpret_cast<const float*>(payload + WindowLayout::payload_weights_offset(kTopk));
    const auto* x_vectors = reinterpret_cast<const int4*>(payload + WindowLayout::payload_x_offset(kTopk));

    if constexpr (kExpand) {
        for (int64_t k = 0; k < kTopk; ++k) {
            const int64_t slot = expanded_slots[row * kTopk + k];
            if (slot < 0) {
                continue;
            }

            if constexpr (kWithWeights) {
                if (lane == 0) {
                    recv_weights[slot] = weights[k];
                }
            }

            auto* output_vectors = reinterpret_cast<int4*>(recv_x + slot * kHidden);
            for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                for (int u = 0; u < kUnrollFactor; ++u) {
                    const int index = (step * kUnrollFactor + u) * warp_size + lane;
                    output_vectors[index] = x_vectors[index];
                }
            }
        }
    } else {
        for (int64_t k = lane; k < kTopk; k += warp_size) {
            recv_topk_idx[row * kTopk + k] = local_topk_idx[row * kTopk + k];
            if constexpr (kWithWeights) {
                recv_weights[row * kTopk + k] = weights[k];
            }
        }
        auto* output_vectors = reinterpret_cast<int4*>(recv_x + row * kHidden);
        for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
            for (int u = 0; u < kUnrollFactor; ++u) {
                const int index = (step * kUnrollFactor + u) * warp_size + lane;
                output_vectors[index] = x_vectors[index];
            }
        }
    }
}

}  // namespace nano_ep
