#pragma once

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdio>
#include <cuda/std/bit>

#include "layout.hpp"

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

template <int kHidden,
          int kTopk,
          int kWorldSize,
          int64_t kMaxTokens,
          typename data_t,
          bool kExpand,
          bool kAllowMultipleReduction,
          bool kWithWeights>
__global__ void combine_return_kernel(const int64_t* peer_addresses,  // [R], includes self
                                      const data_t* x,                // compact [N,H], expanded [C,H]
                                      const float* weights,           // compact [N,K], expanded [C], or nullptr
                                      const int64_t* src_ranks,       // [N]
                                      const int64_t* src_tokens,      // [N]
                                      const int64_t* local_topk_idx,  // [N,K], nonlocal/masked = -1
                                      const int64_t* expanded_slots,  // expanded [N,K], compact nullptr
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

    for (int64_t row = global_warp_idx; row < recv_count; row += token_stride) {
        const int64_t src_rank = src_ranks[row];
        const int64_t src_token = src_tokens[row];
        if (src_rank < 0 || src_rank >= kWorldSize || src_token < 0 || src_token >= kMaxTokens) {
            if (lane == 0)
                printf("NanoEP combine_return invalid source: row=%lld rank=%lld token=%lld\n",
                       static_cast<long long>(row),
                       static_cast<long long>(src_rank),
                       static_cast<long long>(src_token));
            asm volatile("trap;");
            return;
        }

        const bool valid_k = lane < kTopk && local_topk_idx[row * kTopk + lane] >= 0;
        const unsigned valid_mask = __ballot_sync(0xffffffffu, valid_k);
        int64_t lane_slot = -1;
        if constexpr (kExpand) {
            if (valid_k) {
                lane_slot = expanded_slots[row * kTopk + lane];
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
            for (int64_t k = 0; k < kTopk; ++k) {
                // check if current k is valid
                if ((valid_mask & (1u << k)) == 0)
                    continue;
                const int64_t slot = __shfl_sync(0xffffffffu, lane_slot, k);
                const int64_t peer_idx = k * kMaxTokens + src_token;
                auto* peer_row = peer_window + return_offset + peer_idx * return_row_bytes;
                auto* output_vectors = reinterpret_cast<int4*>(peer_row);
                const auto* input_vectors = reinterpret_cast<const int4*>(x + slot * kHidden);
#pragma unroll
                for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                    for (int u = 0; u < kUnrollFactor; ++u) {
                        const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                        output_vectors[index] = input_vectors[index];
                    }
                }
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
            auto* peer_x = reinterpret_cast<at::BFloat16*>(peer_row);

            const bool no_local_reduce = !kExpand || (kAllowMultipleReduction && __popc(valid_mask) == 1);
            if (no_local_reduce) {
                int64_t input_row = row;
                if constexpr (kExpand) {
                    // Broadcast the only valid assignment's slot to the whole warp.
                    input_row = __shfl_sync(0xffffffffu, lane_slot, first_valid_k);
                }
                auto* output_vectors = reinterpret_cast<int4*>(peer_row);
                const auto* input_vectors = reinterpret_cast<const int4*>(x + input_row * kHidden);
#pragma unroll
                for (int step = 0; step < kVectorsPerLane / kUnrollFactor; ++step) {
#pragma unroll
                    for (int u = 0; u < kUnrollFactor; ++u) {
                        const int index = (step * kUnrollFactor + u) * kWarpSize + lane;
                        output_vectors[index] = input_vectors[index];
                    }
                }
            } else if constexpr (kExpand) {
                int64_t slots[kTopk];
                // number of valid k's in this row
                const int count = compute_expanded_slots(lane_slot, valid_mask, slots);

                auto* output_vectors = reinterpret_cast<int4*>(peer_x);
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
    }
}

template <int kHidden, int kTopk, int kWorldSize, int64_t kMaxTokens>
void launch_combine_return(const torch::Tensor& addresses,
                           const torch::Tensor& x,
                           const std::vector<torch::Tensor>& metadata,
                           const c10::optional<torch::Tensor>& weights,
                           int rank,
                           const WindowLayout& layout,
                           bool expand,
                           bool allow_multiple_reduction,
                           cudaStream_t stream) {
    const int64_t recv_count = metadata[0].numel();
    if (recv_count == 0)
        return;

    const auto* weight_ptr = weights.has_value() ? weights->data_ptr<float>() : nullptr;
    constexpr int threads = 256;
    const int blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    using scalar_t = at::BFloat16;

    const auto launch = [&]<bool kExpand, bool kAllowMultipleReduction, bool kWithWeights>() {
        combine_return_kernel<kHidden, kTopk, kWorldSize, kMaxTokens, scalar_t, kExpand, kAllowMultipleReduction, kWithWeights>
            <<<blocks, threads, 0, stream>>>(addresses.data_ptr<int64_t>(),
                                             x.data_ptr<scalar_t>(),
                                             weight_ptr,
                                             metadata[0].data_ptr<int64_t>(),
                                             metadata[1].data_ptr<int64_t>(),
                                             metadata[2].data_ptr<int64_t>(),
                                             kExpand ? metadata[5].data_ptr<int64_t>() : nullptr,
                                             rank,
                                             recv_count,
                                             x.size(0),
                                             layout.return_offset);
    };
    const auto launch_with_weights = [&]<bool kExpand, bool kAllowMultipleReduction>() {
        if (weights.has_value())
            launch.template operator()<kExpand, kAllowMultipleReduction, true>();
        else
            launch.template operator()<kExpand, kAllowMultipleReduction, false>();
    };
    if (expand) {
        if (allow_multiple_reduction)
            launch_with_weights.template operator()<true, true>();
        else
            launch_with_weights.template operator()<true, false>();
    } else {
        if (allow_multiple_reduction)
            launch_with_weights.template operator()<false, true>();
        else
            launch_with_weights.template operator()<false, false>();
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace nano_ep
