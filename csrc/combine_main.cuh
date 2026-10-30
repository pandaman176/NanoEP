#pragma once

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdio>

#include "layout.hpp"

namespace nano_ep {

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
    constexpr int64_t hidden = kHidden;
    constexpr int64_t num_topk = kTopk;
    constexpr int world_size = kWorldSize;
    constexpr int64_t max_tokens = kMaxTokens;
    constexpr int64_t return_row_bytes = WindowLayout::return_weights_offset(kHidden) + ((kTopk * sizeof(float) + 31) / 32) * 32;
    static_assert(kHidden > 0 && kHidden % WindowLayout::kHiddenAlignment == 0);
    static_assert(kTopk > 0 && kTopk <= 32 && kWorldSize > 0 && kWorldSize <= 32 && kMaxTokens > 0);

    // One warp owns one deduplicated receive row.
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    if (row >= recv_count)
        return;

    const int64_t src_rank = src_ranks[row];
    const int64_t src_token = src_tokens[row];
    if (src_rank < 0 || src_rank >= world_size || src_token < 0 || src_token >= max_tokens) {
        if (lane == 0)
            printf("NanoEP combine_return invalid source: row=%lld rank=%lld token=%lld\n",
                   static_cast<long long>(row),
                   static_cast<long long>(src_rank),
                   static_cast<long long>(src_token));
        asm volatile("trap;");
        return;
    }

    const bool valid_k = lane < num_topk && local_topk_idx[row * num_topk + lane] >= 0;
    const unsigned valid_mask = __ballot_sync(0xffffffffu, valid_k);
    if (valid_mask == 0) {
        if (lane == 0)
            printf("NanoEP combine_return row has no local assignment: row=%lld\n", static_cast<long long>(row));
        asm volatile("trap;");
        return;
    }
    if constexpr (kExpand) {
        if (valid_k) {
            const int64_t slot = expanded_slots[row * num_topk + lane];
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
        for (int64_t k = 0; k < num_topk; ++k) {
            // check if current k is valid
            if ((valid_mask & (1u << k)) == 0)
                continue;
            const int64_t slot = expanded_slots[row * num_topk + k];
            const int64_t peer_idx = k * max_tokens + src_token;
            auto* peer_row = peer_window + return_offset + peer_idx * return_row_bytes;
            auto* peer_x = reinterpret_cast<at::BFloat16*>(peer_row);
            for (int64_t h = lane; h < hidden; h += warp_size)
                peer_x[h] = x[slot * hidden + h];
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
        const bool use_rank_layout = kAllowMultipleReduction && world_size <= num_topk;
        const int partition = use_rank_layout ? rank : first_valid_k;
        const int64_t peer_idx = partition * max_tokens + src_token;
        auto* peer_row = peer_window + return_offset + peer_idx * return_row_bytes;
        auto* peer_x = reinterpret_cast<at::BFloat16*>(peer_row);
        if constexpr (kExpand) {
            for (int64_t h = lane; h < hidden; h += warp_size) {
                float accum = 0.0f;
                for (int64_t k = 0; k < num_topk; ++k) {
                    if ((valid_mask & (1u << k)) != 0) {
                        const int64_t slot = expanded_slots[row * num_topk + k];
                        accum += static_cast<float>(x[slot * hidden + h]);
                    }
                }
                peer_x[h] = static_cast<at::BFloat16>(accum);
            }
        } else {
            for (int64_t h = lane; h < hidden; h += warp_size)
                peer_x[h] = x[row * hidden + h];
        }
        if constexpr (kWithWeights) {
            // One lane writes each original top-k position in the same return row.
            if (valid_k) {
                auto* peer_weights = reinterpret_cast<float*>(peer_row + WindowLayout::return_weights_offset(hidden));
                const int64_t weight_idx = kExpand ? expanded_slots[row * num_topk + lane] : row * num_topk + lane;
                peer_weights[lane] = weights[weight_idx];
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
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (recv_count + warps_per_block - 1) / warps_per_block;

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
