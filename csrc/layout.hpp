#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef __CUDACC__
#define NANO_EP_HOST_DEVICE __host__ __device__
#else
#define NANO_EP_HOST_DEVICE
#endif

namespace nano_ep {

struct WindowLayout {
    int64_t signal_offset, count_offset, payload_offset, return_offset, num_bytes;
    int64_t send_next_offset;
    int64_t notify_reduction_offset;
    bool return_rank_layout;
    int return_partitions;
    int64_t return_capacity;

    // num of elements a warp will handle
    // 32 threads * 16 bytes/thread / 2 bytes/element
    static constexpr int64_t kHiddenAlignment = 32 * 16 / 2;
    static constexpr int64_t kAlignment = 32;
    static constexpr int64_t kRegionAlignment = kAlignment;
    static constexpr int64_t kRowAlignment = kAlignment;
    static constexpr int64_t kPayloadTopkOffset = kAlignment;
    static constexpr int64_t kCountNotReady = -1;

    NANO_EP_HOST_DEVICE static constexpr int64_t payload_weights_offset(int64_t topk) { return align_field(kPayloadTopkOffset + topk * 8); }

    NANO_EP_HOST_DEVICE static constexpr int64_t payload_x_offset(int64_t topk) {
        return align_field(payload_weights_offset(topk) + topk * 4);
    }

    NANO_EP_HOST_DEVICE static constexpr int64_t return_weights_offset(int64_t hidden) { return align_field(hidden * 2); }

    // Each field begins at a 32-byte boundary; provenance occupies the first 16 bytes.
    static constexpr int64_t payload_row_bytes(int hidden, int topk) {
        return align(checked_add(payload_x_offset(topk), checked_mul(hidden, 2)), kRowAlignment);
    }

    // Return row: BF16 x[H], align32 padding, float weights[K]. A locally reduced x may carry
    // several independent routing-weight payloads; the source selects each k.
    static constexpr int64_t return_row_bytes(int hidden, int topk) {
        return align(checked_add(return_weights_offset(hidden), checked_mul(topk, 4)), kRowAlignment);
    }

    static constexpr bool use_rank_layout(int ranks, int topk, bool allow_multiple_reduction) {
        return allow_multiple_reduction && ranks <= topk;
    }

    static constexpr int num_return_partitions(int ranks, int topk, bool allow_multiple_reduction) {
        return allow_multiple_reduction ? std::min(ranks, topk) : topk;
    }

    static constexpr WindowLayout make(
        int ranks, int64_t max_tokens, int hidden, int topk, int64_t num_experts, bool allow_multiple_reduction = false) {
        if (ranks <= 0 || ranks > 32 || max_tokens <= 0 || hidden <= 0 || topk <= 0 || topk > 32) {
            throw std::invalid_argument("WindowLayout requires ranks/topk in [1,32] and positive capacity/hidden");
        }
        if (num_experts <= 0 || num_experts % ranks != 0) {
            throw std::invalid_argument("WindowLayout requires positive num_experts divisible by ranks");
        }

        // each rank has a uint64_t signal to store the epoch of the last barrier
        const auto signal_bytes = checked_mul(ranks, 8);

        // int64 rank_count[R], expert_count_by_source[R,E/R].
        // Counts use -1 for pending and >=0 for ready; send_next start at zero.
        const auto notify_bytes = checked_mul(checked_add(ranks, num_experts), 8);
        // int64 send_next[R]
        const auto count_bytes = checked_add(notify_bytes, checked_mul(ranks, 8));

        const auto recv_capacity = checked_mul(ranks, max_tokens);
        const auto payload_bytes = checked_mul(recv_capacity, payload_row_bytes(hidden, topk));

        const auto return_partitions = num_return_partitions(ranks, topk, allow_multiple_reduction);
        const auto return_capacity = checked_mul(return_partitions, max_tokens);
        const auto return_bytes = checked_mul(return_capacity, return_row_bytes(hidden, topk));

        WindowLayout layout{};
        layout.signal_offset = 0;
        // Rank-local reduction words pack CTA arrivals (high 32) and counts (low 32).
        layout.notify_reduction_offset = align(signal_bytes, kRegionAlignment);
        layout.count_offset = align(checked_add(layout.notify_reduction_offset, notify_bytes), kRegionAlignment);
        layout.send_next_offset = checked_add(layout.count_offset, notify_bytes);
        layout.return_rank_layout = use_rank_layout(ranks, topk, allow_multiple_reduction);
        layout.return_partitions = return_partitions;
        layout.return_capacity = return_capacity;
        layout.payload_offset = align(checked_add(layout.count_offset, count_bytes), kRegionAlignment);
        // Reuse Send Buffer For Return Payload
        layout.return_offset = layout.payload_offset;
        layout.num_bytes = align(checked_add(layout.payload_offset, std::max(payload_bytes, return_bytes)), kRegionAlignment);
        return layout;
    }

private:
    NANO_EP_HOST_DEVICE static constexpr int64_t align_field(int64_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

    static constexpr int64_t checked_add(int64_t a, int64_t b) {
        if (a > std::numeric_limits<int64_t>::max() - b) {
            throw std::overflow_error("WindowLayout byte size overflow");
        }
        return a + b;
    }

    static constexpr int64_t checked_mul(int64_t a, int64_t b) {
        if (a > std::numeric_limits<int64_t>::max() / b) {
            throw std::overflow_error("WindowLayout byte size overflow");
        }
        return a * b;
    }

    static constexpr int64_t align(int64_t bytes, int64_t alignment) {
        const auto remainder = bytes % alignment;
        return remainder == 0 ? bytes : checked_add(bytes, alignment - remainder);
    }
};

// Hopper admits 227KB of shared memory per block, static and dynamic together.
inline constexpr int64_t kMaxSmemBytes = 227 * 1024;

// One payload row staged in shared memory per warp; rows are broadcast with bulk copies, so a row
// crosses the memory system once instead of once per destination.
struct PayloadRowPlan {
    int64_t row_bytes;    // one wire-format payload row, 32-byte aligned
    int64_t stage_bytes;  // one row plus its mbarrier slot
    int64_t num_bytes;    // dynamic shared memory request
    int payload_warps;
    int total_warps;

    // reserved_bytes is the static shared memory kept alongside the stages.
    static constexpr PayloadRowPlan make(int hidden, int topk, int notify_warps = 0, int64_t reserved_bytes = 0) {
        const int64_t row_bytes = WindowLayout::payload_row_bytes(hidden, topk);
        const int64_t stage_bytes = row_bytes + 32;  // one mbarrier slot
        const int payload_warps = std::min<int64_t>(32 - notify_warps, (kMaxSmemBytes - reserved_bytes) / stage_bytes);
        return {row_bytes, stage_bytes, payload_warps * stage_bytes, payload_warps, notify_warps + payload_warps};
    }

    NANO_EP_HOST_DEVICE constexpr int64_t stage_offset(int warp) const { return static_cast<int64_t>(warp) * stage_bytes; }
};

// Fresh dispatch reserves warps for counting and notification.
constexpr int dispatch_notify_warps(bool cached) {
    return cached ? 0 : 4;
}

constexpr PayloadRowPlan make_dispatch_payload_plan(int hidden, int topk, int64_t ranks, int64_t experts_per_rank, int notify_warps) {
    const int64_t count_entries = ranks + ranks * experts_per_rank;
    const int64_t reserved_bytes = notify_warps == 0 ? 0 : count_entries * static_cast<int64_t>(sizeof(uint32_t));
    return PayloadRowPlan::make(hidden, topk, notify_warps, reserved_bytes);
}

// One BF16 row and its optional load barrier in shared memory per warp.
struct HiddenRowPlan {
    int64_t row_bytes;    // one BF16 row
    int64_t stage_bytes;  // one row plus its per-warp bookkeeping

    static constexpr HiddenRowPlan make(int hidden, bool incoming = false) {
        const int64_t row_bytes = static_cast<int64_t>(hidden) * 2;  // BF16 hidden dimension
        return {row_bytes, row_bytes + (incoming ? 32 : 0)};
    }

    constexpr int max_warps(int64_t smem_budget = kMaxSmemBytes) const {
        return static_cast<int>(std::min<int64_t>(32, smem_budget / stage_bytes));
    }

    NANO_EP_HOST_DEVICE constexpr int64_t stage_offset(int warp) const { return static_cast<int64_t>(warp) * stage_bytes; }
};

}  // namespace nano_ep

#undef NANO_EP_HOST_DEVICE
