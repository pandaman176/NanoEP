#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace nano_ep {

struct WindowLayout {
    int64_t signal_offset, count_offset, payload_offset, return_offset, num_bytes;
    int64_t recv_next_offset;
    bool return_rank_layout;
    int return_partitions;
    int64_t return_capacity;

    static constexpr int64_t kRegionAlignment = 256;  // cudaMalloc is 256-byte aligned
    static constexpr int64_t kRowAlignment = 16;      // easier for vectorized loads/stores
    static constexpr int64_t kCountNotReady = -1;

    // Dispatch row: int64 source rank/token, int64 topk[K], float routing weights[K], float x[H].
    //                             8  + 8           + 8k             + 4k                + 4H
    // BF16 payloads can be represented exactly in the FP32 communication storage.
    static int64_t payload_row_bytes(int hidden, int topk) {
        return align(checked_add(checked_add(16, checked_mul(topk, 12)), checked_mul(hidden, 4)), kRowAlignment);
    }

    // Return row: float x[H], float weights[K]. A locally reduced x may carry
    // several independent routing-weight payloads; the source selects each k.
    static int64_t return_row_bytes(int hidden, int topk) {
        return align(checked_add(checked_mul(hidden, 4), checked_mul(topk, 4)), kRowAlignment);
    }

    static constexpr bool use_rank_layout(int ranks, int topk, bool allow_multiple_reduction) {
        return allow_multiple_reduction && ranks <= topk;
    }

    static constexpr int num_return_partitions(int ranks, int topk, bool allow_multiple_reduction) {
        return allow_multiple_reduction ? std::min(ranks, topk) : topk;
    }

    static WindowLayout make(
        int ranks, int64_t max_tokens, int hidden, int topk, int64_t num_experts, bool allow_multiple_reduction = false) {
        if (ranks <= 0 || ranks > 32 || max_tokens <= 0 || hidden <= 0 || topk <= 0 || topk > 32) {
            throw std::invalid_argument("WindowLayout requires ranks/topk in [1,32] and positive capacity/hidden");
        }
        if (num_experts <= 0 || num_experts % ranks != 0) {
            throw std::invalid_argument("WindowLayout requires positive num_experts divisible by ranks");
        }

        // each rank has a uint64_t signal to store the epoch of the last barrier
        const auto signal_bytes = checked_mul(ranks, 8);

        // int64 rank_count[R], expert_count_by_source[R,E/R], then recv_next.
        // Counts use -1 for pending and >=0 for ready; recv_next starts at zero.
        const auto notify_bytes = checked_mul(checked_add(ranks, num_experts), 8);
        const auto count_bytes = checked_add(notify_bytes, 8);

        const auto recv_capacity = checked_mul(ranks, max_tokens);
        const auto payload_bytes = checked_mul(recv_capacity, payload_row_bytes(hidden, topk));

        const auto return_partitions = num_return_partitions(ranks, topk, allow_multiple_reduction);
        const auto return_capacity = checked_mul(return_partitions, max_tokens);
        const auto return_bytes = checked_mul(return_capacity, return_row_bytes(hidden, topk));

        WindowLayout layout{};
        layout.signal_offset = 0;
        layout.count_offset = align(signal_bytes, kRegionAlignment);
        layout.recv_next_offset = checked_add(layout.count_offset, notify_bytes);
        layout.return_rank_layout = use_rank_layout(ranks, topk, allow_multiple_reduction);
        layout.return_partitions = return_partitions;
        layout.return_capacity = return_capacity;
        layout.payload_offset = align(checked_add(layout.count_offset, count_bytes), kRegionAlignment);
        // Entry barriers separate the readers and writers of successive phases.
        layout.return_offset = layout.payload_offset;
        layout.num_bytes = align(checked_add(layout.payload_offset, std::max(payload_bytes, return_bytes)), kRegionAlignment);
        return layout;
    }

private:
    static int64_t checked_add(int64_t a, int64_t b) {
        if (a > std::numeric_limits<int64_t>::max() - b) {
            throw std::overflow_error("WindowLayout byte size overflow");
        }
        return a + b;
    }

    static int64_t checked_mul(int64_t a, int64_t b) {
        if (a > std::numeric_limits<int64_t>::max() / b) {
            throw std::overflow_error("WindowLayout byte size overflow");
        }
        return a * b;
    }

    static int64_t align(int64_t bytes, int64_t alignment) {
        const auto remainder = bytes % alignment;
        return remainder == 0 ? bytes : checked_add(bytes, alignment - remainder);
    }
};

}  // namespace nano_ep
