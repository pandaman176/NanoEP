#pragma once
#include <torch/extension.h>

#include <nano_ep/peer.hpp>

namespace nano_ep {

struct WindowLayout;

// metadata tensors are on the caller's GPU. Fresh main returns five inputs:
// expert_counts, expert_offsets, dst_slots, send_counts, source_topk_snapshot.
// After the epilogue prepares receive metadata (and on cached states), it is:
// src_ranks, src_tokens, local_topk_idx, expert_counts, expert_offsets,
// expanded_slots, dst_slots, send_counts, source_topk_snapshot.
// counts is CPU int64 [2], containing (recv_count, expanded_capacity).
// stage_buffer is borrowed uint8 [N,row_bytes] communication storage. Keep its
// runtime alive and consume it on the same stream before the next dispatch or
// combine overwrites the shared arena. Metadata owns its storage independently.
// hidden/x_dtype describe the original x, whose wire representation is FP32.
// metadata_prepared means preparation has been enqueued, not GPU completion.
struct DispatchState {
    std::vector<torch::Tensor> metadata;
    torch::Tensor counts;
    torch::Tensor stage_buffer;
    int64_t hidden = 0;
    at::ScalarType x_dtype = at::ScalarType::Undefined;
    int64_t rank = -1;
    bool metadata_prepared = false;
};

// Collective on the current stream: exchange rank counts [R] and each peer's
// expert-count slice from [E]. Return CUDA int64 rank counts [R] by source and
// aggregated local expert counts [E/R]. Consumed slots are reset to -1.
// The window's count slots must be initialized to -1 before peers can use it.
// Caller enters a GPU barrier before publishing and completes another after
// payload writes. Counts becoming ready does not imply token payload arrival.
std::vector<torch::Tensor> notify_counts(PeerRuntime& runtime,
                                         const torch::Tensor& send_counts,
                                         const torch::Tensor& send_expert_counts,
                                         int64_t count_offset);

// Fresh routing reserves once per (token,destination) and returns CUDA int64
// [T,R] slots, with -1 for absent routes. Cached calls read the supplied slots
// without modifying the mapping or counter; indices must be the old snapshot.
// Valid expert IDs are in [-1,E). Caller resets local recv_next for fresh calls
// before the entry barrier and runs an exit barrier before reading payloads.
torch::Tensor reserve_and_put(PeerRuntime& runtime,
                              const torch::Tensor& x,
                              const torch::Tensor& indices,
                              const c10::optional<torch::Tensor>& weights,
                              int64_t num_experts,
                              int64_t max_tokens,
                              const WindowLayout& layout,
                              const c10::optional<torch::Tensor>& cached_dst_slots = c10::nullopt);

// After the payload-completion barrier, decode uint8 [N,row_bytes] records.
// expert_offsets is padded CUDA int64 [L+1], where L=experts_per_rank. Counts
// and payload routing must agree. Return owned CUDA int64 tensors:
// src_ranks [N], src_tokens [N], local_topk_idx [N,K], expanded_slots [N,K].
// Rows retain physical receive-slot order. Nonlocal/masked choices become -1;
// valid slots are unique within each expert segment, with unspecified order.
// Inputs are read-only. The caller keeps the payload alive and unchanged until
// this current-stream launch finishes; returned metadata does not alias it.
std::vector<torch::Tensor> build_receive_metadata(
    const torch::Tensor& stage_buffer, const torch::Tensor& expert_offsets, int64_t rank, int64_t experts_per_rank, int64_t num_topk);

// All ranks use matching geometry and all enter either fresh or cached mode.
// Local storage checks precede window mutation and the entry barrier. Routing
// values are a caller precondition: -1 or distinct valid experts per token.
// Existing kernels assert bounds/uniqueness on device, without a validation
// launch or status readback. Device failures are asynchronous and can invalidate
// the CUDA context; exact receive-count allocation still reads back N/C.
// Cached metadata is the unmodified vector prepared by dispatch_epilogue on this
// runtime with the same H/K/E/capacity/alignment; the Python handle checks this
// ownership/configuration contract. A local failure is not a collective error
// exchange: the distributed launcher must terminate peers on rank failure.
// Pass the allocation's allow_multiple_reduction mode even though dispatch
// payloads are unchanged: the total window capacity check includes returns.
DispatchState dispatch_main(PeerRuntime& runtime,
                            const torch::Tensor& x,
                            const torch::Tensor& indices,
                            const c10::optional<torch::Tensor>& weights,
                            int64_t num_experts,
                            int64_t max_tokens,
                            int64_t alignment,
                            const std::vector<torch::Tensor>& cached_metadata,
                            bool allow_multiple_reduction = false);

// Return [recv_x, recv_topk_idx, recv_weights]. Absent optional tensors are
// zero-length tensors; Python converts them using explicit mode/presence flags.
// Fresh receive metadata is built once, before the independent copy kernel.
// Cached/prepared states retain their mappings. Use the same communication
// stream as dispatch_main; construct the Python handle after this call and
// expose completion via its event.
std::vector<torch::Tensor> dispatch_epilogue(DispatchState& state, bool expand, bool with_weights, bool zero_padding);

// Collective on the current stream. Use the same reduction mode and capacity
// as the owning allocation. Return borrowed uint8 [P,Tmax,row_bytes] storage,
// where P=K without local reduction, else min(R,K). Rows contain FP32 x[H]
// and FP32 weights[K]; absent/masked weights must not be read.
// Rank layout is selected iff allow_multiple_reduction && R<=K. Otherwise a
// compact/local-reduced row uses its first valid top-k slot as representative;
// expanded single-reduction sends each assignment to its own k partition.
// Entry/exit GPU barriers protect reuse and publication. Expanded padding is
// never sent; local reduction uses FP32 and stores FP32 partial sums.
// Storage validation is host-only; metadata values/ownership and exact expanded
// capacity are handle preconditions. No routing scan or GPU scalar readback is
// added. The device kernel asserts provenance and expanded-slot bounds. Empty
// ranks still participate in both barriers and return the borrowed view.
torch::Tensor combine_main(PeerRuntime& runtime,
                           const torch::Tensor& x,
                           const std::vector<torch::Tensor>& metadata,
                           const c10::optional<torch::Tensor>& weights,
                           int64_t max_tokens,
                           bool expand,
                           bool allow_multiple_reduction);

// Return [combined_x, combined_weights], with source shape from snapshot.
// The source routing and mode determine which return partitions were written.
// Deduplicate destination ranks for compact/local-reduced x; expanded without
// local reduction reads each valid k. Weights select their original k from the
// corresponding writer's row, independently of x reduction. Masked outputs are
// zero. Accumulate FP32, add source bias once, then cast to x_dtype. The wire
// partial sums stay FP32 (unlike DeepEP's BF16 partial storage).
// Bias is optional contiguous [T,H] with x_dtype. Without weights the second
// result is an empty CUDA FP32 [0] tensor. Outputs own their storage; the caller
// orders this read after combine_main and before the next dispatch or combine.
std::vector<torch::Tensor> combine_epilogue(const torch::Tensor& stage_buffer,
                                            const torch::Tensor& source_topk_snapshot,
                                            int64_t hidden,
                                            at::ScalarType x_dtype,
                                            int64_t num_experts,
                                            int64_t world_size,
                                            int64_t max_tokens,
                                            const c10::optional<torch::Tensor>& bias,
                                            bool with_weights,
                                            bool expand,
                                            bool allow_multiple_reduction);

}  // namespace nano_ep
