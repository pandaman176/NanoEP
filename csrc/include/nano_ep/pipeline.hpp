#pragma once
#include <torch/extension.h>

#include <array>
#include <nano_ep/event.hpp>
#include <nano_ep/peer.hpp>
#include <tuple>

namespace nano_ep {

struct WindowLayout;

struct JitKernel;

// metadata tensors are on the caller's GPU. Fresh main returns six inputs:
// expert_counts, expert_offsets, dst_slots, send_counts, source_topk_snapshot,
// recv_rank_prefix[R] (inclusive source counts).
// Prepared metadata: recv_metadata int32 [N,K+2], expert_counts,
// expert_offsets, dst_slots, send_counts, source_topk_snapshot, recv_rank_prefix.
// Receive rows contain source rank/token and local expert IDs (compact) or
// expanded slots (expanded), with -1 for invalid assignments.
// counts contains (recv_count, expanded_capacity).
// Fresh expanded states own a mutable expert cursor outside the handle metadata.
// Notify initializes it from padded starts; the epilogue reserves assignment slots.
// window owns the borrowed source-partitioned storage at stage_offset.
// Only each source's count prefix of its Tmax rows is valid. Keep its
// runtime alive and consume it on the same stream before the next dispatch or
// combine overwrites the shared arena. Metadata owns its storage independently.
// hidden/x_dtype describe BF16 x and its dispatch wire format.
struct DispatchState {
    std::vector<torch::Tensor> metadata;
    std::array<int64_t, 2> counts;
    torch::Tensor window;
    int64_t stage_offset;
    int64_t max_tokens;
    int64_t hidden = 0;
    at::ScalarType x_dtype = at::ScalarType::Undefined;
    int64_t rank = -1;
    torch::Tensor expert_cursor;  // GPU int64 [L], mutable fresh-expanded scratch; absent otherwise
};

std::vector<torch::Tensor> dispatch_epilogue(
    const JitKernel& kernel, DispatchState& state, bool cached, bool expand, bool with_weights, int64_t consumer_stream);
std::vector<torch::Tensor> combine_epilogue(const JitKernel& kernel,
                                            const torch::Tensor& window,
                                            int64_t return_offset,
                                            const torch::Tensor& source_topk_snapshot,
                                            int64_t hidden,
                                            const c10::optional<torch::Tensor>& bias,
                                            bool with_weights,
                                            int64_t consumer_stream);

// Order the communication stream after the compute stream or previous event,
// enqueue both dispatch stages there, and restore the caller's stream.
// Return receive tensors, prepared handle metadata, and host counts (N, C).
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::array<int64_t, 2>, std::shared_ptr<CompletionEvent>> dispatch(
    PeerRuntime& runtime,
    const torch::Tensor& x,
    const torch::Tensor& indices,
    const c10::optional<torch::Tensor>& weights,
    int64_t num_experts,
    int64_t max_tokens,
    int64_t alignment,
    const std::vector<torch::Tensor>& cached_metadata,
    int64_t cached_expanded_capacity,
    bool allow_multiple_reduction,
    bool expand,
    bool zero_padding,
    int num_sms,
    int64_t comm_stream,
    int64_t previous_event = -1,
    bool async_op = true);

// Order the communication stream, enqueue both combine stages, and restore the caller's stream.
std::tuple<std::vector<torch::Tensor>, std::shared_ptr<CompletionEvent>> combine(PeerRuntime& runtime,
                                                                                 const torch::Tensor& x,
                                                                                 const std::vector<torch::Tensor>& metadata,
                                                                                 const c10::optional<torch::Tensor>& weights,
                                                                                 const c10::optional<torch::Tensor>& bias,
                                                                                 int64_t max_tokens,
                                                                                 bool expand,
                                                                                 bool allow_multiple_reduction,
                                                                                 int num_sms,
                                                                                 int64_t comm_stream,
                                                                                 int64_t previous_event = -1,
                                                                                 bool async_op = true);

}  // namespace nano_ep
