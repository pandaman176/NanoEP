#pragma once
#include <torch/extension.h>

#include <nano_ep/peer.hpp>

namespace nano_ep {

struct WindowLayout;

// The core validates geometry and owns IPC; specialization callbacks use tensors
// and stream handles without passing PeerRuntime across extension modules.
struct DispatchMainSpecialization {
    int64_t hidden, topk, world_size, experts_per_rank, max_tokens, expert_alignment;
    void (*prepare)(const torch::Tensor& x, bool cached_mode);
    void (*launch)(const torch::Tensor& addresses,
                   const torch::Tensor& window,
                   const torch::Tensor& x,
                   const torch::Tensor& indices,
                   const c10::optional<torch::Tensor>& weights,
                   const torch::Tensor& dst_slots,
                   const torch::Tensor& send_counts,
                   const torch::Tensor& recv_rank_prefix,
                   const torch::Tensor& recv_expert_counts,
                   const torch::Tensor& expert_offsets,
                   int rank,
                   bool cached_mode,
                   int64_t entry_epoch,
                   cudaStream_t stream);
};

// Geometry-specific modules share this host launcher contract with the core
// extension. Keep the module alive while its launcher is bound to a buffer.
struct CombineMainSpecialization {
    int64_t hidden, topk, world_size, experts_per_rank, max_tokens;
    void (*launch)(const torch::Tensor& addresses,
                   const torch::Tensor& x,
                   const std::vector<torch::Tensor>& metadata,
                   const c10::optional<torch::Tensor>& weights,
                   int rank,
                   const WindowLayout& layout,
                   bool expand,
                   bool allow_multiple_reduction,
                   int64_t entry_epoch,
                   cudaStream_t stream);
};

// metadata tensors are on the caller's GPU. Fresh main returns five inputs:
// expert_counts, expert_offsets, dst_slots, send_counts, source_topk_snapshot.
// After the epilogue prepares receive metadata (and on cached states), it is:
// src_ranks, src_tokens, local_topk_idx, expert_counts, expert_offsets,
// expanded_slots, dst_slots, send_counts, source_topk_snapshot.
// expanded_slots is int64 [N,K] for expanded layout and [0] for compact;
// prepared metadata retains its original layout, including when N is zero.
// counts is CPU int64 [2], containing (recv_count, expanded_capacity).
// stage_buffer is borrowed uint8 [N,row_bytes] communication storage. Keep its
// runtime alive and consume it on the same stream before the next dispatch or
// combine overwrites the shared arena. Metadata owns its storage independently.
// hidden/x_dtype describe BF16 x and its dispatch wire format.
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

// All ranks use BF16 x, matching geometry and either fresh or cached mode.
// Specialize H/K/R/experts-per-rank/Tmax/alignment; actual T and rank remain runtime inputs.
// The core rejects mismatched launchers before window mutation or barriers.
// Keep the specialization module alive while its callbacks are in use.
// Fresh cooperative sends split each CTA into four notify and four payload
// warps. Notify uses shared-memory counts and persistent packed CTA reductions;
// CTA 0 consumes/resets the reductions and exchanges counts while payload work
// proceeds. Incoming counts reuse the same shared buffer after all sends.
// CTA 0's warp 0 builds the rank prefix; warp 1 sums experts, aligns and scans
// them directly into actual counts and padded offsets, using int64 accumulators.
// Both variants launch cooperatively, including T=0 ranks. Cached sends use
// eight payload warps. CTA 0 exchanges entry/exit epochs inside the kernel;
// grid synchronization protects workspace reuse and system fences publish payloads.
// Fresh counts fit uint32; rank/expert counts require per-CTA shared memory.
// Launch resource checks precede window mutation and the entry barrier.
// Local storage checks precede window mutation and the entry barrier. Routing
// values are a caller precondition: -1 or distinct valid experts per token.
// Existing kernels assert bounds/uniqueness on device, without a validation
// launch or status readback. Device failures are asynchronous and can invalidate
// the CUDA context; fresh exact receive-count allocation reads back N/C.
// Cached metadata is the unmodified vector prepared by dispatch_epilogue on this
// runtime with the same H/K/E/capacity/alignment; the Python handle checks this
// ownership/configuration contract. cached_expanded_capacity is the host capacity
// saved with that metadata; fresh calls ignore it. Cached N comes from tensor shape.
// A local failure is not a collective error
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
                            int64_t cached_expanded_capacity,
                            bool allow_multiple_reduction,
                            const DispatchMainSpecialization& specialization);

// Collective on the current stream. Use the same reduction mode and capacity
// as the owning allocation. Return borrowed uint8 [P,Tmax,row_bytes] storage,
// where P=K without local reduction, else min(R,K). Rows contain BF16 x[H],
// padding to a 32-byte boundary, then FP32 weights[K]. Absent/masked weights
// must not be read.
// Rank layout is selected iff allow_multiple_reduction && R<=K. Otherwise a
// compact/local-reduced row uses its first valid top-k slot as representative;
// expanded single-reduction sends each assignment to its own k partition.
// One cooperative kernel contains entry/exit barriers and payload work.
// Entry/exit GPU barriers protect reuse and publication. Expanded padding is
// never sent; local reduction uses FP32 and rounds partial sums to BF16.
// Storage validation is host-only; metadata values/ownership and exact expanded
// capacity are handle preconditions. No routing scan or GPU scalar readback is
// added. The device kernel asserts provenance and expanded-slot bounds. Empty
// ranks still participate in both barriers and return the borrowed view.
// The specialization must match H/K/R/E-per-rank/Tmax before either barrier.
torch::Tensor combine_main(PeerRuntime& runtime,
                           const torch::Tensor& x,
                           const std::vector<torch::Tensor>& metadata,
                           const c10::optional<torch::Tensor>& weights,
                           int64_t max_tokens,
                           bool expand,
                           bool allow_multiple_reduction,
                           const CombineMainSpecialization& specialization);

}  // namespace nano_ep
