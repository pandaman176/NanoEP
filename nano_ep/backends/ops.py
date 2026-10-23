"""CUDA tensor operators. Payloads and indices remain on the same CUDA device."""

from ._build import invoke, load_extension


def route_counts(topk_idx, num_experts: int, world_size: int):
    """Return CUDA int64 (rank_counts[R], expert_counts[E]).

    topk_idx is contiguous int64 [T,K], with -1 masked and distinct valid
    experts in each row. Rank counts deduplicate destinations per token;
    expert counts count every valid assignment. E is divisible by R.
    """
    return invoke(load_extension().route_counts, topk_idx, num_experts, world_size)


def pack_rows(x, indices):
    """Gather contiguous CUDA x[T,H] by int64 indices[P] into [P,H].

    Repeated indices intentionally duplicate rows. P may be zero. Inputs are
    unmodified; the result owns its storage. Supported x: FP32, FP64, BF16.
    """
    return invoke(load_extension().pack_rows, x, indices)


def expand_rows(x, expanded_slots, capacity: int):
    """Copy compact x[N,H] to zero-filled [capacity,H] expert storage.

    CUDA int64 expanded_slots[N,K] maps assignments to output rows. -1 is
    ignored, all nonnegative slots are unique and below capacity. NaN x in a
    fully masked compact row must never enter output or its padding gaps.
    """
    return invoke(load_extension().expand_rows, x, expanded_slots, capacity)


def reduce_rows(x, expanded_slots):
    """Gather/sum expanded x[C,H] to compact [N,H], ignoring -1 slots.

    CUDA int64 expanded_slots[N,K] is the inverse layout. Accumulate FP32/BF16
    in FP32 and FP64 in FP64, then cast once to x.dtype. Fully masked rows are
    zero. Padding rows may contain NaNs: they must not be read or multiplied
    by zero. This operator never applies gate weights.
    """
    return invoke(load_extension().reduce_rows, x, expanded_slots)
