"""Scalar-loop specifications. Never import production routing or layouts here.

The expected types intentionally differ from production handles. Reconstructing
routes from source indices keeps the oracle independent of the implementation.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass
class ExpectedPlan:
    topk_idx: torch.Tensor
    token_in_rank: torch.Tensor
    send_counts: torch.Tensor
    expert_counts: torch.Tensor
    send_offsets: torch.Tensor
    send_token_indices: torch.Tensor


@dataclass
class ExpectedLayout:
    src_ranks: torch.Tensor
    src_tokens: torch.Tensor
    local_topk_idx: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    expanded_slots: torch.Tensor
    recv_count: int
    expanded_capacity: int


@dataclass
class ExpectedBatch:
    x: torch.Tensor
    topk_idx: torch.Tensor | None
    topk_weights: torch.Tensor | None
    layout: ExpectedLayout
    rank: int
    do_expand: bool


def reference_plan(indices: torch.Tensor, num_experts: int, world_size: int) -> ExpectedPlan:
    local_experts = num_experts // world_size
    token_in_rank = torch.zeros((len(indices), world_size), dtype=torch.bool)
    expert_counts = [0] * num_experts
    for token, row in enumerate(indices.tolist()):
        for expert in row:
            if expert >= 0:
                expert_counts[expert] += 1
                token_in_rank[token, expert // local_experts] = True
    counts, offsets, packed = [], [0], []
    for rank in range(world_size):
        tokens = [t for t in range(len(indices)) if bool(token_in_rank[t, rank])]
        counts.append(len(tokens))
        packed.extend(tokens)
        offsets.append(len(packed))
    return ExpectedPlan(
        indices.clone(),
        token_in_rank,
        torch.tensor(counts, dtype=torch.int64),
        torch.tensor(expert_counts, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(packed, dtype=torch.int64),
    )


def reference_layout(indices: Sequence[torch.Tensor], rank: int, num_experts: int, alignment: int = 1) -> ExpectedLayout:
    local_experts = num_experts // len(indices)
    topk = indices[0].shape[1]
    src_ranks, src_tokens, local_rows = [], [], []
    counts = [0] * local_experts
    for source, rank_indices in enumerate(indices):
        for token, row in enumerate(rank_indices.tolist()):
            localized = [e - rank * local_experts if rank * local_experts <= e < (rank + 1) * local_experts else -1 for e in row]
            if any(e >= 0 for e in localized):
                src_ranks.append(source)
                src_tokens.append(token)
                local_rows.append(localized)
                for expert in localized:
                    if expert >= 0:
                        counts[expert] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + ((count + alignment - 1) // alignment) * alignment)
    slots = [[-1] * topk for _ in local_rows]
    next_slot = offsets[:-1].copy()
    for token, row in enumerate(local_rows):
        for k, expert in enumerate(row):
            if expert >= 0:
                slots[token][k] = next_slot[expert]
                next_slot[expert] += 1
    return ExpectedLayout(
        torch.tensor(src_ranks, dtype=torch.int64),
        torch.tensor(src_tokens, dtype=torch.int64),
        torch.tensor(local_rows, dtype=torch.int64).reshape(-1, topk),
        torch.tensor(counts, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(slots, dtype=torch.int64).reshape(-1, topk),
        len(local_rows),
        offsets[-1],
    )


def reference_dispatch(
    xs: Sequence[torch.Tensor],
    indices: Sequence[torch.Tensor],
    num_experts: int,
    weights: Sequence[torch.Tensor] | None = None,
    *,
    do_expand: bool = False,
    alignment: int = 1,
) -> list[ExpectedBatch]:
    """Exact valid values; all oracle padding is zero for convenient inspection."""
    result = []
    for rank in range(len(xs)):
        layout = reference_layout(indices, rank, num_experts, alignment)
        rows = layout.expanded_capacity if do_expand else layout.recv_count
        data = xs[0].new_zeros((rows, xs[0].shape[1]))
        out_weights = None
        if weights is not None:
            shape = (rows,) if do_expand else (rows, indices[0].shape[1])
            out_weights = weights[0].new_zeros(shape)
        for recv, (src, token) in enumerate(zip(layout.src_ranks.tolist(), layout.src_tokens.tolist(), strict=True)):
            if do_expand:
                for k, slot in enumerate(layout.expanded_slots[recv].tolist()):
                    if slot >= 0:
                        data[slot] = xs[src][token]
                        if weights is not None:
                            out_weights[slot] = weights[src][token, k]
            else:
                data[recv] = xs[src][token]
                if weights is not None:
                    out_weights[recv] = weights[src][token]
        result.append(
            ExpectedBatch(
                data,
                None if do_expand else layout.local_topk_idx.clone(),
                out_weights,
                layout,
                rank,
                do_expand,
            )
        )
    return result


def reference_combine(
    outputs: Sequence[torch.Tensor],
    batches: Sequence[ExpectedBatch],
    indices: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor] | None = None,
    biases: Sequence[torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Single-final-reduction Nano contract, independent of production handles."""
    dtype = outputs[0].dtype
    accumulator_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    result = [torch.zeros((len(idx), outputs[0].shape[1]), dtype=accumulator_dtype) for idx in indices]
    returned_weights = None if weights is None else [weights[0].new_zeros(idx.shape) for idx in indices]
    for destination, batch in enumerate(batches):
        for row, (src, token) in enumerate(
            zip(
                batch.layout.src_ranks.tolist(),
                batch.layout.src_tokens.tolist(),
                strict=True,
            )
        ):
            if batch.do_expand:
                for k, slot in enumerate(batch.layout.expanded_slots[row].tolist()):
                    if slot >= 0:
                        result[src][token] += outputs[destination][slot].to(accumulator_dtype)
                        if weights is not None:
                            returned_weights[src][token, k] = weights[destination][slot]
            else:
                result[src][token] += outputs[destination][row].to(accumulator_dtype)
                if weights is not None:
                    for k, local_expert in enumerate(batch.layout.local_topk_idx[row].tolist()):
                        if local_expert >= 0:
                            returned_weights[src][token, k] = weights[destination][row, k]
    if biases is not None:
        for rank, bias in enumerate(biases):
            for part in bias if isinstance(bias, tuple) else (() if bias is None else (bias,)):
                result[rank] += part.to(accumulator_dtype)
    return [value.to(dtype) for value in result], returned_weights


def dense_moe(
    xs: Sequence[torch.Tensor],
    indices: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    expert_matrices: torch.Tensor,
) -> list[torch.Tensor]:
    """Dense expert-loop answer, preserving gradients for every continuous input."""
    outputs = []
    for x, idx, weight in zip(xs, indices, weights, strict=True):
        zero_connection = expert_matrices.sum() * 0 + weight.sum() * 0
        if len(x) == 0:
            outputs.append(x + zero_connection)
            continue
        rows = []
        for token, choices in enumerate(idx.tolist()):
            value = x[token] * 0 + zero_connection
            for k, expert in enumerate(choices):
                if expert >= 0:
                    value = value + torch.tanh(x[token] @ expert_matrices[expert]) * weight[token, k]
            rows.append(value)
        outputs.append(torch.stack(rows))
    return outputs
