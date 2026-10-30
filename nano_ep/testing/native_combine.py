"""Independent source-assignment fixtures for native return-packet tests."""

from dataclasses import dataclass

import torch

from .oracle import reference_layout, reference_plan
from .wire import return_row_bytes, return_weights_offset


@dataclass
class ReturnCase:
    indices: list[torch.Tensor]
    values: list[torch.Tensor]  # per source [T,K,H], already weighted expert outputs
    weights: list[torch.Tensor]  # independent per source [T,K] payload
    hidden: int
    topk: int
    capacity: int
    num_experts: int


def make_return_case(ranks, topk, hidden, dtype, *, kind="mixed", tokens=5):
    generator = torch.Generator().manual_seed(71)
    local_experts = max(3, topk)
    indices, values, weights = [], [], []
    for source in range(ranks):
        count = 0 if kind == "empty" and source == ranks - 1 else tokens
        routing = torch.full((count, topk), -1, dtype=torch.int64)
        for token in range(count):
            owner = (source + token) % ranks
            if kind == "masked" or token % 4 == 2:
                continue
            if token % 4 == 1:
                for k in range(topk):
                    routing[token, k] = ((source + k) % ranks) * local_experts + k
            else:
                for k in range(topk):
                    routing[token, k] = owner * local_experts + k
                if topk == 3:
                    routing[token, 1 if token % 4 == 0 else 0] = -1
                elif topk == 2 and token % 4 == 3:
                    routing[token, 0] = -1
        value = (torch.randn(count, topk, hidden, generator=generator) + source * 3).to(dtype)
        weight = torch.randn(count, topk, generator=generator)
        if kind == "nonfinite" and count:
            value[0, 0, 0] = float("nan")
            value[0, topk - 1, -1] = float("inf")
            weight[0, topk - 1] = float("nan")
        indices.append(routing)
        values.append(value)
        weights.append(weight)
    return ReturnCase(indices, values, weights, hidden, topk, max(1, tokens), ranks * local_experts)


def _local_sum(case, source, token, choices):
    value = torch.zeros(case.hidden, dtype=torch.float32)
    for k in choices:
        value += case.values[source][token, k].float()
    return value


def expert_inputs(case, rank, expand, *, alignment=4):
    """Prepare coherent handle tensors with reversed receive and expert-slot order."""
    ranks = len(case.indices)
    layouts = [reference_layout(case.indices, r, case.num_experts, alignment) for r in range(ranks)]
    layout = layouts[rank]
    order = torch.arange(layout.recv_count - 1, -1, -1, dtype=torch.int64)
    sources, tokens, local = layout.src_ranks[order], layout.src_tokens[order], layout.local_topk_idx[order]
    slots = layout.expanded_slots[order].clone()
    for expert, count in enumerate(layout.expert_counts.tolist()):
        start = int(layout.expert_offsets[expert])
        mask = local == expert
        slots[mask] = 2 * start + count - 1 - slots[mask]
    rows = layout.expanded_capacity if expand else layout.recv_count
    x = torch.full((rows, case.hidden), float("nan"), dtype=case.values[rank].dtype)
    weights = torch.full((rows,) if expand else (rows, case.topk), float("nan"))
    for row, (source, token) in enumerate(zip(sources.tolist(), tokens.tolist(), strict=True)):
        choices = [k for k, expert in enumerate(local[row].tolist()) if expert >= 0]
        if expand:
            for k in choices:
                slot = int(slots[row, k])
                x[slot] = case.values[source][token, k]
                weights[slot] = case.weights[source][token, k]
        else:
            x[row] = _local_sum(case, source, token, choices).to(x.dtype)
            weights[row] = case.weights[source][token]
    dst_slots = torch.full((len(case.indices[rank]), ranks), -1, dtype=torch.int64)
    for destination, recv in enumerate(layouts):
        keys = list(zip(recv.src_ranks.tolist(), recv.src_tokens.tolist(), strict=True))[::-1]
        for row, (source, token) in enumerate(keys):
            if source == rank:
                dst_slots[token, destination] = row
    metadata = [
        sources,
        tokens,
        local,
        layout.expert_counts,
        layout.expert_offsets,
        slots,
        dst_slots,
        reference_plan(case.indices[rank], case.num_experts, ranks).send_counts,
        case.indices[rank].clone(),
    ]
    return x, weights, metadata


def _packet_views(packet, hidden):
    weight_offset = return_weights_offset(hidden)
    return packet[..., : 2 * hidden].view(torch.bfloat16), packet[..., weight_offset:].view(torch.float32)


def assert_return_packets(actual, case, source, expand, mode, weighted):
    """Inspect BF16 x / FP32 weights; all unpublished bytes retain their poison."""
    expected = make_return_packet(case, source, expand, mode, weighted)
    assert actual.dtype == torch.uint8 and actual.shape == expected.shape
    actual = actual.cpu()
    written = torch.zeros(actual.shape, dtype=torch.bool)
    local_experts = case.num_experts // len(case.indices)
    weight_offset = return_weights_offset(case.hidden)
    for token, routing in enumerate(case.indices[source].tolist()):
        for owner in range(len(case.indices)):
            choices = [k for k, expert in enumerate(routing) if expert >= 0 and expert // local_experts == owner]
            groups = [[k] for k in choices] if expand and not mode else ([choices] if choices else [])
            for group in groups:
                partition = owner if mode and len(case.indices) <= case.topk else group[0]
                written[partition, token, : 2 * case.hidden] = True
                if weighted:
                    for k in group:
                        written[partition, token, weight_offset + 4 * k : weight_offset + 4 * (k + 1)] = True
    actual_x, actual_weights = _packet_views(actual, case.hidden)
    expected_x, expected_weights = _packet_views(expected, case.hidden)
    x_mask = written[..., : 2 * case.hidden : 2]
    weight_mask = written[..., weight_offset::4]
    torch.testing.assert_close(actual_x[x_mask], expected_x[x_mask], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(actual_weights[weight_mask], expected_weights[weight_mask], rtol=0, atol=0, equal_nan=True)
    assert (actual[~written] == 165).all(), "unpublished return fields or padding were overwritten"


def make_return_packet(case, source, expand, mode, weighted):
    """Encode BF16 partials and aligned FP32 weights; unpublished fields are NaN.

    Independent of native kernels and receive metadata. Extra capacity and NaN
    gaps expose incorrect partition strides and reads of unpublished fields.
    """
    ranks = len(case.indices)
    local_experts = case.num_experts // ranks
    partitions = min(ranks, case.topk) if mode else case.topk
    stride = return_row_bytes(case.hidden, case.topk)
    packet = torch.full((partitions, case.capacity, stride), 255, dtype=torch.uint8)
    x, weights = _packet_views(packet, case.hidden)
    for token, routing in enumerate(case.indices[source].tolist()):
        for owner in range(ranks):
            choices = [k for k, expert in enumerate(routing) if expert >= 0 and expert // local_experts == owner]
            groups = [[k] for k in choices] if expand and not mode else ([choices] if choices else [])
            for group in groups:
                partition = owner if mode and ranks <= case.topk else group[0]
                if expand and len(group) == 1:
                    x[partition, token] = case.values[source][token, group[0]]
                else:
                    x[partition, token] = _local_sum(case, source, token, group).bfloat16()
                if weighted:
                    for k in group:
                        weights[partition, token, k] = case.weights[source][token, k]
    return packet


def reference_combined(case, source, expand, mode, bias=None):
    """Source-side arithmetic reference, independent of return-buffer addressing.

    Expanded single reduction visits valid k in order. Otherwise partial sums
    visit destination ranks in first-occurrence order. Compact client sums round
    to BF16; expanded local sums also round to BF16 on the wire. Source bias is added once
    in FP32 before the final cast. Routing weights remain an independent payload.
    """
    routing = case.indices[source]
    dtype = case.values[source].dtype
    local_experts = case.num_experts // len(case.indices)
    result = torch.zeros((len(routing), case.hidden), dtype=torch.float32)
    for token, experts in enumerate(routing.tolist()):
        groups = {}
        for k, expert in enumerate(experts):
            if expert >= 0:
                key = k if expand and not mode else expert // local_experts
                groups.setdefault(key, []).append(k)
        for choices in groups.values():
            partial = torch.zeros(case.hidden, dtype=torch.float32)
            for k in choices:
                partial += case.values[source][token, k].float()
            partial = partial.bfloat16().float()
            result[token] += partial
    if bias is not None:
        result += bias.cpu().float()
    weights = torch.where(routing >= 0, case.weights[source], 0.0)
    return result.to(dtype), weights


def assert_combined_output(outputs, case, source, expand, mode, weighted, bias=None):
    """Check final values and public output shapes, including masked/empty rows."""
    x, weights = outputs
    expected_x, expected_weights = reference_combined(case, source, expand, mode, bias)
    assert x.is_contiguous() and weights.is_contiguous()
    torch.testing.assert_close(x.cpu(), expected_x, rtol=0, atol=0, equal_nan=True)
    if weighted:
        torch.testing.assert_close(weights.cpu(), expected_weights, rtol=0, atol=0, equal_nan=True)
    else:
        assert weights.dtype == torch.float32 and weights.shape == (0,)
