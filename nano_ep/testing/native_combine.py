"""Independent source-assignment fixtures for native return-packet tests."""

from dataclasses import dataclass

import torch

from .oracle import reference_layout, reference_plan


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


def assert_return_packets(actual, case, source, expand, mode, weighted):
    """Inspect published fields and require every other byte to retain its poison."""
    ranks = len(case.indices)
    local_experts = case.num_experts // ranks
    partitions = min(ranks, case.topk) if mode else case.topk
    stride = (4 * case.hidden + 4 * case.topk + 15) // 16 * 16
    assert actual.dtype == torch.uint8 and actual.shape == (partitions, case.capacity, stride)
    actual = actual.cpu()
    expected = torch.full(actual.shape, 165, dtype=torch.uint8).view(torch.float32)
    written = torch.zeros(expected.shape, dtype=torch.bool)
    for token, routing in enumerate(case.indices[source].tolist()):
        for owner in range(ranks):
            choices = [k for k, expert in enumerate(routing) if expert >= 0 and expert // local_experts == owner]
            if not choices:
                continue
            groups = [[k] for k in choices] if expand and not mode else [choices]
            for group in groups:
                partition = owner if mode and ranks <= case.topk else group[0]
                value = _local_sum(case, source, token, group)
                if not expand:
                    value = value.to(case.values[source].dtype).float()
                expected[partition, token, : case.hidden] = value
                written[partition, token, : case.hidden] = True
                if weighted:
                    for k in group:
                        expected[partition, token, case.hidden + k] = case.weights[source][token, k]
                        written[partition, token, case.hidden + k] = True
    floats = actual.view(torch.float32)
    torch.testing.assert_close(floats[written], expected[written], rtol=0, atol=0, equal_nan=True)
    untouched_bytes = (~written).repeat_interleave(4, dim=-1)
    assert (actual[untouched_bytes] == 165).all(), "unpublished return fields or padding were overwritten"


def make_return_packet(case, source, expand, mode, weighted):
    """Encode source assignments as return rows; unpublished FP32 fields are NaN.

    This fixture does not invoke a native kernel or use receive metadata. The
    extra capacity and NaN gaps expose readers that use T instead of Tmax or
    consume a partition/weight field that its sender never published.
    """
    ranks = len(case.indices)
    local_experts = case.num_experts // ranks
    partitions = min(ranks, case.topk) if mode else case.topk
    words = (case.hidden + case.topk + 3) // 4 * 4
    fields = torch.full((partitions, case.capacity, words), float("nan"))
    for token, routing in enumerate(case.indices[source].tolist()):
        for owner in range(ranks):
            choices = [k for k, expert in enumerate(routing) if expert >= 0 and expert // local_experts == owner]
            groups = [[k] for k in choices] if expand and not mode else ([choices] if choices else [])
            for group in groups:
                partition = owner if mode and ranks <= case.topk else group[0]
                value = _local_sum(case, source, token, group)
                if not expand:
                    value = value.to(case.values[source].dtype).float()
                fields[partition, token, : case.hidden] = value
                if weighted:
                    for k in group:
                        fields[partition, token, case.hidden + k] = case.weights[source][token, k]
    return fields.view(torch.uint8)


def reference_combined(case, source, expand, mode, bias=None):
    """Source-side arithmetic reference, independent of return-buffer addressing.

    Expanded single reduction visits valid k in order. Otherwise partial sums
    visit destination ranks in first-occurrence order. Compact client sums round
    to the input dtype; expanded local sums stay FP32. Source bias is added once
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
            if not expand:
                partial = partial.to(dtype).float()
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
