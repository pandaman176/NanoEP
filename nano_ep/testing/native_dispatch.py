"""Inspect native dispatch states against independent CPU specifications."""

import torch

from .oracle import reference_dispatch, reference_layout, reference_plan


def assert_dispatch_output(state, outputs, case, rank, alignment, weights, *, expand, zero_padding):
    """Compare actual copy outputs by source identity and assignment, not arrival order.

    Expanded slot order may differ inside each expert segment. Only valid slots
    are inspected unless zero_padding requests initialized gaps. Nano compact
    weights preserve complete source rows, including masked/nonlocal choices.
    """
    expected = reference_dispatch(case.xs, case.indices, case.num_experts, weights, do_expand=expand, alignment=alignment)[rank]
    x, indices, out_weights = outputs
    rows = expected.layout.expanded_capacity if expand else expected.layout.recv_count
    assert x.shape == (rows, case.hidden) and x.dtype == case.xs[rank].dtype
    assert x.device == state.stage_buffer.device and x.is_contiguous()
    assert indices.device == out_weights.device == x.device
    assert indices.dtype == torch.int64 and out_weights.dtype == torch.float32
    assert indices.is_contiguous() and out_weights.is_contiguous()
    actual_x, actual_indices, actual_weights = (tensor.cpu() for tensor in outputs)
    sources, tokens, local = (tensor.cpu() for tensor in state.metadata[:3])
    keys = list(zip(sources.tolist(), tokens.tolist(), strict=True))
    order = torch.tensor(sorted(range(len(keys)), key=keys.__getitem__), dtype=torch.int64)
    layout = expected.layout
    torch.testing.assert_close(sources[order], layout.src_ranks)
    torch.testing.assert_close(tokens[order], layout.src_tokens)
    torch.testing.assert_close(local[order], layout.local_topk_idx)

    if expand:
        assert actual_indices.shape == (0,)
        slots = state.metadata[5].cpu()[order]
        valid = layout.local_topk_idx >= 0
        assert (slots[~valid] == -1).all()
        for expert, count in enumerate(layout.expert_counts.tolist()):
            start = int(layout.expert_offsets[expert])
            torch.testing.assert_close(slots[layout.local_topk_idx == expert].sort().values, torch.arange(start, start + count))
        actual_rows, expected_rows = slots[valid], layout.expanded_slots[valid]
        torch.testing.assert_close(actual_x[actual_rows], expected.x[expected_rows], rtol=0, atol=0, equal_nan=True)
        if weights is not None:
            assert actual_weights.shape == (rows,)
            torch.testing.assert_close(actual_weights[actual_rows], expected.topk_weights[expected_rows], rtol=0, atol=0, equal_nan=True)
        if zero_padding:
            padding = torch.ones(rows, dtype=torch.bool)
            padding[actual_rows] = False
            torch.testing.assert_close(actual_x[padding], torch.zeros_like(actual_x[padding]), rtol=0, atol=0)
            if weights is not None:
                torch.testing.assert_close(actual_weights[padding], torch.zeros_like(actual_weights[padding]), rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual_x[order], expected.x, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(actual_indices, local)
        if weights is not None:
            assert actual_weights.shape == (rows, case.num_topk)
            torch.testing.assert_close(actual_weights[order], expected.topk_weights, rtol=0, atol=0, equal_nan=True)
    if weights is None:
        assert actual_weights.shape == (0,)


def prepare_dispatch_metadata(native, state):
    """Prepare and validate the complete dispatch receive metadata."""
    native.dispatch_epilogue(state, False, False, False)
    assert state.metadata_prepared


def assert_dispatch_state(state, case, rank, alignment, weights):
    expected = reference_layout(case.indices, rank, case.num_experts, alignment)
    world_size = len(case.xs)
    local_experts = case.num_experts // world_size
    rows, topk = expected.recv_count, case.num_topk
    row_bytes = (16 + 12 * topk + 4 * case.hidden + 15) // 16 * 16
    assert state.metadata_prepared and state.rank == rank
    assert state.hidden == case.hidden and state.x_dtype == case.xs[rank].dtype
    assert state.counts.device.type == "cpu" and state.counts.dtype == torch.int64
    assert state.counts.tolist() == [rows, expected.expanded_capacity]
    assert state.stage_buffer.dtype == torch.uint8
    assert state.stage_buffer.shape == (rows, row_bytes)
    assert len(state.metadata) == 9
    for tensor in state.metadata:
        assert tensor.device == state.stage_buffer.device and tensor.dtype == torch.int64
        assert tensor.is_contiguous()
    src_ranks, src_tokens, local, counts, offsets, slots, dst_slots, sends, snapshot = [t.cpu() for t in state.metadata]
    assert src_ranks.shape == src_tokens.shape == (rows,)
    assert local.shape == slots.shape == (rows, topk)
    keys = list(zip(src_ranks.tolist(), src_tokens.tolist(), strict=True))
    permutation = torch.tensor(sorted(range(rows), key=keys.__getitem__), dtype=torch.int64)
    torch.testing.assert_close(src_ranks[permutation], expected.src_ranks)
    torch.testing.assert_close(src_tokens[permutation], expected.src_tokens)
    torch.testing.assert_close(local[permutation], expected.local_topk_idx)
    torch.testing.assert_close(counts, expected.expert_counts)
    torch.testing.assert_close(offsets, expected.expert_offsets)
    assert (slots[local < 0] == -1).all()
    for expert in range(local_experts):
        start, count = offsets[expert].item(), counts[expert].item()
        torch.testing.assert_close(slots[local == expert].sort().values, torch.arange(start, start + count))
    torch.testing.assert_close(snapshot, case.indices[rank])
    torch.testing.assert_close(sends, reference_plan(case.indices[rank], case.num_experts, world_size).send_counts)
    assert dst_slots.shape == (case.indices[rank].shape[0], world_size)
    for token, choices in enumerate(case.indices[rank].tolist()):
        destinations = {expert // local_experts for expert in choices if expert >= 0}
        for destination in range(world_size):
            slot = dst_slots[token, destination].item()
            assert 0 <= slot < world_size * case.capacity if destination in destinations else slot == -1
            if destination == rank and destination in destinations:
                assert keys[slot] == (rank, token)
    payload = state.stage_buffer.cpu()
    for row, (source, token) in enumerate(keys):
        record = payload[row]
        assert record[:16].view(torch.int64).tolist() == [source, token]
        torch.testing.assert_close(record[16 : 16 + 8 * topk].view(torch.int64), case.indices[source][token])
        if weights is not None:
            torch.testing.assert_close(
                record[16 + 8 * topk : 16 + 12 * topk].view(torch.float32), weights[source][token], rtol=0, atol=0, equal_nan=True
            )
        torch.testing.assert_close(
            record[16 + 12 * topk : 16 + 12 * topk + 4 * case.hidden].view(torch.float32),
            case.xs[source][token].float(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
