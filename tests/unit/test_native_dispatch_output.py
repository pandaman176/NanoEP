"""Check the output oracle, including deliberately incorrect copy results."""

from types import SimpleNamespace

import pytest
import torch

from nano_ep.testing.cases import make_case
from nano_ep.testing.native_dispatch import assert_dispatch_output
from nano_ep.testing.oracle import reference_dispatch



def reordered_output(
    *, expand, weighted, zero_padding, kind="mixed", dtype=torch.float32
):
    case = make_case(2, kind=kind, dtype=dtype)
    rank, alignment = 1, 4
    weights = case.weights if weighted else None
    expected = reference_dispatch(
        case.xs,
        case.indices,
        case.num_experts,
        weights,
        do_expand=expand,
        alignment=alignment,
    )[rank]
    layout = expected.layout
    order = torch.arange(layout.recv_count - 1, -1, -1, dtype=torch.int64)
    local = layout.local_topk_idx[order]
    slots = layout.expanded_slots[order].clone()
    # Row arrival and per-expert atomic reservation order can both vary.
    for expert, count in enumerate(layout.expert_counts.tolist()):
        mask = local == expert
        start = int(layout.expert_offsets[expert])
        slots[mask] = 2 * start + count - 1 - slots[mask]
    state = SimpleNamespace(
        stage_buffer=torch.empty(0, dtype=torch.uint8),
        metadata=[
            layout.src_ranks[order],
            layout.src_tokens[order],
            local,
            layout.expert_counts,
            layout.expert_offsets,
            slots,
        ],
    )
    out_weights = torch.empty(0, dtype=torch.float32)
    if expand:
        x = torch.full_like(expected.x, 0 if zero_padding else float("nan"))
        indices = torch.empty(0, dtype=torch.int64)
        if weighted:
            out_weights = torch.full_like(
                expected.topk_weights, 0 if zero_padding else float("nan")
            )
        for row, source_row in enumerate(order.tolist()):
            for k, slot in enumerate(slots[row].tolist()):
                if slot >= 0:
                    reference_slot = int(layout.expanded_slots[source_row, k])
                    x[slot] = expected.x[reference_slot]
                    if weighted:
                        out_weights[slot] = expected.topk_weights[reference_slot]
    else:
        x, indices = expected.x[order].clone(), local.clone()
        if weighted:
            out_weights = expected.topk_weights[order].clone()
    options = dict(expand=expand, zero_padding=zero_padding)
    return state, [x, indices, out_weights], case, rank, alignment, weights, options


def check(fixture):
    state, outputs, case, rank, alignment, weights, options = fixture
    assert_dispatch_output(state, outputs, case, rank, alignment, weights, **options)


@pytest.mark.parametrize(
    "expand,zero_padding", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["mixed", "empty", "masked", "skew"])
def test_output_checker_accepts_reordered_assignments(
    expand, zero_padding, weighted, dtype, kind
):
    check(
        reordered_output(
            expand=expand,
            weighted=weighted,
            zero_padding=zero_padding,
            dtype=dtype,
            kind=kind,
        )
    )


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(
    "corruption",
    ["payload", "weights", "indices", "missing_weights", "duplicate_source", "padding"],
)
def test_output_checker_rejects_incorrect_copy(expand, corruption):
    fixture = reordered_output(expand=expand, weighted=True, zero_padding=True)
    state, outputs, _, _, _, _, _ = fixture
    x, indices, weights = outputs
    valid = state.metadata[2] >= 0
    slots = state.metadata[5][valid]
    row = int(slots[0]) if expand else 0
    if corruption == "payload":
        x[row, 0] += 1
    elif corruption == "weights":
        if expand:
            weights[row] += 1
        else:
            # Compact copies must preserve masked/nonlocal choices too.
            masked = (state.metadata[2] < 0).nonzero()[0]
            weights[masked[0], masked[1]] += 1
    elif corruption == "indices":
        if expand:
            outputs[1] = state.metadata[2].clone()
        else:
            indices[0, 0] = 99
    elif corruption == "missing_weights":
        outputs[2] = torch.empty(0)
    elif corruption == "duplicate_source":
        state.metadata[0][1] = state.metadata[0][0]
        state.metadata[1][1] = state.metadata[1][0]
    else:
        if expand:
            padding = torch.ones(len(x), dtype=torch.bool)
            padding[slots] = False
            assert padding.any()
            x[padding] = float("nan")
            weights[padding] = float("nan")
        else:
            outputs[0] = x[:-1]
    with pytest.raises(AssertionError):
        check(fixture)


def test_output_checker_rejects_weight_padding_alone():
    fixture = reordered_output(expand=True, weighted=True, zero_padding=True)
    state, outputs, *_ = fixture
    padding = torch.ones(len(outputs[0]), dtype=torch.bool)
    padding[state.metadata[5][state.metadata[2] >= 0]] = False
    assert padding.any()
    outputs[2][padding] = float("nan")
    with pytest.raises(AssertionError):
        check(fixture)
