"""Check return-packet fixtures against scalar metadata-driven transfers."""

import pytest
import torch

from nano_ep.testing.native_combine import (
    assert_combined_output,
    assert_return_packets,
    expert_inputs,
    make_return_case,
    make_return_packet,
    reference_combined,
)



def materialize_packets(case, expand, mode, weighted):
    ranks = len(case.indices)
    partitions = min(ranks, case.topk) if mode else case.topk
    stride = (4 * case.hidden + 4 * case.topk + 15) // 16 * 16
    packets = [
        torch.full((partitions, case.capacity, stride), 165, dtype=torch.uint8)
        for _ in range(ranks)
    ]
    for owner in range(ranks):
        x, weights, metadata = expert_inputs(case, owner, expand)
        sources, tokens, local, _, _, slots, *_ = metadata
        for row, (source, token) in enumerate(
            zip(sources.tolist(), tokens.tolist(), strict=True)
        ):
            choices = [k for k, expert in enumerate(local[row].tolist()) if expert >= 0]
            groups = [[k] for k in choices] if expand and not mode else [choices]
            for group in groups:
                partition = owner if mode and ranks <= case.topk else group[0]
                output = packets[source].view(torch.float32)[partition, token]
                if expand:
                    value = torch.zeros(case.hidden)
                    for k in group:
                        value += x[int(slots[row, k])].float()
                else:
                    value = x[row].float()
                output[: case.hidden] = value
                if weighted:
                    for k in group:
                        output[case.hidden + k] = (
                            weights[int(slots[row, k])] if expand else weights[row, k]
                        )
    return packets


@pytest.mark.parametrize("ranks,topk", [(1, 3), (2, 3), (2, 2), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_packet_fixture_matches_source_assignments(
    ranks, topk, dtype, expand, mode, weighted
):
    case = make_return_case(ranks, topk, 4, dtype)
    for source, packet in enumerate(materialize_packets(case, expand, mode, weighted)):
        assert_return_packets(packet, case, source, expand, mode, weighted)


@pytest.mark.parametrize(
    "kind,tokens", [("empty", 5), ("masked", 5), ("mixed", 0), ("nonfinite", 5)]
)
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_packet_fixture_edge_cases(kind, tokens, expand, mode):
    case = make_return_case(4, 2, 33, torch.bfloat16, kind=kind, tokens=tokens)
    for source, packet in enumerate(materialize_packets(case, expand, mode, True)):
        assert_return_packets(packet, case, source, expand, mode, True)


@pytest.mark.parametrize(
    "corruption",
    ["missing_weight", "nonlocal_x", "masked_token", "wrong_sum", "padding"],
)
def test_packet_checker_rejects_wrong_transfer(corruption):
    case = make_return_case(2, 3, 4, torch.float32)
    packet = materialize_packets(case, False, False, True)[0]
    fields = packet.view(torch.float32)
    if corruption == "missing_weight":
        packet[0, 0, (case.hidden + 2) * 4 : (case.hidden + 3) * 4] = 165
    elif corruption == "nonlocal_x":
        fields[1, 0, : case.hidden] = 0
    elif corruption == "masked_token":
        fields[0, 2, : case.hidden] = 0
    elif corruption == "wrong_sum":
        fields[0, 0, 0] += 1
    else:
        fields[0, 0, -1] = 0
    with pytest.raises(AssertionError):
        assert_return_packets(packet, case, 0, False, False, True)


def test_local_reduction_keeps_each_hidden_coordinate_and_weight():
    case = make_return_case(4, 2, 3, torch.float32)
    case.values[0][0] = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    case.weights[0][0] = torch.tensor([0.2, 0.7])
    packet = materialize_packets(case, True, True, True)[0].view(torch.float32)
    torch.testing.assert_close(packet[0, 0, :3], torch.tensor([11.0, 22.0, 33.0]))
    torch.testing.assert_close(packet[0, 0, 3:5], torch.tensor([0.2, 0.7]))


def decode_packet(packet, case, source, expand, mode, bias):
    """Scalar wire decoder for fixture verification; no native execution."""
    fields = packet.view(torch.float32)
    ranks = len(case.indices)
    local_experts = case.num_experts // ranks
    x = torch.zeros((len(case.indices[source]), case.hidden))
    weights = torch.zeros(case.indices[source].shape)
    for token, experts in enumerate(case.indices[source].tolist()):
        seen = set()
        for k, expert in enumerate(experts):
            if expert < 0:
                continue
            owner = expert // local_experts
            representative = next(
                i
                for i, e in enumerate(experts)
                if e >= 0 and e // local_experts == owner
            )
            partition = (
                k
                if expand and not mode
                else (owner if mode and ranks <= case.topk else representative)
            )
            weights[token, k] = fields[partition, token, case.hidden + k]
            if expand and not mode or owner not in seen:
                x[token] += fields[partition, token, : case.hidden]
                seen.add(owner)
    return (x + bias.float()).to(case.values[source].dtype), weights


@pytest.mark.parametrize("ranks,topk", [(2, 3), (4, 2), (2, 32), (32, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_epilogue_packet_fixture_matches_source_reference(
    ranks, topk, dtype, expand, mode
):
    case = make_return_case(ranks, topk, 33, dtype, tokens=3)
    case.capacity += 7
    for source in (0, ranks - 1):
        packet = make_return_packet(case, source, expand, mode, True)
        bias = torch.linspace(-1, 1, 3 * 33).reshape(3, 33).to(dtype)
        outputs = decode_packet(packet, case, source, expand, mode, bias)
        assert_combined_output(outputs, case, source, expand, mode, True, bias)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_epilogue_reference_hand_calculation_and_masked_nan(expand, mode):
    case = make_return_case(2, 3, 3, torch.float32, tokens=2)
    case.indices[0] = torch.tensor([[0, 1, 3], [-1, -1, -1]])
    case.values[0][0] = torch.tensor(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]
    )
    case.values[0][1].fill_(float("nan"))
    case.weights[0] = torch.tensor([[0.2, 0.7, -0.4], [float("nan")] * 3])
    bias = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_x = torch.tensor([[112.0, 220.0, 336.0], [4.0, 5.0, 6.0]])
    expected_w = torch.tensor([[0.2, 0.7, -0.4], [0.0, 0.0, 0.0]])
    x, w = reference_combined(case, 0, expand, mode, bias)
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(w, expected_w, rtol=0, atol=0)
    packet = make_return_packet(case, 0, expand, mode, True)
    decoded = decode_packet(packet, case, 0, expand, mode, bias)
    torch.testing.assert_close(decoded[0], expected_x, rtol=0, atol=0)
    torch.testing.assert_close(decoded[1], expected_w, rtol=0, atol=0)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_epilogue_reference_compact_rounding_is_explicit(expand, mode):
    case = make_return_case(2, 3, 1, torch.bfloat16, tokens=1)
    case.indices[0][0] = torch.tensor([0, 1, 3])
    case.values[0][0, :, 0] = torch.tensor([256.0, 1.0, -256.0])
    x, _ = reference_combined(case, 0, expand, mode)
    assert x.item() == (1 if expand else 0)


@pytest.mark.parametrize(
    "corruption",
    [
        "hidden_tail",
        "duplicate_local_sum",
        "bias_twice",
        "weight_from_other_k",
        "masked_weight",
        "x_dtype",
        "empty_weight_shape",
    ],
)
def test_epilogue_checker_rejects_wrong_outputs(corruption):
    case = make_return_case(2, 3, 65, torch.bfloat16)
    bias = torch.full((5, 65), 0.5, dtype=torch.bfloat16)
    x, weights = reference_combined(case, 0, False, False, bias)
    weighted = True
    if corruption == "hidden_tail":
        x[:, 32:] = bias[:, 32:]
    elif corruption == "duplicate_local_sum":
        x[0] += 3
    elif corruption == "bias_twice":
        x += bias
    elif corruption == "weight_from_other_k":
        weights[0, 2] = weights[0, 0]
    elif corruption == "masked_weight":
        weights[2].fill_(float("nan"))
    elif corruption == "x_dtype":
        x = x.float()
    else:
        weighted = False
    with pytest.raises(AssertionError):
        assert_combined_output((x, weights), case, 0, False, False, weighted, bias)
