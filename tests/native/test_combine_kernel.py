"""Actual CUDA return code with logical peer windows on one physical GPU."""

import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

from nano_ep.backends._build import native_source_dir
from nano_ep.testing.native_combine import (
    assert_return_packets,
    expert_inputs,
    make_return_case,
)
from nano_ep.testing.wire import return_row_bytes

pytestmark = [pytest.mark.cuda]


@pytest.fixture(scope="module")
def probe(request, tmp_path_factory):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda or --run-distributed")
    if not torch.cuda.is_available():
        pytest.fail("CUDA tests requested but NVIDIA CUDA is unavailable")
    source_dir = native_source_dir()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

    @lru_cache(maxsize=None)
    def prepare(hidden, topk, ranks, capacity):
        geometry = (hidden, topk, ranks, capacity)
        defines = [
            f"-DNANO_EP_{name}={value}"
            for name, value in zip(
                ("HIDDEN", "TOPK", "WORLD_SIZE", "MAX_TOKENS"), geometry, strict=True
            )
        ]
        return load(
            name="nano_ep_combine_probe_" + "_".join(map(str, geometry)),
            sources=[str(Path(__file__).with_name("combine_probe.cu"))],
            extra_include_paths=[str(source_dir)],
            extra_cflags=["-O2", "-std=c++20"],
            extra_cuda_cflags=["-O2", "-std=c++20", *defines],
        )

    return prepare


@pytest.mark.parametrize("topk", [1, 3, 32])
def test_slot_shuffle_compacts_int64_values_for_every_lane(probe, topk):
    module = probe(256, topk, 2, 5)
    values = (1 << 40) + torch.arange(7 * topk, dtype=torch.int64).reshape(7, topk)
    valid = torch.zeros((7, topk), dtype=torch.bool)
    valid[1, 0] = True
    valid[2, -1] = True
    valid[3, [0, topk - 1]] = True
    valid[4] = True
    valid[5, ::2] = True
    valid[6, 1::2] = True
    slots = values.masked_fill(~valid, -1)
    expected = torch.full((7, topk), -1, dtype=torch.int64)
    counts = valid.sum(1).to(torch.int32)
    for row in range(7):
        expected[row, : counts[row].item()] = slots[row, valid[row]]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual, actual_counts = module.compact_slots(slots.cuda())
        torch.testing.assert_close(
            actual.cpu(), expected[:, None].expand(-1, 32, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_counts.cpu(), counts[:, None].expand(-1, 32), rtol=0, atol=0
        )
        empty, empty_counts = module.compact_slots(
            torch.empty((0, topk), dtype=torch.int64, device="cuda")
        )
        assert empty.shape == (0, 32, topk) and empty_counts.shape == (0, 32)
    stream.synchronize()


def check_kernel(
    probe,
    case,
    expand,
    mode,
    weighted,
    *,
    expected_first_recv_count=None,
    packet_check=None,
):
    ranks = len(case.indices)
    probe = probe(case.hidden, case.topk, ranks, case.capacity)
    offset, total = probe.layout(ranks, case.capacity, case.hidden, case.topk, mode)
    partitions = min(ranks, case.topk) if mode else case.topk
    stride = return_row_bytes(case.hidden, case.topk)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        windows = [
            torch.full((total,), 165, dtype=torch.uint8, device="cuda")
            for _ in range(ranks)
        ]
        for owner in range(ranks):
            x, weights, metadata = expert_inputs(case, owner, expand)
            if owner == 0 and expected_first_recv_count is not None:
                assert metadata[0].numel() == expected_first_recv_count
            device_x, device_weights = x.cuda(), weights.cuda()
            device_metadata = [tensor.cuda() for tensor in metadata]
            probe.run(
                windows,
                device_x,
                device_metadata,
                device_weights if weighted else None,
                owner,
                case.capacity,
                expand,
                mode,
            )
            torch.testing.assert_close(
                device_x.cpu(), x, rtol=0, atol=0, equal_nan=True
            )
            torch.testing.assert_close(
                device_weights.cpu(), weights, rtol=0, atol=0, equal_nan=True
            )
            for actual, expected in zip(device_metadata, metadata, strict=True):
                torch.testing.assert_close(actual.cpu(), expected)
        for source, window in enumerate(windows):
            returned = window[
                offset : offset + partitions * case.capacity * stride
            ].view(partitions, case.capacity, stride)
            assert_return_packets(returned, case, source, expand, mode, weighted)
            if packet_check is not None:
                packet_check(returned, source)
            assert (window[:offset] == 165).all().item()
            assert (window[offset + returned.numel() :] == 165).all().item()
    stream.synchronize()


@pytest.mark.parametrize(
    "ranks,topk,hidden", [(1, 3, 256), (2, 3, 512), (2, 2, 768), (4, 2, 256)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_return_kernel_packets(
    probe, ranks, topk, hidden, dtype, expand, mode, weighted
):
    case = make_return_case(ranks, topk, hidden, dtype)
    check_kernel(probe, case, expand, mode, weighted)


@pytest.mark.parametrize(
    "kind,tokens",
    [("empty", 5), ("masked", 5), ("mixed", 0), ("mixed", 19), ("nonfinite", 5)],
)
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_return_kernel_edge_cases(probe, kind, tokens, expand, mode):
    case = make_return_case(4, 2, 256, torch.bfloat16, kind=kind, tokens=tokens)
    check_kernel(probe, case, expand, mode, True)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_return_kernel_rounds_local_sum_to_bf16(probe, expand, mode):
    case = make_return_case(2, 3, 256, torch.bfloat16, tokens=1)
    case.indices[0][0] = torch.tensor([0, 1, 3])
    case.values[0][0, 0].fill_(256)
    case.values[0][0, 1].fill_(1)
    case.values[0][0, 2].fill_(-256)
    check_kernel(probe, case, expand, mode, True)


@pytest.mark.parametrize(
    "ranks,topk,k", [(2, 1, 0), (2, 3, 0), (2, 3, 2), (4, 3, 1), (2, 32, 31)]
)
@pytest.mark.parametrize("weighted", [False, True])
def test_expanded_single_assignment_preserves_payload_bits(
    probe, ranks, topk, k, weighted
):
    case = make_return_case(ranks, topk, 256, torch.bfloat16, tokens=2)
    case.capacity = 5
    local_experts = case.num_experts // ranks
    bits = torch.tensor(
        [0x8000, 0, 0x3F80, 0x7F80, 0xFF80, 0x7FC5, 1, 0x8001], dtype=torch.uint16
    )
    payload = bits.view(torch.bfloat16).repeat(case.hidden // bits.numel())
    for source in range(ranks):
        case.indices[source].fill_(-1)
        case.values[source].fill_(float("nan"))
        for token in range(2):
            destination = (source + token + 1) % ranks
            case.indices[source][token, k] = destination * local_experts + 2
            case.values[source][token, k] = payload.roll(source + token)

    def check_bits(returned, source):
        for token in range(2):
            destination = int(case.indices[source][token, k]) // local_experts
            partition = destination if ranks <= topk else k
            actual = (
                returned[partition, token, : case.hidden * 2].cpu().view(torch.int16)
            )
            expected = case.values[source][token, k].view(torch.int16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    check_kernel(probe, case, True, True, weighted, packet_check=check_bits)


@pytest.mark.parametrize("ranks,topk", [(2, 3), (4, 2)])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    "boundary", ["sm", "wave_minus_one", "wave", "wave_plus_one", "multiple_waves"]
)
def test_return_kernel_across_sm_grid_iterations(
    probe, ranks, topk, expand, mode, weighted, boundary
):
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    wave = sms * 8
    receive_count = {
        "sm": sms + 1,
        "wave_minus_one": wave - 1,
        "wave": wave,
        "wave_plus_one": wave + 1,
        "multiple_waves": 2 * wave + 17,
    }[boundary]
    source_counts = [receive_count // ranks] * ranks
    for source in range(receive_count % ranks):
        source_counts[source] += 1
    if source_counts[-1] > 0:
        source_counts[0] += 1
        source_counts[-1] -= 1

    case = make_return_case(ranks, topk, 256, torch.bfloat16, tokens=max(source_counts))
    case.capacity = 2 * wave + 23
    last_rank_expert = (ranks - 1) * (case.num_experts // ranks)
    patterns = (
        [[0, last_rank_expert, -1], [-1, 1, last_rank_expert + 1], [2, -1, 0]]
        if topk == 3
        else [[0, last_rank_expert], [-1, 1], [2, 0]]
    )
    position = 0
    for source, count in enumerate(source_counts):
        routing = torch.empty((count, topk), dtype=torch.int64)
        for token in range(count):
            # Shift routing between waves so reused warps see different masks.
            routing[token] = torch.tensor(
                patterns[(position % wave + position // wave) % len(patterns)]
            )
            position += 1
        case.indices[source] = routing
        case.values[source] = case.values[source][:count].clone()
        case.weights[source] = case.weights[source][:count].clone()
        case.values[source][routing < 0] = float("nan")
        case.weights[source][routing < 0] = float("nan")

    check_kernel(
        probe, case, expand, mode, weighted, expected_first_recv_count=receive_count
    )
