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


def check_kernel(probe, case, expand, mode, weighted):
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
