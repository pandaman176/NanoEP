"""Actual CUDA return code with logical peer windows on one physical GPU."""

import os
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
    code = (source_dir / "pipeline.cu").read_text()
    start = code.index(
        "template <typename data_t, bool kExpand, bool kAllowMultipleReduction, bool kWithWeights>"
    )
    end = code.index("\n}  // namespace", start)
    directory = tmp_path_factory.mktemp("combine-kernel")
    # Compile the production kernel and launcher, with no duplicate implementation.
    (directory / "combine_kernel_under_test.cuh").write_text(
        '#include "layout.hpp"\nnamespace nano_ep {\n' + code[start:end] + "\n}\n"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    return load(
        name="nano_ep_combine_probe",
        sources=[str(Path(__file__).with_name("combine_probe.cu"))],
        extra_include_paths=[str(directory), str(source_dir)],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O2", "-std=c++20"],
    )


def check_kernel(probe, case, expand, mode, weighted):
    ranks = len(case.indices)
    offset, total = probe.layout(ranks, case.capacity, case.hidden, case.topk, mode)
    partitions = min(ranks, case.topk) if mode else case.topk
    stride = (4 * case.hidden + 4 * case.topk + 15) // 16 * 16
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
    "ranks,topk,hidden", [(1, 3, 4), (2, 3, 33), (2, 2, 65), (4, 2, 65)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
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
    case = make_return_case(4, 2, 33, torch.bfloat16, kind=kind, tokens=tokens)
    check_kernel(probe, case, expand, mode, True)
