"""Validate combine setup, real self transfers, and empty collectives."""

from contextlib import contextmanager

import pytest
import torch

from nano_ep.backends._build import load_extension, prepare_combine_main
from nano_ep.testing.native_combine import (
    assert_return_packets,
    expert_inputs,
    make_return_case,
)

pytestmark = [pytest.mark.cuda]


@pytest.fixture
def native(request):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda or --run-distributed")
    if not torch.cuda.is_available():
        pytest.fail("CUDA tests requested but NVIDIA CUDA is unavailable")
    return load_extension()


@pytest.fixture
def launcher(native):
    return prepare_combine_main(256, 3, 1, 3, 5).launcher


@contextmanager
def empty_inputs(native, mode=False, dtype=torch.bfloat16, tokens=2):
    # A source may have tokens while receiving no assignments at all.
    capacity, hidden, topk, experts = 5, 256, 3, 3
    layout = native.window_layout(1, capacity, hidden, topk, experts, mode)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    window[layout[3] :].fill_(165)
    metadata = [
        torch.empty(0, dtype=torch.int64, device="cuda"),
        torch.empty(0, dtype=torch.int64, device="cuda"),
        torch.empty((0, topk), dtype=torch.int64, device="cuda"),
        torch.zeros(experts, dtype=torch.int64, device="cuda"),
        torch.zeros(experts + 1, dtype=torch.int64, device="cuda"),
        torch.empty((0, topk), dtype=torch.int64, device="cuda"),
        torch.full((tokens, 1), -1, dtype=torch.int64, device="cuda"),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        torch.full((tokens, topk), -1, dtype=torch.int64, device="cuda"),
    ]
    x = torch.empty((0, hidden), dtype=dtype, device="cuda")
    torch.cuda.synchronize()
    runtime.import_peer_handles([runtime.export_local_handle()])
    try:
        yield runtime, window, layout, x, metadata
    finally:
        torch.cuda.synchronize()
        del window
        runtime.close_peer_mappings()
        runtime.close()


@pytest.mark.parametrize("hidden", [1, 33, 128, 255, 257])
def test_combine_hidden_constraint_precedes_barrier(native, launcher, hidden):
    with empty_inputs(native) as (runtime, window, _, _, metadata):
        x = torch.empty((0, hidden), dtype=torch.bfloat16, device="cuda")
        before = window.clone()
        with pytest.raises(RuntimeError, match="multiple of 256"):
            native.combine_main(
                runtime, x, metadata, None, 5, False, False, launcher=launcher
            )
        torch.testing.assert_close(window, before, rtol=0, atol=0)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("offset", [1, 4, 7, 8])
def test_combine_input_alignment_before_barrier(native, launcher, expand, mode, offset):
    case = make_return_case(1, 3, 256, torch.bfloat16, tokens=3)
    case.capacity = 5
    host_x, _, host_metadata = expert_inputs(case, 0, expand)
    with empty_inputs(native, mode) as (runtime, window, _, _, _):
        storage = torch.empty(
            host_x.numel() + offset, dtype=torch.bfloat16, device="cuda"
        )
        x = storage[offset:].view(host_x.shape)
        x.copy_(host_x)
        assert x.is_contiguous() and x.storage_offset() == offset
        assert x.data_ptr() % 16 == offset * x.element_size() % 16
        metadata = [tensor.cuda() for tensor in host_metadata]
        before = window.clone()
        if offset % 8:
            with pytest.raises(RuntimeError, match="16-byte aligned"):
                native.combine_main(
                    runtime, x, metadata, None, 5, expand, mode, launcher=launcher
                )
            torch.testing.assert_close(window, before, rtol=0, atol=0)
        else:
            returned = native.combine_main(
                runtime, x, metadata, None, 5, expand, mode, launcher=launcher
            )
            assert_return_packets(returned, case, 0, expand, mode, False)
            assert window[:8].view(torch.int64).item() == 2


@pytest.mark.parametrize("dimension,value", [(0, 512), (1, 2), (2, 2), (3, 4), (4, 6)])
def test_compiled_geometry_mismatch_precedes_barrier(native, dimension, value):
    geometry = [256, 3, 1, 3, 5]
    geometry[dimension] = value
    specialized = prepare_combine_main(*geometry)
    assert specialized.geometry == tuple(geometry)
    with empty_inputs(native) as (runtime, window, _, x, metadata):
        before = window.clone()
        with pytest.raises(RuntimeError, match="compiled geometry"):
            native.combine_main(
                runtime,
                x,
                metadata,
                None,
                5,
                False,
                False,
                launcher=specialized.launcher,
            )
        torch.testing.assert_close(window, before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 2])
def test_empty_combine_barriers_and_borrowed_return_view(
    native, launcher, mode, expand, weighted, dtype, tokens
):
    with empty_inputs(native, mode, dtype, tokens) as (
        runtime,
        window,
        layout,
        x,
        metadata,
    ):
        shape = (0,) if expand else (0, 3)
        weights = torch.empty(shape, device="cuda") if weighted else None
        saved_metadata = [tensor.cpu().clone() for tensor in metadata]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            returned = native.combine_main(
                runtime, x, metadata, weights, 5, expand, mode, launcher=launcher
            )
        stream.synchronize()
        partitions = 1 if mode else 3
        assert returned.shape == (partitions, 5, 544)
        assert returned.dtype == torch.uint8 and returned.device == x.device
        assert returned.is_contiguous()
        assert returned.data_ptr() == window.data_ptr() + layout[3]
        assert torch.all(returned == 165).item()
        assert window[:8].view(torch.int64).item() == 2
        with torch.cuda.stream(stream):
            replay = native.combine_main(
                runtime, x, metadata, weights, 5, expand, mode, launcher=launcher
            )
        stream.synchronize()
        assert replay.data_ptr() == returned.data_ptr()
        assert window[:8].view(torch.int64).item() == 4
        for tensor, expected in zip(metadata, saved_metadata, strict=True):
            torch.testing.assert_close(tensor.cpu(), expected)


def test_fused_combine_epochs_interleave_with_standalone_barriers(native, launcher):
    with empty_inputs(native) as (runtime, window, _, x, metadata):
        stream = torch.cuda.current_stream()
        runtime.barrier(stream.cuda_stream, 7)
        native.combine_main(
            runtime, x, metadata, None, 5, False, False, launcher=launcher
        )
        assert window[:8].view(torch.int64).item() == 9
        runtime.barrier(stream.cuda_stream)
        assert window[:8].view(torch.int64).item() == 10
        native.combine_main(
            runtime, x, metadata, None, 5, False, False, launcher=launcher
        )
        assert window[:8].view(torch.int64).item() == 12
        with pytest.raises(RuntimeError, match="strictly increase"):
            runtime.barrier(stream.cuda_stream, 12)
        runtime.barrier(stream.cuda_stream, 17)
        native.combine_main(
            runtime, x, metadata, None, 5, False, False, launcher=launcher
        )
        assert window[:8].view(torch.int64).item() == 19
        other_stream = torch.cuda.Stream()
        before = window.clone()
        with (
            torch.cuda.stream(other_stream),
            pytest.raises(RuntimeError, match="same stream"),
        ):
            native.combine_main(
                runtime, x, metadata, None, 5, False, False, launcher=launcher
            )
        torch.testing.assert_close(window, before, rtol=0, atol=0)
        runtime.barrier(stream.cuda_stream)
        assert window[:8].view(torch.int64).item() == 20


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_nonempty_combine_self_return_packets(
    native, launcher, mode, expand, weighted, dtype
):
    case = make_return_case(1, 3, 256, dtype)
    layout = native.window_layout(
        1, case.capacity, case.hidden, case.topk, case.num_experts, mode
    )
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for step in range(2):
                if step:
                    case.values = [value + 0.25 for value in case.values]
                    case.weights = [weight - 0.5 for weight in case.weights]
                x, weights, metadata = expert_inputs(case, 0, expand)
                runtime.local_window()[layout[3] :].fill_(165)
                received = native.combine_main(
                    runtime,
                    x.cuda(),
                    [tensor.cuda() for tensor in metadata],
                    weights.cuda() if weighted else None,
                    case.capacity,
                    expand,
                    mode,
                    launcher=launcher,
                )
                assert_return_packets(received, case, 0, expand, mode, weighted)
        stream.synchronize()
    finally:
        torch.cuda.synchronize()
        runtime.close_peer_mappings()
        runtime.close()


@pytest.mark.parametrize(
    "kind",
    [
        "metadata_count",
        "source_shape",
        "metadata_dtype",
        "metadata_device",
        "metadata_stride",
        "snapshot_rank",
        "source_capacity",
        "x_dtype",
        "x_fp32",
        "x_device",
        "x_rank",
        "x_stride",
        "compact_rows",
        "expanded_rows",
        "weight_dtype",
        "weight_device",
        "weight_shape",
        "weight_stride",
        "capacity",
        "window",
    ],
)
def test_invalid_combine_storage_precedes_barrier(native, launcher, kind):
    with empty_inputs(native) as (runtime, window, _, x, metadata):
        capacity, expand, weights = 5, False, None
        if kind == "metadata_count":
            metadata = metadata[:5]
        elif kind == "source_shape":
            metadata[1] = torch.empty(1, dtype=torch.int64, device="cuda")
        elif kind == "metadata_dtype":
            metadata[3] = metadata[3].float()
        elif kind == "metadata_device":
            metadata[3] = metadata[3].cpu()
        elif kind == "metadata_stride":
            metadata[8] = torch.full((2, 6), -1, dtype=torch.int64, device="cuda")[
                :, ::2
            ]
        elif kind == "snapshot_rank":
            metadata[8] = metadata[8].flatten()
        elif kind == "source_capacity":
            metadata[8] = torch.full((6, 3), -1, dtype=torch.int64, device="cuda")
        elif kind == "x_dtype":
            x = x.double()
        elif kind == "x_fp32":
            x = x.float()
        elif kind == "x_device":
            x = x.cpu()
        elif kind == "x_rank":
            x = x.flatten()
        elif kind == "x_stride":
            metadata[0] = torch.zeros(2, dtype=torch.int64, device="cuda")
            x = torch.zeros((2, 512), dtype=torch.bfloat16, device="cuda")[:, ::2]
        elif kind in ("compact_rows", "expanded_rows"):
            x = torch.empty((1, 256), dtype=torch.bfloat16, device="cuda")
            expand = kind == "expanded_rows"
        elif kind == "weight_dtype":
            weights = torch.empty((0, 3), dtype=torch.float64, device="cuda")
        elif kind == "weight_device":
            weights = torch.empty((0, 3))
        elif kind == "weight_shape":
            weights = torch.empty(0, device="cuda")
        elif kind == "weight_stride":
            expand = True
            metadata[0] = torch.zeros(2, dtype=torch.int64, device="cuda")
            metadata[1] = torch.zeros(2, dtype=torch.int64, device="cuda")
            metadata[2] = torch.zeros((2, 3), dtype=torch.int64, device="cuda")
            metadata[5] = torch.zeros((2, 3), dtype=torch.int64, device="cuda")
            x = torch.zeros((2, 256), dtype=torch.bfloat16, device="cuda")
            weights = torch.zeros(4, device="cuda")[::2]
        elif kind == "capacity":
            capacity = 0
        else:
            capacity = 500
        before = window.cpu().clone()
        with pytest.raises((RuntimeError, ValueError)):
            native.combine_main(
                runtime,
                x,
                metadata,
                weights,
                capacity,
                expand,
                False,
                launcher=launcher,
            )
        torch.testing.assert_close(window.cpu(), before, rtol=0, atol=0)
