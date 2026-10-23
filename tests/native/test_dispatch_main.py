from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nano_ep.backends._build import load_extension
from nano_ep.testing.cases import make_case
from nano_ep.testing.native_dispatch import (
    assert_dispatch_state,
    prepare_dispatch_metadata,
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
def runtime(native):
    layout = native.window_layout(1, 5, 4, 3, 3)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    window[layout[1] : layout[1] + 32].view(torch.int64).fill_(-1)
    torch.cuda.synchronize()
    runtime.import_peer_handles([runtime.export_local_handle()])
    yield runtime, layout
    torch.cuda.synchronize()
    del window
    runtime.close_peer_mappings()
    runtime.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["mixed", "empty", "masked", "skew"])
@pytest.mark.parametrize("alignment", [1, 4, 16])
def test_dispatch_state_and_old_cache(native, runtime, dtype, kind, alignment):
    runtime, layout = runtime
    case = make_case(1, kind=kind, dtype=dtype)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, indices, weights = (
            case.xs[0].cuda(),
            case.indices[0].cuda(),
            case.weights[0].cuda(),
        )
        state = native.dispatch_main(runtime, x, indices, weights, 3, 5, alignment, [])
        assert not state.metadata_prepared
        assert len(state.metadata) == 5
        assert all(isinstance(tensor, torch.Tensor) for tensor in state.metadata)
        torch.testing.assert_close(state.metadata[4].cpu(), case.indices[0])
        assert (
            state.rank == 0 and state.hidden == case.hidden and state.x_dtype == dtype
        )
        prepare_dispatch_metadata(native, state)
        pointers = [tensor.data_ptr() for tensor in state.metadata]
        prepare_dispatch_metadata(native, state)
        assert [tensor.data_ptr() for tensor in state.metadata] == pointers
        assert_dispatch_state(state, case, 0, alignment, case.weights)
        torch.testing.assert_close(x.cpu(), case.xs[0])
        torch.testing.assert_close(indices.cpu(), case.indices[0])
        torch.testing.assert_close(weights.cpu(), case.weights[0])
        if indices.numel():
            assert state.metadata[8].data_ptr() != indices.data_ptr()
        saved_metadata = [tensor.cpu().clone() for tensor in state.metadata]
        indices.fill_(99)
        torch.testing.assert_close(state.metadata[8].cpu(), case.indices[0])

        # A different fresh dispatch overwrites the window and resets recv_next.
        masked = replace(case, indices=[torch.full_like(case.indices[0], -1)])
        fresh = native.dispatch_main(
            runtime, x, masked.indices[0].cuda(), None, 3, 5, 1, []
        )
        prepare_dispatch_metadata(native, fresh)
        assert_dispatch_state(fresh, masked, 0, 1, None)
        assert (
            runtime.local_window()[layout[1] + 32 : layout[1] + 40]
            .view(torch.int64)
            .item()
            == 0
        )

        replay_case = replace(case, xs=[case.xs[0] * 2])
        replay = native.dispatch_main(
            runtime, x * 2, indices, None, 3, 5, alignment, state.metadata
        )
        assert replay.metadata_prepared
        prepare_dispatch_metadata(native, replay)
        assert_dispatch_state(replay, replay_case, 0, alignment, None)
        for before, after, saved in zip(
            state.metadata, replay.metadata, saved_metadata, strict=True
        ):
            assert before.data_ptr() == after.data_ptr()
            torch.testing.assert_close(after.cpu(), saved)
        assert (
            runtime.local_window()[layout[1] + 32 : layout[1] + 40]
            .view(torch.int64)
            .item()
            == 0
        )

        replay_case = replace(case, xs=[case.xs[0] * 3], weights=[case.weights[0] * 4])
        replay = native.dispatch_main(
            runtime, x * 3, indices, weights * 4, 3, 5, alignment, state.metadata
        )
        prepare_dispatch_metadata(native, replay)
        assert_dispatch_state(replay, replay_case, 0, alignment, replay_case.weights)
        # Metadata from the intervening fresh call must survive subsequent writes.
        torch.testing.assert_close(fresh.metadata[8].cpu(), masked.indices[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nonfinite_payload_is_preserved(native, runtime, dtype):
    runtime, _ = runtime
    case = make_case(1, kind="mixed", dtype=dtype)
    case.xs[0][0, 0] = float("nan")
    case.xs[0][1, 0] = float("inf")
    state = native.dispatch_main(
        runtime, case.xs[0].cuda(), case.indices[0].cuda(), None, 3, 5, 4, []
    )
    prepare_dispatch_metadata(native, state)
    assert_dispatch_state(state, case, 0, 4, None)


@pytest.mark.parametrize(
    "kind",
    [
        "x_cpu",
        "x_dtype",
        "x_shape",
        "x_stride",
        "capacity",
        "idx_cpu",
        "idx_dtype",
        "idx_shape",
        "idx_stride",
        "weights_cpu",
        "weights_dtype",
        "weights_shape",
        "weights_stride",
        "alignment",
        "experts",
        "window",
        "cache_length",
        "cache_shape",
        "cache_unprepared",
    ],
)
def test_invalid_input_precedes_communication(native, runtime, kind):
    runtime, layout = runtime
    x = torch.ones((2, 4), device="cuda")
    indices = torch.tensor([[0, 1, -1], [2, -1, 0]], device="cuda")
    weights = torch.ones((2, 3), device="cuda")
    alignment, experts, capacity, cached = 4, 3, 5, []
    if kind == "x_cpu":
        x = x.cpu()
    elif kind == "x_dtype":
        x = x.double()
    elif kind == "x_shape":
        x = x.flatten()
    elif kind == "x_stride":
        x = torch.ones((2, 8), device="cuda")[:, ::2]
    elif kind == "capacity":
        capacity = 1
    elif kind == "idx_cpu":
        indices = indices.cpu()
    elif kind == "idx_dtype":
        indices = indices.float()
    elif kind == "idx_shape":
        indices = indices[:1]
    elif kind == "idx_stride":
        indices = torch.zeros((2, 6), device="cuda", dtype=torch.int64)[:, ::2]
    elif kind == "weights_cpu":
        weights = weights.cpu()
    elif kind == "weights_dtype":
        weights = weights.double()
    elif kind == "weights_shape":
        weights = weights[:1]
    elif kind == "weights_stride":
        weights = torch.ones((2, 6), device="cuda")[:, ::2]
    elif kind == "alignment":
        alignment = 0
    elif kind == "experts":
        experts = 0
    elif kind == "window":
        capacity = 500
    elif kind == "cache_length":
        cached = [indices]
    else:
        state = native.dispatch_main(
            runtime, x, indices, weights, experts, capacity, alignment, []
        )
        if kind == "cache_shape":
            prepare_dispatch_metadata(native, state)
        cached = state.metadata
        if kind == "cache_shape":
            cached[6] = torch.zeros((2, 2), device="cuda", dtype=torch.int64)
    before = runtime.local_window().cpu().clone()
    with pytest.raises((RuntimeError, ValueError)):
        native.dispatch_main(
            runtime, x, indices, weights, experts, capacity, alignment, cached
        )
    torch.testing.assert_close(runtime.local_window().cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "stage,kind",
    [
        ("route_counts", "negative"),
        ("route_counts", "overflow"),
        ("reserve_and_put", "negative"),
        ("reserve_and_put", "overflow"),
        ("dispatch_main", "negative"),
        ("dispatch_main", "overflow"),
        ("dispatch_epilogue", "duplicate"),
    ],
)
def test_routing_device_assert_in_subprocess(native, stage, kind):
    worker = Path(__file__).with_name("routing_assert_worker.py")
    result = subprocess.run(
        [sys.executable, str(worker), stage, kind],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"NANO_EP_ROUTING_ASSERT_OK={stage}:{kind}" in result.stdout, (
        result.stdout + result.stderr
    )
