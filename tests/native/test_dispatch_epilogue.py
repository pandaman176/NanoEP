"""Exercise actual copy outputs independently of the unfinished CudaBuffer API."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from torch.utils import deterministic

from nano_ep.backends._build import load_extension
from nano_ep.testing.cases import make_case
from nano_ep.testing.native_dispatch import (
    assert_dispatch_output,
    assert_dispatch_state,
)

pytestmark = [pytest.mark.cuda]

LAYOUTS = [(False, False), (True, False), (True, True)]
LAYOUT_IDS = ["compact", "expanded", "expanded-zero-padding"]


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


@pytest.fixture(autouse=True)
def poison_empty_allocations(native):
    """Make missed initialization observable even when the CUDA allocator reuses zeros."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        deterministic.fill_uninitialized_memory = fill
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def copy_case(kind, dtype, hidden, tokens=None):
    case = make_case(1, kind=kind, dtype=dtype)
    indices, weights = case.indices[0], case.weights[0]
    if tokens is not None:
        repeats = (tokens + len(indices) - 1) // len(indices)
        indices, weights = (
            indices.repeat(repeats, 1)[:tokens],
            weights.repeat(repeats, 1)[:tokens],
        )
    x = torch.randn(
        len(indices), hidden, generator=torch.Generator().manual_seed(123), dtype=dtype
    )
    return replace(
        case,
        xs=[x],
        indices=[indices],
        weights=[weights],
        hidden=hidden,
        capacity=max(case.capacity, len(indices)),
        expert_matrices=torch.zeros(case.num_experts, hidden, hidden, dtype=dtype),
    )


@contextmanager
def local_runtime(native, case, allow_multiple_reduction=False):
    layout = native.window_layout(
        1,
        case.capacity,
        case.hidden,
        case.num_topk,
        case.num_experts,
        allow_multiple_reduction,
    )
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    window[layout[1] : layout[1] + (1 + case.num_experts) * 8].view(torch.int64).fill_(
        -1
    )
    torch.cuda.synchronize()
    runtime.import_peer_handles([runtime.export_local_handle()])
    try:
        yield runtime, window, layout
    finally:
        torch.cuda.synchronize()
        del window
        runtime.close_peer_mappings()
        runtime.close()


def check_fresh_and_cached(
    native,
    case,
    alignment,
    expand,
    weighted,
    zero_padding,
    allow_multiple_reduction=False,
):
    stream = torch.cuda.Stream()
    with (
        local_runtime(native, case, allow_multiple_reduction) as (
            runtime,
            window,
            layout,
        ),
        torch.cuda.stream(stream),
    ):
        x, indices, weights = (
            case.xs[0].cuda(),
            case.indices[0].cuda(),
            case.weights[0].cuda(),
        )
        # Absent weight fields contain NaNs; the unweighted path must not consume them.
        window[layout[2] :].fill_(255)
        state = native.dispatch_main(
            runtime,
            x,
            indices,
            weights if weighted else None,
            case.num_experts,
            case.capacity,
            alignment,
            [],
            allow_multiple_reduction,
        )
        outputs = native.dispatch_epilogue(state, expand, weighted, zero_padding)
        assert_dispatch_state(
            state, case, 0, alignment, case.weights if weighted else None
        )
        assert_dispatch_output(
            state,
            outputs,
            case,
            0,
            alignment,
            case.weights if weighted else None,
            expand=expand,
            zero_padding=zero_padding,
        )
        saved_outputs = [tensor.cpu().clone() for tensor in outputs]
        saved_metadata = [tensor.cpu().clone() for tensor in state.metadata]
        pointers = [tensor.data_ptr() for tensor in state.metadata]
        torch.testing.assert_close(x.cpu(), case.xs[0], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(indices.cpu(), case.indices[0])
        torch.testing.assert_close(
            weights.cpu(), case.weights[0], rtol=0, atol=0, equal_nan=True
        )
        indices.fill_(99)

        for step, replay_weighted in enumerate((not weighted, weighted), start=1):
            replay_case = replace(
                case, xs=[case.xs[0] + 2 * step], weights=[-case.weights[0] + step]
            )
            replay_x, replay_weights = (
                replay_case.xs[0].cuda(),
                replay_case.weights[0].cuda(),
            )
            window[layout[2] :].fill_(255)
            replay = native.dispatch_main(
                runtime,
                replay_x,
                indices,
                replay_weights if replay_weighted else None,
                case.num_experts,
                case.capacity,
                alignment,
                state.metadata,
                allow_multiple_reduction,
            )
            assert replay.metadata_prepared
            replay_outputs = native.dispatch_epilogue(
                replay, expand, replay_weighted, zero_padding
            )
            assert_dispatch_output(
                replay,
                replay_outputs,
                replay_case,
                0,
                alignment,
                replay_case.weights if replay_weighted else None,
                expand=expand,
                zero_padding=zero_padding,
            )
            assert [tensor.data_ptr() for tensor in replay.metadata] == pointers
            for actual, expected in zip(replay.metadata, saved_metadata, strict=True):
                torch.testing.assert_close(actual.cpu(), expected)
            for actual, expected in zip(outputs, saved_outputs, strict=True):
                torch.testing.assert_close(
                    actual.cpu(), expected, rtol=0, atol=0, equal_nan=True
                )
            torch.testing.assert_close(
                replay_x.cpu(), replay_case.xs[0], rtol=0, atol=0, equal_nan=True
            )
            torch.testing.assert_close(
                replay_weights.cpu(),
                replay_case.weights[0],
                rtol=0,
                atol=0,
                equal_nan=True,
            )


@pytest.mark.parametrize("expand,zero_padding", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["mixed", "empty", "masked", "skew"])
@pytest.mark.parametrize("hidden,alignment", [(4, 1), (33, 4), (65, 16)])
def test_dispatch_copy_fresh_and_cached(
    native, expand, zero_padding, weighted, dtype, kind, hidden, alignment
):
    check_fresh_and_cached(
        native,
        copy_case(kind, dtype, hidden),
        alignment,
        expand,
        weighted,
        zero_padding,
    )


@pytest.mark.parametrize("expand,zero_padding", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dispatch_copy_multiple_blocks(native, expand, zero_padding, dtype):
    check_fresh_and_cached(
        native,
        copy_case("mixed", dtype, 129, tokens=19),
        16,
        expand,
        True,
        zero_padding,
    )


@pytest.mark.parametrize("expand,zero_padding", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
def test_dispatch_uses_allocated_reduction_mode(
    native, expand, zero_padding, dtype, allow_multiple_reduction
):
    check_fresh_and_cached(
        native,
        copy_case("mixed", dtype, 33, tokens=19),
        4,
        expand,
        True,
        zero_padding,
        allow_multiple_reduction,
    )


@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
def test_reserve_binding_uses_allocated_reduction_mode(
    native, allow_multiple_reduction
):
    case = copy_case("mixed", torch.float32, 33, tokens=19)
    with local_runtime(native, case, allow_multiple_reduction) as (
        runtime,
        window,
        layout,
    ):
        x, indices = case.xs[0].cuda(), case.indices[0].cuda()
        stream = torch.cuda.current_stream()
        runtime.barrier(stream.cuda_stream)
        slots = native.reserve_and_put(
            runtime,
            x,
            indices,
            None,
            case.num_experts,
            case.capacity,
            allow_multiple_reduction=allow_multiple_reduction,
        )
        runtime.barrier(stream.cuda_stream)
        torch.testing.assert_close(slots[:, 0].cpu().sort().values, torch.arange(19))
        counter = window[
            layout[1] + (1 + case.num_experts) * 8 : layout[1]
            + (2 + case.num_experts) * 8
        ].view(torch.int64)
        assert counter.item() == 19
        runtime.barrier(stream.cuda_stream)
        replay = native.reserve_and_put(
            runtime,
            x * 2,
            indices,
            None,
            case.num_experts,
            case.capacity,
            cached_dst_slots=slots,
            allow_multiple_reduction=allow_multiple_reduction,
        )
        runtime.barrier(stream.cuda_stream)
        assert replay.data_ptr() == slots.data_ptr()
        assert counter.item() == 19


@pytest.mark.parametrize("expand,zero_padding", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dispatch_copy_nonfinite_payload(native, expand, zero_padding, dtype):
    case = copy_case("mixed", dtype, 33)
    case.xs[0][0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], dtype=dtype
    )
    case.weights[0][0, 0] = float("nan")
    case.weights[0][1, 2] = float("inf")
    check_fresh_and_cached(native, case, 4, expand, True, zero_padding)
