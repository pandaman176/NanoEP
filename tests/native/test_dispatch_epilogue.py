"""Exercise actual copy outputs independently of the unfinished CudaBuffer API."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from torch.utils import deterministic

from nano_ep.backends._build import load_extension, prepare_dispatch_epilogue
from nano_ep.testing.cases import make_case
from nano_ep.testing.native_dispatch import (
    assert_dispatch_output,
    assert_dispatch_state,
)
from nano_ep.testing.wire import align32, dispatch_offsets

pytestmark = [pytest.mark.cuda]

LAYOUTS = [(False, False), (False, True), (True, False), (True, True)]
LAYOUT_IDS = ["compact", "compact-padding-ignored", "expanded", "expanded-zero-padding"]


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
    epilogues = tuple(
        prepare_dispatch_epilogue(
            case.hidden, case.num_topk, case.num_experts, cached_mode=mode
        ).dispatch_epilogue
        for mode in (False, True)
    )
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
        outputs = epilogues[False](state, expand, weighted, zero_padding)
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
            replay_outputs = epilogues[True](
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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kind", ["mixed", "empty", "masked", "skew"])
@pytest.mark.parametrize("hidden,alignment", [(256, 1), (512, 4), (768, 16)])
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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dispatch_copy_multiple_blocks(native, expand, zero_padding, dtype):
    check_fresh_and_cached(
        native,
        copy_case("mixed", dtype, 512, tokens=19),
        16,
        expand,
        True,
        zero_padding,
    )


@pytest.mark.parametrize("expand,zero_padding", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
def test_dispatch_uses_allocated_reduction_mode(
    native, expand, zero_padding, dtype, allow_multiple_reduction
):
    check_fresh_and_cached(
        native,
        copy_case("mixed", dtype, 256, tokens=19),
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
    case = copy_case("mixed", torch.bfloat16, 256, tokens=19)
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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dispatch_copy_nonfinite_payload(native, expand, zero_padding, dtype):
    case = copy_case("mixed", dtype, 256)
    case.xs[0][0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], dtype=dtype
    )
    case.weights[0][0, 0] = float("nan")
    case.weights[0][1, 2] = float("inf")
    check_fresh_and_cached(native, case, 4, expand, True, zero_padding)


@pytest.mark.parametrize(
    "topk,hidden", [(1, 256), (3, 512), (3, 768), (8, 1280), (8, 4096), (32, 256)]
)
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_bf16_wire_bits_and_cached_payload(native, topk, hidden, expand, weighted):
    # Include signed zero, subnormals, infinities and distinct NaN encodings.
    bits = torch.tensor(
        [0, -32768, 1, -32767, 16257, 32639, 32640, -128, 32705, 32641],
        dtype=torch.int16,
    )
    tokens = 11
    x = (
        bits.repeat((tokens * hidden + len(bits) - 1) // len(bits))[: tokens * hidden]
        .view(torch.bfloat16)
        .reshape(tokens, hidden)
    )
    indices = torch.arange(topk).repeat(tokens, 1)
    indices[1].fill_(-1)
    weights = torch.arange(tokens * topk, dtype=torch.float32).reshape(tokens, topk) / 7
    case = replace(
        make_case(1, dtype=torch.bfloat16),
        xs=[x],
        indices=[indices],
        weights=[weights],
        num_experts=max(3, topk),
        num_topk=topk,
        hidden=hidden,
        capacity=tokens,
        expert_matrices=torch.empty(0),
    )
    epilogues = tuple(
        prepare_dispatch_epilogue(
            hidden, topk, case.num_experts, cached_mode=mode
        ).dispatch_epilogue
        for mode in (False, True)
    )
    with local_runtime(native, case) as (runtime, window, layout):
        cached = []
        for replay in (False, True):
            if replay:
                case = replace(case, xs=[x.flip(0).contiguous()], weights=[-weights])
            window[layout[2] :].fill_(165)
            state = native.dispatch_main(
                runtime,
                case.xs[0].cuda(),
                case.indices[0].cuda(),
                case.weights[0].cuda() if weighted else None,
                case.num_experts,
                tokens,
                4,
                cached,
            )
            outputs = epilogues[replay](state, expand, weighted, expand)
            assert_dispatch_state(state, case, 0, 4, case.weights if weighted else None)
            assert_dispatch_output(
                state,
                outputs,
                case,
                0,
                4,
                case.weights if weighted else None,
                expand=expand,
                zero_padding=expand,
            )
            records = state.stage_buffer.cpu()
            start = dispatch_offsets(topk)[2]
            assert records.shape[1] == align32(start + 2 * hidden)
            assert state.stage_buffer.data_ptr() % 32 == 0
            assert all(
                (state.stage_buffer.data_ptr() + offset) % 32 == 0
                for offset in dispatch_offsets(topk)
            )
            output_bits = outputs[0].view(torch.int16).cpu()
            source_bits = case.xs[0].view(torch.int16)
            expanded_slots = state.metadata[5].cpu()
            for row, token in enumerate(state.metadata[1].tolist()):
                destinations = expanded_slots[row].tolist() if expand else [row]
                for slot in destinations:
                    if slot >= 0:
                        torch.testing.assert_close(
                            output_bits[slot], source_bits[token], rtol=0, atol=0
                        )
                torch.testing.assert_close(
                    records[row, start : start + 2 * hidden],
                    case.xs[0][token].view(torch.uint8),
                )
                assert (records[row, start + 2 * hidden :] == 165).all()
            assert (window[layout[2] + records.numel() :].cpu() == 165).all()
            cached = state.metadata


@pytest.mark.parametrize("geometry", [(512, 3, 3), (256, 2, 3), (256, 3, 4)])
@pytest.mark.parametrize("prepared", [False, True])
def test_dispatch_geometry_mismatch_preserves_state(native, geometry, prepared):
    correct = prepare_dispatch_epilogue(256, 3, 3, cached_mode=False).dispatch_epilogue
    wrong = prepare_dispatch_epilogue(*geometry, cached_mode=prepared).dispatch_epilogue
    case = copy_case("mixed", torch.bfloat16, 256)
    with local_runtime(native, case) as (runtime, window, layout):
        state = native.dispatch_main(
            runtime,
            case.xs[0].cuda(),
            case.indices[0].cuda(),
            None,
            case.num_experts,
            case.capacity,
            4,
            [],
        )
        if prepared:
            correct(state, False, False, False)
        tensors = [tensor.clone() for tensor in state.metadata]
        pointers = [tensor.data_ptr() for tensor in state.metadata]
        with pytest.raises(RuntimeError, match="compiled geometry"):
            wrong(state, False, False, False)
        assert state.metadata_prepared is prepared
        assert [tensor.data_ptr() for tensor in state.metadata] == pointers
        for actual, expected in zip(state.metadata, tensors, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("zero_padding", [False, True])
@pytest.mark.parametrize(
    "experts,tokens,alignment",
    [
        (3, 1, 128),
        (40, 1, 16),
        (65, 1, 128),
        (65, 9, 4),
        (40, 9, 1),
    ],
)
def test_padding_gaps_across_expert_waves(
    native, experts, tokens, alignment, weighted, zero_padding
):
    hidden = 256
    selected = [2, 33, experts - 1] if experts > 32 else [0, 1, 2]
    indices = torch.tensor([selected] * tokens, dtype=torch.int64)
    if tokens > 1:
        indices[::2, 1] = -1
        indices[::3, 2] = -1
    x = (torch.arange(tokens * hidden).reshape(tokens, hidden) % 31 + 1).to(
        torch.bfloat16
    )
    x[0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], dtype=torch.bfloat16
    )
    weights = torch.arange(tokens * 3, dtype=torch.float32).reshape(tokens, 3) + 0.5
    case = replace(
        make_case(1, dtype=torch.bfloat16),
        xs=[x],
        indices=[indices],
        weights=[weights],
        hidden=hidden,
        num_experts=experts,
        num_topk=3,
        capacity=tokens,
        expert_matrices=torch.empty(0),
    )
    # One receive row leaves seven idle copy warps; padding may require many rounds.
    check_fresh_and_cached(native, case, alignment, True, weighted, zero_padding)


@pytest.mark.parametrize("prepared", [False, True])
def test_wrong_jit_cache_mode_preserves_state(native, prepared):
    epilogues = tuple(
        prepare_dispatch_epilogue(256, 3, 3, cached_mode=mode).dispatch_epilogue
        for mode in (False, True)
    )
    case = copy_case("mixed", torch.bfloat16, 256)
    with local_runtime(native, case) as (runtime, window, layout):
        state = native.dispatch_main(
            runtime,
            case.xs[0].cuda(),
            case.indices[0].cuda(),
            None,
            case.num_experts,
            case.capacity,
            4,
            [],
        )
        if prepared:
            epilogues[False](state, False, False, False)
        saved = [tensor.clone() for tensor in state.metadata]
        pointers = [tensor.data_ptr() for tensor in state.metadata]
        with pytest.raises(RuntimeError, match="cached_mode does not match"):
            epilogues[not prepared](state, True, False, True)
        assert state.metadata_prepared is prepared
        assert [tensor.data_ptr() for tensor in state.metadata] == pointers
        for actual, expected in zip(state.metadata, saved, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("topk", [3, 32])
@pytest.mark.parametrize("tokens", [1, 257])
def test_prepared_state_switches_layout_without_reserving_again(
    native, expand, topk, tokens
):
    hidden, alignment = 256, 128
    indices = torch.arange(topk).repeat(tokens, 1)
    indices[::2, 1::2] = -1
    case = replace(
        copy_case("mixed", torch.bfloat16, hidden, tokens=tokens),
        indices=[indices],
        weights=[torch.arange(tokens * topk).reshape(tokens, topk).float() + 0.5],
        num_topk=topk,
        num_experts=topk,
    )
    epilogues = tuple(
        prepare_dispatch_epilogue(
            hidden, topk, topk, cached_mode=mode
        ).dispatch_epilogue
        for mode in (False, True)
    )
    stream = torch.cuda.Stream()
    with (
        local_runtime(native, case) as (runtime, window, layout),
        torch.cuda.stream(stream),
    ):
        state = native.dispatch_main(
            runtime,
            case.xs[0].cuda(),
            case.indices[0].cuda(),
            case.weights[0].cuda(),
            topk,
            case.capacity,
            alignment,
            [],
        )
        assert not state.metadata_prepared
        initial_offsets = state.metadata[1].clone()
        outputs = epilogues[False](state, expand, True, True)
        assert_dispatch_state(state, case, 0, alignment, case.weights)
        assert_dispatch_output(
            state,
            outputs,
            case,
            0,
            alignment,
            case.weights,
            expand=expand,
            zero_padding=True,
        )
        saved = [tensor.clone() for tensor in state.metadata]
        pointers = [tensor.data_ptr() for tensor in state.metadata]
        saved_outputs = [tensor.clone() for tensor in outputs]

        # Both layouts must retain the full mapping, including slots not used by compact copy.
        replay_outputs = epilogues[True](state, not expand, False, True)
        assert_dispatch_output(
            state,
            replay_outputs,
            case,
            0,
            alignment,
            None,
            expand=not expand,
            zero_padding=True,
        )
        assert [tensor.data_ptr() for tensor in state.metadata] == pointers
        for actual, expected in zip(state.metadata, saved, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state.metadata[4], initial_offsets)
        for actual, expected in zip(outputs, saved_outputs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
