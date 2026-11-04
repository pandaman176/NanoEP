import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nano_ep.backends._build import (
    load_extension,
    prepare_dispatch_epilogue,
    prepare_dispatch_main,
)
from nano_ep.testing.cases import make_case
from nano_ep.testing.native_dispatch import (
    assert_dispatch_output,
    assert_dispatch_state,
    prepare_dispatch_metadata,
)

pytestmark = [pytest.mark.cuda]


@pytest.mark.parametrize("hidden", [1, 33, 128, 255, 257])
@pytest.mark.parametrize("tokens", [0, 2])
def test_hidden_constraint_precedes_communication(native, runtime, hidden, tokens):
    peer, _ = runtime
    x = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    indices = torch.full((tokens, 3), -1, dtype=torch.int64, device="cuda")
    before = peer.local_window().clone()
    with pytest.raises(RuntimeError, match="multiple of 256"):
        native.dispatch_main(
            peer,
            x,
            indices,
            None,
            3,
            5,
            1,
            [],
            0,
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(256, 3, 1, 3, 5, 1).launcher,
        )
    torch.testing.assert_close(peer.local_window(), before, rtol=0, atol=0)


@pytest.fixture
def native(request):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda or --run-distributed")
    if not torch.cuda.is_available():
        pytest.fail("CUDA tests requested but NVIDIA CUDA is unavailable")
    native = load_extension()
    for cached_mode in (False, True):
        prepare_dispatch_epilogue(256, 3, 3, cached_mode=cached_mode)
    return native


@pytest.fixture
def runtime(native):
    layout = native.window_layout(1, 5, 256, 3, 3)
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


@pytest.mark.parametrize("experts", [3, 260])
def test_notify_and_payload_warps_across_multiple_token_rounds(native, experts):
    tokens = (
        8
        * torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        + 3
    )
    hidden, alignment = 256, 4
    layout = native.window_layout(1, tokens, hidden, 3, experts)
    epilogue = prepare_dispatch_epilogue(
        hidden, 3, experts, cached_mode=False
    ).dispatch_epilogue
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    count_bytes = (1 + experts) * 8
    window[layout[1] : layout[1] + count_bytes].view(torch.int64).fill_(-1)
    torch.cuda.synchronize()
    runtime.import_peer_handles([runtime.export_local_handle()])
    base = make_case(1, dtype=torch.bfloat16, hidden=hidden)
    x = (torch.arange(tokens * hidden).reshape(tokens, hidden) % 29).to(torch.bfloat16)
    weights = torch.arange(tokens * 3).reshape(tokens, 3).float() / 7
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for kind in ("mixed", "masked", "empty", "skew"):
                pattern = {
                    "mixed": [
                        [0, 1, experts - 1],
                        [-1, -1, -1],
                        [experts - 1, -1, 0],
                        [-1, 1, -1],
                    ],
                    "masked": [[-1, -1, -1]],
                    "empty": [[-1, -1, -1]],
                    "skew": [[experts - 1, -1, -1]],
                }[kind]
                n = 0 if kind == "empty" else tokens
                indices = (
                    torch.tensor(pattern)
                    .repeat((n + len(pattern) - 1) // len(pattern), 1)[:n]
                    .contiguous()
                )
                case = replace(
                    base,
                    xs=[x[:n]],
                    indices=[indices],
                    weights=[weights[:n]],
                    capacity=tokens,
                    num_experts=experts,
                    expert_matrices=torch.empty(0),
                )
                state = native.dispatch_main(
                    runtime,
                    case.xs[0].cuda(),
                    indices.cuda(),
                    case.weights[0].cuda(),
                    experts,
                    tokens,
                    alignment,
                    [],
                    0,
                    allow_multiple_reduction=False,
                    launcher=prepare_dispatch_main(
                        case.xs[0].shape[1],
                        indices.shape[1],
                        1,
                        experts,
                        tokens,
                        alignment,
                    ).launcher,
                )
                outputs = epilogue(state, False, True, False)
                assert_dispatch_state(
                    state, case, 0, alignment, case.weights, expand=False
                )
                assert_dispatch_output(
                    state,
                    outputs,
                    case,
                    0,
                    alignment,
                    case.weights,
                    expand=False,
                    zero_padding=False,
                )
                assert window[layout[1] : layout[1] + count_bytes].view(
                    torch.int64
                ).tolist() == [-1] * (1 + experts)
                # One rank's barrier region aligns to 32 bytes; the reduction words follow it.
                assert torch.count_nonzero(window[32 : 32 + count_bytes]).item() == 0
    finally:
        stream.synchronize()
        del window
        runtime.close_peer_mappings()
        runtime.close()


def test_static_notify_shared_memory_limit_rejected_at_build(native):
    # Static notify storage is checked by the CUDA compiler before IPC setup.
    with pytest.raises(RuntimeError, match="(?i)(shared|smem|local_counts)"):
        prepare_dispatch_main(256, 3, 1, 1_000_000, 1, 1)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kind", ["mixed", "empty", "masked", "skew"])
@pytest.mark.parametrize("alignment", [1, 4, 16])
def test_dispatch_state_and_old_cache(native, runtime, dtype, kind, alignment):
    runtime, layout = runtime
    case = make_case(1, kind=kind, dtype=dtype, hidden=256)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, indices, weights = (
            case.xs[0].cuda(),
            case.indices[0].cuda(),
            case.weights[0].cuda(),
        )
        state = native.dispatch_main(
            runtime,
            x,
            indices,
            weights,
            3,
            5,
            alignment,
            [],
            0,
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(
                x.shape[1], indices.shape[1], 1, 3, 5, alignment
            ).launcher,
        )
        assert not state.metadata_prepared
        assert len(state.metadata) == 5
        assert all(isinstance(tensor, torch.Tensor) for tensor in state.metadata)
        torch.testing.assert_close(state.metadata[4].cpu(), case.indices[0])
        assert (
            state.rank == 0 and state.hidden == case.hidden and state.x_dtype == dtype
        )
        prepare_dispatch_metadata(state)
        pointers = [tensor.data_ptr() for tensor in state.metadata]
        prepare_dispatch_metadata(state)
        assert [tensor.data_ptr() for tensor in state.metadata] == pointers
        assert_dispatch_state(state, case, 0, alignment, case.weights, expand=False)
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
            runtime,
            x,
            masked.indices[0].cuda(),
            None,
            3,
            5,
            1,
            [],
            0,
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(
                x.shape[1], masked.indices[0].shape[1], 1, 3, 5, 1
            ).launcher,
        )
        prepare_dispatch_metadata(fresh)
        assert_dispatch_state(fresh, masked, 0, 1, None, expand=False)
        assert (
            runtime.local_window()[layout[1] + 32 : layout[1] + 40]
            .view(torch.int64)
            .item()
            == 0
        )

        replay_case = replace(case, xs=[case.xs[0] * 2])
        replay = native.dispatch_main(
            runtime,
            x * 2,
            indices,
            None,
            3,
            5,
            alignment,
            state.metadata,
            int(state.counts[1]),
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(
                (x * 2).shape[1], indices.shape[1], 1, 3, 5, alignment
            ).launcher,
        )
        assert replay.metadata_prepared
        prepare_dispatch_metadata(replay)
        assert_dispatch_state(replay, replay_case, 0, alignment, None, expand=False)
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
            runtime,
            x * 3,
            indices,
            weights * 4,
            3,
            5,
            alignment,
            state.metadata,
            int(state.counts[1]),
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(
                (x * 3).shape[1], indices.shape[1], 1, 3, 5, alignment
            ).launcher,
        )
        prepare_dispatch_metadata(replay)
        assert_dispatch_state(
            replay, replay_case, 0, alignment, replay_case.weights, expand=False
        )
        # Metadata from the intervening fresh call must survive subsequent writes.
        torch.testing.assert_close(fresh.metadata[8].cpu(), masked.indices[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_nonfinite_payload_is_preserved(native, runtime, dtype):
    runtime, _ = runtime
    case = make_case(1, kind="mixed", dtype=dtype, hidden=256)
    case.xs[0][0, 0] = float("nan")
    case.xs[0][1, 0] = float("inf")
    state = native.dispatch_main(
        runtime,
        case.xs[0].cuda(),
        case.indices[0].cuda(),
        None,
        3,
        5,
        4,
        [],
        0,
        allow_multiple_reduction=False,
        launcher=prepare_dispatch_main(
            case.xs[0].shape[1], case.indices[0].shape[1], 1, 3, 5, 4
        ).launcher,
    )
    prepare_dispatch_metadata(state)
    assert_dispatch_state(state, case, 0, 4, None, expand=False)


@pytest.mark.parametrize(
    "kind",
    [
        "x_cpu",
        "x_dtype",
        "x_fp32",
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
        "cache_slots",
        "cache_capacity",
        "cache_unprepared",
    ],
)
def test_invalid_input_precedes_communication(native, runtime, kind):
    runtime, layout = runtime
    x = torch.ones((2, 256), dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([[0, 1, -1], [2, -1, 0]], device="cuda")
    weights = torch.ones((2, 3), device="cuda")
    alignment, experts, capacity, cached = 4, 3, 5, []
    cached_expanded_capacity = 0
    if kind == "x_cpu":
        x = x.cpu()
    elif kind == "x_dtype":
        x = x.double()
    elif kind == "x_fp32":
        x = x.float()
    elif kind == "x_shape":
        x = x.flatten()
    elif kind == "x_stride":
        x = torch.ones((2, 512), dtype=torch.bfloat16, device="cuda")[:, ::2]
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
            runtime,
            x,
            indices,
            weights,
            experts,
            capacity,
            alignment,
            [],
            0,
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(256, 3, 1, 3, 5, alignment).launcher,
        )
        if kind in ("cache_shape", "cache_slots", "cache_capacity"):
            prepare_dispatch_metadata(state)
        cached = state.metadata
        cached_expanded_capacity = int(state.counts[1])
        if kind == "cache_shape":
            cached[6] = torch.zeros((2, 2), device="cuda", dtype=torch.int64)
        elif kind == "cache_slots":
            cached[5] = torch.zeros(1, device="cuda", dtype=torch.int64)
        elif kind == "cache_capacity":
            cached_expanded_capacity = -1
    before = runtime.local_window().cpu().clone()
    with pytest.raises((RuntimeError, ValueError)):
        native.dispatch_main(
            runtime,
            x,
            indices,
            weights,
            experts,
            capacity,
            alignment,
            cached,
            cached_expanded_capacity,
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(256, 3, 1, 3, 5, alignment).launcher,
        )
    torch.testing.assert_close(runtime.local_window().cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("kind", ["mixed", "empty", "masked"])
def test_cached_capacity_without_gpu_scalar_readback(native, runtime, expand, kind):
    peer, _ = runtime
    case = make_case(1, kind=kind, dtype=torch.bfloat16, hidden=256)
    x, indices, weights = (
        case.xs[0].cuda(),
        case.indices[0].cuda(),
        case.weights[0].cuda(),
    )
    state = native.dispatch_main(
        peer,
        x,
        indices,
        weights,
        3,
        5,
        16,
        [],
        0,
        allow_multiple_reduction=False,
        launcher=prepare_dispatch_main(
            x.shape[1], indices.shape[1], 1, 3, 5, 16
        ).launcher,
    )
    fresh = prepare_dispatch_epilogue(256, 3, 3, cached_mode=False).dispatch_epilogue
    cached = prepare_dispatch_epilogue(256, 3, 3, cached_mode=True).dispatch_epilogue
    fresh(state, expand, True, expand)
    capacity = int(state.counts[1])
    replay_case = replace(case, xs=[case.xs[0] + 2])
    replay_x = replay_case.xs[0].cuda()

    # Limit profiling to the native cached call; oracle reads occur afterwards.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        replay = native.dispatch_main(
            peer,
            replay_x,
            indices,
            weights,
            3,
            5,
            16,
            state.metadata,
            capacity,
            allow_multiple_reduction=False,
            launcher=prepare_dispatch_main(
                replay_x.shape[1], indices.shape[1], 1, 3, 5, 16
            ).launcher,
        )
    operations = {event.key for event in profile.key_averages()}
    assert not operations & {"aten::item", "aten::_local_scalar_dense"}
    assert replay.counts.tolist() == state.counts.tolist()
    outputs = cached(replay, expand, True, expand)
    assert_dispatch_state(replay, replay_case, 0, 16, case.weights, expand=expand)
    assert_dispatch_output(
        replay,
        outputs,
        replay_case,
        0,
        16,
        case.weights,
        expand=expand,
        zero_padding=expand,
    )


@pytest.mark.parametrize(
    "stage,kind",
    [
        ("dispatch_payload", "negative"),
        ("dispatch_payload", "overflow"),
        ("dispatch_main", "negative"),
        ("dispatch_main", "overflow"),
        ("dispatch_epilogue", "duplicate"),
    ],
)
def test_routing_device_assert_in_subprocess(native, stage, kind, request):
    worker = Path(__file__).with_name("routing_assert_worker.py")
    environment = os.environ.copy()
    if stage == "dispatch_payload":
        environment["NANO_EP_PEER_PROBE_PATH"] = request.getfixturevalue(
            "payload_probe"
        ).__file__
    result = subprocess.run(
        [sys.executable, str(worker), stage, kind],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"NANO_EP_ROUTING_ASSERT_OK={stage}:{kind}" in result.stdout, (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize(
    "position,value", [(0, 512), (1, 2), (2, 2), (3, 4), (4, 7), (5, 8)]
)
def test_specialization_mismatch_precedes_window_mutation(
    native, runtime, position, value
):
    peer, _ = runtime
    geometry = [256, 3, 1, 3, 5, 4]
    geometry[position] = value
    module = prepare_dispatch_main(*geometry)
    x = torch.ones((2, 256), dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([[0, 1, -1], [2, -1, 0]], device="cuda")
    before = peer.local_window().clone()
    with pytest.raises(RuntimeError, match="compiled geometry"):
        native.dispatch_main(
            peer, x, indices, None, 3, 5, 4, [], 0, False, module.launcher
        )
    torch.testing.assert_close(peer.local_window(), before, rtol=0, atol=0)


def test_dispatch_rejects_combine_launcher_before_window_mutation(native, runtime):
    from nano_ep.backends._build import prepare_combine_main

    peer, _ = runtime
    x = torch.ones((2, 256), dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([[0, 1, -1], [2, -1, 0]], device="cuda")
    wrong = prepare_combine_main(256, 3, 1, 3, 5)
    before = peer.local_window().clone()
    with pytest.raises(ValueError, match="dispatch main specialization launcher"):
        native.dispatch_main(
            peer, x, indices, None, 3, 5, 4, [], 0, False, wrong.launcher
        )
    torch.testing.assert_close(peer.local_window(), before, rtol=0, atol=0)
