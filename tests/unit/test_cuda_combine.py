"""CPU checks for public combine validation, wiring and allocator bookkeeping."""

from types import SimpleNamespace

import pytest
import torch

import nano_ep.backends.cuda as cuda
from test_cuda_dispatch_orchestration import dispatch_runtime as dispatch_runtime



@pytest.fixture
def combine_runtime(dispatch_runtime):
    r = dispatch_runtime

    def integers(name, values):
        return r.tensor(name, torch.tensor(values, dtype=torch.int64))

    layout = cuda.NativeLayout(
        src_ranks=integers("src_ranks", [0, 1]),
        src_tokens=integers("src_tokens", [0, 1]),
        local_topk_idx=integers("local_topk", [[0, 1, -1], [-1, -1, 2]]),
        expert_counts=integers("expert_counts", [1, 1, 1]),
        expert_offsets=integers("expert_offsets", [0, 4, 8, 12]),
        expanded_slots=integers("expanded_slots", [[0, 4, -1], [-1, -1, 8]]),
        recv_count=2,
        expanded_capacity=12,
    )
    handle = cuda.NativeHandle(
        config=r.buffer._config,
        rank=0,
        world_size=2,
        do_expand=False,
        alignment=4,
        topk_idx=integers("snapshot", [[0, 1, 3], [3, -1, 5], [-1, -1, -1]]),
        layout=layout,
        dst_slots=integers("dst_slots", [[0, 0], [-1, 1], [-1, -1]]),
        send_counts=integers("send_counts", [1, 2]),
        owner=r.buffer,
        x_dtype=torch.bfloat16,
    )
    calls = SimpleNamespace(main=None, epilogue=None, outputs=None, window=None)

    class BorrowedWindow(r.tensor):
        def record_stream(self, stream):
            raise AssertionError(
                "runtime window must not be treated as allocator-owned output"
            )

    def main(*args):
        assert cuda.torch.cuda.current_stream() is r.comm
        r.events.append(("combine_main",))
        if r.settings["failure"] == "combine_main":
            raise RuntimeError("injected combine_main failure")
        calls.main = args
        calls.window = BorrowedWindow(
            "window", torch.empty((3, 5, 32), dtype=torch.uint8)
        )
        return calls.window

    def epilogue(*args):
        assert cuda.torch.cuda.current_stream() is r.comm
        r.events.append(("combine_epilogue",))
        if r.settings["failure"] == "combine_epilogue":
            raise RuntimeError("injected combine_epilogue failure")
        calls.epilogue = args
        _, snapshot, hidden, dtype, _, _, _, _, weighted, _, _ = args
        source_tokens = snapshot.shape[0]
        calls.outputs = (
            r.tensor("combined_x", torch.zeros((source_tokens, hidden), dtype=dtype)),
            r.tensor(
                "combined_weights",
                torch.zeros((source_tokens, 3)) if weighted else torch.empty(0),
            ),
        )
        return calls.outputs

    r.buffer._backend.combine_main = main
    r.buffer._backend.combine_epilogue = epilogue
    return SimpleNamespace(**vars(r), handle=handle, combine_calls=calls)


def handle_metadata(handle):
    layout = handle.layout
    return [
        layout.src_ranks,
        layout.src_tokens,
        layout.local_topk_idx,
        layout.expert_counts,
        layout.expert_offsets,
        layout.expanded_slots,
        handle.dst_slots,
        handle.send_counts,
        handle.topk_idx,
    ]


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit_event", [False, True])
def test_combine_wires_native_reverse_path(
    combine_runtime, expand, dtype, mode, weighted, biased, asynchronous, explicit_event
):
    r = combine_runtime
    r.handle.do_expand = expand
    r.buffer._allow_multiple_reduction = mode
    rows = 12 if expand else 2
    x = r.tensor("expert_output", torch.zeros((rows, 4), dtype=dtype))
    weights = (
        r.tensor("weights", torch.ones((rows,) if expand else (rows, 3)))
        if weighted
        else None
    )
    bias = r.tensor("bias", torch.zeros((3, 4), dtype=dtype)) if biased else None
    ready = r.event() if explicit_event else None
    got = r.buffer.combine(
        x,
        r.handle,
        weights=weights,
        bias=bias,
        previous_event=ready,
        async_op=asynchronous,
    )
    calls = r.combine_calls
    metadata = handle_metadata(r.handle)
    assert calls.main == (r.buffer._runtime, x, metadata, weights, 5, expand, mode)
    assert calls.epilogue == (
        calls.window,
        r.handle.topk_idx,
        4,
        dtype,
        6,
        2,
        5,
        bias,
        weighted,
        expand,
        mode,
    )
    assert got.x is calls.outputs[0]
    assert got.x.shape == (3, 4) and got.x.dtype == dtype
    assert got.topk_weights is (calls.outputs[1] if weighted else None)
    dependency = (
        ("wait_event", "comm", ready)
        if explicit_event
        else ("wait_stream", "comm", "compute")
    )
    assert r.events[0] == dependency
    for tensor in [
        x,
        *metadata,
        *([weights] if weighted else []),
        *([bias] if biased else []),
    ]:
        assert r.events.index(("record_tensor", tensor.name, "comm")) < r.events.index(
            ("combine_main",)
        )
    assert (
        r.events.index(("combine_main",))
        < r.events.index(("combine_epilogue",))
        < r.events.index(("record_event", "comm", got.event.event))
    )
    assert (("wait_event", "compute", got.event.event) in r.events) is not asynchronous
    for tensor in calls.outputs:
        assert (
            ("record_tensor", tensor.name, "compute") in r.events
        ) is not asynchronous
    assert any(item is calls.window for item in got.event.keepalive)
    assert any(item is r.handle for item in got.event.keepalive)
    assert any(item is x for item in got.event.keepalive)
    with r.use_stream(r.consumer):
        got.event.current_stream_wait()
    assert ("wait_event", "consumer", got.event.event) in r.events
    for tensor in calls.outputs:
        assert ("record_tensor", tensor.name, "consumer") in r.events


@pytest.mark.parametrize(
    "kind",
    [
        "closed",
        "owner",
        "config",
        "x_device",
        "x_other_gpu",
        "x_dtype",
        "x_rank",
        "x_stride",
        "x_hidden",
        "compact_rows",
        "expanded_rows",
        "weight_device",
        "weight_dtype",
        "weight_shape",
        "weight_stride",
        "expanded_weight_shape",
        "bias_device",
        "bias_dtype",
        "bias_rank",
        "bias_receive_rows",
        "bias_stride",
    ],
)
def test_combine_invalid_inputs_precede_native_work(combine_runtime, kind):
    r = combine_runtime
    x = r.tensor("x", torch.zeros((2, 4)))
    weights, bias = None, None
    error = ValueError
    if kind == "closed":
        r.buffer._closed = True
        error = RuntimeError
    elif kind == "owner":
        r.handle.owner = object()
    elif kind == "config":
        r.handle.config = cuda.EPConfig(12, 3, 4, 5)
    elif kind.startswith("x_") or kind in ("compact_rows", "expanded_rows"):
        if kind == "x_device":
            x.device = torch.device("cpu")
        elif kind == "x_other_gpu":
            x.device = torch.device("cuda", 1)
        elif kind == "x_dtype":
            x.payload = x.payload.double()
        elif kind == "x_rank":
            x.payload = x.payload.flatten()
        elif kind == "x_stride":
            x.payload = torch.zeros((2, 8))[:, ::2]
        elif kind == "x_hidden":
            x.payload = torch.zeros((2, 5))
        elif kind == "compact_rows":
            x.payload = torch.zeros((3, 4))
        else:
            r.handle.do_expand = True
    elif "weight" in kind:
        weights = r.tensor("weights", torch.zeros((2, 3)))
        if kind == "weight_device":
            weights.device = torch.device("cpu")
        elif kind == "weight_dtype":
            weights.payload = weights.payload.to(torch.bfloat16)
        elif kind == "weight_shape":
            weights.payload = torch.zeros((2,))
        elif kind == "weight_stride":
            weights.payload = torch.zeros((2, 6))[:, ::2]
        else:
            r.handle.do_expand = True
            x.payload = torch.zeros((12, 4))
            weights.payload = torch.zeros((12, 3))
    else:
        bias = r.tensor("bias", torch.zeros((3, 4)))
        if kind == "bias_device":
            bias.device = torch.device("cpu")
        elif kind == "bias_dtype":
            bias.payload = bias.payload.to(torch.bfloat16)
        elif kind == "bias_rank":
            bias.payload = bias.payload.flatten()
        elif kind == "bias_receive_rows":
            bias.payload = torch.zeros((2, 4))
        else:
            bias.payload = torch.zeros((3, 8))[:, ::2]
    with pytest.raises(error):
        r.buffer.combine(x, r.handle, weights=weights, bias=bias)
    assert not r.events and r.combine_calls.main is None


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_empty_source_still_returns_remote_expert_contributions(
    combine_runtime, expand, weighted
):
    r = combine_runtime
    r.handle.do_expand = expand
    r.handle.topk_idx = r.tensor(
        "empty_snapshot", torch.empty((0, 3), dtype=torch.int64)
    )
    r.handle.dst_slots = r.tensor("empty_dst", torch.empty((0, 2), dtype=torch.int64))
    r.handle.send_counts.payload.zero_()
    r.handle.layout.src_ranks.payload.fill_(1)
    rows = 12 if expand else 2
    x = r.tensor("x", torch.zeros((rows, 4)))
    weights = (
        r.tensor("weights", torch.zeros((rows,) if expand else (rows, 3)))
        if weighted
        else None
    )
    got = r.buffer.combine(x, r.handle, weights=weights, async_op=True)
    assert r.combine_calls.main is not None and r.combine_calls.epilogue is not None
    assert got.x.shape == (0, 4)
    if weighted:
        assert got.topk_weights is not None and got.topk_weights.shape == (0, 3)
    else:
        assert got.topk_weights is None


@pytest.mark.parametrize("failure", ["combine_main", "combine_epilogue"])
def test_combine_native_failure_propagates(combine_runtime, failure):
    r = combine_runtime
    r.settings["failure"] = failure
    x = r.tensor("x", torch.zeros((2, 4)))
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        r.buffer.combine(x, r.handle)
    assert ("record_tensor", "x", "comm") in r.events
    assert not any(event[0] == "record_event" for event in r.events)
