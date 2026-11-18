"""CPU checks for public combine validation, wiring and allocator bookkeeping."""

from types import SimpleNamespace

import pytest
import torch
from test_cuda_dispatch_orchestration import dispatch_runtime as dispatch_runtime

import nano_ep.backends.cuda as cuda



@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_non_bf16_combine_rejected_before_stream_work(combine_runtime, dtype):
    r = combine_runtime
    x = r.tensor("invalid", torch.zeros((2, 256), dtype=dtype))
    with pytest.raises(ValueError, match="bfloat16"):
        r.buffer.combine(x, r.handle)
    assert not r.events and r.combine_calls.combine is None


@pytest.fixture
def combine_runtime(dispatch_runtime):
    r = dispatch_runtime

    def integers(name, values):
        return r.tensor(name, torch.tensor(values, dtype=torch.int64))

    layout = cuda.NativeLayout(
        recv_metadata=r.tensor(
            "recv_metadata",
            torch.tensor([[0, 0, 0, 1, -1], [1, 1, -1, -1, 2]], dtype=torch.int32),
        ),
        expert_counts=integers("expert_counts", [1, 1, 1]),
        expert_offsets=integers("expert_offsets", [0, 4, 8, 12]),
        recv_rank_prefix=integers("recv_rank_prefix", [1, 2]),
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
        num_sms=24,
    )
    calls = SimpleNamespace(combine=None, outputs=None)

    def combine(*args):
        assert cuda.torch.cuda.current_stream() is r.compute
        r.events.append(("combine",))
        _, x, metadata, weights, bias, *rest = args
        comm_stream, previous_event, asynchronous = args[-3:]
        assert comm_stream == r.comm.cuda_stream
        if previous_event == -1:
            r.comm.wait_stream(r.compute)
        else:
            if not asynchronous:
                r.comm.wait_stream(r.compute)
            r.comm.wait_event(previous_event)
        if asynchronous:
            for tensor in (
                x,
                *metadata,
                *((weights,) if weights is not None else ()),
                *((bias,) if bias is not None else ()),
            ):
                tensor.record_stream(r.comm)
        if r.settings["failure"] == "combine":
            raise RuntimeError("injected combine failure")
        calls.combine = args
        source_tokens = metadata[5].shape[0]
        hidden = x.shape[1]
        calls.outputs = (
            r.tensor("combined_x", torch.zeros((source_tokens, hidden), dtype=x.dtype)),
            r.tensor(
                "combined_weights",
                torch.zeros((source_tokens, 3))
                if weights is not None
                else torch.empty(0),
            ),
        )

        class Completion(r.event):
            def current_stream_wait(self):
                stream = cuda.torch.cuda.current_stream()
                stream.wait_event(self)
                for tensor in calls.outputs:
                    tensor.record_stream(stream)

        completed = Completion()
        completed.record(r.comm)
        if not asynchronous:
            r.compute.wait_event(completed)
            completed = None
        return calls.outputs, completed

    r.buffer._backend.combine = combine
    return SimpleNamespace(**vars(r), handle=handle, combine_calls=calls)


def handle_metadata(handle):
    layout = handle.layout
    return [
        layout.recv_metadata,
        layout.expert_counts,
        layout.expert_offsets,
        handle.dst_slots,
        handle.send_counts,
        handle.topk_idx,
        layout.recv_rank_prefix,
    ]


def test_combine_sms_override_does_not_change_handle(combine_runtime):
    r = combine_runtime
    x = r.tensor("expert_output", torch.zeros((2, 256), dtype=torch.bfloat16))
    r.buffer.combine(x, r.handle, num_sms=7)
    assert r.combine_calls.combine[-4] == 7
    assert r.handle.num_sms == 24
    r.buffer.combine(x, r.handle)
    assert r.combine_calls.combine[-4] == 24


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
    if expand:
        r.handle.layout.recv_metadata.payload[:, 2:] = torch.tensor(
            [[0, 4, -1], [-1, -1, 8]]
        )
    r.buffer._allow_multiple_reduction = mode
    rows = 12 if expand else 2
    x = r.tensor("expert_output", torch.zeros((rows, 256), dtype=dtype))
    weights = (
        r.tensor("weights", torch.ones((rows,) if expand else (rows, 3)))
        if weighted
        else None
    )
    bias = r.tensor("bias", torch.zeros((3, 256), dtype=dtype)) if biased else None
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
    assert calls.combine == (
        r.buffer._runtime,
        x,
        metadata,
        weights,
        bias,
        5,
        expand,
        mode,
        24,
        r.comm.cuda_stream,
        ready.cuda_event if explicit_event else -1,
        asynchronous,
    )
    assert got.x is calls.outputs[0]
    assert got.x.shape == (3, 256) and got.x.dtype == dtype
    assert got.topk_weights is (calls.outputs[1] if weighted else None)
    if explicit_event:
        assert ("wait_event", "comm", ready) in r.events
    else:
        assert ("wait_stream", "comm", "compute") in r.events
    if asynchronous:
        for tensor in [
            x,
            *metadata,
            *([weights] if weighted else []),
            *([bias] if biased else []),
        ]:
            assert r.events.index(
                ("record_tensor", tensor.name, "comm")
            ) > r.events.index(("combine",))
        assert got.event is not None
        assert any(item is r.handle for item in got.event.keepalive)
        with r.use_stream(r.consumer):
            got.event.current_stream_wait()
        assert ("wait_event", "consumer", got.event.event) in r.events
        for tensor in calls.outputs:
            assert ("record_tensor", tensor.name, "consumer") in r.events
    else:
        assert got.event is None
        assert not any(event[0] == "record_tensor" for event in r.events)
        assert any(event[:2] == ("wait_event", "compute") for event in r.events)


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
    x = r.tensor("x", torch.zeros((2, 256), dtype=torch.bfloat16))
    weights, bias = None, None
    error = ValueError
    if kind == "closed":
        r.buffer._closed = True
        error = RuntimeError
    elif kind == "owner":
        r.handle.owner = object()
    elif kind == "config":
        r.handle.config = cuda.EPConfig(12, 3, 256, 5)
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
            x.payload = torch.zeros((2, 8), dtype=torch.bfloat16)[:, ::2]
        elif kind == "x_hidden":
            x.payload = torch.zeros((2, 5), dtype=torch.bfloat16)
        elif kind == "compact_rows":
            x.payload = torch.zeros((3, 256), dtype=torch.bfloat16)
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
            x.payload = torch.zeros((12, 256), dtype=torch.bfloat16)
            weights.payload = torch.zeros((12, 3))
    else:
        bias = r.tensor("bias", torch.zeros((3, 256), dtype=torch.bfloat16))
        if kind == "bias_device":
            bias.device = torch.device("cpu")
        elif kind == "bias_dtype":
            bias.payload = bias.payload.float()
        elif kind == "bias_rank":
            bias.payload = bias.payload.flatten()
        elif kind == "bias_receive_rows":
            bias.payload = torch.zeros((2, 256), dtype=torch.bfloat16)
        else:
            bias.payload = torch.zeros((3, 8), dtype=torch.bfloat16)[:, ::2]
    with pytest.raises(error):
        r.buffer.combine(x, r.handle, weights=weights, bias=bias)
    assert not r.events and r.combine_calls.combine is None


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_empty_source_still_returns_remote_expert_contributions(
    combine_runtime, expand, weighted
):
    r = combine_runtime
    r.handle.do_expand = expand
    if expand:
        r.handle.layout.recv_metadata.payload[:, 2:] = torch.tensor(
            [[0, 4, -1], [-1, -1, 8]]
        )
    r.handle.topk_idx = r.tensor(
        "empty_snapshot", torch.empty((0, 3), dtype=torch.int64)
    )
    r.handle.dst_slots = r.tensor("empty_dst", torch.empty((0, 2), dtype=torch.int64))
    r.handle.send_counts.payload.zero_()
    r.handle.layout.recv_metadata.payload[:, 0].fill_(1)
    rows = 12 if expand else 2
    x = r.tensor("x", torch.zeros((rows, 256), dtype=torch.bfloat16))
    weights = (
        r.tensor("weights", torch.zeros((rows,) if expand else (rows, 3)))
        if weighted
        else None
    )
    got = r.buffer.combine(x, r.handle, weights=weights, async_op=True)
    assert r.combine_calls.combine is not None
    assert got.x.shape == (0, 256)
    if weighted:
        assert got.topk_weights is not None and got.topk_weights.shape == (0, 3)
    else:
        assert got.topk_weights is None


def test_combine_native_failure_propagates(combine_runtime):
    r = combine_runtime
    r.settings["failure"] = "combine"
    x = r.tensor("x", torch.zeros((2, 256), dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="injected combine failure"):
        r.buffer.combine(x, r.handle)
    assert not any(event[0] == "record_tensor" for event in r.events)
    assert not any(event[0] == "record_event" for event in r.events)
