"""CPU checks for combine metadata and stream handling."""

from types import SimpleNamespace

import pytest
import torch
from test_cuda_dispatch import (
    dispatch_runtime as dispatch_runtime,  # noqa: PLC0414
)

from nano_ep.backends import cuda



@pytest.fixture
def combine_runtime(dispatch_runtime):
    context = dispatch_runtime

    def make_int64_tensor(name, values):
        return context.tensor(name, torch.tensor(values, dtype=torch.int64))

    layout = cuda.NativeLayout(
        recv_metadata=context.tensor(
            "recv_metadata",
            torch.tensor([[0, 0, 0, 1, -1], [1, 1, -1, -1, 2]], dtype=torch.int32),
        ),
        expert_counts=make_int64_tensor("expert_counts", [1, 1, 1]),
        expert_offsets=make_int64_tensor("expert_offsets", [0, 4, 8, 12]),
        recv_rank_prefix=make_int64_tensor("recv_rank_prefix", [1, 2]),
        recv_count=2,
        expanded_capacity=12,
    )
    handle = cuda.NativeHandle(
        config=context.buffer._config,
        rank=0,
        world_size=2,
        do_expand=False,
        alignment=4,
        topk_idx=make_int64_tensor("snapshot", [[0, 1, 3], [3, -1, 5], [-1, -1, -1]]),
        layout=layout,
        dst_slots=make_int64_tensor("dst_slots", [[0, 0], [-1, 1], [-1, -1]]),
        send_counts=make_int64_tensor("send_counts", [1, 2]),
        owner=context.buffer,
        x_dtype=torch.bfloat16,
        num_sms=24,
    )
    calls = SimpleNamespace(combine=None, outputs=None)

    def combine(*args):
        assert cuda.torch.cuda.current_stream() is context.compute
        context.events.append(("combine",))
        _, x, metadata, weights, bias, *_rest = args
        comm_stream, previous_event, asynchronous = args[-3:]
        assert comm_stream == context.comm.cuda_stream
        if previous_event == -1:
            context.comm.wait_stream(context.compute)
        else:
            if not asynchronous:
                context.comm.wait_stream(context.compute)
            context.comm.wait_event(previous_event)
        if asynchronous:
            for tensor in (
                x,
                *metadata,
                *((weights,) if weights is not None else ()),
                *((bias,) if bias is not None else ()),
            ):
                tensor.record_stream(context.comm)
        calls.combine = args
        source_tokens = metadata[5].shape[0]
        hidden = x.shape[1]
        calls.outputs = (
            context.tensor(
                "combined_x", torch.zeros((source_tokens, hidden), dtype=x.dtype)
            ),
            context.tensor(
                "combined_weights",
                torch.zeros((source_tokens, 3))
                if weights is not None
                else torch.empty(0),
            ),
        )

        class FakeCompletion(context.event):
            def current_stream_wait(self):
                stream = cuda.torch.cuda.current_stream()
                stream.wait_event(self)
                for tensor in calls.outputs:
                    tensor.record_stream(stream)

        completed = FakeCompletion()
        completed.record(context.comm)
        if not asynchronous:
            context.compute.wait_event(completed)
            completed = None
        return calls.outputs, completed

    context.buffer._backend.combine = combine
    return SimpleNamespace(**vars(context), handle=handle, combine_calls=calls)


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


@pytest.mark.parametrize(
    "expand,mode,weighted,biased,asynchronous,explicit_event",
    [
        pytest.param(False, False, False, False, False, False, id="compact-sync"),
        pytest.param(False, True, True, True, True, True, id="compact-async-weighted"),
        pytest.param(
            True, False, True, False, True, False, id="expanded-async-weighted"
        ),
        pytest.param(True, True, False, True, False, True, id="expanded-sync-biased"),
    ],
)
def test_combine_metadata_and_streams(
    combine_runtime, expand, mode, weighted, biased, asynchronous, explicit_event
):
    context = combine_runtime
    dtype = torch.bfloat16
    context.handle.do_expand = expand
    if expand:
        context.handle.layout.recv_metadata.payload[:, 2:] = torch.tensor(
            [[0, 4, -1], [-1, -1, 8]]
        )
    context.buffer._allow_multiple_reduction = mode
    rows = 12 if expand else 2
    x = context.tensor("expert_output", torch.zeros((rows, 256), dtype=dtype))
    weights = (
        context.tensor("weights", torch.ones((rows,) if expand else (rows, 3)))
        if weighted
        else None
    )
    bias = (
        context.tensor("bias", torch.zeros((3, 256), dtype=dtype)) if biased else None
    )
    ready = context.event() if explicit_event else None
    result = context.buffer.combine(
        x,
        context.handle,
        weights=weights,
        bias=bias,
        previous_event=ready,
        async_op=asynchronous,
    )
    calls = context.combine_calls
    metadata = handle_metadata(context.handle)
    assert calls.combine == (
        context.buffer._runtime,
        x,
        metadata,
        weights,
        bias,
        5,
        expand,
        mode,
        24,
        context.comm.cuda_stream,
        ready.cuda_event if explicit_event else -1,
        asynchronous,
    )
    assert result.x is calls.outputs[0]
    assert result.x.shape == (3, 256) and result.x.dtype == dtype
    assert result.topk_weights is (calls.outputs[1] if weighted else None)
    if explicit_event:
        assert ("wait_event", "comm", ready) in context.events
    else:
        assert ("wait_stream", "comm", "compute") in context.events
    if asynchronous:
        for tensor in [
            x,
            *metadata,
            *([weights] if weighted else []),
            *([bias] if biased else []),
        ]:
            assert context.events.index(
                ("record_tensor", tensor.name, "comm")
            ) > context.events.index(("combine",))
        assert result.event is not None
        assert any(item is context.handle for item in result.event.keepalive)
        with context.use_stream(context.consumer):
            result.event.current_stream_wait()
        assert ("wait_event", "consumer", result.event.event) in context.events
        for tensor in calls.outputs:
            assert ("record_tensor", tensor.name, "consumer") in context.events
    else:
        assert result.event is None
        assert not any(event[0] == "record_tensor" for event in context.events)
        assert any(event[:2] == ("wait_event", "compute") for event in context.events)
