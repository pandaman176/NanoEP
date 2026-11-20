"""CPU doubles verify Python wiring and stream bookkeeping, not CUDA execution."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from nano_ep import EPConfig
from nano_ep.backends import cuda


@pytest.fixture
def dispatch_runtime(monkeypatch):
    events, calls = [], []

    class FakeStream:
        def __init__(self, name):
            self.name = name
            self.cuda_stream = {"compute": 10, "comm": 11, "consumer": 12}[name]

        def wait_stream(self, other):
            events.append(("wait_stream", self.name, other.name))

        def wait_event(self, event):
            events.append(("wait_event", self.name, event))

    compute, comm, consumer = (FakeStream(name) for name in ("compute", "comm", "consumer"))
    current = [compute]

    @contextmanager
    def use_stream(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    class FakeEvent:
        def __init__(self):
            self.cuda_event = self

        def record(self, stream):
            assert stream is comm
            events.append(("record_event", stream.name, self))

    class FakeTensor:
        device = torch.device("cuda", 0)

        def __init__(self, name, payload):
            self.name, self.payload = name, payload

        @property
        def dtype(self):
            return self.payload.dtype

        @property
        def shape(self):
            return self.payload.shape

        @property
        def ndim(self):
            return self.payload.ndim

        def is_contiguous(self):
            return self.payload.is_contiguous()

        def record_stream(self, stream):
            events.append(("record_tensor", self.name, stream.name))

    def make_int64_tensor(name, data):
        return FakeTensor(name, torch.tensor(data, dtype=torch.int64))

    runtime = object()

    def dispatch(
        actual_runtime,
        x,
        routing,
        weights,
        experts,
        capacity,
        alignment,
        cached,
        cached_expanded_capacity,
        mode,
        expand,
        zero_padding,
        num_sms,
        comm_stream,
        previous_event,
        async_op,
    ):
        assert actual_runtime is runtime and current[0] is compute
        assert comm_stream == comm.cuda_stream
        consumer_stream = -1 if async_op else compute.cuda_stream
        if previous_event == -1:
            comm.wait_stream(compute)
        else:
            if not async_op:
                comm.wait_stream(compute)
            comm.wait_event(previous_event)
        assert (experts, capacity) == (6, 5)
        events.append(("dispatch",))
        # Record asynchronous inputs on the communication stream.
        if consumer_stream == -1:
            for tensor in (
                x,
                *((routing,) if not cached else ()),
                *((weights,) if weights is not None else ()),
                *cached,
            ):
                tensor.record_stream(comm)
        metadata = cached or [
            FakeTensor(
                "recv_metadata",
                torch.tensor([[0, 0, 0, 1, -1], [1, 0, -1, -1, 2]], dtype=torch.int32),
            ),
            make_int64_tensor("expert_counts", [1, 1, 1]),
            make_int64_tensor("expert_offsets", [0, alignment, 2 * alignment, 3 * alignment]),
            make_int64_tensor("dst_slots", [[0, 1], [-1, -1]]),
            make_int64_tensor("send_counts", [1, 1]),
            FakeTensor("snapshot", routing.payload.clone()),
            make_int64_tensor("recv_rank_prefix", [1, 2]),
        ]
        n, expanded_capacity = 2, cached_expanded_capacity if cached else 3 * alignment
        rows = expanded_capacity if expand else n
        outputs = (
            FakeTensor("recv_x", torch.zeros((rows, 256), dtype=x.dtype)),
            FakeTensor(
                "recv_idx",
                torch.empty(0, dtype=torch.int64) if expand else metadata[0].payload[:, 2:].to(torch.int64).clone(),
            ),
            FakeTensor(
                "recv_weights",
                torch.zeros((rows,) if expand else (rows, 3)) if weights is not None else torch.empty(0),
            ),
        )
        calls.append(
            SimpleNamespace(
                x=x,
                routing=routing,
                weights=weights,
                alignment=alignment,
                cached=cached,
                cached_expanded_capacity=cached_expanded_capacity,
                mode=mode,
                expand=expand,
                zero_padding=zero_padding,
                num_sms=num_sms,
                metadata=metadata,
                outputs=outputs,
            )
        )

        class FakeCompletion(FakeEvent):
            def current_stream_wait(self):
                current[0].wait_event(self)
                for tensor in (*outputs, *metadata):
                    tensor.record_stream(current[0])

        completed = FakeCompletion()
        completed.record(comm)
        if consumer_stream != -1:
            assert consumer_stream == compute.cuda_stream
            compute.wait_event(completed)
            completed = None
        return outputs, metadata, (n, expanded_capacity), completed

    monkeypatch.setattr(cuda.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(cuda.torch.cuda, "current_stream", lambda device=None: current[0])
    monkeypatch.setattr(cuda.torch.cuda, "stream", use_stream)
    monkeypatch.setattr(cuda.torch.cuda, "Event", FakeEvent)
    buffer = cuda.CudaBuffer.__new__(cuda.CudaBuffer)
    buffer._config = EPConfig(6, 3, 256, 5)
    buffer._device = buffer._rank = 0
    buffer._device_sms = 78
    buffer._gpu_uuids = ("gpu-0", "gpu-1")
    buffer._auto_num_sms = None
    buffer._closed = False
    buffer._world_size = 2
    buffer._runtime = runtime
    buffer._comm_stream = comm
    buffer._backend = SimpleNamespace(dispatch=dispatch)
    buffer._allow_multiple_reduction = False
    return SimpleNamespace(
        buffer=buffer,
        tensor=FakeTensor,
        event=FakeEvent,
        compute=compute,
        comm=comm,
        consumer=consumer,
        use_stream=use_stream,
        events=events,
        calls=calls,
        indices=make_int64_tensor("indices", [[0, 1, 3], [-1, -1, -1]]),
    )


@pytest.mark.parametrize(
    "layout,weighted,mode,asynchronous,explicit_event",
    [
        pytest.param(None, False, False, False, False, id="default-sync"),
        pytest.param(False, True, True, True, True, id="compact-async-weighted"),
        pytest.param(True, True, False, True, False, id="expanded-async-weighted"),
        pytest.param(True, False, True, False, True, id="expanded-sync-unweighted"),
    ],
)
def test_dispatch_fresh_metadata_and_streams(dispatch_runtime, layout, weighted, mode, asynchronous, explicit_event):
    context = dispatch_runtime
    dtype = torch.bfloat16
    context.buffer._allow_multiple_reduction = mode
    x = context.tensor("x", torch.zeros((2, 256), dtype=dtype))
    weights = context.tensor("weights", torch.ones((2, 3))) if weighted else None
    ready = context.event() if explicit_event else None
    alignment = None if layout is None else 4
    result = context.buffer.dispatch(
        x,
        topk_idx=context.indices,
        weights=weights,
        do_expand=layout,
        alignment=alignment,
        zero_padding=True,
        previous_event=ready,
        async_op=asynchronous,
    )
    call = context.calls[0]
    assert call.mode is mode and call.cached == []
    assert call.cached_expanded_capacity == 0
    assert call.x is x and call.routing is context.indices and call.weights is weights
    assert call.alignment == (1 if alignment is None else alignment)
    expand = layout is True
    assert call.expand is expand and call.zero_padding is True
    metadata = call.metadata
    assert metadata[0].shape == (2, 5)
    handle = result.handle
    assert call.num_sms == handle.num_sms == 78
    assert handle.owner is context.buffer and handle.config == context.buffer._config
    assert (
        handle.rank,
        handle.world_size,
        handle.x_dtype,
        handle.do_expand,
        handle.alignment,
    ) == (
        0,
        2,
        dtype,
        expand,
        call.alignment,
    )
    assert (handle.layout.recv_count, handle.layout.expanded_capacity) == (
        2,
        3 * call.alignment,
    )
    assert [
        handle.layout.recv_metadata,
        handle.layout.expert_counts,
        handle.layout.expert_offsets,
        handle.dst_slots,
        handle.send_counts,
        handle.topk_idx,
        handle.layout.recv_rank_prefix,
    ] == metadata
    assert result.x is call.outputs[0]
    assert result.topk_idx is (None if expand else call.outputs[1])
    assert result.topk_weights is (call.outputs[2] if weighted else None)
    if explicit_event:
        assert ("wait_event", "comm", ready) in context.events
    else:
        assert ("wait_stream", "comm", "compute") in context.events
    if asynchronous:
        assert context.events.index(("record_tensor", "x", "comm")) > context.events.index(("dispatch",))
        assert result.event is not None
        assert any(item is handle for item in result.event.keepalive)
        with context.use_stream(context.consumer):
            result.event.current_stream_wait()
        assert ("wait_event", "consumer", result.event.event) in context.events
        for tensor in (*call.outputs, *metadata):
            assert ("record_tensor", tensor.name, "consumer") in context.events
    else:
        assert result.event is None
        assert not any(event[0] == "record_tensor" for event in context.events)
        assert any(event[:2] == ("wait_event", "compute") for event in context.events)
    context.indices.payload.fill_(-1)
    assert handle.topk_idx.payload[0].tolist() == [0, 1, 3]


@pytest.mark.parametrize(
    "expand,weighted,asynchronous",
    [
        pytest.param(False, True, False, id="compact-sync-weighted"),
        pytest.param(True, False, True, id="expanded-async-unweighted"),
    ],
)
def test_dispatch_cached_reuses_handle_with_new_payload(dispatch_runtime, expand, weighted, asynchronous):
    context = dispatch_runtime
    x = context.tensor("x", torch.zeros((2, 256), dtype=torch.bfloat16))
    weights = context.tensor("weights", torch.ones((2, 3)))
    first = context.buffer.dispatch(
        x,
        topk_idx=context.indices,
        weights=weights if weighted else None,
        do_expand=expand,
        alignment=4,
        async_op=True,
    )
    old_metadata = context.calls[0].metadata
    context.events.clear()
    new_x = context.tensor("new_x", torch.full((2, 256), 7.0, dtype=torch.bfloat16))
    new_weights = None if weighted else weights
    result = context.buffer.dispatch(new_x, weights=new_weights, handle=first.handle, async_op=asynchronous)
    call = context.calls[-1]
    assert result.handle is first.handle
    assert call.routing is first.handle.topk_idx and call.x is new_x and call.weights is new_weights
    assert call.cached == old_metadata and call.metadata == old_metadata
    assert call.cached_expanded_capacity == first.handle.layout.expanded_capacity
    assert call.alignment == 4 and call.expand is expand and not call.zero_padding
    for tensor in old_metadata:
        assert (("record_tensor", tensor.name, "comm") in context.events) is asynchronous
    assert context.events.count(("record_tensor", first.handle.topk_idx.name, "comm")) == int(asynchronous)
