"""CPU doubles verify Python wiring and stream bookkeeping, not CUDA execution."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

import nano_ep.backends.cuda as cuda
from nano_ep import EPConfig



@pytest.fixture
def dispatch_runtime(monkeypatch):
    events, calls = [], []
    settings = {"failure": None}

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait_stream", self.name, other.name))

        def wait_event(self, event):
            events.append(("wait_event", self.name, event))

    compute, comm, consumer = (Stream(name) for name in ("compute", "comm", "consumer"))
    current = [compute]

    @contextmanager
    def use_stream(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    class Event:
        def record(self, stream):
            assert current[0] is stream is comm
            events.append(("record_event", stream.name, self))

    class Tensor:
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

    def integers(name, data):
        return Tensor(name, torch.tensor(data, dtype=torch.int64))

    runtime = object()

    def main(
        actual_runtime, x, routing, weights, experts, capacity, alignment, cached, mode
    ):
        assert actual_runtime is runtime and current[0] is comm
        assert (experts, capacity) == (6, 5)
        events.append(("main",))
        if settings["failure"] == "main":
            raise RuntimeError("injected main failure")
        if cached:
            metadata = cached
        else:
            metadata = [
                integers("expert_counts", [1, 1, 1]),
                integers(
                    "expert_offsets", [0, alignment, 2 * alignment, 3 * alignment]
                ),
                integers("dst_slots", [[0, 1], [-1, -1]]),
                integers("send_counts", [1, 1]),
                Tensor("snapshot", routing.payload.clone()),
            ]
        state = SimpleNamespace(
            metadata=metadata,
            counts=torch.tensor([2, 3 * alignment]),
            metadata_prepared=bool(cached),
            dtype=x.dtype,
        )
        calls.append(
            SimpleNamespace(
                x=x,
                routing=routing,
                weights=weights,
                alignment=alignment,
                cached=cached,
                mode=mode,
                state=state,
            )
        )
        return state

    def epilogue(state, expand, weighted, zero_padding):
        assert current[0] is comm
        events.append(("epilogue",))
        if settings["failure"] == "epilogue":
            raise RuntimeError("injected epilogue failure")
        if not state.metadata_prepared:
            counts, offsets, dst, sends, snapshot = state.metadata
            alignment = int(offsets.payload[1])
            state.metadata = [
                integers("src_ranks", [0, 1]),
                integers("src_tokens", [0, 0]),
                integers("local_topk", [[0, 1, -1], [-1, -1, 2]]),
                counts,
                offsets,
                integers(
                    "expanded_slots", [[0, alignment, -1], [-1, -1, 2 * alignment]]
                ),
                dst,
                sends,
                snapshot,
            ]
            state.metadata_prepared = True
        n, capacity = state.counts.tolist()
        rows = capacity if expand else n
        outputs = (
            Tensor("recv_x", torch.zeros((rows, 4), dtype=state.dtype)),
            Tensor(
                "recv_idx",
                torch.empty(0, dtype=torch.int64)
                if expand
                else state.metadata[2].payload.clone(),
            ),
            Tensor(
                "recv_weights",
                torch.zeros((rows,) if expand else (rows, 3))
                if weighted
                else torch.empty(0),
            ),
        )
        state.outputs = outputs
        state.epilogue_args = (expand, weighted, zero_padding)
        return outputs

    monkeypatch.setattr(cuda.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        cuda.torch.cuda, "current_stream", lambda device=None: current[0]
    )
    monkeypatch.setattr(cuda.torch.cuda, "stream", use_stream)
    monkeypatch.setattr(cuda.torch.cuda, "Event", Event)
    buffer = cuda.CudaBuffer.__new__(cuda.CudaBuffer)
    buffer._config = EPConfig(6, 3, 4, 5)
    buffer._device = buffer._rank = 0
    buffer._closed = False
    buffer._world_size = 2
    buffer._runtime = runtime
    buffer._comm_stream = comm
    buffer._backend = SimpleNamespace(dispatch_main=main, dispatch_epilogue=epilogue)
    buffer._allow_multiple_reduction = False
    return SimpleNamespace(
        buffer=buffer,
        tensor=Tensor,
        event=Event,
        compute=compute,
        comm=comm,
        consumer=consumer,
        use_stream=use_stream,
        events=events,
        calls=calls,
        settings=settings,
        indices=integers("indices", [[0, 1, 3], [-1, -1, -1]]),
    )


@pytest.mark.parametrize("layout", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit_event", [False, True])
def test_fresh_dispatch_wires_state_and_streams(
    dispatch_runtime, layout, dtype, weighted, mode, asynchronous, explicit_event
):
    r = dispatch_runtime
    r.buffer._allow_multiple_reduction = mode
    x = r.tensor("x", torch.zeros((2, 4), dtype=dtype))
    weights = r.tensor("weights", torch.ones((2, 3))) if weighted else None
    ready = r.event() if explicit_event else None
    alignment = None if layout is None else 4
    got = r.buffer.dispatch(
        x,
        topk_idx=r.indices,
        weights=weights,
        do_expand=layout,
        alignment=alignment,
        zero_padding=True,
        previous_event=ready,
        async_op=asynchronous,
    )
    call = r.calls[0]
    assert call.mode is mode and call.cached == []
    assert call.x is x and call.routing is r.indices and call.weights is weights
    assert call.alignment == (1 if alignment is None else alignment)
    expand = layout is True
    assert call.state.epilogue_args == (expand, weighted, True)
    metadata = call.state.metadata
    h = got.handle
    assert h.owner is r.buffer and h.config == r.buffer._config
    assert (h.rank, h.world_size, h.x_dtype, h.do_expand, h.alignment) == (
        0,
        2,
        dtype,
        expand,
        call.alignment,
    )
    assert (h.layout.recv_count, h.layout.expanded_capacity) == (2, 3 * call.alignment)
    assert [
        h.layout.src_ranks,
        h.layout.src_tokens,
        h.layout.local_topk_idx,
        h.layout.expert_counts,
        h.layout.expert_offsets,
        h.layout.expanded_slots,
        h.dst_slots,
        h.send_counts,
        h.topk_idx,
    ] == metadata
    assert got.x is call.state.outputs[0]
    assert got.topk_idx is (None if expand else call.state.outputs[1])
    assert got.topk_weights is (call.state.outputs[2] if weighted else None)
    dependency = (
        ("wait_event", "comm", ready)
        if explicit_event
        else ("wait_stream", "comm", "compute")
    )
    assert r.events[0] == dependency
    assert r.events.index(("record_tensor", "x", "comm")) < r.events.index(("main",))
    if weighted:
        assert r.events.index(("record_tensor", "weights", "comm")) < r.events.index(
            ("main",)
        )
    assert (
        r.events.index(("main",))
        < r.events.index(("epilogue",))
        < r.events.index(("record_event", "comm", got.event.event))
    )
    compute_wait = ("wait_event", "compute", got.event.event)
    assert (compute_wait in r.events) is not asynchronous
    for tensor in (*call.state.outputs, *metadata):
        assert (
            ("record_tensor", tensor.name, "compute") in r.events
        ) is not asynchronous
    assert any(item is call.state for item in got.event.keepalive)
    r.indices.payload.fill_(-1)
    assert h.topk_idx.payload[0].tolist() == [0, 1, 3]
    with r.use_stream(r.consumer):
        got.event.current_stream_wait()
    assert ("wait_event", "consumer", got.event.event) in r.events
    for tensor in (*call.state.outputs, *metadata):
        assert ("record_tensor", tensor.name, "consumer") in r.events


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cached_dispatch_reuses_handle_with_new_payload(
    dispatch_runtime, expand, weighted, asynchronous
):
    r = dispatch_runtime
    x = r.tensor("x", torch.zeros((2, 4)))
    weights = r.tensor("weights", torch.ones((2, 3)))
    first = r.buffer.dispatch(
        x,
        topk_idx=r.indices,
        weights=weights if weighted else None,
        do_expand=expand,
        alignment=4,
        async_op=True,
    )
    old_metadata = r.calls[0].state.metadata
    r.events.clear()
    new_x = r.tensor("new_x", torch.full((2, 4), 7.0))
    new_weights = None if weighted else weights
    got = r.buffer.dispatch(
        new_x, weights=new_weights, handle=first.handle, async_op=asynchronous
    )
    call = r.calls[-1]
    assert got.handle is first.handle
    assert (
        call.routing is first.handle.topk_idx
        and call.x is new_x
        and call.weights is new_weights
    )
    assert call.cached == old_metadata and call.state.metadata == old_metadata
    assert call.alignment == 4 and call.state.epilogue_args == (
        expand,
        not weighted,
        False,
    )
    for tensor in old_metadata:
        assert ("record_tensor", tensor.name, "comm") in r.events


@pytest.mark.parametrize("failure", ["main", "epilogue"])
def test_native_failure_does_not_return_completion(dispatch_runtime, failure):
    r = dispatch_runtime
    r.settings["failure"] = failure
    x = r.tensor("x", torch.zeros((2, 4)))
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        r.buffer.dispatch(x, topk_idx=r.indices)
    assert ("record_tensor", "x", "comm") in r.events
    assert not any(event[0] == "record_event" for event in r.events)
