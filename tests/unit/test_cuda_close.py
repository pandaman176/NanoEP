"""CPU checks for teardown orchestration, not CUDA IPC or device ordering."""

import pytest

import nano_ep.backends.cuda as cuda



@pytest.fixture
def closing_buffer(monkeypatch):
    events = []
    fail = {"at": None}

    def record(name):
        events.append(name)
        if name == fail["at"]:
            raise RuntimeError(name)

    device, group = object(), object()

    def synchronize(actual_device):
        assert actual_device is device
        record("drain")

    barriers = 0

    def barrier(*, group):
        nonlocal barriers
        assert group is buffer._group
        barriers += 1
        record(f"barrier{barriers}")

    class Runtime:
        def close_peer_mappings(self):
            record("unmap")

        def close(self):
            record("free")

    buffer = cuda.CudaBuffer.__new__(cuda.CudaBuffer)
    buffer._runtime = Runtime()
    buffer._group = group
    buffer._device = device
    buffer._closed = False
    monkeypatch.setattr(cuda.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(cuda.dist, "barrier", barrier)
    return buffer, events, fail


def test_collective_close_order_and_idempotence(closing_buffer):
    buffer, events, _ = closing_buffer
    buffer.close()
    assert events == ["drain", "barrier1", "unmap", "barrier2", "free"]
    assert buffer._closed
    buffer.close()
    assert events == ["drain", "barrier1", "unmap", "barrier2", "free"]


@pytest.mark.parametrize("failure", ["drain", "barrier1", "unmap", "barrier2", "free"])
def test_collective_close_stops_on_failure(closing_buffer, failure):
    buffer, events, fail = closing_buffer
    fail["at"] = failure
    with pytest.raises(RuntimeError, match=failure):
        buffer.close()
    order = ["drain", "barrier1", "unmap", "barrier2", "free"]
    assert events == order[: order.index(failure) + 1]
    assert not buffer._closed
