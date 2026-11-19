"""Two CPU participants exercise bootstrap control flow with simulated Gloo/CUDA."""

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from nano_ep import EPConfig
from nano_ep.backends import cuda



@pytest.fixture
def bootstrap(monkeypatch):
    local = threading.local()
    condition = threading.Condition()
    rounds = {}
    events = []
    runtimes = {}
    settings = {"allow_multiple_reduction": False}
    group = object()

    def record_operation(name):
        events.append((name, local.rank))

    def gather(output, value, *, group):
        assert group is expected_group
        index = local.round
        local.round += 1
        with condition:
            bucket = rounds.setdefault(index, {})
            bucket[local.rank] = pickle.loads(pickle.dumps(value))
            condition.notify_all()
            assert condition.wait_for(lambda: len(bucket) == 2, timeout=5), (
                f"missing rank in collective {index}"
            )
            output[:] = [pickle.loads(pickle.dumps(bucket[rank])) for rank in range(2)]

    def barrier(*, group):
        gather([None, None], None, group=group)

    expected_group = group
    monkeypatch.setattr(cuda.dist, "get_world_size", lambda **kwargs: 2)
    monkeypatch.setattr(cuda.dist, "get_rank", lambda **kwargs: local.rank)
    monkeypatch.setattr(cuda.dist, "get_backend", lambda actual: "gloo")
    monkeypatch.setattr(cuda.dist, "all_gather_object", gather)
    monkeypatch.setattr(cuda.dist, "barrier", barrier)
    monkeypatch.setattr(cuda.torch.cuda, "current_device", lambda: local.rank)

    def properties(device):
        record_operation("properties")
        return SimpleNamespace(uuid=f"gpu-{device}", multi_processor_count=78)

    monkeypatch.setattr(cuda.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(
        cuda,
        "socket",
        SimpleNamespace(gethostname=lambda: "node"),
    )

    class FakeEnvironment:
        def __getitem__(self, key):
            if key == "LOCAL_RANK":
                return str(local.rank)
            assert key == "LOCAL_WORLD_SIZE"
            return "2"

    monkeypatch.setattr(cuda, "os", SimpleNamespace(environ=FakeEnvironment()))

    class FakePeerRuntime:
        def __init__(self, rank, world_size, num_bytes, /):
            record_operation("allocate")
            assert rank == local.rank and world_size == 2 and num_bytes == 9888
            self.rank = rank
            self.unmapped = False
            self.freed = False
            self.window = torch.zeros(num_bytes, dtype=torch.uint8)
            runtimes[rank] = self

        def local_window(self):
            record_operation("initialize")
            return self.window

        def export_local_handle(self):
            record_operation("export")
            return f"handle-{self.rank}".encode()

        def import_peer_handles(self, handles):
            record_operation("import")
            assert handles == [b"handle-0", b"handle-1"]

        def close_peer_mappings(self):
            record_operation("unmap")
            self.unmapped = True

        def close(self):
            record_operation("free")
            assert all(runtime.unmapped for runtime in runtimes.values())
            assert not self.freed
            self.freed = True

    def window_layout(
        ranks, tokens, hidden, topk, experts, allow_multiple_reduction, /
    ):
        record_operation("layout")
        assert (ranks, tokens, hidden, topk, experts) == (2, 8, 256, 2, 4)
        assert allow_multiple_reduction is settings["allow_multiple_reduction"]
        return [0, 96, 160, 160, 9888]

    native = SimpleNamespace(
        PeerRuntime=FakePeerRuntime,
        window_layout=window_layout,
    )

    def load():
        record_operation("extension")
        return native

    def stream(*, device):
        record_operation("stream")
        assert device == local.rank
        return object()

    monkeypatch.setattr(cuda, "load_extension", load)

    monkeypatch.setattr(cuda.torch.cuda, "Stream", stream)
    monkeypatch.setattr(
        cuda.torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(
            synchronize=lambda: record_operation("window_ready")
        ),
    )
    monkeypatch.setattr(
        cuda.torch.cuda, "synchronize", lambda device: record_operation("drain")
    )

    def run(*, close=False):
        def participant(rank):
            local.rank = rank
            local.round = 0
            buffer = cuda.CudaBuffer(
                group,
                EPConfig(4, 2, 256, 8),
                allow_multiple_reduction=settings["allow_multiple_reduction"],
            )
            if close:
                buffer.close()
                buffer.close()
            return buffer

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(participant, rank) for rank in range(2)]
            return [future.result(timeout=15) for future in futures]

    return SimpleNamespace(
        run=run, settings=settings, events=events, runtimes=runtimes, rounds=rounds
    )


@pytest.mark.parametrize(
    "allow_multiple_reduction",
    [False, True],
    ids=["without-reduction", "with-reduction"],
)
def test_bootstrap_collective_close(bootstrap, allow_multiple_reduction):
    bootstrap.settings["allow_multiple_reduction"] = allow_multiple_reduction
    buffers = bootstrap.run(close=True)
    assert all(
        isinstance(buffer, cuda.CudaBuffer) and buffer._closed for buffer in buffers
    )
    assert all(runtime.freed for runtime in bootstrap.runtimes.values())
    for rank, buffer in enumerate(buffers):
        assert buffer._device == buffer._rank == rank
        assert buffer._world_size == 2
        assert buffer._allow_multiple_reduction is allow_multiple_reduction
        assert (
            buffer._signal_offset,
            buffer._count_offset,
            buffer._payload_offset,
            buffer._return_offset,
            buffer._num_bytes,
        ) == (0, 96, 160, 160, 9888)
        assert buffer._comm_stream is not None
        window = bootstrap.runtimes[rank].window
        assert window.view(torch.int64)[12:18].tolist() == [-1] * 6
        assert torch.count_nonzero(window[:96]).item() == 0
        assert torch.count_nonzero(window[144:]).item() == 0
        operations = [
            name for name, actual_rank in bootstrap.events if actual_rank == rank
        ]
        assert (
            operations.index("initialize")
            < operations.index("window_ready")
            < operations.index("export")
        )
