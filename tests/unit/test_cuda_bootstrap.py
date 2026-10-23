"""Two CPU participants exercise bootstrap control flow with simulated Gloo/CUDA."""

from concurrent.futures import ThreadPoolExecutor
import pickle
import threading
from types import SimpleNamespace

import pytest
import torch

import nano_ep.backends.cuda as cuda
from nano_ep import EPConfig



@pytest.fixture
def bootstrap(monkeypatch):
    local = threading.local()
    condition = threading.Condition()
    rounds = {}
    events = []
    runtimes = {}
    settings = {
        "failure": None,
        "cleanup_failure": None,
        "metadata": None,
        "backend": "gloo",
        "allow_multiple_reduction": False,
    }
    group = object()

    def operation(name):
        events.append((name, local.rank))
        if local.rank == 0 and name in (
            settings["failure"],
            settings["cleanup_failure"],
        ):
            raise RuntimeError(f"injected {name}")

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
    monkeypatch.setattr(cuda.dist, "get_backend", lambda actual: settings["backend"])
    monkeypatch.setattr(cuda.dist, "all_gather_object", gather)
    monkeypatch.setattr(cuda.dist, "barrier", barrier)
    monkeypatch.setattr(cuda.torch.cuda, "current_device", lambda: local.rank)

    def properties(device):
        operation("properties")
        uuid = 0 if settings["metadata"] == "duplicate_gpu" else device
        return SimpleNamespace(uuid=f"gpu-{uuid}")

    monkeypatch.setattr(cuda.torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(
        cuda,
        "socket",
        SimpleNamespace(
            gethostname=lambda: (
                f"node-{local.rank}" if settings["metadata"] == "nodes" else "node"
            )
        ),
    )

    class Environment:
        def __getitem__(self, key):
            if key == "LOCAL_RANK":
                return str(0 if settings["metadata"] == "local_ranks" else local.rank)
            assert key == "LOCAL_WORLD_SIZE"
            return "8" if settings["metadata"] == "local_size" else "2"

    monkeypatch.setattr(cuda, "os", SimpleNamespace(environ=Environment()))

    class Runtime:
        def __init__(self, rank, world_size, num_bytes, /):
            operation("allocate")
            assert rank == local.rank and world_size == 2 and num_bytes == 4096
            self.rank = rank
            self.unmapped = False
            self.freed = False
            self.window = torch.zeros(num_bytes, dtype=torch.uint8)
            runtimes[rank] = self

        def local_window(self):
            operation("initialize")
            return self.window

        def export_local_handle(self):
            operation("export")
            return f"handle-{self.rank}".encode()

        def import_peer_handles(self, handles):
            operation("import")
            assert handles == [b"handle-0", b"handle-1"]

        def close_peer_mappings(self):
            operation("unmap")
            self.unmapped = True

        def close(self):
            operation("free")
            assert all(runtime.unmapped for runtime in runtimes.values())
            assert not self.freed
            self.freed = True

    def window_layout(
        ranks, tokens, hidden, topk, experts, allow_multiple_reduction, /
    ):
        operation("layout")
        assert (ranks, tokens, hidden, topk, experts) == (2, 8, 128, 2, 4)
        assert allow_multiple_reduction is settings["allow_multiple_reduction"]
        return [0, 256, 512, 512, 4096]

    native = SimpleNamespace(PeerRuntime=Runtime, window_layout=window_layout)

    def load():
        operation("extension")
        return native

    def stream(*, device):
        operation("stream")
        assert device == local.rank
        return object()

    monkeypatch.setattr(cuda, "load_extension", load)
    monkeypatch.setattr(cuda.torch.cuda, "Stream", stream)
    monkeypatch.setattr(
        cuda.torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: operation("window_ready")),
    )
    monkeypatch.setattr(
        cuda.torch.cuda, "synchronize", lambda device: operation("drain")
    )

    def run(*, close=False):
        def participant(rank):
            local.rank = rank
            local.round = 0
            experts = 6 if settings["metadata"] == "config" and rank == 0 else 4
            if settings["metadata"] == "experts":
                experts = 3
            try:
                mode = settings["allow_multiple_reduction"]
                if settings["metadata"] == "reduction" and rank == 0:
                    mode = not mode
                buffer = cuda.CudaBuffer(
                    group, EPConfig(experts, 2, 128, 8), allow_multiple_reduction=mode
                )
                if close:
                    buffer.close()
                    buffer.close()
                return buffer
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(participant, rank) for rank in range(2)]
            return [future.result(timeout=15) for future in futures]

    return SimpleNamespace(
        run=run, settings=settings, events=events, runtimes=runtimes, rounds=rounds
    )


@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
def test_bootstrap_and_collective_close(bootstrap, allow_multiple_reduction):
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
        ) == (0, 256, 512, 512, 4096)
        assert buffer._comm_stream is not None
        window = bootstrap.runtimes[rank].window
        assert window.view(torch.int64)[32:38].tolist() == [-1] * 6
        assert torch.count_nonzero(window[:256]).item() == 0
        assert torch.count_nonzero(window[304:]).item() == 0
        operations = [
            name for name, actual_rank in bootstrap.events if actual_rank == rank
        ]
        assert (
            operations.index("initialize")
            < operations.index("window_ready")
            < operations.index("export")
        )


@pytest.mark.parametrize(
    "failure",
    [
        "properties",
        "extension",
        "layout",
        "allocate",
        "initialize",
        "window_ready",
        "export",
        "import",
        "stream",
    ],
)
def test_one_rank_failure_reaches_both_ranks(bootstrap, failure):
    bootstrap.settings["failure"] = failure
    results = bootstrap.run()
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert str(results[0]) == str(results[1])
    assert f"rank 0: RuntimeError: injected {failure}" in str(results[0])
    assert all(runtime.freed for runtime in bootstrap.runtimes.values())
    if failure in (
        "extension",
        "layout",
        "allocate",
        "initialize",
        "window_ready",
        "export",
    ):
        assert not any(name == "import" for name, rank in bootstrap.events)


@pytest.mark.parametrize("cleanup_failure", ["drain", "unmap", "free"])
def test_cleanup_failure_is_collectively_reported(bootstrap, cleanup_failure):
    bootstrap.settings.update(failure="import", cleanup_failure=cleanup_failure)
    results = bootstrap.run()
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert str(results[0]) == str(results[1])
    assert "injected import" in str(
        results[0]
    ) and f"injected {cleanup_failure}" in str(results[0])
    if cleanup_failure in ("drain", "unmap"):
        assert not any(name == "free" for name, rank in bootstrap.events)
    else:
        assert not bootstrap.runtimes[0].freed and bootstrap.runtimes[1].freed


@pytest.mark.parametrize(
    "metadata",
    [
        "config",
        "nodes",
        "duplicate_gpu",
        "local_size",
        "local_ranks",
        "experts",
        "reduction",
    ],
)
def test_invalid_group_metadata_fails_before_allocation(bootstrap, metadata):
    bootstrap.settings["metadata"] = metadata
    results = bootstrap.run()
    assert all(isinstance(result, ValueError) for result in results), results
    assert str(results[0]) == str(results[1])
    assert not bootstrap.runtimes
    assert not any(name == "extension" for name, rank in bootstrap.events)


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_invalid_reduction_mode_is_collectively_reported(bootstrap, value):
    bootstrap.settings["allow_multiple_reduction"] = value
    results = bootstrap.run()
    assert all(isinstance(result, RuntimeError) for result in results)
    assert str(results[0]) == str(results[1])
    assert "allow_multiple_reduction must be bool" in str(results[0])
    assert not bootstrap.runtimes


def test_control_group_must_be_gloo(bootstrap):
    bootstrap.settings["backend"] = "nccl"
    results = bootstrap.run()
    assert all(
        isinstance(result, ValueError) and "Gloo" in str(result) for result in results
    )
    assert not bootstrap.rounds and not bootstrap.runtimes
