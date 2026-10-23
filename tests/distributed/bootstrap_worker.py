"""CudaBuffer initialization/rollback acceptance without dispatch or combine."""

from datetime import timedelta
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from nano_ep import EPConfig
import nano_ep.backends.cuda as cuda
from nano_ep.backends._build import load_extension


def run():
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    assert size == int(os.environ["NANO_EP_EXPECTED_WORLD"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    try:
        native = load_extension()
        config = EPConfig(size * 2, 2, 128, 8)

        for failure in ("allocate", "export", "import"):

            def create_runtime(
                actual_rank, actual_size, num_bytes, failure_mode=failure
            ):
                if rank == 0 and failure_mode == "allocate":
                    raise RuntimeError("injected allocate")
                runtime = native.PeerRuntime(actual_rank, actual_size, num_bytes)

                def export():
                    if rank == 0 and failure_mode == "export":
                        raise RuntimeError("injected export")
                    return runtime.export_local_handle()

                def import_handles(handles):
                    if rank == 0 and failure_mode == "import":
                        raise RuntimeError("injected import")
                    runtime.import_peer_handles(handles)

                return SimpleNamespace(
                    local_window=runtime.local_window,
                    export_local_handle=export,
                    import_peer_handles=import_handles,
                    close_peer_mappings=runtime.close_peer_mappings,
                    close=runtime.close,
                )

            backend = SimpleNamespace(
                window_layout=native.window_layout, PeerRuntime=create_runtime
            )
            error = None
            with patch.object(cuda, "load_extension", return_value=backend):
                try:
                    cuda.CudaBuffer(dist.group.WORLD, config)
                except RuntimeError as exc:
                    error = str(exc)
            errors = [None] * size
            dist.all_gather_object(errors, error)
            assert all(item == errors[0] and item is not None for item in errors), (
                errors
            )
            assert f"rank 0: RuntimeError: injected {failure}" in errors[0], errors
            print(f"NANO_EP_BOOTSTRAP_ROLLBACK_OK={rank}:{failure}", flush=True)

        for lifetime in range(2):
            mode = bool(lifetime)
            with cuda.CudaBuffer(
                dist.group.WORLD, config, allow_multiple_reduction=mode
            ) as buffer:
                assert buffer._rank == rank and buffer._world_size == size
                assert buffer._allow_multiple_reduction is mode
                assert (
                    buffer._num_bytes
                    == native.window_layout(size, 8, 128, 2, config.num_experts, mode)[
                        -1
                    ]
                )
                addresses = buffer._runtime.peer_addresses()
                window = buffer._runtime.local_window()
                assert addresses[rank].item() == window.data_ptr()
                counts = window[
                    buffer._count_offset : buffer._count_offset
                    + (size + config.num_experts) * 8
                ].view(torch.int64)
                assert counts.tolist() == [-1] * (size + config.num_experts)
                assert (
                    window[
                        buffer._count_offset
                        + (size + config.num_experts) * 8 : buffer._count_offset
                        + (size + config.num_experts + 1) * 8
                    ]
                    .view(torch.int64)
                    .item()
                    == 0
                )
                del addresses, window, counts
                with torch.cuda.stream(buffer._comm_stream):
                    for epoch in (1, 2, 3):
                        buffer._runtime.barrier(buffer._comm_stream.cuda_stream, epoch)
                buffer._comm_stream.synchronize()
            buffer.close()
            print(f"NANO_EP_BOOTSTRAP_LIFETIME_OK={rank}:{lifetime}", flush=True)
        print(f"NANO_EP_BOOTSTRAP_RANK_OK={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
