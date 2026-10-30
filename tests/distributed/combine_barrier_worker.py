"""Exercise fused combine barriers, empty peers and arena reuse on real GPUs."""

import os
from datetime import timedelta
from itertools import product

import torch
import torch.distributed as dist
from peer_worker import load_probe

from nano_ep.backends._build import load_extension, prepare_combine_main
from nano_ep.testing.native_combine import (
    assert_return_packets,
    expert_inputs,
    make_return_case,
)


def run():
    rank, size, local = (
        int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    native, probe = load_extension(), load_probe()
    hidden, topk, capacity = 256, 3, 2153
    specialization = prepare_combine_main(hidden, topk, size, 3, capacity)
    stream = torch.cuda.Stream()
    checks = 0
    for mode in (False, True):
        layout = native.window_layout(size, capacity, hidden, topk, size * 3, mode)
        runtime = native.PeerRuntime(rank, size, layout[-1])
        handles = [None] * size
        dist.all_gather_object(handles, runtime.export_local_handle())
        runtime.import_peer_handles(handles)
        window = runtime.local_window()
        stream.wait_stream(torch.cuda.current_stream())
        epoch = 0
        for expand, weighted in product((False, True), repeat=2):
            # Warm the exact kernel variant before queuing a GPU-side wait.
            empty = make_return_case(size, topk, hidden, torch.bfloat16, tokens=0)
            empty.capacity = capacity
            with torch.cuda.stream(stream):
                x, w, metadata = expert_inputs(empty, rank, expand)
                device_metadata = [tensor.cuda() for tensor in metadata]
                native.combine_main(
                    runtime,
                    x.cuda(),
                    device_metadata,
                    w.cuda() if weighted else None,
                    capacity,
                    expand,
                    mode,
                    launcher=specialization.launcher,
                )
            stream.synchronize()
            epoch += 2
            dist.barrier()
            for kind in ("mixed", "one_receiver", "masked"):
                case = make_return_case(
                    size, topk, hidden, torch.bfloat16, tokens=capacity, kind=kind
                )
                if kind == "one_receiver":
                    for routing in case.indices:
                        routing[:] = torch.arange(topk) + (size - 1) * 3
                x, w, metadata = expert_inputs(case, rank, expand)
                if kind == "one_receiver":
                    assert metadata[0].numel() == (
                        size * capacity if rank == size - 1 else 0
                    )
                elif kind == "masked":
                    assert metadata[0].numel() == 0
                with torch.cuda.stream(stream):
                    device_x, device_w = x.cuda(), w.cuda()
                    device_metadata = [tensor.cuda() for tensor in metadata]
                    arena = window[layout[3] :]
                    arena.fill_(165)
                    if rank == 0:
                        before = arena.clone()
                        # Peer 1 must publish its entry before the old reader ends.
                        probe.wait_for_arrival(window, 1, epoch + 1)
                        after = arena.clone()
                    returned = native.combine_main(
                        runtime,
                        device_x,
                        device_metadata,
                        device_w if weighted else None,
                        capacity,
                        expand,
                        mode,
                        launcher=specialization.launcher,
                    )
                    # Kernel completion must include writes from every local CTA
                    # and every peer, including peers that have no receive rows.
                    assert_return_packets(returned, case, rank, expand, mode, weighted)
                    if rank == 0:
                        torch.testing.assert_close(after, before, rtol=0, atol=0)
                    epoch += 2
                    assert (
                        window[: size * 8].view(torch.int64).tolist() == [epoch] * size
                    )
                dist.barrier()
                # Standalone dispatch-style phases share the same epoch sequence.
                runtime.barrier(stream.cuda_stream, epoch + 3)
                epoch += 3
                stream.synchronize()
                checks += 1
                dist.barrier()
        stream.synchronize()
        dist.barrier()
        runtime.close_peer_mappings()
        del returned, arena, window
        dist.barrier()
        runtime.close()
    print(f"NANO_EP_COMBINE_BARRIER_RANK_OK={rank}:{checks}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
