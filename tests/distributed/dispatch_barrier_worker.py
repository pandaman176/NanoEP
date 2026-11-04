"""Check fused dispatch barriers with delayed readers and multiple CTA rounds."""

import os
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
from peer_worker import load_probe

from nano_ep.backends._build import (
    load_extension,
    prepare_dispatch_epilogue,
    prepare_dispatch_main,
)
from nano_ep.testing.cases import Case
from nano_ep.testing.native_dispatch import assert_dispatch_output


def make_case(size, tokens, kind):
    xs, indices, weights = [], [], []
    for source in range(size):
        count = (
            0
            if kind == "all_empty" or kind == "one_empty" and source == size - 1
            else tokens
        )
        row = torch.arange(count)
        x = (row[:, None] % 127 + torch.arange(256)[None, :] % 7 + source * 11).to(
            torch.bfloat16
        )
        routing = torch.stack(
            (
                source * 3 + row % 3,
                source * 3 + (row + 1) % 3,
                ((source + 1) % size) * 3 + row % 3,
            ),
            dim=1,
        )
        routing[::5, -1] = -1
        routing[::7] = -1
        if kind == "masked":
            routing.fill_(-1)
        xs.append(x)
        indices.append(routing)
        weights.append((torch.arange(count * 3).reshape(count, 3).float() % 11 - 5) / 8)
    return Case(xs, indices, weights, torch.empty(0), size * 3, 3, 256, tokens)


def run():
    rank, size, local = (
        int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    native, probe = load_extension(), load_probe()
    capacity, alignment = 2153, 4
    module = prepare_dispatch_main(256, 3, size, 3, capacity, alignment)
    epilogues = tuple(
        prepare_dispatch_epilogue(256, 3, 3, cached_mode=mode).dispatch_epilogue
        for mode in (False, True)
    )
    layout = native.window_layout(size, capacity, 256, 3, size * 3)
    runtime = native.PeerRuntime(rank, size, layout[-1])
    handles = [None] * size
    dist.all_gather_object(handles, runtime.export_local_handle())
    runtime.import_peer_handles(handles)
    window = runtime.local_window()
    window[layout[1] : layout[1] + (size + size * 3) * 8].fill_(255)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    epoch, checks = 0, 0

    # Warm both cooperative variants before queuing any GPU-side arrival gate.
    with torch.cuda.stream(stream):
        empty_x = torch.empty((0, 256), dtype=torch.bfloat16, device="cuda")
        empty_indices = torch.empty((0, 3), dtype=torch.int64, device="cuda")
        warm = native.dispatch_main(
            runtime,
            empty_x,
            empty_indices,
            None,
            size * 3,
            capacity,
            alignment,
            [],
            0,
            False,
            module.launcher,
        )
        epilogues[False](warm, False, False, False)
        warm = native.dispatch_main(
            runtime,
            empty_x,
            empty_indices,
            None,
            size * 3,
            capacity,
            alignment,
            warm.metadata,
            0,
            False,
            module.launcher,
        )
        epilogues[True](warm, False, False, False)
    epoch += 4
    stream.synchronize()
    dist.barrier()
    del warm

    for expand in (False, True):
        for kind in ("mixed", "one_empty", "masked", "all_empty"):
            case = make_case(size, capacity, kind)
            metadata, expanded_capacity = [], 0
            for step in range(3):
                cached = step != 0
                current = replace(
                    case,
                    xs=[x + 8 * step for x in case.xs],
                    weights=[w + step for w in case.weights],
                )
                with torch.cuda.stream(stream):
                    x = current.xs[rank].cuda()
                    indices = case.indices[rank].cuda()
                    weights = current.weights[rank].cuda()
                stream.synchronize()
                dist.barrier()
                with torch.cuda.stream(stream):
                    arena = window[layout[2] :]
                    arena.fill_(165)
                    if rank == 0:
                        before = arena.clone()
                        # Peer 1 publishes entry while rank 0's old reader is pending.
                        probe.wait_for_arrival(window, 1, epoch + 1)
                        after = arena.clone()
                    state = native.dispatch_main(
                        runtime,
                        x,
                        indices,
                        weights,
                        size * 3,
                        capacity,
                        alignment,
                        metadata if cached else [],
                        expanded_capacity if cached else 0,
                        False,
                        module.launcher,
                    )
                    outputs = epilogues[cached](state, expand, True, expand)
                    if not cached:
                        metadata, expanded_capacity = (
                            state.metadata,
                            int(state.counts[1]),
                        )
                    # The epilogue must see every CTA and every peer's completed writes.
                    assert_dispatch_output(
                        state,
                        outputs,
                        current,
                        rank,
                        alignment,
                        current.weights,
                        expand=expand,
                        zero_padding=expand,
                    )
                    if rank == 0:
                        torch.testing.assert_close(after, before, rtol=0, atol=0)
                stream.synchronize()
                epoch += 2
                dist.barrier()
                assert window[: size * 8].view(torch.int64).tolist() == [epoch] * size
                dist.barrier()
                checks += 1
            # Standalone barriers and fused phases share one advancing epoch sequence.
            runtime.barrier(stream.cuda_stream, epoch + 3)
            epoch += 3
            stream.synchronize()
            dist.barrier()

    stream.synchronize()
    dist.barrier()
    runtime.close_peer_mappings()
    del state, outputs, arena, window
    dist.barrier()
    runtime.close()
    print(f"NANO_EP_DISPATCH_BARRIER_RANK_OK={rank}:{checks}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
