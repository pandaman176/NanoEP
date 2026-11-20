"""Independent PeerRuntime acceptance; never constructs CudaBuffer."""

import json
import importlib.util
import os
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist

from nano_ep.backends._build import load_extension, native_source_dir
from nano_ep.testing.oracle import reference_plan
from nano_ep.testing.wire import dispatch_offsets, dispatch_row_bytes


@lru_cache(maxsize=1)
def build_probe():
    """Build in the launcher before starting distributed workers."""
    from torch.utils.cpp_extension import load

    return load(
        name="nano_ep_peer_probe",
        sources=[str(Path(__file__).with_name("peer_probe.cu"))],
        extra_include_paths=[
            str(native_source_dir()),
            str(native_source_dir() / "include"),
        ],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo"],
    )


@lru_cache(maxsize=1)
def load_probe():
    """Load the launcher's binary without entering the extension build lock."""
    binary = Path(os.environ["NANO_EP_PEER_PROBE_PATH"])
    spec = importlib.util.spec_from_file_location(binary.name.split(".")[0], binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def delayed_barrier(runtime, window, stream, epoch, rank, waiter):
    """Observe a real arrival before letting the delayed peer enter."""
    peer = 1 - rank
    message = torch.tensor([epoch], dtype=torch.int64)
    if rank == waiter:
        runtime.barrier(stream.cuda_stream, epoch)
        completed = torch.cuda.Event()
        completed.record(stream)
        dist.send(message, peer)
        dist.recv(message, peer)
        assert message.item() == epoch
        assert not completed.query(), "barrier completed before the delayed peer arrived"
        dist.send(message, peer)
    else:
        dist.recv(message, peer)
        deadline = time.monotonic() + 10
        while window.view(torch.int64)[peer].item() != epoch:
            if time.monotonic() >= deadline:
                raise AssertionError(f"rank {peer} did not publish epoch {epoch}")
            time.sleep(0.01)
        dist.send(message, peer)
        dist.recv(message, peer)
        time.sleep(0.2)
        runtime.barrier(stream.cuda_stream, epoch)


def check_lifetime(native, probe, rank, lifetime):
    size = 2
    layout = native.window_layout(size, 3, 4, 2, size * 2)
    num_bytes = layout[-1]
    runtime = native.PeerRuntime(rank, size, num_bytes)
    handles = [None] * size
    dist.all_gather_object(handles, runtime.export_local_handle())
    assert all(isinstance(handle, bytes) and len(handle) == 64 for handle in handles)
    assert handles[0] != handles[1]
    runtime.import_peer_handles(handles)
    addresses, window = runtime.peer_addresses(), runtime.local_window()
    assert addresses.shape == (size,) and addresses.dtype == torch.int64
    assert addresses.device == window.device == torch.device("cuda", rank)
    assert window.dtype == torch.uint8 and window.numel() == num_bytes
    assert addresses[rank].item() == window.data_ptr()
    assert addresses.unique().numel() == size and torch.all(addresses != 0).item()
    assert torch.count_nonzero(window).item() == 0

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    offset, words = layout[2] // 8, 4
    for round_id in range(6):
        epoch = 2 * round_id + 1
        expected = torch.tensor(
            [[lifetime * 10000 + round_id * 100 + peer * 10 + i for i in range(words)] for peer in range(size)],
            dtype=torch.int64,
        )
        with torch.cuda.stream(stream):
            window.view(torch.int64)[offset : offset + words].copy_(expected[rank])
            if round_id in (1, 4):
                # The delayed rank polls arrivals on its idle default stream.
                torch.cuda.current_stream().synchronize()
        if round_id in (1, 4):
            delayed_barrier(runtime, window, stream, epoch, rank, waiter=round_id % 2)
        else:
            runtime.barrier(stream.cuda_stream, epoch)
        with torch.cuda.stream(stream):
            actual = probe.read_peer_words(addresses, offset, words)
        stream.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        signals = window.view(torch.int64)[:size].cpu()
        assert torch.all(signals >= epoch).item()
        assert torch.all(signals <= epoch + 1).item()
        # Readers finish before any rank overwrites the next round's payload.
        runtime.barrier(stream.cuda_stream, epoch + 1)
        stream.synchronize()

    assert torch.count_nonzero(window[layout[1] : layout[2]]).item() == 0
    assert layout[2] == layout[3]
    assert torch.count_nonzero(window[layout[2] + words * 8 : layout[4]]).item() == 0
    with_error = False
    try:
        runtime.barrier(stream.cuda_stream, 12)
    except RuntimeError as error:
        assert "strictly increase" in str(error)
        with_error = True
    assert with_error
    try:
        runtime.close()
    except RuntimeError as error:
        assert "close_peer_mappings" in str(error)
    else:
        raise AssertionError("close must reject freeing before unmapping")

    torch.cuda.synchronize()
    dist.barrier()
    runtime.close_peer_mappings()
    runtime.close_peer_mappings()
    # Allocation remains locally readable after imported mappings are closed.
    torch.testing.assert_close(window.view(torch.int64)[offset : offset + words].cpu(), expected[rank])
    for method in (
        runtime.local_window,
        runtime.peer_addresses,
        runtime.export_local_handle,
    ):
        try:
            method()
        except RuntimeError as error:
            assert "closing or closed" in str(error)
        else:
            raise AssertionError("closing runtime exposed an active resource")
    del actual, window, addresses
    torch.cuda.synchronize()
    if rank == 1:
        time.sleep(0.2)
    # No rank frees its allocation until every importer has closed its mapping.
    dist.barrier()
    runtime.close()
    runtime.close()
    runtime.close_peer_mappings()
    dist.barrier()
    print(
        f"NANO_EP_PEER_LIFETIME_OK={json.dumps({'rank': rank, 'lifetime': lifetime, 'epochs': 12})}",
        flush=True,
    )


def check_notify(native, probe, rank):
    num_experts = 260
    local_experts = num_experts // 2
    layout = native.window_layout(2, 8, 4, 2, num_experts)
    runtime = native.PeerRuntime(rank, 2, layout[-1])
    window = runtime.local_window()
    counts = window[layout[1] : layout[1] + (2 + num_experts) * 8].view(torch.int64)
    counts.fill_(-1)
    torch.cuda.synchronize()
    handles = [None, None]
    dist.all_gather_object(handles, runtime.export_local_handle())
    runtime.import_peer_handles(handles)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Rows are sources, columns are destinations; zeros are ready messages.
    rounds = (((0, 3), (7, 0)), ((2, 0), (0, 5)), ((0, 0), (0, 0))) * 2
    with torch.cuda.stream(stream):
        for round_id, matrix in enumerate(rounds):
            # All prior consumers finish before observing the reset slots.
            runtime.barrier(stream.cuda_stream)
            stream.synchronize()
            assert counts.tolist() == [-1] * (2 + num_experts)
            send = torch.tensor(matrix[rank], dtype=torch.int64, device="cuda")
            all_expert_counts = [
                [
                    matrix[source][destination] if expert in (0, local_experts - 1) else 0
                    for destination in range(2)
                    for expert in range(local_experts)
                ]
                for source in range(2)
            ]
            send_experts = torch.tensor(all_expert_counts[rank], dtype=torch.int64, device="cuda")
            peer = 1 - rank
            waiter = round_id % 2
            message = torch.tensor([round_id], dtype=torch.int64)

            # Delay alternate publishers after both ranks passed the entry barrier.
            if rank == waiter:
                received_raw = probe.run_notify_peer_counts(
                    runtime.peer_addresses(),
                    window,
                    send,
                    send_experts,
                    rank,
                    layout[1],
                )
                done = torch.cuda.Event()
                done.record(stream)
                dist.send(message, peer)
                dist.recv(message, peer)
                assert not done.query(), "notify accepted a pending slot before the peer published its count"
                dist.send(message, peer)
            else:
                dist.recv(message, peer)
                deadline = time.monotonic() + 10
                while counts[peer].item() < 0:
                    if time.monotonic() >= deadline:
                        raise AssertionError("peer did not publish its notify count")
                    time.sleep(0.01)
                assert counts[peer].item() == matrix[peer][rank]
                dist.send(message, peer)
                dist.recv(message, peer)
                time.sleep(0.1)
                received_raw = probe.run_notify_peer_counts(
                    runtime.peer_addresses(),
                    window,
                    send,
                    send_experts,
                    rank,
                    layout[1],
                )

            received = received_raw[:2]
            received_experts = received_raw[2:].view(2, local_experts).sum(0)
            expected = [matrix[source][rank] for source in range(2)]
            assert received.tolist() == expected
            expected_experts = [
                sum(all_expert_counts[source][rank * local_experts + expert] for source in range(2)) for expert in range(local_experts)
            ]
            assert received_experts.tolist() == expected_experts
            assert counts.tolist() == [-1] * (2 + num_experts)
            # Consumption completes on every peer before the next publication.
            runtime.barrier(stream.cuda_stream)
            stream.synchronize()

    torch.cuda.synchronize()
    dist.barrier()
    runtime.close_peer_mappings()
    del counts, window
    dist.barrier()
    runtime.close()
    print(f"NANO_EP_NOTIFY_RANK_OK={rank}", flush=True)


def check_payload(native, probe, rank):
    hidden, topk, capacity = 256, 3, 11
    dist.barrier()
    layout = native.window_layout(2, capacity, hidden, topk, 4)
    runtime = native.PeerRuntime(rank, 2, layout[-1])
    window = runtime.local_window()
    counts = window[layout[1] : layout[1] + 48].view(torch.int64)
    send_next = window[layout[1] + 48 : layout[1] + 64].view(torch.int64)
    counts.fill_(-1)
    torch.cuda.synchronize()
    handles = [None, None]
    dist.all_gather_object(handles, runtime.export_local_handle())
    runtime.import_peer_handles(handles)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def inputs(source, round_id, dtype, weighted):
        tokens = ((11, 7), (0, 11), (11, 11), (3, 0))[round_id][source]
        x = (torch.arange(tokens * hidden).reshape(tokens, hidden) / 8 + source * 100 + round_id * 1000).to(dtype)
        if round_id == 2:
            indices = torch.tensor([[0, 1, -1]]).repeat(tokens, 1)
        else:
            pattern = torch.tensor([[0, 1, 2], [2, 3, 0], [-1, -1, -1], [1, -1, -1]])
            indices = pattern.repeat(((tokens + 3) // 4, 1))[:tokens].contiguous()
        weights = torch.arange(tokens * topk, dtype=torch.float32).reshape(tokens, topk) / 7 + source if weighted else None
        return x, indices, weights

    def validate_payload(received, slots_cpu, sources, weighted):
        dtype = sources[0][0].dtype
        x_bytes = sources[0][0].element_size() * hidden
        row_bytes = dispatch_row_bytes(hidden, topk)
        expected = {
            (source, token): (
                x[token],
                indices[token],
                weights[token] if weighted else None,
            )
            for source, (x, indices, weights) in enumerate(sources)
            for token in range(x.shape[0])
            if any(expert >= 0 and expert // 2 == rank for expert in indices[token].tolist())
        }
        expected_sends = [
            sum(any(e >= 0 and e // 2 == destination for e in choices) for choices in sources[rank][1].tolist()) for destination in range(2)
        ]
        assert received[layout[1] + 48 : layout[1] + 64].view(torch.int64).tolist() == expected_sends
        assert received[layout[1] : layout[1] + 48].view(torch.int64).tolist() == [-1] * 6
        for token, experts in enumerate(sources[rank][1].tolist()):
            destinations = {expert // 2 for expert in experts if expert >= 0}
            for destination in range(2):
                slot = slots_cpu[token, destination].item()
                assert rank * capacity <= slot < (rank + 1) * capacity if destination in destinations else slot == -1
        seen = set()
        published = [
            source * capacity + local_slot for source in range(2) for local_slot in range(sum(key[0] == source for key in expected))
        ]
        for slot in published:
            row = received[layout[2] + slot * row_bytes : layout[2] + (slot + 1) * row_bytes]
            key = tuple(row[:16].view(torch.int64).tolist())
            assert key in expected and key not in seen
            assert key[0] == slot // capacity
            seen.add(key)
            wanted_x, wanted_indices, wanted_weights = expected[key]
            torch.testing.assert_close(
                row[dispatch_offsets(topk)[0] : dispatch_offsets(topk)[0] + 8 * topk].view(torch.int64),
                wanted_indices,
            )
            if weighted:
                torch.testing.assert_close(
                    row[dispatch_offsets(topk)[1] : dispatch_offsets(topk)[1] + 4 * topk].view(torch.float32),
                    wanted_weights,
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                row[dispatch_offsets(topk)[2] : dispatch_offsets(topk)[2] + x_bytes].view(dtype),
                wanted_x,
                rtol=0,
                atol=0,
            )
            if key[0] == rank:
                assert slots_cpu[key[1], rank].item() == slot
        assert seen == set(expected)
        return len(expected)

    for round_id, (dtype, weighted) in enumerate(
        (
            (torch.bfloat16, False),
            (torch.bfloat16, True),
            (torch.bfloat16, False),
            (torch.bfloat16, True),
        )
    ):
        sources = [inputs(source, round_id, dtype, weighted) for source in range(2)]
        source_x, source_indices, source_weights = sources[rank]
        plan = reference_plan(source_indices, 4, 2)
        with torch.cuda.stream(stream):
            x, indices = source_x.cuda(), source_indices.cuda()
            weights = source_weights.cuda() if weighted else None
            send_counts, expert_counts = (
                plan.send_counts.cuda(),
                plan.expert_counts.cuda(),
            )
            send_next.zero_()
            runtime.barrier(stream.cuda_stream)
            received_raw = probe.run_notify_peer_counts(
                runtime.peer_addresses(),
                window,
                send_counts,
                expert_counts,
                rank,
                layout[1],
            )
            received_counts = received_raw[:2]
            received_experts = received_raw[2:].view(2, 2).sum(0)
            slots = torch.full((x.shape[0], 2), 0x5A5A, dtype=torch.int64, device="cuda")
            snapshot = torch.full_like(indices, 0x5A5A)
            probe.run_dispatch_payload(
                runtime.peer_addresses(),
                x,
                indices,
                snapshot,
                weights,
                slots,
                rank,
                4,
                capacity,
                False,
            )
            runtime.barrier(stream.cuda_stream)
            received, slots_cpu = window.cpu(), slots.cpu()
            torch.testing.assert_close(snapshot.cpu(), source_indices, rtol=0, atol=0)
            expected_count = received_counts.sum().item()
            expected_experts = [sum(int((source_idx == rank * 2 + expert).sum()) for _, source_idx, _ in sources) for expert in range(2)]
            assert received_experts.tolist() == expected_experts
        assert validate_payload(received, slots_cpu, sources, weighted) == expected_count

        # Keep the old routing/slots and replace payloads, toggling weights.
        cached_sources = [
            (
                x + 64,
                idx,
                None if weighted else torch.full(idx.shape, source + 0.375, dtype=torch.float32),
            )
            for source, (x, idx, _) in enumerate(sources)
        ]
        with torch.cuda.stream(stream):
            cached_x, _, cached_weights = cached_sources[rank]
            x = cached_x.cuda()
            weights = cached_weights.cuda() if cached_weights is not None else None
            runtime.barrier(stream.cuda_stream)
            probe.run_dispatch_payload(
                runtime.peer_addresses(),
                x,
                indices,
                snapshot,
                weights,
                slots,
                rank,
                4,
                capacity,
                True,
            )
            runtime.barrier(stream.cuda_stream)
            cached_received, cached_slots = window.cpu(), slots.cpu()
        torch.testing.assert_close(cached_slots, slots_cpu)
        assert validate_payload(cached_received, cached_slots, cached_sources, not weighted) == expected_count

    torch.cuda.synchronize()
    dist.barrier()
    runtime.close_peer_mappings()
    del counts, send_next, window
    dist.barrier()
    runtime.close()
    print(f"NANO_EP_PAYLOAD_RANK_OK={rank}", flush=True)


def run():
    rank, size, local = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    assert size == int(os.environ["LOCAL_WORLD_SIZE"]) == 2
    torch.cuda.set_device(local)
    native, probe = load_extension(), load_probe()
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        for lifetime in range(2):
            check_lifetime(native, probe, rank, lifetime)
        check_notify(native, probe, rank)
        check_payload(native, probe, rank)
        print(f"NANO_EP_PEER_RANK_OK={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
