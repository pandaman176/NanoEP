"""Independent PeerRuntime acceptance; never constructs CudaBuffer."""

from datetime import timedelta
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from nano_ep.backends._build import load_extension


@lru_cache(maxsize=1)
def load_probe():
    from torch.utils.cpp_extension import load

    return load(
        name="nano_ep_peer_probe",
        sources=[str(Path(__file__).with_name("peer_probe.cu"))],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo"],
    )


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
        assert not completed.query(), (
            "barrier completed before the delayed peer arrived"
        )
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
            [
                [
                    lifetime * 10000 + round_id * 100 + peer * 10 + i
                    for i in range(words)
                ]
                for peer in range(size)
            ],
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
    torch.testing.assert_close(
        window.view(torch.int64)[offset : offset + words].cpu(), expected[rank]
    )
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


def check_notify(native, rank):
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
                    matrix[source][destination]
                    if expert in (0, local_experts - 1)
                    else 0
                    for destination in range(2)
                    for expert in range(local_experts)
                ]
                for source in range(2)
            ]
            send_experts = torch.tensor(
                all_expert_counts[rank], dtype=torch.int64, device="cuda"
            )
            peer = 1 - rank
            waiter = round_id % 2
            message = torch.tensor([round_id], dtype=torch.int64)

            # Delay alternate publishers after both ranks passed the entry barrier.
            if rank == waiter:
                received, received_experts = native.notify_counts(
                    runtime, send, send_experts, layout[1]
                )
                done = torch.cuda.Event()
                done.record(stream)
                dist.send(message, peer)
                dist.recv(message, peer)
                assert not done.query(), (
                    "notify accepted a pending slot before the peer published its count"
                )
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
                received, received_experts = native.notify_counts(
                    runtime, send, send_experts, layout[1]
                )

            expected = [matrix[source][rank] for source in range(2)]
            assert received.tolist() == expected
            expected_experts = [
                sum(
                    all_expert_counts[source][rank * local_experts + expert]
                    for source in range(2)
                )
                for expert in range(local_experts)
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


def check_payload(native, rank):
    hidden, topk, capacity = 37, 3, 11
    layout = native.window_layout(2, capacity, hidden, topk, 4)
    runtime = native.PeerRuntime(rank, 2, layout[-1])
    window = runtime.local_window()
    counts = window[layout[1] : layout[1] + 48].view(torch.int64)
    recv_next = window[layout[1] + 48 : layout[1] + 56].view(torch.int64)
    counts.fill_(-1)
    torch.cuda.synchronize()
    handles = [None, None]
    dist.all_gather_object(handles, runtime.export_local_handle())
    runtime.import_peer_handles(handles)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    row_bytes = (16 + 12 * topk + 4 * hidden + 15) // 16 * 16

    def inputs(source, round_id, dtype, weighted):
        tokens = ((11, 7), (0, 11), (11, 11), (3, 0))[round_id][source]
        x = (
            torch.arange(tokens * hidden).reshape(tokens, hidden) / 8
            + source * 100
            + round_id * 1000
        ).to(dtype)
        if round_id == 2:
            indices = torch.tensor([[0, 1, -1]]).repeat(tokens, 1)
        else:
            pattern = torch.tensor([[0, 1, 2], [2, 3, 0], [-1, -1, -1], [1, -1, -1]])
            indices = pattern.repeat(((tokens + 3) // 4, 1))[:tokens].contiguous()
        weights = (
            torch.arange(tokens * topk, dtype=torch.float32).reshape(tokens, topk) / 7
            + source
            if weighted
            else None
        )
        return x, indices, weights

    def validate_payload(received, slots_cpu, sources, weighted):
        expected = {
            (source, token): (
                x[token].float(),
                indices[token],
                weights[token] if weighted else None,
            )
            for source, (x, indices, weights) in enumerate(sources)
            for token in range(x.shape[0])
            if any(
                expert >= 0 and expert // 2 == rank
                for expert in indices[token].tolist()
            )
        }
        assert received[layout[1] + 48 : layout[1] + 56].view(
            torch.int64
        ).item() == len(expected)
        assert (
            received[layout[1] : layout[1] + 48].view(torch.int64).tolist() == [-1] * 6
        )
        for token, experts in enumerate(sources[rank][1].tolist()):
            destinations = {expert // 2 for expert in experts if expert >= 0}
            for destination in range(2):
                slot = slots_cpu[token, destination].item()
                assert (
                    0 <= slot < 2 * capacity
                    if destination in destinations
                    else slot == -1
                )
        seen = set()
        for slot in range(len(expected)):
            row = received[
                layout[2] + slot * row_bytes : layout[2] + (slot + 1) * row_bytes
            ]
            key = tuple(row[:16].view(torch.int64).tolist())
            assert key in expected and key not in seen
            seen.add(key)
            wanted_x, wanted_indices, wanted_weights = expected[key]
            torch.testing.assert_close(
                row[16 : 16 + 8 * topk].view(torch.int64), wanted_indices
            )
            if weighted:
                torch.testing.assert_close(
                    row[16 + 8 * topk : 16 + 12 * topk].view(torch.float32),
                    wanted_weights,
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                row[16 + 12 * topk : 16 + 12 * topk + 4 * hidden].view(torch.float32),
                wanted_x,
                rtol=0,
                atol=0,
            )
            if key[0] == rank:
                assert slots_cpu[key[1], rank].item() == slot
        assert seen == set(expected)
        return len(expected)

    def validate_metadata(metadata, received, expert_counts, offsets):
        src_ranks, src_tokens, local_indices, expanded_slots = metadata
        recv_count = src_ranks.numel()
        assert src_tokens.shape == (recv_count,)
        assert local_indices.shape == expanded_slots.shape == (recv_count, topk)
        expected_local = torch.full((recv_count, topk), -1, dtype=torch.int64)
        for slot in range(recv_count):
            row = received[
                layout[2] + slot * row_bytes : layout[2] + (slot + 1) * row_bytes
            ]
            assert [src_ranks[slot].item(), src_tokens[slot].item()] == row[:16].view(
                torch.int64
            ).tolist()
            for k, expert in enumerate(
                row[16 : 16 + 8 * topk].view(torch.int64).tolist()
            ):
                if rank * 2 <= expert < (rank + 1) * 2:
                    expected_local[slot, k] = expert - rank * 2
        torch.testing.assert_close(local_indices, expected_local)
        assert (expanded_slots[expected_local < 0] == -1).all()
        for expert, count in enumerate(expert_counts):
            start = offsets[expert].item()
            torch.testing.assert_close(
                expanded_slots[expected_local == expert].sort().values,
                torch.arange(start, start + count),
            )

    for round_id, (dtype, weighted) in enumerate(
        (
            (torch.float32, False),
            (torch.bfloat16, True),
            (torch.bfloat16, False),
            (torch.float32, True),
        )
    ):
        sources = [inputs(source, round_id, dtype, weighted) for source in range(2)]
        source_x, source_indices, source_weights = sources[rank]
        with torch.cuda.stream(stream):
            x, indices = source_x.cuda(), source_indices.cuda()
            weights = source_weights.cuda() if weighted else None
            recv_next.zero_()
            runtime.barrier(stream.cuda_stream)
            send_counts, expert_counts = native.route_counts(indices, 4, 2)
            received_counts, received_experts = native.notify_counts(
                runtime, send_counts, expert_counts, layout[1]
            )
            slots = native.reserve_and_put(runtime, x, indices, weights, 4, capacity)
            runtime.barrier(stream.cuda_stream)
            received, slots_cpu = window.cpu(), slots.cpu()
            expected_count = received_counts.sum().item()
            expected_experts = [
                sum(
                    int((source_idx == rank * 2 + expert).sum())
                    for _, source_idx, _ in sources
                )
                for expert in range(2)
            ]
            assert received_experts.tolist() == expected_experts
            offsets = torch.zeros(3, dtype=torch.int64, device=x.device)
            offsets[1:] = ((received_experts + 3) // 4 * 4).cumsum(0)
            stage_buffer = window.narrow(0, layout[2], expected_count * row_bytes).view(
                expected_count, row_bytes
            )
            metadata = native.build_receive_metadata(
                stage_buffer, offsets, rank, 2, topk
            )
            metadata_cpu = [tensor.cpu() for tensor in metadata]
            offsets_cpu = offsets.cpu()
        assert (
            validate_payload(received, slots_cpu, sources, weighted) == expected_count
        )
        validate_metadata(metadata_cpu, received, expected_experts, offsets_cpu)

        # Keep the old routing/slots and replace payloads, toggling weights.
        cached_sources = [
            (
                x + 64,
                idx,
                None
                if weighted
                else torch.full(idx.shape, source + 0.375, dtype=torch.float32),
            )
            for source, (x, idx, _) in enumerate(sources)
        ]
        with torch.cuda.stream(stream):
            cached_x, _, cached_weights = cached_sources[rank]
            x = cached_x.cuda()
            weights = cached_weights.cuda() if cached_weights is not None else None
            runtime.barrier(stream.cuda_stream)
            reused = native.reserve_and_put(
                runtime, x, indices, weights, 4, capacity, cached_dst_slots=slots
            )
            runtime.barrier(stream.cuda_stream)
            cached_received, cached_slots = window.cpu(), reused.cpu()
            metadata_cpu = [tensor.cpu() for tensor in metadata]
        assert reused.data_ptr() == slots.data_ptr()
        torch.testing.assert_close(cached_slots, slots_cpu)
        assert (
            validate_payload(
                cached_received, cached_slots, cached_sources, not weighted
            )
            == expected_count
        )
        validate_metadata(metadata_cpu, cached_received, expected_experts, offsets_cpu)

    torch.cuda.synchronize()
    dist.barrier()
    runtime.close_peer_mappings()
    del stage_buffer, counts, recv_next, window
    dist.barrier()
    runtime.close()
    print(f"NANO_EP_PAYLOAD_RANK_OK={rank}", flush=True)


def check_dispatch_main(native, rank):
    from dataclasses import replace

    from nano_ep.testing.cases import make_case
    from nano_ep.testing.native_dispatch import (
        assert_dispatch_output,
        assert_dispatch_state,
        prepare_dispatch_metadata,
    )

    def check_copy(state, case, weights, alignment=4):
        for expand, zero_padding in ((False, False), (True, False), (True, True)):
            outputs = native.dispatch_epilogue(
                state, expand, weights is not None, zero_padding
            )
            assert_dispatch_output(
                state,
                outputs,
                case,
                rank,
                alignment,
                weights,
                expand=expand,
                zero_padding=zero_padding,
            )

    layout = native.window_layout(2, 5, 4, 3, 6)
    runtime = native.PeerRuntime(rank, 2, layout[-1])
    window = runtime.local_window()
    window[layout[1] : layout[1] + 64].view(torch.int64).fill_(-1)
    torch.cuda.synchronize()
    handles = [None, None]
    dist.all_gather_object(handles, runtime.export_local_handle())
    runtime.import_peer_handles(handles)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for kind, dtype in (
        ("mixed", torch.float32),
        ("empty", torch.bfloat16),
        ("masked", torch.float32),
        ("skew", torch.bfloat16),
    ):
        case = make_case(2, kind=kind, dtype=dtype)
        with torch.cuda.stream(stream):
            x, indices, weights = (
                case.xs[rank].cuda(),
                case.indices[rank].cuda(),
                case.weights[rank].cuda(),
            )
            state = native.dispatch_main(runtime, x, indices, weights, 6, 5, 4, [])
            assert not state.metadata_prepared and state.rank == rank
            prepare_dispatch_metadata(native, state)
            assert_dispatch_state(state, case, rank, 4, case.weights)
            check_copy(state, case, case.weights)
            indices.fill_(99)
            torch.testing.assert_close(state.metadata[8].cpu(), case.indices[rank])
            saved_metadata = [tensor.cpu().clone() for tensor in state.metadata]

            # Reuse an older handle after a fresh operation with a different receive count.
            masked = replace(
                case, indices=[torch.full_like(idx, -1) for idx in case.indices]
            )
            fresh = native.dispatch_main(
                runtime, x, masked.indices[rank].cuda(), None, 6, 5, 1, []
            )
            prepare_dispatch_metadata(native, fresh)
            assert_dispatch_state(fresh, masked, rank, 1, None)
            check_copy(fresh, masked, None, alignment=1)
            replay_case = replace(case, xs=[x * 2 for x in case.xs])
            replay = native.dispatch_main(
                runtime, x * 2, indices, None, 6, 5, 4, state.metadata
            )
            assert replay.metadata_prepared
            prepare_dispatch_metadata(native, replay)
            assert_dispatch_state(replay, replay_case, rank, 4, None)
            check_copy(replay, replay_case, None)
            replay_case = replace(
                case, xs=[x * 3 for x in case.xs], weights=[w * 4 for w in case.weights]
            )
            replay = native.dispatch_main(
                runtime, x * 3, indices, weights * 4, 6, 5, 4, state.metadata
            )
            prepare_dispatch_metadata(native, replay)
            assert_dispatch_state(replay, replay_case, rank, 4, replay_case.weights)
            check_copy(replay, replay_case, replay_case.weights)
            for actual, expected in zip(replay.metadata, saved_metadata, strict=True):
                torch.testing.assert_close(actual.cpu(), expected)
            assert window[layout[1] + 64 : layout[1] + 72].view(torch.int64).item() == 0
            assert (
                (window[layout[1] : layout[1] + 64].view(torch.int64) == -1)
                .all()
                .item()
            )
    torch.cuda.synchronize()
    dist.barrier()
    runtime.close_peer_mappings()
    del state, fresh, replay, window
    dist.barrier()
    runtime.close()
    print(f"NANO_EP_DISPATCH_MAIN_RANK_OK={rank}", flush=True)
    print(f"NANO_EP_DISPATCH_COPY_RANK_OK={rank}", flush=True)


def check_combine_main(native, rank):
    from itertools import product

    from nano_ep.testing.native_combine import (
        assert_combined_output,
        assert_return_packets,
        expert_inputs,
        make_return_case,
    )

    for mode in (False, True):
        layout = native.window_layout(2, 5, 33, 3, 6, mode)
        runtime = native.PeerRuntime(rank, 2, layout[-1])
        handles = [None, None]
        dist.all_gather_object(handles, runtime.export_local_handle())
        runtime.import_peer_handles(handles)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for dtype, expand, weighted, kind in product(
                (torch.float32, torch.bfloat16),
                (False, True),
                (False, True),
                ("mixed", "empty", "masked"),
            ):
                case = make_return_case(2, 3, 33, dtype, kind=kind)
                x, weights, metadata = expert_inputs(case, rank, expand)
                runtime.local_window()[layout[3] :].fill_(165)
                device_x, device_weights = x.cuda(), weights.cuda()
                device_metadata = [tensor.cuda() for tensor in metadata]
                returned = native.combine_main(
                    runtime,
                    device_x,
                    device_metadata,
                    device_weights if weighted else None,
                    case.capacity,
                    expand,
                    mode,
                )
                assert_return_packets(returned, case, rank, expand, mode, weighted)
                bias = (
                    torch.full(
                        (len(case.indices[rank]), case.hidden),
                        rank + 0.125,
                        dtype=dtype,
                    )
                    if weighted
                    else None
                )
                combined = native.combine_epilogue(
                    returned,
                    device_metadata[8],
                    case.hidden,
                    dtype,
                    case.num_experts,
                    2,
                    case.capacity,
                    bias.cuda() if bias is not None else None,
                    weighted,
                    expand,
                    mode,
                )
                assert_combined_output(
                    combined, case, rank, expand, mode, weighted, bias
                )
                torch.testing.assert_close(
                    device_x.cpu(), x, rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    device_weights.cpu(), weights, rtol=0, atol=0, equal_nan=True
                )
        torch.cuda.synchronize()
        dist.barrier()
        runtime.close_peer_mappings()
        del returned
        dist.barrier()
        runtime.close()
    print(f"NANO_EP_COMBINE_MAIN_RANK_OK={rank}", flush=True)
    print(f"NANO_EP_COMBINE_EPILOGUE_RANK_OK={rank}", flush=True)


def run():
    rank, size, local = (
        int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    assert size == int(os.environ["LOCAL_WORLD_SIZE"]) == 2
    torch.cuda.set_device(local)
    native, probe = load_extension(), load_probe()
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        for lifetime in range(2):
            check_lifetime(native, probe, rank, lifetime)
        check_notify(native, rank)
        check_payload(native, rank)
        check_dispatch_main(native, rank)
        check_combine_main(native, rank)
        print(f"NANO_EP_PEER_RANK_OK={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
