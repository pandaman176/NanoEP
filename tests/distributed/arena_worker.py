"""Exercise shared-arena ownership and delayed readers with real CUDA IPC."""

from dataclasses import replace
from datetime import timedelta
from itertools import product
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nano_ep import EPConfig
from nano_ep.backends.cuda import CudaBuffer
from nano_ep.testing.cases import Case
from nano_ep.testing.oracle import reference_dispatch
from peer_worker import load_probe


def make_case(size, topk, dtype, step):
    generator = torch.Generator().manual_seed(71 + step)
    xs, indices, weights = [], [], []
    experts, hidden, capacity = size * 8, 65, 11
    for rank in range(size):
        tokens = 7 - rank % 3 if step == 0 else 4 + rank % 3
        if step == 1 and rank == size - 1:
            tokens = 0
        xs.append(torch.full((tokens, hidden), rank + 1 + step * 8, dtype=dtype))
        idx = (
            torch.stack(
                [
                    torch.randperm(experts, generator=generator)[:topk]
                    for _ in range(tokens)
                ]
            )
            if tokens
            else torch.empty((0, topk), dtype=torch.int64)
        )
        if tokens:
            idx[0] = -1
            idx[1::2, -1] = -1
        indices.append(idx)
        weights.append(
            (torch.arange(tokens * topk).view(tokens, topk).float() % 9 - 4) / 8
        )
    return Case(xs, indices, weights, torch.empty(0), experts, topk, hidden, capacity)


def check_dispatch(got, case, rank, expand, weighted):
    expected = reference_dispatch(
        case.xs,
        case.indices,
        case.num_experts,
        case.weights if weighted else None,
        do_expand=expand,
        alignment=16,
    )[rank]
    layout, ref = got.handle.layout, expected.layout
    order = (layout.src_ranks * case.capacity + layout.src_tokens).argsort()
    for name in ("src_ranks", "src_tokens", "local_topk_idx"):
        torch.testing.assert_close(
            getattr(layout, name)[order].cpu(), getattr(ref, name)
        )
    for name in ("expert_counts", "expert_offsets"):
        torch.testing.assert_close(getattr(layout, name).cpu(), getattr(ref, name))
    assert (layout.recv_count, layout.expanded_capacity) == (
        ref.recv_count,
        ref.expanded_capacity,
    )
    if expand:
        actual_slots = layout.expanded_slots[order].cpu()
        mask = ref.expanded_slots >= 0
        assert torch.equal(actual_slots >= 0, mask)
        actual, target = actual_slots[mask].cuda(), ref.expanded_slots[mask]
        torch.testing.assert_close(
            got.x[actual].cpu(), expected.x[target], rtol=0, atol=0
        )
        padding = torch.ones(layout.expanded_capacity, dtype=torch.bool, device="cuda")
        padding[actual] = False
        assert torch.count_nonzero(got.x[padding]).item() == 0
        if weighted:
            torch.testing.assert_close(
                got.topk_weights[actual].cpu(),
                expected.topk_weights[target],
                rtol=0,
                atol=0,
            )
            assert torch.count_nonzero(got.topk_weights[padding]).item() == 0
    else:
        torch.testing.assert_close(got.x[order].cpu(), expected.x, rtol=0, atol=0)
        torch.testing.assert_close(got.topk_idx[order].cpu(), expected.topk_idx)
        if weighted:
            torch.testing.assert_close(
                got.topk_weights[order].cpu(), expected.topk_weights, rtol=0, atol=0
            )
    if not weighted:
        assert got.topk_weights is None


def check_combine(got, case, rank, expand, weighted):
    multiplicity = [
        sum(e >= 0 for e in row) if expand else len({e // 8 for e in row if e >= 0})
        for row in case.indices[rank].tolist()
    ]
    expected = case.xs[rank] * torch.tensor(
        multiplicity, dtype=case.xs[rank].dtype
    ).unsqueeze(1)
    torch.testing.assert_close(got.x.cpu(), expected, rtol=0, atol=0)
    if weighted:
        torch.testing.assert_close(
            got.topk_weights.cpu(),
            case.weights[rank].masked_fill(case.indices[rank] == -1, 0),
            rtol=0,
            atol=0,
        )
    else:
        assert got.topk_weights is None


def run_case(rank, size, probe, *, expand, mode, weighted, dtype, topk):
    original = make_case(size, topk, dtype, 0)
    other = make_case(size, topk, dtype, 1)
    config = EPConfig(original.num_experts, topk, original.hidden, original.capacity)
    with CudaBuffer(dist.group.WORLD, config, allow_multiple_reduction=mode) as buffer:
        native = buffer._backend
        assert buffer._payload_offset == buffer._return_offset
        payload_bytes = (
            size
            * original.capacity
            * ((16 + 12 * topk + 4 * original.hidden + 15) // 16 * 16)
        )
        return_bytes = (
            (min(size, topk) if mode else topk)
            * original.capacity
            * ((4 * original.hidden + 4 * topk + 15) // 16 * 16)
        )
        assert (
            buffer._num_bytes
            == (buffer._payload_offset + max(payload_bytes, return_bytes) + 255)
            // 256
            * 256
        )
        window = buffer._runtime.local_window()
        dispatch_gates, combine_gates = [], []
        snapshots = []

        def delay_reader(stage, epoch):
            before, after = stage.clone(), torch.empty_like(stage)
            probe.wait_for_arrival(buffer._runtime.local_window(), 1, epoch)
            after.copy_(stage)
            snapshots.append((before, after))

        def dispatch_epilogue(*args):
            outputs = native.dispatch_epilogue(*args)
            if rank == 0 and not dispatch_gates:
                # Keep an old arena reader pending until peer 1 attempts combine.
                delay_reader(args[0].stage_buffer, 3)
                dispatch_gates.append(3)
            return outputs

        def combine_epilogue(*args):
            outputs = native.combine_epilogue(*args)
            if rank == 0 and not combine_gates:
                # Peer 1 attempts fresh dispatch while this return read is pending.
                delay_reader(args[0], 5)
                combine_gates.append(5)
            return outputs

        buffer._backend = SimpleNamespace(
            dispatch_main=native.dispatch_main,
            dispatch_epilogue=dispatch_epilogue,
            combine_main=native.combine_main,
            combine_epilogue=combine_epilogue,
        )
        with torch.cuda.stream(buffer._comm_stream):
            # All-one FP32 bytes are NaN. Unwritten records must never contribute.
            window[buffer._payload_offset :].fill_(255)
        first = buffer.dispatch(
            original.xs[rank].cuda(),
            topk_idx=original.indices[rank].cuda(),
            weights=original.weights[rank].cuda() if weighted else None,
            do_expand=expand,
            alignment=16,
            zero_padding=expand,
            async_op=True,
        )
        combined = buffer.combine(
            first.x,
            first.handle,
            weights=first.topk_weights,
            previous_event=first.event.event,
            async_op=True,
        )
        second = buffer.dispatch(
            other.xs[rank].cuda(),
            topk_idx=other.indices[rank].cuda(),
            weights=other.weights[rank].cuda() if weighted else None,
            do_expand=expand,
            alignment=16,
            zero_padding=expand,
            async_op=True,
        )
        second.event.current_stream_wait()
        check_dispatch(first, original, rank, expand, weighted)
        check_combine(combined, original, rank, expand, weighted)
        check_dispatch(second, other, rank, expand, weighted)
        if rank == 0:
            assert dispatch_gates == [3] and combine_gates == [5]
            for before, after in snapshots:
                torch.testing.assert_close(after.cpu(), before.cpu(), rtol=0, atol=0)
        saved_metadata = [
            tensor.cpu().clone()
            for tensor in (
                first.handle.layout.src_ranks,
                first.handle.layout.src_tokens,
                first.handle.layout.local_topk_idx,
                first.handle.layout.expanded_slots,
                first.handle.dst_slots,
                first.handle.topk_idx,
            )
        ]
        replay_case = replace(
            original,
            xs=[x + 2 for x in original.xs],
            weights=[-w for w in original.weights],
        )
        replay = buffer.dispatch(
            replay_case.xs[rank].cuda(),
            handle=first.handle,
            weights=replay_case.weights[rank].cuda() if weighted else None,
            zero_padding=expand,
            async_op=True,
        )
        replay.event.current_stream_wait()
        check_dispatch(replay, replay_case, rank, expand, weighted)
        assert replay.handle is first.handle
        returned = buffer.combine(
            replay.x, replay.handle, weights=replay.topk_weights, async_op=True
        )
        returned.event.current_stream_wait()
        check_combine(returned, replay_case, rank, expand, weighted)
        check_dispatch(first, original, rank, expand, weighted)
        check_combine(combined, original, rank, expand, weighted)
        check_dispatch(second, other, rank, expand, weighted)
        for tensor, expected in zip(
            (
                first.handle.layout.src_ranks,
                first.handle.layout.src_tokens,
                first.handle.layout.local_topk_idx,
                first.handle.layout.expanded_slots,
                first.handle.dst_slots,
                first.handle.topk_idx,
            ),
            saved_metadata,
            strict=True,
        ):
            torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
        del window


def run():
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        probe = load_probe()
        for expand, mode, weighted, dtype, topk in product(
            (False, True),
            (False, True),
            (False, True),
            (torch.float32, torch.bfloat16),
            (3, 8),
        ):
            run_case(
                rank,
                size,
                probe,
                expand=expand,
                mode=mode,
                weighted=weighted,
                dtype=dtype,
                topk=topk,
            )
        dist.barrier()
        print(f"NANO_EP_ARENA_RANK_OK={rank}:32", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
