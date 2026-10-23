"""Actual rank-local integration worker; launched by test_launch.py/torchrun."""

from dataclasses import replace
from datetime import timedelta
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils import deterministic

from nano_ep import EPConfig
from nano_ep.backends.cuda import CudaBuffer
from nano_ep.backends.cuda import close_peer_runtime
from nano_ep.backends._build import invoke, load_extension
from nano_ep.testing.cases import make_case
from nano_ep.testing.oracle import reference_dispatch


def check_peer_runtime(rank, size):
    native = load_extension()
    runtime = invoke(native.PeerRuntime, rank, size, 1 << 20)
    handles = [None] * size
    dist.all_gather_object(handles, invoke(runtime.export_local_handle))
    invoke(runtime.import_peer_handles, handles)
    addresses = invoke(runtime.peer_addresses)
    local_window = invoke(runtime.local_window)
    assert addresses.is_cuda and addresses.dtype == torch.int64
    assert addresses.shape == (size,) and torch.all(addresses != 0)
    assert local_window.is_cuda and local_window.dtype == torch.uint8
    assert local_window.numel() == 1 << 20
    assert addresses[rank].item() == local_window.data_ptr()
    for epoch in (1, 2, 3):
        invoke(runtime.barrier, torch.cuda.current_stream().cuda_stream, epoch)
        torch.cuda.synchronize()
    # Borrowed views cannot outlive the runtime window.
    del local_window, addresses
    try:
        invoke(runtime.close)
    except RuntimeError as error:
        assert "close_peer_mappings" in str(error)
    else:
        raise AssertionError("freeing before the unmap phase must fail")
    close_peer_runtime(runtime, dist.group.WORLD, torch.cuda.current_device())
    invoke(runtime.close_peer_mappings)
    invoke(runtime.close)


def assert_dispatch(got, expected, config, rank):
    layout = got.handle.layout
    assert layout.recv_count == expected.layout.recv_count
    assert layout.expanded_capacity == expected.layout.expanded_capacity
    keys = layout.src_ranks * config.num_max_tokens_per_rank + layout.src_tokens
    perm = keys.argsort()
    torch.testing.assert_close(
        keys[perm].cpu(),
        expected.layout.src_ranks * config.num_max_tokens_per_rank
        + expected.layout.src_tokens,
    )
    torch.testing.assert_close(
        layout.local_topk_idx[perm].cpu(), expected.layout.local_topk_idx
    )
    torch.testing.assert_close(
        layout.expert_counts.cpu(), expected.layout.expert_counts
    )
    torch.testing.assert_close(
        layout.expert_offsets.cpu(), expected.layout.expert_offsets
    )
    if not got.handle.do_expand:
        torch.testing.assert_close(got.x[perm].cpu(), expected.x, rtol=0, atol=0)
        torch.testing.assert_close(got.topk_idx[perm].cpu(), expected.topk_idx)
        torch.testing.assert_close(got.topk_weights[perm].cpu(), expected.topk_weights)
    else:
        assert got.topk_idx is None
        slots = layout.expanded_slots[perm].cpu()
        assert torch.equal(slots >= 0, expected.layout.expanded_slots >= 0)
        valid = slots >= 0
        assert slots[valid].unique().numel() == valid.sum()
        for e in range(config.num_experts // got.handle.world_size):
            selected = slots[expected.layout.local_topk_idx == e]
            lo = int(layout.expert_offsets[e])
            hi = lo + int(layout.expert_counts[e])
            assert torch.all((selected >= lo) & (selected < hi))
        src = slots[valid].to(got.x.device)
        dst = expected.layout.expanded_slots[valid]
        torch.testing.assert_close(got.x[src].cpu(), expected.x[dst], rtol=0, atol=0)
        torch.testing.assert_close(
            got.topk_weights[src].cpu(), expected.topk_weights[dst]
        )


def expert_output(received):
    handle, layout = received.handle, received.handle.layout
    local_experts = handle.config.num_experts // handle.world_size
    y = (
        torch.full_like(received.x, float("nan"))
        if handle.do_expand
        else torch.zeros_like(received.x)
    )
    for row in range(layout.recv_count):
        for k, expert in enumerate(layout.local_topk_idx[row].tolist()):
            if expert < 0:
                continue
            slot = int(layout.expanded_slots[row, k]) if handle.do_expand else row
            weight = (
                received.topk_weights[slot]
                if handle.do_expand
                else received.topk_weights[row, k]
            )
            contribution = (
                received.x[slot] * (handle.rank * local_experts + expert + 1) + 0.5
            ) * weight
            if handle.do_expand:
                y[slot] = contribution
            else:
                y[row] += contribution
    return y


def source_reference(case, rank):
    """Source assignment reference, independent of receive ordering and layouts."""
    expected = torch.zeros_like(case.xs[rank])
    for token, choices in enumerate(case.indices[rank].tolist()):
        for k, expert in enumerate(choices):
            if expert >= 0:
                expected[token] += (
                    case.xs[rank][token] * (expert + 1) + 0.5
                ) * case.weights[rank][token, k]
    return expected


def delayed_backend(native):
    def epilogue(*args):
        torch.cuda._sleep(20_000_000)
        return native.combine_epilogue(*args)

    return SimpleNamespace(
        dispatch_main=native.dispatch_main,
        dispatch_epilogue=native.dispatch_epilogue,
        combine_main=native.combine_main,
        combine_epilogue=epilogue,
    )


def combine_experts(buffer, received, bias, *, explicit_event, asynchronous):
    device_bias = bias.cuda() if bias is not None else None
    producer = torch.cuda.Stream() if explicit_event else torch.cuda.current_stream()
    if explicit_event:
        producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        received.event.current_stream_wait()
        y = expert_output(received)
        if explicit_event:
            delayed_y = torch.empty_like(y)
            torch.cuda._sleep(20_000_000)
            delayed_y.copy_(y)
            y = delayed_y
        ready = torch.cuda.Event() if explicit_event else None
        if ready is not None:
            ready.record(producer)
    result = buffer.combine(
        y,
        received.handle,
        weights=received.topk_weights,
        bias=device_bias,
        previous_event=ready,
        async_op=asynchronous,
    )
    consumer = torch.cuda.Stream() if asynchronous else torch.cuda.current_stream()
    return result, consumer


def assert_combined(result, case, rank, bias):
    expected = source_reference(case, rank)
    if bias is not None:
        expected += bias
    torch.testing.assert_close(result.x.cpu(), expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        result.topk_weights.cpu(),
        case.weights[rank].masked_fill(case.indices[rank] == -1, 0),
        rtol=0,
        atol=0,
    )


def run():
    rank, size, local = (
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
    )
    assert size == int(os.environ["NANO_EP_EXPECTED_WORLD"])
    assert int(os.environ["LOCAL_WORLD_SIZE"]) == size
    torch.cuda.set_device(local)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = True
    try:
        check_peer_runtime(rank, size)
        cfg = EPConfig(size * 3, 3, 128, 5)
        # A second lifetime catches stale IPC handles and counters after close.
        for lifetime in range(2):
            buffer = CudaBuffer(
                dist.group.WORLD, cfg, allow_multiple_reduction=bool(lifetime)
            )
            buffer._backend = delayed_backend(buffer._backend)
            for expand in (False, True):
                for kind in ("mixed", "empty", "masked", "skew"):
                    c = make_case(size, kind=kind, seed=lifetime + 7)
                    c.xs = [x.repeat(1, 32) for x in c.xs]
                    expected = reference_dispatch(
                        c.xs,
                        c.indices,
                        c.num_experts,
                        c.weights,
                        do_expand=expand,
                        alignment=4,
                    )[rank]
                    asynchronous = kind in ("masked", "skew")
                    explicit_combine_event = kind in ("mixed", "skew")
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        x = c.xs[rank].cuda()
                        idx, weights = c.indices[rank].cuda(), c.weights[rank].cuda()
                        ready = torch.cuda.Event()
                        ready.record()
                    got = buffer.dispatch(
                        x,
                        topk_idx=idx,
                        weights=weights,
                        do_expand=expand,
                        alignment=4,
                        zero_padding=expand,
                        previous_event=ready,
                        async_op=True,
                    )
                    got.event.current_stream_wait()
                    assert_dispatch(got, expected, cfg, rank)
                    expected_sends = torch.zeros(size, dtype=torch.int64)
                    for row in c.indices[rank].tolist():
                        for dest in {e // 3 for e in row if e >= 0}:
                            expected_sends[dest] += 1
                    torch.testing.assert_close(
                        got.handle.send_counts.cpu(), expected_sends
                    )
                    bias = (
                        torch.full_like(c.xs[rank], 0.25) if kind == "mixed" else None
                    )
                    result, consumer = combine_experts(
                        buffer,
                        got,
                        bias,
                        explicit_event=explicit_combine_event,
                        asynchronous=asynchronous,
                    )
                    with torch.cuda.stream(consumer):
                        if asynchronous:
                            result.event.current_stream_wait()
                        assert_combined(result, c, rank, bias)
                        saved_result = (
                            result.x.cpu().clone(),
                            result.topk_weights.cpu().clone(),
                        )
                    # Snapshot isolation + changed payload/weights, same cached slots.
                    idx.fill_(-1)
                    replay = buffer.dispatch(
                        x * 2,
                        weights=weights * 3,
                        handle=got.handle,
                        zero_padding=expand,
                        async_op=True,
                    )
                    replay.event.current_stream_wait()
                    assert replay.handle is got.handle
                    expected_replay = reference_dispatch(
                        [a * 2 for a in c.xs],
                        c.indices,
                        c.num_experts,
                        [w * 3 for w in c.weights],
                        do_expand=expand,
                        alignment=4,
                    )[rank]
                    assert_dispatch(replay, expected_replay, cfg, rank)
                    replay_case = replace(
                        c, xs=[a * 2 for a in c.xs], weights=[w * 3 for w in c.weights]
                    )
                    combined_replay, consumer = combine_experts(
                        buffer,
                        replay,
                        bias,
                        explicit_event=not explicit_combine_event,
                        asynchronous=asynchronous,
                    )
                    with torch.cuda.stream(consumer):
                        if asynchronous:
                            combined_replay.event.current_stream_wait()
                        assert_combined(combined_replay, replay_case, rank, bias)
                        torch.testing.assert_close(
                            result.x.cpu(), saved_result[0], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            result.topk_weights.cpu(), saved_result[1], rtol=0, atol=0
                        )
            # Exact BF16 roundtrips distinguish rank duplication from expert
            # duplication, with no weights, plus a source bias added once.
            c = make_case(size, kind="mixed", seed=31)
            x_cpu = torch.full((len(c.xs[rank]), 128), rank + 1, dtype=torch.bfloat16)
            for expand in (False, True):
                idx = c.indices[rank].cuda()
                received = buffer.dispatch(
                    x_cpu.cuda() + 0,
                    topk_idx=idx,
                    do_expand=expand,
                    alignment=4,
                    zero_padding=expand,
                    async_op=True,
                )
                # Input expression is already dead. Exercise allocator ownership
                # while communication may still be in flight.
                scratch = torch.empty_like(x_cpu, device="cuda")
                scratch.fill_(-77)
                received.event.current_stream_wait()
                assert received.topk_weights is None
                combined = buffer.combine(
                    received.x,
                    received.handle,
                    bias=torch.ones_like(x_cpu, device="cuda"),
                    async_op=expand,
                )
                if expand:
                    combined.event.current_stream_wait()
                multiplicity = []
                for row in c.indices[rank].tolist():
                    multiplicity.append(
                        sum(e >= 0 for e in row)
                        if expand
                        else len({e // 3 for e in row if e >= 0})
                    )
                expected = (
                    x_cpu
                    * torch.tensor(multiplicity, dtype=torch.bfloat16).unsqueeze(1)
                    + 1
                )
                torch.testing.assert_close(combined.x.cpu(), expected, rtol=0, atol=0)
                assert combined.topk_weights is None
                promoted = buffer.combine(
                    received.x.float(),
                    received.handle,
                    bias=torch.ones_like(x_cpu, dtype=torch.float32, device="cuda"),
                    async_op=True,
                )
                promoted.event.current_stream_wait()
                assert (
                    promoted.x.dtype == torch.float32 and promoted.topk_weights is None
                )
                torch.testing.assert_close(
                    promoted.x.cpu(), expected.float(), rtol=0, atol=0
                )
            buffer.close()
            buffer.close()
            try:
                buffer.dispatch(x_cpu.cuda(), topk_idx=c.indices[rank].cuda())
            except RuntimeError:
                pass
            else:
                raise AssertionError("dispatch after close must fail")
            try:
                buffer.combine(received.x, received.handle)
            except RuntimeError as error:
                assert "closed" in str(error)
            else:
                raise AssertionError("combine after close must fail")
        torch.cuda.synchronize()
        dist.barrier()
        print(f"NANO_EP_RANK_OK={rank}", flush=True)
        print(f"NANO_EP_PUBLIC_COMBINE_RANK_OK={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
