"""Public CudaBuffer dispatch acceptance, independent of combine execution."""

from dataclasses import replace
from datetime import timedelta
from itertools import product
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils import deterministic

from nano_ep import EPConfig
from nano_ep.backends.cuda import CudaBuffer
from nano_ep.testing.cases import make_case
from nano_ep.testing.native_dispatch import assert_dispatch_output
from nano_ep.testing.oracle import reference_layout, reference_plan


def check_output(got, case, rank, alignment, weights, zero_padding):
    handle, layout = got.handle, got.handle.layout
    assert handle.rank == rank and handle.world_size == len(case.xs)
    assert handle.x_dtype == case.xs[rank].dtype
    assert got.x.device == torch.device("cuda", torch.cuda.current_device())
    assert (got.topk_idx is None) is handle.do_expand
    assert (got.topk_weights is None) is (weights is None)
    metadata = [
        layout.src_ranks,
        layout.src_tokens,
        layout.local_topk_idx,
        layout.expert_counts,
        layout.expert_offsets,
        layout.expanded_slots,
        handle.dst_slots,
        handle.send_counts,
        handle.topk_idx,
    ]
    # The shared checker uses device and receive mappings; it never reads raw staging here.
    state = SimpleNamespace(stage_buffer=got.x, metadata=metadata)
    outputs = (
        got.x,
        got.topk_idx
        if got.topk_idx is not None
        else got.x.new_empty((0,), dtype=torch.int64),
        got.topk_weights
        if got.topk_weights is not None
        else got.x.new_empty((0,), dtype=torch.float32),
    )
    assert_dispatch_output(
        state,
        outputs,
        case,
        rank,
        alignment,
        weights,
        expand=handle.do_expand,
        zero_padding=zero_padding,
    )
    expected = reference_layout(case.indices, rank, case.num_experts, alignment)
    assert (layout.recv_count, layout.expanded_capacity) == (
        expected.recv_count,
        expected.expanded_capacity,
    )
    torch.testing.assert_close(layout.expert_counts.cpu(), expected.expert_counts)
    torch.testing.assert_close(layout.expert_offsets.cpu(), expected.expert_offsets)
    torch.testing.assert_close(handle.topk_idx.cpu(), case.indices[rank])
    torch.testing.assert_close(
        handle.send_counts.cpu(),
        reference_plan(case.indices[rank], case.num_experts, len(case.xs)).send_counts,
    )


def make_payloads(size, kind, dtype, hidden):
    case = make_case(size, kind=kind, dtype=dtype, seed=23)
    xs = [
        (torch.arange(len(x) * hidden).reshape(len(x), hidden) / 8 + source * 10).to(
            dtype
        )
        for source, x in enumerate(case.xs)
    ]
    return replace(case, xs=xs, hidden=hidden)


def check_rounds(buffer, rank, size):
    native = buffer._backend
    delay = [False]

    def epilogue(*args):
        if delay[0]:
            torch.cuda._sleep(20_000_000)
        return native.dispatch_epilogue(*args)

    buffer._backend = SimpleNamespace(
        dispatch_main=native.dispatch_main, dispatch_epilogue=epilogue
    )
    for expand, dtype, weighted, asynchronous, kind in product(
        (False, True),
        (torch.float32, torch.bfloat16),
        (False, True),
        (False, True),
        ("mixed", "empty", "masked", "skew"),
    ):
        case = make_payloads(size, kind, dtype, buffer._config.hidden)
        all_weights = case.weights if weighted else None
        zero_padding = expand and kind != "skew"
        alignment = 1 if not expand and kind == "mixed" else 4
        explicit = kind in ("mixed", "masked")
        source_x, source_indices = case.xs[rank].cuda(), case.indices[rank].cuda()
        source_weights = all_weights[rank].cuda() if weighted else None
        x, indices = torch.empty_like(source_x), torch.empty_like(source_indices)
        weights = torch.empty_like(source_weights) if weighted else None
        producer = torch.cuda.Stream() if explicit else torch.cuda.current_stream()
        if explicit:
            producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            if kind in ("mixed", "empty"):
                torch.cuda._sleep(10_000_000)
            x.copy_(source_x)
            indices.copy_(source_indices)
            if weighted:
                weights.copy_(source_weights)
            ready = torch.cuda.Event() if explicit else None
            if ready is not None:
                ready.record(producer)
        delay[0] = kind == "mixed"
        got = buffer.dispatch(
            x,
            topk_idx=indices,
            weights=weights,
            do_expand=None if alignment == 1 else expand,
            alignment=None if alignment == 1 else alignment,
            zero_padding=zero_padding,
            previous_event=ready,
            async_op=asynchronous,
        )
        consumer = torch.cuda.Stream() if asynchronous else torch.cuda.current_stream()
        with torch.cuda.stream(consumer):
            if asynchronous:
                got.event.current_stream_wait()
            check_output(got, case, rank, alignment, all_weights, zero_padding)
            torch.testing.assert_close(x.cpu(), case.xs[rank], rtol=0, atol=0)
            torch.testing.assert_close(indices.cpu(), case.indices[rank])
            if weights is not None:
                torch.testing.assert_close(weights.cpu(), all_weights[rank])
        indices.fill_(-1)
        if kind == "mixed":
            # Interleave a different route before replaying the independently owned handle.
            buffer.dispatch(
                x,
                topk_idx=indices,
                weights=weights,
                do_expand=expand,
                alignment=alignment,
                zero_padding=zero_padding,
                async_op=True,
            )
        replay_case = replace(case, xs=[value * 2 for value in case.xs])
        replay_weights = None if weighted else [value * 3 for value in case.weights]
        replay = buffer.dispatch(
            x * 2,
            weights=replay_weights[rank].cuda() if replay_weights is not None else None,
            handle=got.handle,
            zero_padding=zero_padding,
            async_op=asynchronous,
        )
        assert replay.handle is got.handle
        with torch.cuda.stream(consumer):
            if asynchronous:
                replay.event.current_stream_wait()
            check_output(
                replay, replay_case, rank, alignment, replay_weights, zero_padding
            )
            check_output(got, case, rank, alignment, all_weights, zero_padding)


def check_consumer_lifetime(buffer, rank, size):
    case = make_payloads(size, "mixed", torch.bfloat16, buffer._config.hidden)
    got = buffer.dispatch(
        case.xs[rank].cuda(),
        topk_idx=case.indices[rank].cuda(),
        weights=case.weights[rank].cuda(),
        do_expand=True,
        alignment=4,
        zero_padding=True,
        async_op=True,
    )
    handle = got.handle
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        got.event.current_stream_wait()
        torch.cuda._sleep(30_000_000)
        captured = SimpleNamespace(
            x=got.x.clone(),
            topk_idx=None,
            topk_weights=got.topk_weights.clone(),
            handle=handle,
        )
    del got
    # Matching allocations on comm must not recycle storage still read by consumer.
    newer = buffer.dispatch(
        case.xs[rank].cuda() * 3,
        weights=case.weights[rank].cuda() * 2,
        handle=handle,
        zero_padding=True,
        async_op=True,
    )
    consumer.synchronize()
    check_output(captured, case, rank, 4, case.weights, True)
    newer.event.current_stream_wait()


def run():
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert size == int(os.environ["NANO_EP_EXPECTED_WORLD"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = True
    try:
        for mode in (False, True):
            config = EPConfig(size * 3, 3, 65 if mode else 33, 5)
            with CudaBuffer(
                dist.group.WORLD, config, allow_multiple_reduction=mode
            ) as buffer:
                check_rounds(buffer, rank, size)
                check_consumer_lifetime(buffer, rank, size)
            try:
                buffer.dispatch(
                    torch.empty((0, config.hidden), device="cuda"),
                    topk_idx=torch.empty((0, 3), dtype=torch.int64, device="cuda"),
                )
            except RuntimeError as error:
                assert "closed" in str(error)
            else:
                raise AssertionError("dispatch after close must fail")
            print(f"NANO_EP_PUBLIC_DISPATCH_MODE_OK={rank}:{mode}", flush=True)
        print(f"NANO_EP_PUBLIC_DISPATCH_RANK_OK={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
