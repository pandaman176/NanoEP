"""Single-node forward MoE: torchrun --standalone --nproc-per-node=N -m nano_ep.examples.moe.

Experts are distinct elementwise tanh transforms, keeping the communication
example small. Gate weights are applied explicitly before combine. No autograd
or CPU simulator is used in the communication path.
"""

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from nano_ep import CudaBuffer, EPConfig


def expert_forward(received):
    """Evaluate local experts using GPU metadata, including padded layouts."""
    handle = received.handle
    local_experts = handle.config.num_experts // handle.world_size
    if handle.do_expand:
        layout = handle.layout
        rows = torch.arange(received.x.shape[0], device=received.x.device)
        expert = torch.bucketize(rows, layout.expert_offsets[1:], right=True)
        valid = rows < layout.expert_offsets[expert] + layout.expert_counts[expert]
        global_expert = expert + handle.rank * local_experts
        value = torch.tanh(received.x.float() * (1 + global_expert[:, None] / 32) + global_expert[:, None] / 64)
        value = value * received.topk_weights[:, None]
        return value.masked_fill(~valid[:, None], 0).to(received.x.dtype)

    value = torch.zeros_like(received.x, dtype=torch.float32)
    for k in range(handle.config.num_topk):
        expert = received.topk_idx[:, k]
        global_expert = expert + handle.rank * local_experts
        contribution = torch.tanh(received.x.float() * (1 + global_expert[:, None] / 32) + global_expert[:, None] / 64)
        contribution = contribution * received.topk_weights[:, k, None]
        value += contribution.masked_fill((expert < 0)[:, None], 0)
    return value.to(received.x.dtype)


def dense_reference(x, indices, weights):
    """Source-local mathematical MoE reference, independent of receive metadata."""
    result = torch.zeros_like(x, dtype=torch.float32)
    for k in range(indices.shape[1]):
        expert = indices[:, k]
        value = torch.tanh(x.float() * (1 + expert[:, None] / 32) + expert[:, None] / 64)
        result += (value * weights[:, k, None]).masked_fill((expert < 0)[:, None], 0)
    return result.to(x.dtype)


def make_inputs(args, rank, size, iteration, device):
    tokens = args.tokens
    if args.pattern == "uneven":
        tokens = 0 if rank == size - 1 else max(0, tokens - rank * 3)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    row = torch.arange(tokens, device=device)
    col = torch.arange(args.hidden, device=device)
    x = (((row[:, None] * 7 + col + rank * 11 + iteration * 3) % 127 - 63) / 64).to(dtype)
    experts = args.experts_per_rank * size
    base = row + rank * args.experts_per_rank
    if args.pattern == "skew":
        base = torch.zeros_like(row)
    indices = (base[:, None] + torch.arange(args.topk, device=device)) % experts
    indices = indices.clone()
    if args.pattern == "masked":
        indices.fill_(-1)
    else:
        indices[:, -1].masked_fill_(row % 7 == 0, -1)
    weights = ((row[:, None] + torch.arange(args.topk, device=device) + iteration) % 5 + 1).float() / (5 * args.topk)
    return x.contiguous(), indices.contiguous(), weights.contiguous()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("compact", "expanded"), default="expanded")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--experts-per-rank", type=int, default=8)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pattern", choices=("mixed", "skew", "masked", "uneven"), default="mixed")
    parser.add_argument("--async-op", action="store_true")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    if args.tokens < 0 or min(args.hidden, args.topk, args.experts_per_rank, args.alignment, args.iterations) <= 0:
        parser.error("tokens must be nonnegative; other dimensions/iterations must be positive")
    return args


@torch.no_grad()
def run(args):
    rank, size, local = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if int(os.environ["LOCAL_WORLD_SIZE"]) != size:
        raise ValueError("This example requires a single node")
    torch.cuda.set_device(local)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        cfg = EPConfig(size * args.experts_per_rank, args.topk, args.hidden, max(1, args.tokens))
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        checks = []
        handle = None
        with CudaBuffer(dist.group.WORLD, cfg) as buffer:
            for iteration in range(args.iterations):
                with torch.cuda.stream(producer):
                    x, indices, weights = make_inputs(args, rank, size, iteration, "cuda")
                    expected = dense_reference(x, indices, weights)
                    expected_weights = weights.masked_fill(indices < 0, 0)
                    ready = torch.cuda.Event()
                    ready.record()
                with torch.cuda.stream(consumer):
                    # Wait only on the result event before consuming the payload.
                    received = buffer.dispatch(
                        x,
                        topk_idx=indices if handle is None else None,
                        weights=weights,
                        handle=handle,
                        do_expand=args.layout == "expanded",
                        alignment=args.alignment,
                        zero_padding=args.layout == "expanded",
                        previous_event=ready,
                        async_op=args.async_op,
                    )
                    if handle is not None and received.handle is not handle:
                        raise AssertionError("cached dispatch must preserve handle identity")
                    received.event.current_stream_wait()
                    y = expert_forward(received)
                    result = buffer.combine(y, received.handle, weights=received.topk_weights, async_op=args.async_op)
                    result.event.current_stream_wait()
                    # Keep the original outputs: later rounds must not overwrite them.
                    checks.append((result.x, expected, result.topk_weights, expected_weights))
                    if args.cached:
                        handle = received.handle
                del x, indices, weights, y, received, result
                # Reuse producer allocations while comm/consumer work may be pending.
                with torch.cuda.stream(producer):
                    churn = torch.empty((args.tokens, args.hidden), device="cuda", dtype=expected.dtype)
                    churn.fill_(-77)
                    del churn
            # No .cpu(), .item(), .tolist() or explicit host wait between dispatch
            # and combine or between rounds. Exact-count readback remains allowed.
            torch.cuda.synchronize()
            for actual, expected, actual_weights, expected_weights in checks:
                tolerance = 0.03 if args.dtype == "bf16" else 2e-5
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
        reports = [None] * size if rank == 0 else None
        dist.gather_object({"rank": rank, "world_size": size, **vars(args)}, reports, dst=0)
        if rank == 0:
            for report in reports:
                print("NANO_EP_MOE_OK=" + json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
