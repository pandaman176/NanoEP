# API

`CudaBuffer(group, config, *, allow_multiple_reduction=False)` is a collective, single-node buffer. Use a Gloo group containing all local workers and one distinct GPU per process. Let `R` be the number of ranks, `E` the number of experts, `K` the top-k size, `H` the hidden dimension, and `Tmax` the source token capacity per rank. Expert placement is contiguous: rank `r` owns `[r * E/R, (r+1) * E/R)`. `EPConfig(E, K, H, Tmax)` requires positive dimensions, `E % R == 0`, `K <= min(32, E)` and source token count `T <= Tmax`.

## Dispatch

```python
received = buffer.dispatch(
    x,
    topk_idx=indices,
    weights=weights,
    do_expand=False,
    alignment=1,
    zero_padding=False,
    previous_event=None,
    async_op=False,
)
```

Inputs must be contiguous on the local GPU: `x[T,H]` is FP32/BF16, `indices[T,K]` is int64 and optional `weights[T,K]` is FP32. Indices are global expert IDs or `-1`; valid expert IDs in each token must be distinct. Empty source ranks are supported and must participate. All ranks agree on payload dtype, weight presence and collective order.

Compact output has `x[N,H]`, local `topk_idx[N,K]`, and optional weights `[N,K]`. A source token appears once per destination rank, even if it selects several local experts. Nonlocal/masked IDs are `-1`; weights retain the complete original row.

Expanded output has expert-major `x[C,H]`, `topk_idx=None`, and optional weights `[C]`. `C` includes per-expert alignment gaps. Padding is only initialized when `zero_padding=True`; otherwise do not read it. Receive order is unspecified.

`received.handle.layout` contains CUDA int64 `src_ranks[N]`, `src_tokens[N]`, `local_topk_idx[N,K]`, `expert_counts[L]`, `expert_offsets[L+1]` and `expanded_slots[N,K]`, where `L=E/R`. Counts describe actual assignments; offsets include alignment. Invalid expanded slots are `-1`. Metadata rows remain rank-deduplicated in both layouts.

## Cached routing

```python
replay = buffer.dispatch(new_x, weights=new_weights, handle=received.handle)
```

Omit `topk_idx`; reuse source token count, input dtype, routing, buffer, layout and alignment. Omitted layout/alignment settings inherit from the handle. All ranks replay corresponding handles. Handles snapshot routing and remain valid across subsequent calls on the same live buffer. Treat their contents as read-only. Outputs own storage independently of the communication arena.

## Combine

```python
result = buffer.combine(
    expert_x,
    received.handle,
    weights=None,
    bias=None,
    previous_event=None,
    async_op=False,
)
```

The caller computes experts and applies gate weights. Compact `expert_x[N,H]` must already contain the sum of the local weighted expert outputs. Expanded `expert_x[C,H]` contains one output per assignment; padding is ignored. FP32/BF16 output dtype may differ from dispatch, but must agree across ranks.

Optional FP32 `weights` uses the receive layout and is returned independently as `result.topk_weights[T,K]`, with masked positions zero; it never multiplies `expert_x`. With no weight payload, the result field is `None`. Optional source-local `bias[T,H]` has the expert-output dtype and is added once.

`result.x[T,H]` accumulates FP32 then casts to the expert-output dtype. Default `allow_multiple_reduction=False` keeps expanded contributions separate until final reduction. True permits FP32 local reduction before source reduction. Compact inputs have already been locally reduced by the caller. Grouping and client BF16 rounding can change numerical results between modes/layouts; no bitwise equivalence is promised.

## Events and lifetime

With `async_op=False`, completion is ordered onto the caller's CUDA stream. With True, wait on the consumer stream:

```python
received = buffer.dispatch(x, topk_idx=indices, async_op=True)
received.event.current_stream_wait()
```

`previous_event` is an optional native `torch.cuda.Event` for an input producer on another stream. Otherwise the communication stream waits for the caller's current stream. The returned event tracks tensor ownership on the consumer stream. Keep the current device set to this process's GPU and do not mutate inputs until communication completes. Exact receive-count readback can block the host; asynchronous mode does not imply CUDA Graph compatibility.

Close collectively after consumers finish, preferably with `with CudaBuffer(...)`. Closing drains GPU work, coordinates unmapping across ranks, then frees allocations. Do not use handles after close or across buffers. Fatal rank failure relies on the process-group/launcher timeout.
