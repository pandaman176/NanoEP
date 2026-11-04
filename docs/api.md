# API

`CudaBuffer(group, config, *, allow_multiple_reduction=False)` collectively creates a single-node CUDA IPC buffer. Use a Gloo group, one GPU per process, and the same collective order on every rank. Keep each process on its assigned GPU.

`EPConfig(num_experts=E, num_topk=K, hidden=H, num_max_tokens_per_rank=Tmax)` uses contiguous expert placement: rank `r` owns `[r * E/R, (r+1) * E/R)`, where `R` is the group size.

- `R` is a power of two in `[1,32]`; `E > 0` and `E % R == 0`.
- `1 <= K <= min(32,E)`; `H` is a positive multiple of 256; `Tmax > 0`.
- Each source has `0 <= T <= Tmax`. Empty ranks still participate.

## Dispatch

```python
received = buffer.dispatch(
    x, topk_idx=indices, weights=weights,
    do_expand=False, alignment=1, zero_padding=False,
    previous_event=None, async_op=False,
)
```

Inputs are contiguous tensors on the local GPU: BF16 `x[T,H]`, int64 `indices[T,K]`, and optional FP32 `weights[T,K]`. Nonempty x must have a 16-byte-aligned data pointer; use `clone()` to realign an offset view. Indices are global expert IDs or `-1`; valid experts within each token must be distinct. All ranks agree on weight presence and collective settings. `alignment` is a positive integer.

`N` counts received `(source token, destination rank)` pairs; `C` is the padded assignment capacity. The same token is sent once to a rank even when it selects several experts there. Receive order is unspecified.

| Result field | Compact | Expanded |
| --- | --- | --- |
| `x` | BF16 `[N,H]` | BF16 `[C,H]`, expert-major |
| `topk_idx` | int64 `[N,K]`, local IDs or `-1` | `None` |
| `topk_weights` | FP32 `[N,K]`, complete source rows | FP32 `[C]` |

Without weights, `topk_weights` is `None`. In expanded layout, only `zero_padding=True` initializes expert gaps; otherwise do not read them. Compact ignores zero padding. Outputs own storage independently of the communication arena.

## Handles and cached routing

```python
replay = buffer.dispatch(new_x, weights=new_weights, handle=received.handle)
```

Omit `topk_idx`; preserve source token count, dtype, routing, buffer and receive layout. Omitted layout/alignment settings inherit from the handle. All ranks replay corresponding handles. Handles snapshot routing, remain valid across calls on the same live buffer, and must be treated as read-only.

`handle.layout` contains CUDA int64 metadata, with `L=E/R`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `src_ranks`, `src_tokens` | `[N]` | Source of each deduplicated receive row |
| `local_topk_idx` | `[N,K]` | Local expert IDs or `-1` |
| `expert_counts` | `[L]` | Actual assignments per expert |
| `expert_offsets` | `[L+1]` | Padded exclusive offsets, ending at C |
| `expanded_slots` | Expanded `[N,K]`; compact `[0]` | Assignment slots or `-1` |
| `recv_rank_prefix` | `[R]` | Inclusive receive-row count by source rank |

`recv_count` and `expanded_capacity` are host integers N/C. Metadata rows remain deduplicated in both layouts.

## Combine

```python
result = buffer.combine(
    expert_x, received.handle, weights=None, bias=None,
    previous_event=None, async_op=False,
)
```

The caller computes experts and applies gate weights. Compact BF16 `expert_x[N,H]` contains the sum of weighted local expert outputs; expanded BF16 `expert_x[C,H]` contains one output per assignment. Padding is ignored. Inputs are contiguous on the local GPU with 16-byte-aligned data pointers.

`result.x[T,H]` accumulates FP32, adds optional source-local BF16 `bias[T,H]` once, then casts to BF16. Optional FP32 weights use the receive layout and return independently as `result.topk_weights[T,K]`, with masked slots zero. They never multiply `expert_x`; without weights the result field is `None`.

`allow_multiple_reduction=False` keeps expanded contributions separate until source reduction. True permits FP32 local reduction, BF16 communication storage, then FP32 source reduction. Compact inputs are already locally reduced. Grouping and BF16 rounding can change results across layouts or modes.

## Streams and lifetime

With `async_op=False`, completion is ordered onto the caller's CUDA stream. With True, wait on the consumer stream before using the output:

```python
received = buffer.dispatch(x, topk_idx=indices, async_op=True)
received.event.current_stream_wait()
```

`previous_event` is an optional native `torch.cuda.Event` from another input-producing stream. Without it, communication waits for the caller's current stream. `current_stream_wait()` also records tensor use on the consumer stream. Do not modify inputs until communication completes.

Fresh dispatch waits on mapped host N/C publication to allocate exact outputs. Cached dispatch uses the handle's saved counts. CUDA Graph capture is not supported.

Use `with CudaBuffer(...)` or close collectively after consumers finish. Close drains GPU work, unmaps peer memory collectively, then releases allocations. Handles cannot cross buffers or outlive close. Fatal rank failure relies on process-group/launcher timeouts.

## Native execution

PyTorch caches compiled CUDA modules. `CudaBuffer` prepares core and geometry modules before IPC setup; additional dispatch alignments compile on first use. Reuse a fixed configuration for steady-state calls.

| Module | Specialization |
| --- | --- |
| Dispatch main | H, K, R, L, Tmax, expert alignment |
| Dispatch epilogue | H, K, L, cached mode |
| Combine main / epilogue | H, K, R, L, Tmax |

Main kernels use cooperative SM-sized grids and entry/completion peer barriers. Fresh dispatch counts routes, publishes receive counts and reserves local per-destination slots; cached dispatch reads saved mappings. Receive storage is source-partitioned `[R,Tmax,row_bytes]`, and epilogue maps its valid rows through the source prefix. Expanded epilogue builds expert slots and copies payloads in one kernel.

Communication fields and row strides are 32-byte aligned. BF16 payloads use int4 bit copies and bounded register tiles for source reuse across destinations; reductions accumulate FP32. Dispatch and combine share one communication arena. TMA and PDL are not implemented.
