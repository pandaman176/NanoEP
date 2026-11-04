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

Inputs must be contiguous on the local GPU: `x[T,H]` is BF16, `indices[T,K]` is int64 and optional `weights[T,K]` is FP32. Indices are global expert IDs or `-1`; valid expert IDs in each token must be distinct. Empty source ranks are supported and must participate. All ranks agree on payload dtype, weight presence and collective order.

Compact output has `x[N,H]`, local `topk_idx[N,K]`, and optional weights `[N,K]`. A source token appears once per destination rank, even if it selects several local experts. Nonlocal/masked IDs are `-1`; weights retain the complete original row.

Expanded output has expert-major `x[C,H]`, `topk_idx=None`, and optional weights `[C]`. `C` includes per-expert alignment gaps. Padding is only initialized when `zero_padding=True`; otherwise do not read it. Receive order is unspecified.

`received.handle.layout` contains CUDA int64 `src_ranks[N]`, `src_tokens[N]`, `local_topk_idx[N,K]`, `expert_counts[L]`, `expert_offsets[L+1]` and `expanded_slots`, where `L=E/R`. Counts describe actual assignments; offsets include alignment. Expanded slots have shape `[N,K]`, with invalid entries set to `-1`; compact slots have shape `[0]`. Metadata rows remain rank-deduplicated in both layouts.

Compact dispatch skips expanded slot allocation and expert cursor reservation. Code that inspected a compact handle's previous `[N,K]` expanded mapping must adapt to `[0]`. Prepared native states, including empty states, must retain their original compact/expanded layout on replay. Rebuild native core and epilogue modules together and restart the process.

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

The caller computes experts and applies gate weights. Compact `expert_x[N,H]` must already contain the sum of the local weighted expert outputs. Expanded `expert_x[C,H]` contains one output per assignment; padding is ignored. Expert outputs are BF16 on every rank; FP32 EP inputs are not supported. The input data pointer must be 16-byte aligned for vector loads; contiguous views with an unaligned storage offset are rejected before the entry GPU barrier. Aligned nonzero storage offsets are supported.

Optional FP32 `weights` uses the receive layout and is returned independently as `result.topk_weights[T,K]`, with masked positions zero; it never multiplies `expert_x`. With no weight payload, the result field is `None`. Optional source-local `bias[T,H]` has the expert-output dtype and is added once.

`result.x[T,H]` accumulates FP32 then casts to the expert-output dtype. Default `allow_multiple_reduction=False` keeps expanded contributions separate until final reduction. True permits FP32 local reduction, BF16 communication storage, then FP32 source reduction. Compact inputs have already been locally reduced by the caller. Grouping and client BF16 rounding can change numerical results between modes/layouts; no bitwise equivalence is promised.

## Events and lifetime

With `async_op=False`, completion is ordered onto the caller's CUDA stream. With True, wait on the consumer stream:

```python
received = buffer.dispatch(x, topk_idx=indices, async_op=True)
received.event.current_stream_wait()
```

`previous_event` is an optional native `torch.cuda.Event` for an input producer on another stream. Otherwise the communication stream waits for the caller's current stream. The returned event tracks tensor ownership on the consumer stream. Keep the current device set to this process's GPU and do not mutate inputs until communication completes. Fresh exact receive-count readback can block the host. Cached dispatch uses metadata shape for N and the handle's saved host C, without reading GPU offsets back. Asynchronous mode does not imply CUDA Graph compatibility.

Native `dispatch_main` requires `cached_expanded_capacity` immediately after `cached_metadata`: fresh calls pass `0` (ignored), cached calls pass the original handle's C. The nine metadata tensors remain on the GPU, with unchanged ordering. Direct native callers must update this argument, rebuild core and restart the process; public `CudaBuffer.dispatch` arguments are unchanged.

Close collectively after consumers finish, preferably with `with CudaBuffer(...)`. Closing drains GPU work, coordinates unmapping across ranks, then frees allocations. Do not use handles after close or across buffers. Fatal rank failure relies on the process-group/launcher timeout.

Communication region starts, row strides and top-k/weights/x field starts are aligned to 32 bytes. Public tensors retain their contiguous logical shapes; CUDA EP hidden sizes must be positive multiples of 256. Native extensions and buffers must be rebuilt after this byte-ABI change.

## Specialized combine epilogue

CUDA EP requires BF16 x and a positive hidden size divisible by 256 (`32 * sizeof(int4) / sizeof(BF16)`). Independent row operators and the byte-layout arithmetic utility retain their existing hidden-size support.

The combine epilogue is compiled separately for `(H,K,R,experts_per_rank,Tmax)` using the existing PyTorch C++/CUDA JIT. `CudaBuffer` prepares this module during collective initialization, before allocating/exporting IPC windows; initialization errors are gathered before peers continue. It binds `_dispatch_epilogue` and `_combine_epilogue` to their separately specialized module entries. Public calls invoke these callables directly. Low-level callers explicitly use `prepare_combine_epilogue(...).combine_epilogue`; the core native module does not expose a combine epilogue dispatcher. A new geometry incurs a cold compilation; reuse a fixed buffer configuration for steady-state calls.

The kernel derives P from R/K/reduction mode and derives return-row bytes from H/K. It uses 16-byte int4 loads/stores, an unroll factor chosen from H at compile time, and ordered FP32 accumulation followed by one BF16 cast. Bias is added after the partial sums, and routing weights remain an independent FP32 payload. There is no hidden tail path. This does not implement TMA or guarantee equality with every DeepEP rounding recipe.

Combine main also specializes `(H,K,R,experts_per_rank,Tmax)`. `CudaBuffer` prepares and retains its module before allocating/exporting IPC windows. The core extension keeps validation, runtime ownership and epoch allocation; it calls the compiled host launcher through a named capsule. One cooperative kernel performs both barriers and payload work. Geometry mismatches fail before entering either barrier. `experts_per_rank` validates metadata geometry; the return kernel does not divide expert IDs.

H/K/R/Tmax and the derived return-row stride are compile-time constants; actual N and input row capacity remain runtime values. The kernel uses int4 BF16 transfers, ascending-K FP32 local reduction, 256-thread blocks and one block per reported SM. Warp `w` in block `b` starts at receive row `w * gridDim.x + b` and advances by `(blockDim.x / 32) * gridDim.x`, refreshing per-row metadata each iteration. The warp count remains fixed at eight per block; this grid does not pin blocks to physical SMs. Each valid lane loads its expanded slot once for validation and weight return. Warp shuffles compact slots in ascending original-k order before local reduction, or broadcast each assignment slot for cooperative copying. All lanes participate in the shuffles. Compact inputs and expanded rows with exactly one local assignment share a direct BF16 copy path. This bypass preserves payload bits, including signed zero and NaN payloads; rows with multiple local assignments retain ordered FP32 reduction. TMA and PDL are not implemented. Dispatch main still uses runtime geometry; dispatch epilogue specializes H/K/local-expert-count.

CTA 0 exchanges entry epochs before a grid sync permits writes. All writers synchronize the grid after their payload work, then CTA 0 exchanges completion epochs with system-scope release/acquire. For the current ordinary peer stores, the grid sync and cumulative system-scope release establish visibility without a separate per-thread system fence. No entry-side starting grid sync or exit-side ending grid sync is needed: earlier stream work has completed, and the separate epilogue waits for the entire cooperative kernel. Empty ranks still launch and participate. The launcher uses `has_pass` to check cooperative support and actual-kernel occupancy once per variant/host thread; each host thread must use a fixed GPU, since switching devices does not repeat the checks. It retains one block per SM. Each combine reserves two consecutive epochs in the same sequence and stream as standalone barriers. CUDA Graph replay still needs a device-side epoch protocol.

Low-level `native.combine_main(...)` requires `launcher=prepare_combine_main(H,K,R,experts_per_rank,Tmax).launcher`. Keep the module alive while using the launcher. The cooperative launcher uses capsule name `nano_ep.CombineMainSpecialization`; rebuild the core extension together with specialized modules and restart the process. Public `CudaBuffer.combine` arguments, window byte layout and output layouts are unchanged.


## Specialized dispatch epilogue

`prepare_dispatch_epilogue(H, K, experts_per_rank, cached_mode=...)` compiles a
fused dispatch epilogue for one geometry and cache mode. `CudaBuffer` prewarms
both fresh and cached modules before allocating/exporting IPC windows and
reports build failures collectively.
H/K/local-expert-count are compile-time constants; row bytes and field offsets
are derived from H/K. Actual N/C and rank remain runtime values. R, Tmax and
alignment are not specialization keys: the kernels use the existing padded
expert offsets. Fresh dispatch localizes IDs, reserves slots, saves all receive
metadata and copies payloads in one kernel. Each lane retains its local ID and
slot in registers; expanded copies broadcast slots with warp shuffles instead
of rereading freshly written metadata. `NANO_EP_CACHED_MODE` fixes `kCachedMode`
at JIT compilation; module names and cache keys distinguish both modes. The
cached specialization reads existing mappings without allocating cursors or
reserving slots. Host launchers do not dispatch between cache-mode templates.
Payload copies use 16-byte `int4` loads/stores
in both compact and expanded layouts, preserving eight BF16 bit patterns per
vector. H%256 and the aligned communication fields cover complete warp vectors;
CUDA-allocated outputs preserve row alignment. A compile-time unroll factor is
the largest divisor up to four of H/256, matching combine's selection rule;
the inner vector loop is explicitly unrolled in both layouts. Outputs use `empty`.
The `kExpand && kZeroPadding` variant clears only expert gaps, including
optional weight gaps, in the same copy kernel. All warps participate even when
they have no receive row. A warp scan over each wave of 32 experts maps padding
ordinals to `[offset[e] + count[e], offset[e+1])`; immutable counts and offsets
are retained for cached replay. Compact ignores zero padding. The 256-thread
blocks and `ceil(N/8)` grids are unchanged. No TMA is used.

Low-level callers must load the core extension for `DispatchState` registration,
then use the geometry module:

```python
module = prepare_dispatch_epilogue(H, K, experts_per_rank, cached_mode=state.metadata_prepared)
x, indices, weights = module.dispatch_epilogue(state, expand, with_weights, zero_padding)
```

The old core `native.dispatch_epilogue` entry and all standalone
`build_receive_metadata` entries are removed. Geometry mismatches fail before metadata preparation. Rebuild the
core extension and restart the process; public `CudaBuffer.dispatch` arguments,
wire layout, metadata ordering and output ownership are unchanged. The internal
factory now requires keyword-only `cached_mode`; a mode mismatch is rejected
before state mutation. Repeated epilogue calls on a prepared state use the
cached module, including states originally created by a fresh dispatch.


### Fused fresh notify

Fresh dispatch uses a cooperative SM-sized kernel: four notify warps per CTA
count routes in shared memory while four payload warps reserve/copy tokens.
A persistent `notify_reduction[R+E]` region packs CTA arrivals in the high 32
bits and counts in the low 32 bits. CTA 0 consumes/resets each word before
exchanging peer counts; notify-only named barriers preserve payload progress.
Empty ranks still launch and contribute zeros. Resource checks precede window
mutation and communication. Cached dispatch keeps eight payload warps.

The window now stores barrier signals, aligned notify reduction, aligned peer
counts plus recv_next, and the shared dispatch/return arena. Rebuild extensions,
restart the process and recreate IPC buffers/handles after this byte-layout
change. Python signatures and metadata formats are unchanged. Peer counts keep
-1 pending / nonnegative ready. Expert reduction, alignment and prefix sums run inside the fused notify stage.
Exact host count readback remains on host; entry/exit peer barriers are fused into the cooperative dispatch kernel.


The standalone native `reserve_and_put` API has been removed. Fresh/cached
main calls share one internal launcher whose mode follows the presence of
cached slots; there is no `count_routes` argument. Test-only probes call
the payload device helper from `dispatch_main.cuh` with caller-owned slots.
Public `CudaBuffer.dispatch` is unchanged; low-level callers use dispatch_main.

Standalone native `notify_counts` and its independent kernel are also removed.
Notify warps call the exchange device helper directly. Test-only probes cover
delayed peers, zero counts and receive-slot reuse through that same helper.


Dispatch main specializes `(H,K,R,experts_per_rank,Tmax,expert_alignment)` through
`prepare_dispatch_main(..., expert_alignment).launcher`. CudaBuffer prewarms
alignment=1 before IPC allocation/export, then compiles other alignments on first
use before stream work. Modules are retained by alignment; cached calls select
the handle's original alignment. Geometry mismatches fail before window mutation.

CTA 0 reuses shared outgoing counts for incoming counts after every send is
published. After notify-group synchronization, warp 0 emits int64 rank inclusive
prefix[R]; warp 1 sums source counts into actual expert counts[L], aligns in
int64 registers and scans padded counts into offsets[L+1]. No raw received
[R+E] tensor or standalone Torch count/padding/scan kernels are needed. N/C still
come from host reads of the final prefixes for exact allocation. The payload completion barrier now runs inside the cooperative main kernel;
the cached receive metadata contract is unchanged.

The internal launcher ABI and prepare_dispatch_main signature changed; rebuild
extensions and restart processes. Device helpers encode rank/expert geometry and timeout as templates; total
experts derives from R*experts_per_rank. Public dispatch signatures, metadata
order and wire layout remain unchanged.


Standalone route_counts/pack_rows/expand_rows/reduce_rows operators and
backends.ops have been removed. Their work belongs to the integrated
dispatch/combine pipeline; old direct operator calls are no longer supported.


Fresh and cached dispatch both launch cooperatively, even for T=0. The core
validates support/occupancy for the selected variant before reserving two epochs
or mutating the window. CTA 0 exchanges entry readiness, then every CTA grid-syncs
before writing. A grid sync orders all writers before CTA 0's system-scope release
publication of completion; release/acquire cumulativity carries their writes to peers.
The following epilogue waits for kernel completion on the same stream.
Fresh recv_next is reset by CTA 0's thread 0, followed by CTA synchronization
and system-scope release publication of entry readiness. Cached calls preserve
the counter; no standalone host memset is queued. Exact N/C reads remain on host. The
internal callback ABI changed; rebuild core and geometry modules together.
