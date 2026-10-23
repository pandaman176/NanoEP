"""Rank-local CUDA IPC dispatch and combine."""

import os
import socket
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from nano_ep.backends._build import invoke, load_extension
from nano_ep.config import EPConfig


@dataclass
class NativeLayout:
    src_ranks: torch.Tensor
    src_tokens: torch.Tensor
    local_topk_idx: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    expanded_slots: torch.Tensor
    recv_count: int
    expanded_capacity: int


@dataclass
class NativeHandle:
    config: EPConfig
    rank: int
    world_size: int
    do_expand: bool
    alignment: int
    topk_idx: torch.Tensor
    layout: NativeLayout
    dst_slots: torch.Tensor
    send_counts: torch.Tensor
    owner: object
    x_dtype: torch.dtype


@dataclass
class NativeDispatchResult:
    x: torch.Tensor
    topk_idx: torch.Tensor | None
    topk_weights: torch.Tensor | None
    handle: NativeHandle
    event: "CudaEvent"


@dataclass
class NativeCombineResult:
    x: torch.Tensor
    topk_weights: torch.Tensor | None
    event: "CudaEvent"


class CudaEvent:
    def __init__(self, event: torch.cuda.Event, keepalive: tuple = (), tensors: tuple[torch.Tensor, ...] = ()):
        self.event = event
        self.keepalive = keepalive
        self._tensors = tensors

    def current_stream_wait(self):
        """Wait on the consumer stream and protect tensors used on that stream."""
        stream = torch.cuda.current_stream()
        stream.wait_event(self.event)
        for tensor in self._tensors:
            tensor.record_stream(stream)


def close_peer_runtime(runtime, group, device):
    """Collectively drain, unmap all importers, then free local windows.

    Every rank participates in both host barriers. Borrowed window views must
    no longer be used. Phase 2 is not entered if draining, unmapping or either
    host barrier fails; errors propagate to the distributed launcher.
    """
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    invoke(runtime.close_peer_mappings)
    dist.barrier(group=group)
    invoke(runtime.close)


class CudaBuffer:
    def __init__(self, group, config: EPConfig, *, allow_multiple_reduction: bool = False):
        'Collectively create a single-node CUDA IPC buffer and communication stream.'
        self._group = group
        self._config = config
        self._allow_multiple_reduction = allow_multiple_reduction
        self._world_size = dist.get_world_size(group=group)
        self._rank = dist.get_rank(group=group)
        self._runtime = None
        self._comm_stream = None
        self._closed = True
        if dist.get_backend(group) != "gloo":
            raise ValueError("CudaBuffer requires a Gloo control group")

        local_info = {"error": None}
        try:
            if not isinstance(allow_multiple_reduction, bool):
                raise TypeError("allow_multiple_reduction must be bool")
            self._device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(self._device)
            local_info.update(
                config=asdict(config),
                allow_multiple_reduction=allow_multiple_reduction,
                hostname=socket.gethostname(),
                gpu_uuid=str(properties.uuid),
                local_rank=int(os.environ["LOCAL_RANK"]),
                local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            )
        except Exception as exc:  # noqa: BLE001 - Report local failures before any rank raises.
            local_info["error"] = f"{type(exc).__name__}: {exc}"

        all_info = [None] * self._world_size
        dist.all_gather_object(all_info, local_info, group=group)
        errors = [f"rank {rank}: {info['error']}" for rank, info in enumerate(all_info) if info["error"] is not None]
        if errors:
            raise RuntimeError("CudaBuffer environment check failed: " + "; ".join(errors))

        expected_config = all_info[0]["config"]
        if any(info["config"] != expected_config for info in all_info):
            raise ValueError("EPConfig mismatch across ranks")
        if any(info["allow_multiple_reduction"] != allow_multiple_reduction for info in all_info):
            raise ValueError("allow_multiple_reduction mismatch across ranks")

        hostnames = {info["hostname"] for info in all_info}
        if len(hostnames) != 1:
            raise ValueError(f"CudaBuffer requires one node, got: {hostnames}")

        gpu_uuids = [info["gpu_uuid"] for info in all_info]
        if len(set(gpu_uuids)) != self._world_size:
            raise ValueError(f"Ranks must use distinct GPUs: {gpu_uuids}")

        if any(info["local_world_size"] != self._world_size for info in all_info):
            raise ValueError("Group must contain all local workers")

        local_ranks = sorted(info["local_rank"] for info in all_info)
        if local_ranks != list(range(self._world_size)):
            raise ValueError(f"Invalid local rank assignment: {local_ranks}")

        if config.num_experts % self._world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")

        allocation = {"handle": None, "error": None}
        try:
            self._backend = load_extension()
            (
                self._signal_offset,
                self._count_offset,
                self._payload_offset,
                self._return_offset,
                self._num_bytes,
            ) = invoke(
                self._backend.window_layout,
                self._world_size,
                config.num_max_tokens_per_rank,
                config.hidden,
                config.num_topk,
                config.num_experts,
                allow_multiple_reduction,
            )
            self._runtime = invoke(self._backend.PeerRuntime, self._rank, self._world_size, self._num_bytes)
            # All-one bytes represent int64 -1; recv_next and barrier signals stay zero.
            window = invoke(self._runtime.local_window)
            window[self._count_offset : self._count_offset + (self._world_size + config.num_experts) * 8].fill_(255)
            torch.cuda.current_stream(self._device).synchronize()
            del window
            self._handle = invoke(self._runtime.export_local_handle)
            allocation["handle"] = self._handle
        except Exception as exc:  # noqa: BLE001 - Report local failures before any rank raises.
            allocation["error"] = f"{type(exc).__name__}: {exc}"

        allocations = [None] * self._world_size
        dist.all_gather_object(allocations, allocation, group=group)
        errors = [f"rank {rank}: {result['error']}" for rank, result in enumerate(allocations) if result["error"] is not None]
        stage = "allocation/export"

        if not errors:
            handles = [result["handle"] for result in allocations]
            import_error = None
            try:
                invoke(self._runtime.import_peer_handles, handles)
                self._comm_stream = torch.cuda.Stream(device=self._device)
            except Exception as exc:  # noqa: BLE001 - Report local failures before any rank raises.
                import_error = f"{type(exc).__name__}: {exc}"

            import_errors = [None] * self._world_size
            dist.all_gather_object(import_errors, import_error, group=group)
            errors = [f"rank {rank}: {error}" for rank, error in enumerate(import_errors) if error is not None]
            stage = "import/stream"

        if errors:
            # No window users are exposed before collective initialization succeeds.
            unmap_error = None
            try:
                if self._runtime is not None:
                    torch.cuda.synchronize(self._device)
                    invoke(self._runtime.close_peer_mappings)
            except Exception as exc:  # noqa: BLE001 - Keep allocations alive if any importer cannot unmap.
                unmap_error = f"{type(exc).__name__}: {exc}"

            unmap_errors = [None] * self._world_size
            dist.all_gather_object(unmap_errors, unmap_error, group=group)
            cleanup_errors = [f"rank {rank} unmap: {error}" for rank, error in enumerate(unmap_errors) if error is not None]

            # A successful exchange confirms every importer has closed its mappings.
            if not cleanup_errors:
                free_error = None
                try:
                    if self._runtime is not None:
                        invoke(self._runtime.close)
                except Exception as exc:  # noqa: BLE001 - Include cleanup errors in the collective failure.
                    free_error = f"{type(exc).__name__}: {exc}"
                free_errors = [None] * self._world_size
                dist.all_gather_object(free_errors, free_error, group=group)
                cleanup_errors = [f"rank {rank} free: {error}" for rank, error in enumerate(free_errors) if error is not None]

            self._comm_stream = None
            raise RuntimeError(f"CudaBuffer {stage} failed: " + "; ".join(errors + cleanup_errors))

        self._closed = False

    def dispatch(
        self,
        x: torch.Tensor,
        *,
        topk_idx: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        handle: NativeHandle | None = None,
        do_expand: bool | None = None,
        alignment: int | None = None,
        zero_padding: bool = False,
        previous_event: torch.cuda.Event | None = None,
        async_op: bool = False,
    ) -> NativeDispatchResult:
        'Dispatch payloads using fresh or cached routing; return exact receive layouts.'
        if self._closed:
            raise RuntimeError("CudaBuffer is closed")
        if handle is not None:
            if handle.owner is not self:
                raise ValueError("cached handle belongs to another CudaBuffer")
            if handle.config != self._config:
                raise ValueError("cached handle configuration does not match this CudaBuffer")
            if topk_idx is not None:
                raise ValueError("cached dispatch uses the handle snapshot; omit topk_idx")
            if alignment is not None and alignment != handle.alignment:
                raise ValueError("cached dispatch must retain the handle alignment")
            if do_expand is not None and do_expand != handle.do_expand:
                raise ValueError("cached dispatch must retain the handle layout")
        elif topk_idx is None:
            raise ValueError("fresh dispatch requires topk_idx")
        if alignment is not None and (isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0):
            raise ValueError("alignment must be a positive integer")
        if do_expand is not None and not isinstance(do_expand, bool):
            raise ValueError("do_expand must be bool or None")

        device = torch.device("cuda", self._device)
        if x.device != device or x.ndim != 2 or not x.is_contiguous() or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("x must be contiguous CUDA float32/bfloat16 [T,H] on this buffer's device")
        if x.shape[1] != self._config.hidden or x.shape[0] > self._config.num_max_tokens_per_rank:
            raise ValueError("x hidden dimension or source token count does not match the buffer configuration")
        routing = handle.topk_idx if handle is not None else topk_idx
        shape = (x.shape[0], self._config.num_topk)
        if routing.device != device or routing.dtype != torch.int64 or routing.shape != shape or not routing.is_contiguous():
            raise ValueError("routing snapshot/topk_idx must be contiguous CUDA int64 [T,K] matching the buffer configuration")
        if weights is not None and (
            weights.device != device or weights.dtype != torch.float32 or weights.shape != shape or not weights.is_contiguous()
        ):
            raise ValueError("weights must be contiguous CUDA float32 [T,K] matching x and routing")

        if handle is not None and x.dtype != handle.x_dtype:
            raise ValueError("cached dispatch must retain the handle x dtype")

        if handle is None:
            expand = False if do_expand is None else do_expand
            expert_alignment = 1 if alignment is None else alignment
            cached_metadata = []
        else:
            expand = handle.do_expand
            expert_alignment = handle.alignment
            layout = handle.layout
            cached_metadata = [
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

        with torch.cuda.device(self._device):
            compute_stream = torch.cuda.current_stream(self._device)
            if previous_event is None:
                self._comm_stream.wait_stream(compute_stream)
            else:
                self._comm_stream.wait_event(previous_event)

            with torch.cuda.stream(self._comm_stream):
                # Inputs can be released on their allocation stream while comm reads them.
                input_tensors = (x, routing) + ((weights,) if weights is not None else ())
                for tensor in (*input_tensors, *cached_metadata):
                    tensor.record_stream(self._comm_stream)

                state = invoke(
                    self._backend.dispatch_main,
                    self._runtime,
                    x,
                    routing,
                    weights,
                    self._config.num_experts,
                    self._config.num_max_tokens_per_rank,
                    expert_alignment,
                    cached_metadata,
                    self._allow_multiple_reduction,
                )
                outputs = invoke(self._backend.dispatch_epilogue, state, expand, weights is not None, zero_padding)
                metadata = state.metadata
                recv_count, expanded_capacity = state.counts.tolist()  # Native-owned CPU int64 [2].

                if handle is None:
                    layout = NativeLayout(
                        src_ranks=metadata[0],
                        src_tokens=metadata[1],
                        local_topk_idx=metadata[2],
                        expert_counts=metadata[3],
                        expert_offsets=metadata[4],
                        expanded_slots=metadata[5],
                        recv_count=recv_count,
                        expanded_capacity=expanded_capacity,
                    )
                    handle = NativeHandle(
                        config=self._config,
                        rank=self._rank,
                        world_size=self._world_size,
                        do_expand=expand,
                        alignment=expert_alignment,
                        topk_idx=metadata[8],
                        layout=layout,
                        dst_slots=metadata[6],
                        send_counts=metadata[7],
                        owner=self,
                        x_dtype=x.dtype,
                    )

                completed = torch.cuda.Event()
                completed.record(self._comm_stream)

            event = CudaEvent(
                completed,
                keepalive=(self, x, routing, weights, state, handle),
                tensors=(*outputs, *metadata),
            )
            if not async_op:
                event.current_stream_wait()

        return NativeDispatchResult(
            x=outputs[0],
            topk_idx=None if expand else outputs[1],
            topk_weights=outputs[2] if weights is not None else None,
            handle=handle,
            event=event,
        )

    def combine(
        self,
        x: torch.Tensor,
        handle: NativeHandle,
        *,
        weights: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        previous_event: torch.cuda.Event | None = None,
        async_op: bool = False,
    ) -> NativeCombineResult:
        'Return expert outputs to their source and reduce in FP32, without gate weighting.'
        if self._closed:
            raise RuntimeError("CudaBuffer is closed")
        if handle.owner is not self:
            raise ValueError("combine handle belongs to another CudaBuffer")
        if handle.config != self._config:
            raise ValueError("combine handle configuration does not match this CudaBuffer")

        device = torch.device("cuda", self._device)
        layout = handle.layout
        rows = layout.expanded_capacity if handle.do_expand else layout.recv_count
        hidden = self._config.hidden
        if x.device != device or x.dtype not in (torch.float32, torch.bfloat16) or x.shape != (rows, hidden) or not x.is_contiguous():
            raise ValueError("combine x must be contiguous CUDA float32/bfloat16 with the handle's exact receive shape")
        weight_shape = (rows,) if handle.do_expand else (rows, self._config.num_topk)
        if weights is not None and (
            weights.device != device or weights.dtype != torch.float32 or weights.shape != weight_shape or not weights.is_contiguous()
        ):
            raise ValueError("combine weights must be contiguous CUDA float32 with the handle's receive layout")
        source_tokens = handle.topk_idx.shape[0]
        if bias is not None and (
            bias.device != device or bias.dtype != x.dtype or bias.shape != (source_tokens, hidden) or not bias.is_contiguous()
        ):
            raise ValueError("combine bias must be contiguous source [T,H], with the x dtype on this buffer's device")

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
        with torch.cuda.device(self._device):
            compute_stream = torch.cuda.current_stream(self._device)
            if previous_event is None:
                self._comm_stream.wait_stream(compute_stream)
            else:
                self._comm_stream.wait_event(previous_event)

            with torch.cuda.stream(self._comm_stream):
                input_tensors = (x, *metadata)
                if weights is not None:
                    input_tensors += (weights,)
                if bias is not None:
                    input_tensors += (bias,)
                for tensor in input_tensors:
                    tensor.record_stream(self._comm_stream)

                returned = invoke(
                    self._backend.combine_main,
                    self._runtime,
                    x,
                    metadata,
                    weights,
                    self._config.num_max_tokens_per_rank,
                    handle.do_expand,
                    self._allow_multiple_reduction,
                )
                outputs = invoke(
                    self._backend.combine_epilogue,
                    returned,
                    handle.topk_idx,
                    hidden,
                    x.dtype,
                    self._config.num_experts,
                    self._world_size,
                    self._config.num_max_tokens_per_rank,
                    bias,
                    weights is not None,
                    handle.do_expand,
                    self._allow_multiple_reduction,
                )
                completed = torch.cuda.Event()
                completed.record(self._comm_stream)

            # The borrowed window is owned by runtime, not the PyTorch allocator.
            event = CudaEvent(completed, keepalive=(self, x, handle, weights, bias, returned), tensors=tuple(outputs))
            if not async_op:
                event.current_stream_wait()

        return NativeCombineResult(x=outputs[0], topk_weights=outputs[1] if weights is not None else None, event=event)

    def close(self):
        """Collectively drain, close peer mappings, then free local windows.

        Idempotent on this object. No outstanding GPU operation may refer to
        freed windows. The destructor must not start collectives; callers close
        explicitly. Reject dispatch/combine after close or with foreign handles.
        """
        if self._closed:
            return
        close_peer_runtime(self._runtime, self._group, self._device)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
