"""Rank-local CUDA IPC dispatch and combine."""

import os
import socket
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch.utils import deterministic

from nano_ep.backends._build import load_extension
from nano_ep.backends._sms import nvlink_gbs, theoretical_num_sms
from nano_ep.config import EPConfig


def check_torch_deterministic():
    if torch.are_deterministic_algorithms_enabled() and deterministic.fill_uninitialized_memory:
        raise RuntimeError(
            "NanoEP does not support deterministic algorithms with fill_uninitialized_memory enabled; disable one of these PyTorch settings"
        )


@dataclass
class NativeLayout:
    recv_metadata: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    recv_rank_prefix: torch.Tensor
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
    num_sms: int | None = None


@dataclass
class NativeDispatchResult:
    x: torch.Tensor
    topk_idx: torch.Tensor | None
    topk_weights: torch.Tensor | None
    handle: NativeHandle
    event: "CudaEvent | None"


@dataclass
class NativeCombineResult:
    x: torch.Tensor
    topk_weights: torch.Tensor | None
    event: "CudaEvent | None"


class CudaEvent:
    def __init__(self, event, keepalive: tuple = ()):
        self.event = event
        self.keepalive = keepalive

    def current_stream_wait(self):
        """Wait on the consumer stream and protect tensors used on that stream."""
        self.event.current_stream_wait()


def close_peer_runtime(runtime, group, device):
    """Collectively drain, unmap all importers, then free local windows.

    Every rank participates in both host barriers. Borrowed window views must
    no longer be used. Phase 2 is not entered if draining, unmapping or either
    host barrier fails; errors propagate to the distributed launcher.
    """
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    runtime.close_peer_mappings()
    dist.barrier(group=group)
    runtime.close()


class CudaBuffer:
    def __init__(self, group, config: EPConfig, *, allow_multiple_reduction: bool = False):
        "Collectively create a single-node CUDA IPC buffer and communication stream."
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
            self._device_sms = properties.multi_processor_count
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
        self._gpu_uuids = tuple(gpu_uuids)
        self._auto_num_sms = None

        if any(info["local_world_size"] != self._world_size for info in all_info):
            raise ValueError("Group must contain all local workers")

        local_ranks = sorted(info["local_rank"] for info in all_info)
        if local_ranks != list(range(self._world_size)):
            raise ValueError(f"Invalid local rank assignment: {local_ranks}")

        if config.num_experts % self._world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")
        if config.hidden % 256 != 0:
            raise ValueError("CUDA EP hidden must be a positive multiple of 256")

        allocation = {"handle": None, "error": None}
        try:
            self._backend = load_extension()
            (
                self._signal_offset,
                self._count_offset,
                self._payload_offset,
                self._return_offset,
                self._num_bytes,
            ) = self._backend.window_layout(
                self._world_size,
                config.num_max_tokens_per_rank,
                config.hidden,
                config.num_topk,
                config.num_experts,
                allow_multiple_reduction,
            )
            self._runtime = self._backend.PeerRuntime(self._rank, self._world_size, self._num_bytes)
            # All-one bytes represent int64 -1; send_next and barrier signals stay zero.
            window = self._runtime.local_window()
            window[self._count_offset : self._count_offset + (self._world_size + config.num_experts) * 8].fill_(255)
            torch.cuda.current_stream(self._device).synchronize()
            del window
            self._handle = self._runtime.export_local_handle()
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
                self._runtime.import_peer_handles(handles)
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
                    self._runtime.close_peer_mappings()
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
                        self._runtime.close()
                except Exception as exc:  # noqa: BLE001 - Include cleanup errors in the collective failure.
                    free_error = f"{type(exc).__name__}: {exc}"
                free_errors = [None] * self._world_size
                dist.all_gather_object(free_errors, free_error, group=group)
                cleanup_errors = [f"rank {rank} free: {error}" for rank, error in enumerate(free_errors) if error is not None]

            self._comm_stream = None
            raise RuntimeError(f"CudaBuffer {stage} failed: " + "; ".join(errors + cleanup_errors))

        self._closed = False

    def _resolve_num_sms(self, num_sms: int | None, handle: NativeHandle | None) -> int:
        if num_sms is None:
            return self._device_sms if handle is None or handle.num_sms is None else handle.num_sms
        if isinstance(num_sms, bool) or not isinstance(num_sms, int) or not 0 <= num_sms <= self._device_sms:
            raise ValueError(f"num_sms must be an integer between 0 and {self._device_sms}")
        if num_sms > 0:
            return num_sms
        if self._auto_num_sms is None:
            self._auto_num_sms = theoretical_num_sms(
                self._device_sms, self._world_size, self._config.num_experts, self._config.num_topk, nvlink_gbs(self._gpu_uuids)
            )
        return self._auto_num_sms

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
        num_sms: int | None = None,
    ) -> NativeDispatchResult:
        "Dispatch payloads using fresh or cached routing; return exact receive layouts."
        check_torch_deterministic()
        if self._closed:
            raise RuntimeError("CudaBuffer is closed")
        if handle is not None:
            if handle.owner is not self:
                raise ValueError("cached handle belongs to another CudaBuffer")
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

        routing = handle.topk_idx if handle is not None else topk_idx
        if x.shape[1] != self._config.hidden:
            raise ValueError("x hidden dimension does not match the buffer configuration")
        if routing.shape != (x.shape[0], self._config.num_topk):
            raise ValueError("routing snapshot/topk_idx shape must match the buffer configuration")

        if handle is None:
            expand = False if do_expand is None else do_expand
            expert_alignment = 1 if alignment is None else alignment
            cached_metadata = []
            cached_expanded_capacity = 0
        else:
            expand = handle.do_expand
            expert_alignment = handle.alignment
            layout = handle.layout
            cached_expanded_capacity = layout.expanded_capacity
            cached_metadata = [
                layout.recv_metadata,
                layout.expert_counts,
                layout.expert_offsets,
                handle.dst_slots,
                handle.send_counts,
                handle.topk_idx,
                layout.recv_rank_prefix,
            ]

        main_sms = self._resolve_num_sms(num_sms, handle)
        outputs, metadata, counts, completed = self._backend.dispatch(
            self._runtime,
            x,
            routing,
            weights,
            self._config.num_experts,
            self._config.num_max_tokens_per_rank,
            expert_alignment,
            cached_metadata,
            cached_expanded_capacity,
            self._allow_multiple_reduction,
            expand,
            zero_padding,
            main_sms,
            self._comm_stream.cuda_stream,
            -1 if previous_event is None else previous_event.cuda_event,
            async_op,
        )
        recv_count, expanded_capacity = counts

        if handle is None:
            layout = NativeLayout(
                recv_metadata=metadata[0],
                expert_counts=metadata[1],
                expert_offsets=metadata[2],
                recv_rank_prefix=metadata[6],
                recv_count=recv_count,
                expanded_capacity=expanded_capacity,
            )
            handle = NativeHandle(
                config=self._config,
                rank=self._rank,
                world_size=self._world_size,
                do_expand=expand,
                alignment=expert_alignment,
                topk_idx=metadata[5],
                layout=layout,
                dst_slots=metadata[3],
                send_counts=metadata[4],
                owner=self,
                x_dtype=x.dtype,
                num_sms=main_sms,
            )

        event = CudaEvent(completed, keepalive=(self, x, routing, weights, handle)) if async_op else None

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
        num_sms: int | None = None,
    ) -> NativeCombineResult:
        "Return expert outputs to their source and reduce in FP32, without gate weighting."
        check_torch_deterministic()
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
        if x.device != device or x.dtype != torch.bfloat16 or x.shape != (rows, hidden) or not x.is_contiguous():
            raise ValueError("combine x must be contiguous CUDA bfloat16 with the handle's exact receive shape")
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
            layout.recv_metadata,
            layout.expert_counts,
            layout.expert_offsets,
            handle.dst_slots,
            handle.send_counts,
            handle.topk_idx,
            layout.recv_rank_prefix,
        ]
        main_sms = self._resolve_num_sms(num_sms, handle)
        with torch.cuda.device(self._device):
            outputs, completed = self._backend.combine(
                self._runtime,
                x,
                metadata,
                weights,
                bias,
                self._config.num_max_tokens_per_rank,
                handle.do_expand,
                self._allow_multiple_reduction,
                main_sms,
                self._comm_stream.cuda_stream,
                -1 if previous_event is None else previous_event.cuda_event,
                async_op,
            )

            event = CudaEvent(completed, keepalive=(self, x, handle, weights, bias)) if async_op else None

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
