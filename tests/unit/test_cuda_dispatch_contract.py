"""CPU-only checks of Python argument validation; no CUDA work is simulated."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from nano_ep import EPConfig
from nano_ep.backends.cuda import CudaBuffer



@dataclass(frozen=True)
class TensorStorage:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cuda", 0)
    contiguous: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def inputs():
    buffer = CudaBuffer.__new__(CudaBuffer)
    buffer._closed, buffer._device = False, 0
    buffer._config = EPConfig(6, 3, 4, 5)
    x = TensorStorage((2, 4))
    idx = TensorStorage((2, 3), torch.int64)
    # Only the Python boundary's scalar fields/storage descriptors are supplied.
    handle = SimpleNamespace(
        owner=buffer,
        config=buffer._config,
        topk_idx=idx,
        alignment=4,
        do_expand=True,
        x_dtype=torch.float32,
    )
    return buffer, x, idx, handle


@pytest.mark.parametrize(
    "kind",
    [
        "closed",
        "missing_indices",
        "foreign_owner",
        "config",
        "cached_indices",
        "alignment_change",
        "layout_change",
        "alignment_zero",
        "alignment_float",
        "alignment_bool",
        "layout_type",
        "x_cpu",
        "x_dtype",
        "x_shape",
        "x_stride",
        "hidden",
        "capacity",
        "idx_dtype",
        "idx_shape",
        "idx_stride",
        "idx_device",
        "cached_tokens",
        "cached_dtype",
        "weight_dtype",
        "weight_shape",
        "weight_stride",
        "weight_device",
    ],
)
def test_dispatch_contract_before_native_work(inputs, kind):
    buffer, x, idx, handle = inputs
    kwargs = {"topk_idx": idx}
    error = ValueError
    if kind == "closed":
        buffer._closed = True
        error = RuntimeError
    elif kind == "missing_indices":
        kwargs = {}
    elif kind in (
        "foreign_owner",
        "config",
        "cached_indices",
        "alignment_change",
        "layout_change",
        "cached_tokens",
        "cached_dtype",
    ):
        kwargs = {"handle": handle}
        if kind == "foreign_owner":
            handle.owner = object()
        elif kind == "config":
            handle.config = EPConfig(12, 3, 4, 5)
        elif kind == "cached_indices":
            kwargs["topk_idx"] = idx
        elif kind == "alignment_change":
            kwargs["alignment"] = 16
        elif kind == "layout_change":
            kwargs["do_expand"] = False
        elif kind == "cached_tokens":
            x = replace(x, shape=(1, 4))
        else:
            x = replace(x, dtype=torch.bfloat16)
    elif kind.startswith("alignment_"):
        kwargs["alignment"] = {
            "alignment_zero": 0,
            "alignment_float": 4.0,
            "alignment_bool": True,
        }[kind]
    elif kind == "layout_type":
        kwargs["do_expand"] = 1
    elif kind == "x_cpu":
        x = replace(x, device=torch.device("cpu"))
    elif kind == "x_dtype":
        x = replace(x, dtype=torch.float64)
    elif kind == "x_shape":
        x = replace(x, shape=(8,))
    elif kind == "x_stride":
        x = replace(x, contiguous=False)
    elif kind == "hidden":
        x = replace(x, shape=(2, 8))
    elif kind == "capacity":
        x = replace(x, shape=(6, 4))
    elif kind == "idx_dtype":
        kwargs["topk_idx"] = replace(idx, dtype=torch.float32)
    elif kind == "idx_shape":
        kwargs["topk_idx"] = replace(idx, shape=(2, 2))
    elif kind == "idx_stride":
        kwargs["topk_idx"] = replace(idx, contiguous=False)
    elif kind == "idx_device":
        kwargs["topk_idx"] = replace(idx, device=torch.device("cuda", 1))
    else:
        weight = TensorStorage((2, 3))
        if kind == "weight_dtype":
            weight = replace(weight, dtype=torch.float64)
        elif kind == "weight_shape":
            weight = replace(weight, shape=(1, 3))
        elif kind == "weight_stride":
            weight = replace(weight, contiguous=False)
        else:
            weight = replace(weight, device=torch.device("cpu"))
        kwargs["weights"] = weight
    with pytest.raises(error):
        buffer.dispatch(x, **kwargs)
