"""CPU-only checks of dispatch's Python-level argument contract."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from nano_ep import EPConfig
from nano_ep.backends.cuda import CudaBuffer



@dataclass(frozen=True)
class TensorStorage:
    shape: tuple[int, ...]


@pytest.fixture
def inputs():
    buffer = CudaBuffer.__new__(CudaBuffer)
    buffer._closed, buffer._device = False, 0
    buffer._config = EPConfig(6, 3, 256, 5)
    x = TensorStorage((2, 256))
    idx = TensorStorage((2, 3))
    handle = SimpleNamespace(owner=buffer, topk_idx=idx, alignment=4, do_expand=True)
    return buffer, x, idx, handle


@pytest.mark.parametrize(
    "kind",
    [
        "closed",
        "missing_indices",
        "foreign_owner",
        "cached_indices",
        "alignment_change",
        "layout_change",
        "alignment_zero",
        "alignment_float",
        "alignment_bool",
        "layout_type",
        "hidden",
        "idx_shape",
        "cached_tokens",
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
        "cached_indices",
        "alignment_change",
        "layout_change",
        "cached_tokens",
    ):
        kwargs = {"handle": handle}
        if kind == "foreign_owner":
            handle.owner = object()
        elif kind == "cached_indices":
            kwargs["topk_idx"] = idx
        elif kind == "alignment_change":
            kwargs["alignment"] = 16
        elif kind == "layout_change":
            kwargs["do_expand"] = False
        else:
            x = replace(x, shape=(1, 256))
    elif kind.startswith("alignment_"):
        kwargs["alignment"] = {
            "alignment_zero": 0,
            "alignment_float": 4.0,
            "alignment_bool": True,
        }[kind]
    elif kind == "layout_type":
        kwargs["do_expand"] = 1
    elif kind == "hidden":
        x = replace(x, shape=(2, 8))
    else:
        kwargs["topk_idx"] = replace(idx, shape=(2, 2))
    with pytest.raises(error):
        buffer.dispatch(x, **kwargs)
