"""Load the CUDA extension lazily from self-contained native sources."""

import os
from functools import lru_cache
from pathlib import Path


def native_source_dir() -> Path:
    package = Path(__file__).resolve().parents[1]
    bundled = package / "_csrc"
    return bundled if bundled.is_dir() else package.parent / "csrc"


@lru_cache(maxsize=1)
def load_extension():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise RuntimeError("NanoEP requires an NVIDIA CUDA GPU")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME/nvcc is required to compile NanoEP")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    source = native_source_dir()
    return load(
        name="nano_ep_native",
        sources=[str(source / name) for name in ("bindings.cpp", "ops.cu", "peer.cu", "pipeline.cu")],
        extra_include_paths=[str(source / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo"],
        verbose=os.environ.get("NANO_EP_BUILD_VERBOSE") == "1",
    )


def invoke(function, *args):
    return function(*args)
