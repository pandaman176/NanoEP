"""Load the CUDA extension lazily from self-contained native sources."""

import os
from functools import cache, lru_cache
from pathlib import Path

import torch


def native_source_dir() -> Path:
    package = Path(__file__).resolve().parents[1]
    bundled = package / "_csrc"
    return bundled if bundled.is_dir() else package.parent / "csrc"


@lru_cache(maxsize=1)
def load_extension():
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


@cache
def prepare_combine_epilogue(hidden: int, topk: int, world_size: int, experts_per_rank: int, max_tokens: int):
    """Compile one geometry; CudaBuffer prewarms it before IPC collectives."""
    from torch.utils.cpp_extension import load

    if (
        hidden <= 0
        or hidden > 2**31 - 1
        or hidden % 256
        or not 1 <= topk <= 32
        or not 1 <= world_size <= 32
        or experts_per_rank <= 0
        or max_tokens <= 0
    ):
        raise RuntimeError("invalid combine epilogue geometry: H must be a positive multiple of 256, K/R in [1,32], L/Tmax positive")
    source = native_source_dir()
    geometry = (hidden, topk, world_size, experts_per_rank, max_tokens)
    names = ("HIDDEN", "TOPK", "WORLD_SIZE", "EXPERTS_PER_RANK", "MAX_TOKENS")
    defines = [f"-DNANO_EP_{name}={value}" for name, value in zip(names, geometry, strict=True)]
    return load(
        name="nano_ep_combine_" + "_".join(map(str, geometry)),
        sources=[str(source / "combine_epilogue.cu")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo", *defines],
        verbose=os.environ.get("NANO_EP_BUILD_VERBOSE") == "1",
    )


@cache
def prepare_combine_main(hidden: int, topk: int, world_size: int, experts_per_rank: int, max_tokens: int):
    """Compile one geometry; CudaBuffer prewarms it before IPC collectives."""
    from torch.utils.cpp_extension import load

    if (
        hidden <= 0
        or hidden > 2**31 - 1
        or hidden % 256
        or not 1 <= topk <= 32
        or not 1 <= world_size <= 32
        or experts_per_rank <= 0
        or max_tokens <= 0
    ):
        raise RuntimeError("invalid combine main geometry: H must be a positive multiple of 256, K/R in [1,32], L/Tmax positive")
    source = native_source_dir()
    geometry = (hidden, topk, world_size, experts_per_rank, max_tokens)
    names = ("HIDDEN", "TOPK", "WORLD_SIZE", "EXPERTS_PER_RANK", "MAX_TOKENS")
    defines = [f"-DNANO_EP_{name}={value}" for name, value in zip(names, geometry, strict=True)]
    return load(
        name="nano_ep_combine_main_" + "_".join(map(str, geometry)),
        sources=[str(source / "combine_main.cu")],
        extra_include_paths=[str(source / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo", *defines],
        verbose=os.environ.get("NANO_EP_BUILD_VERBOSE") == "1",
    )


@cache
def prepare_dispatch_epilogue(hidden: int, topk: int, experts_per_rank: int):
    """Compile H/K/local-expert geometry before entering IPC collectives."""
    from torch.utils.cpp_extension import load

    if hidden <= 0 or hidden > 2**31 - 1 or hidden % 256 or not 1 <= topk <= 32 or experts_per_rank <= 0:
        raise RuntimeError("invalid dispatch epilogue geometry: H must be a positive multiple of 256, K in [1,32], L positive")
    source = native_source_dir()
    geometry = (hidden, topk, experts_per_rank)
    names = ("HIDDEN", "TOPK", "EXPERTS_PER_RANK")
    defines = [f"-DNANO_EP_{name}={value}" for name, value in zip(names, geometry, strict=True)]
    return load(
        name="nano_ep_dispatch_epilogue_" + "_".join(map(str, geometry)),
        sources=[str(source / "dispatch_epilogue.cu")],
        extra_include_paths=[str(source / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo", *defines],
        verbose=os.environ.get("NANO_EP_BUILD_VERBOSE") == "1",
    )
