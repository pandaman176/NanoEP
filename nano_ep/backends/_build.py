"""Build the host extension once; DeepJIT compiles device kernels on demand."""

import os
import shlex
from functools import cache
from pathlib import Path


def native_source_dir() -> Path:
    package = Path(__file__).resolve().parents[1]
    for source in (package / "_csrc", package / "nano_csrc", package.parent / "nano_csrc", package.parent / "csrc"):
        if source.is_dir():
            return source
    raise RuntimeError("NanoEP native sources are missing")


def deepjit_source_dir(source: Path) -> Path:
    for dependency in (source / "deep_jit", source.parent / "third-party" / "deep_jit"):
        if (dependency / "include/deep_jit/backend/cuda/backend.hpp").is_file():
            return dependency
    raise RuntimeError("DeepJIT headers are missing. Run git submodule update --init third-party/deep_jit")


@cache
def load_extension():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME/nvcc is required to compile NanoEP")
    source = native_source_dir()
    deepjit = deepjit_source_dir(source)
    # This flag targets the extension's IPC barrier. DeepJIT selects the device
    # architecture for the four communication kernels separately.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    module = load(
        name="nano_ep_native",
        sources=[
            str(source / name)
            for name in ("bindings.cpp", "jit.cpp", "peer.cu", "pipeline.cu", "dispatch_epilogue.cpp", "combine_epilogue.cpp")
        ],
        extra_include_paths=[str(source / "include"), str(deepjit / "include")],
        extra_cflags=["-O2", "-std=c++20", "-DUSE_CUDA"],
        extra_cuda_cflags=["-O2", "-std=c++20", "-lineinfo"],
        extra_ldflags=["-ldl", *shlex.split(os.environ.get("LDFLAGS", ""))],
        verbose=os.environ.get("NANO_EP_BUILD_VERBOSE") == "1",
    )
    module.init_jit(str(source))
    return module
