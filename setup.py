"""Ship native sources in wheels; compilation happens on the target system."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
DEEPJIT_INCLUDE = ROOT / "third-party" / "deep_jit" / "include"
DEEPJIT_LICENSE = ROOT / "third-party" / "deep_jit" / "LICENSE"


def require_deepjit():
    if not (DEEPJIT_INCLUDE / "deep_jit/backend/cuda/backend.hpp").is_file() or not DEEPJIT_LICENSE.is_file():
        raise RuntimeError("DeepJIT headers or license are missing; run git submodule update --init --recursive")


def ignore_legacy_deepjit(directory, names):
    if Path(directory) == ROOT / "csrc" and "deep_jit" in names:
        return {"deep_jit"}
    return set()


class BuildPy(build_py):
    def run(self):
        require_deepjit()
        super().run()
        destination = Path(self.build_lib) / "nano_ep" / "_csrc"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(ROOT / "csrc", destination, ignore=ignore_legacy_deepjit)
        shutil.copytree(DEEPJIT_INCLUDE, destination / "deep_jit" / "include")
        shutil.copy2(DEEPJIT_LICENSE, destination / "deep_jit" / "LICENSE")


require_deepjit()
setup(cmdclass={"build_py": BuildPy})
