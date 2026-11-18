"""Ship native sources in wheels; compilation happens on the target system."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
DEEPJIT_INCLUDE = ROOT / "third-party" / "deep_jit" / "include"


def require_deepjit():
    if not (DEEPJIT_INCLUDE / "deep_jit/backend/cuda/backend.hpp").is_file():
        raise RuntimeError("DeepJIT headers are missing; run git submodule update --init --recursive")


class BuildPy(build_py):
    def run(self):
        require_deepjit()
        super().run()
        destination = Path(self.build_lib) / "nano_ep" / "_csrc"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(ROOT / "csrc", destination)
        shutil.copytree(DEEPJIT_INCLUDE, destination / "deep_jit" / "include")


require_deepjit()
setup(cmdclass={"build_py": BuildPy})
