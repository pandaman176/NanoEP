"""Ship native sources in wheels; compilation happens on the target system."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BuildPy(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "nano_ep" / "_csrc"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(ROOT / "csrc", destination)


setup(cmdclass={"build_py": BuildPy})
