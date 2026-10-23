"""Build sdist -> wheel and import the installed package outside the checkout."""

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Also compile the installed CUDA extension on a GPU host")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="nanoep-package-") as directory:
        scratch = Path(directory)
        artifacts, installed = scratch / "dist", scratch / "installed"
        subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(artifacts), str(project)],
            cwd=scratch,
            check=True,
        )
        (wheel,) = artifacts.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            for source in (project / "csrc").rglob("*"):
                if source.is_file():
                    relative = source.relative_to(project / "csrc")
                    assert archive.read("nano_ep/_csrc/" + relative.as_posix()) == source.read_bytes(), relative
            assert not any(name.startswith("nano_ep/sim") for name in archive.namelist())
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(wheel)],
            cwd=scratch,
            check=True,
        )
        probe = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(installed)!r}); "
            "import nano_ep; from nano_ep.backends._build import native_source_dir, load_extension; "
            f"assert Path(nano_ep.__file__).is_relative_to({str(installed)!r}); "
            "assert native_source_dir().name == '_csrc'; "
            "assert (native_source_dir() / 'include/nano_ep/peer.hpp').is_file(); "
            "assert load_extension.cache_info().currsize == 0; "
            "assert not hasattr(nano_ep, 'SimWorld'); "
            "print('INSTALLED_NANO_EP=' + nano_ep.__file__)"
        )
        if args.compile:
            probe += "; print('CUDA_EXTENSION=' + load_extension().__file__)"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        subprocess.run([sys.executable, "-I", "-c", probe], cwd=scratch, env=environment, check=True)
        print("Package check passed (CUDA compiled: " + str(args.compile) + ")")


if __name__ == "__main__":
    main()
