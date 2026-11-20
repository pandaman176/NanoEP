"""Build sdist -> wheel and import the installed package outside the checkout."""

import argparse
import os
import subprocess
import sys
import tarfile
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
        (sdist,) = artifacts.glob("*.tar.gz")
        with tarfile.open(sdist) as archive:
            names = archive.getnames()
            assert any(name.endswith("/third-party/deep_jit/include/deep_jit/backend/cuda/backend.hpp") for name in names)
            license_path = next(name for name in names if name.endswith("/third-party/deep_jit/LICENSE"))
            assert archive.extractfile(license_path).read() == (project / "third-party/deep_jit/LICENSE").read_bytes()
            assert not any("/csrc/deep_jit/" in name for name in names)
        with zipfile.ZipFile(wheel) as archive:
            for source in (project / "csrc").rglob("*"):
                if source.is_file():
                    relative = source.relative_to(project / "csrc")
                    if relative.parts[0] == "deep_jit":
                        continue
                    assert archive.read("nano_ep/_csrc/" + relative.as_posix()) == source.read_bytes(), relative
            assert not any(name.startswith("nano_ep/sim") for name in archive.namelist())
            deepjit_include = project / "third-party/deep_jit/include"
            for source in deepjit_include.rglob("*"):
                if source.is_file():
                    relative = source.relative_to(deepjit_include)
                    assert archive.read("nano_ep/_csrc/deep_jit/include/" + relative.as_posix()) == source.read_bytes(), relative
            expected_headers = {
                "nano_ep/_csrc/deep_jit/include/" + source.relative_to(deepjit_include).as_posix()
                for source in deepjit_include.rglob("*")
                if source.is_file()
            }
            assert {name for name in archive.namelist() if name.startswith("nano_ep/_csrc/deep_jit/include/")} == expected_headers
            assert archive.read("nano_ep/_csrc/deep_jit/LICENSE") == (project / "third-party/deep_jit/LICENSE").read_bytes()
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(wheel)],
            cwd=scratch,
            check=True,
        )
        probe = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(installed)!r}); "
            "import nano_ep; from nano_ep.backends._build import native_source_dir, deepjit_source_dir, load_extension; "
            f"assert Path(nano_ep.__file__).is_relative_to({str(installed)!r}); "
            "assert native_source_dir().name == '_csrc'; "
            "assert (native_source_dir() / 'include/nano_ep/peer.hpp').is_file(); "
            "assert (deepjit_source_dir(native_source_dir()) / 'include/deep_jit/backend/cuda/backend.hpp').is_file(); "
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
