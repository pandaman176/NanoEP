"""Workers load the launcher's probe without invoking the extension builder."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.utils.cpp_extension



@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "distributed/peer_worker.py"
    spec = importlib.util.spec_from_file_location("nanoep_probe_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_loads_prepared_module_without_building(worker, monkeypatch, tmp_path):
    binary = tmp_path / "nano_ep_peer_probe.py"
    binary.write_text("value = 73\n")
    monkeypatch.setenv("NANO_EP_PEER_PROBE_PATH", str(binary))

    def reject_build(*args, **kwargs):
        raise AssertionError("distributed workers must not enter the build path")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", reject_build)
    module = worker.load_probe()
    assert module.value == 73
    assert worker.load_probe() is module


def test_launcher_builds_probe_once(worker, monkeypatch):
    calls = []
    module = SimpleNamespace(__file__="/tmp/nano_ep_peer_probe.so")

    def build(**kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch.utils.cpp_extension, "load", build)
    assert worker.build_probe() is worker.build_probe() is module
    assert len(calls) == 1
    assert calls[0]["name"] == "nano_ep_peer_probe"
