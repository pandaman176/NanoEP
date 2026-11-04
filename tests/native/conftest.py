"""Build test-only kernel launchers before exercising CUDA payloads."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def payload_probe(request):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda or --run-distributed")
    if not torch.cuda.is_available():
        pytest.fail("CUDA payload probe requires an NVIDIA GPU")
    path = Path(__file__).resolve().parents[1] / "distributed/peer_worker.py"
    spec = importlib.util.spec_from_file_location("nano_ep_payload_probe_builder", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker.build_probe()
