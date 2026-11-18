"""Check the NVLink SM model without running a CUDA kernel."""

from subprocess import TimeoutExpired

import pytest

from nano_ep.backends import _sms



@pytest.mark.parametrize("ranks,expected", [(2, 24), (4, 16), (8, 16)])
def test_theoretical_num_sms_for_nvlink(ranks, expected):
    assert _sms.theoretical_num_sms(78, ranks, 8 * ranks, 8, 400.0) == expected


def test_theoretical_num_sms_falls_back_or_caps_at_device_size():
    assert _sms.theoretical_num_sms(78, 1, 8, 8, 400.0) == 78
    assert _sms.theoretical_num_sms(78, 8, 64, 8, 0.0) == 78
    assert _sms.theoretical_num_sms(78, 8, 64, 8, 10000.0) == 78


@pytest.mark.parametrize("link,expected", [("NV18", 45.0), ("PIX", 0.0)])
def test_nvlink_probe_maps_selected_gpu_uuids(monkeypatch, link, expected):
    replies = {
        (
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ): "0, GPU-AA-00\n1, GPU-bb-11\n2, GPU-cc-22\n",
        (
            "topo",
            "-m",
        ): f"\x1b[4mGPU0 GPU1 GPU2 CPU Affinity\x1b[0m\nGPU0 X PIX {link}\nGPU1 PIX X PIX\nGPU2 {link} PIX X\n",
        (
            "nvlink",
            "-s",
        ): "GPU 0: H20\n Link 0: 25.000 GB/s\n Link 1: 25.000 GB/s\nGPU 2: H20\n Link 0: 55.000 GB/s\n",
    }
    calls = []

    def smi(*args):
        calls.append(args)
        return replies[args]

    monkeypatch.setattr(_sms, "_smi", smi)
    _sms.nvlink_gbs.cache_clear()
    try:
        group = ("cc-22", "GPU-aa-00")
        assert _sms.nvlink_gbs(group) == _sms.nvlink_gbs(group) == expected
        assert len(calls) == (3 if link == "NV18" else 2)
    finally:
        _sms.nvlink_gbs.cache_clear()


def test_nvlink_probe_falls_back_when_smi_times_out(monkeypatch):
    def smi(*args):
        raise TimeoutExpired(args, timeout=5)

    monkeypatch.setattr(_sms, "_smi", smi)
    try:
        assert _sms.nvlink_gbs(("gpu-a", "gpu-b")) == 0.0
    finally:
        _sms.nvlink_gbs.cache_clear()
