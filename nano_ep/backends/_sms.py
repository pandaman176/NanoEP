"""SM budget for single-node NVLink EP, following DeepEP's bandwidth model."""

import math
import re
import subprocess
from functools import cache


def _smi(*args: str) -> str:
    return subprocess.run(("nvidia-smi", *args), check=True, capture_output=True, text=True, timeout=5).stdout


def _gpu_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


@cache
def nvlink_gbs(gpu_uuids: tuple[str, ...]) -> float:
    """Return the effective per-GPU bandwidth for an all-NVLink group, or zero."""
    if len(gpu_uuids) < 2:
        return 0.0
    try:
        # UUIDs keep CUDA_VISIBLE_DEVICES remapping independent of nvidia-smi indices.
        devices = {
            _gpu_uuid(uuid): f"GPU{index.strip()}"
            for index, uuid in (line.split(",", 1) for line in _smi("--query-gpu=index,uuid", "--format=csv,noheader").splitlines())
        }
        group = tuple(devices[_gpu_uuid(uuid)] for uuid in gpu_uuids)
        topology = re.sub(r"\x1b\[[0-9;]*m", "", _smi("topo", "-m"))
        lines = [line.split() for line in topology.splitlines() if line.strip()]
        header = next(fields for fields in lines if len(fields) >= 2 and all(re.fullmatch(r"GPU\d+", field) for field in fields[:2]))
        columns = [field for field in header if re.fullmatch(r"GPU\d+", field)]
        links = {row[0]: dict(zip(columns, row[1:])) for row in lines if row is not header and row[0] in columns}
        if any(not links[src][dst].startswith("NV") for src in group for dst in group if src != dst):
            return 0.0

        bandwidth = dict.fromkeys(group, 0.0)
        gpu = None
        for line in _smi("nvlink", "-s").splitlines():
            if match := re.match(r"\s*GPU\s+(\d+):", line):
                gpu = f"GPU{match[1]}"
            elif gpu in bandwidth and (match := re.match(r"\s*Link\s+\d+:\s*([\d.]+)\s+GB/s", line)):
                bandwidth[gpu] += float(match[1])
        return min(bandwidth.values()) * 0.9
    except (OSError, subprocess.SubprocessError, KeyError, StopIteration, ValueError):
        return 0.0


def theoretical_num_sms(device_sms: int, world_size: int, num_experts: int, topk: int, nvlink_bandwidth: float) -> int:
    if world_size == 1 or nvlink_bandwidth <= 0:
        return device_sms

    # probability that the token need to send to a rank
    p_need_send = 1 - math.comb(num_experts - num_experts // world_size, topk) / math.comb(num_experts, topk)
    expected_ranks = world_size * p_need_send
    sm_read = 1 / expected_ranks
    sm_write = 1.0
    nvlink_traffic = 1 - 1 / world_size
    traffic_gbs = nvlink_bandwidth / nvlink_traffic
    sm_read_gbs, sm_write_gbs = 180, 45
    required_sms = traffic_gbs * max(
        sm_write / sm_write_gbs,  # dispatch write
        sm_read / sm_read_gbs,  # dispatch read
        (sm_read + nvlink_traffic) / sm_write_gbs,  # combine write
        (sm_write - nvlink_traffic) / sm_read_gbs,  # combine read
    )
    device_limit = device_sms // 2 * 2 or device_sms
    return min(4 * math.ceil(required_sms * 1.3 / 4), device_limit)
