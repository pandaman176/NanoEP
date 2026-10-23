"""Each profile runs the real public dispatch and combine API."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nano_ep.testing.launcher import run_logged

pytestmark = [pytest.mark.distributed]

# name, layout, dtype, tokens, hidden, topk, pattern, iterations, async/cached
PROFILES = [
    ("compact-fp32", "compact", "fp32", 17, 128, 3, "mixed", 3, False),
    ("expanded-bf16", "expanded", "bf16", 17, 128, 3, "mixed", 3, False),
    ("large-compact", "compact", "fp32", 256, 4096, 8, "mixed", 4, True),
    ("large-expanded", "expanded", "bf16", 256, 7168, 8, "mixed", 8, True),
    ("skew-compact", "compact", "bf16", 129, 1024, 3, "skew", 16, True),
    ("skew-expanded", "expanded", "fp32", 129, 1024, 3, "skew", 16, True),
    ("empty-compact", "compact", "fp32", 0, 128, 3, "mixed", 4, True),
    ("empty-expanded", "expanded", "bf16", 0, 128, 3, "mixed", 4, True),
    ("masked-compact", "compact", "bf16", 33, 128, 3, "masked", 4, True),
    ("masked-expanded", "expanded", "fp32", 33, 128, 3, "masked", 4, True),
    ("uneven-compact", "compact", "fp32", 65, 1024, 3, "uneven", 8, True),
    ("uneven-expanded", "expanded", "bf16", 65, 1024, 3, "uneven", 8, True),
]


@pytest.mark.parametrize("profile", PROFILES, ids=[p[0] for p in PROFILES])
def test_delivery_profile(request, profile, tmp_path):
    if not request.config.getoption("--run-acceptance"):
        pytest.skip("opt in to delivery tests with --run-acceptance --run-distributed")
    if not request.config.getoption("--run-distributed"):
        pytest.fail("--run-acceptance requires --run-distributed")
    nproc = request.config.getoption("--nproc")
    if nproc not in (1, 2, 4, 8) or torch.cuda.device_count() < nproc:
        pytest.fail(f"need {nproc} visible GPUs and nproc in 1/2/4/8")
    from nano_ep.backends._build import load_extension

    load_extension()
    name, layout, dtype, tokens, hidden, topk, pattern, iterations, asynchronous = (
        profile
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        "-m",
        "nano_ep.examples.moe",
        "--layout",
        layout,
        "--dtype",
        dtype,
        "--tokens",
        str(tokens),
        "--hidden",
        str(hidden),
        "--topk",
        str(topk),
        "--pattern",
        pattern,
        "--iterations",
        str(iterations),
    ]
    if asynchronous:
        command += ["--async-op", "--cached"]
    log_dir = Path(os.environ.get("NANO_EP_LOG_DIR", tmp_path)).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"acceptance-{nproc}-{name}.log"
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    try:
        returncode = run_logged(command, log=log, cwd=tmp_path, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail(f"torchrun exceeded 300s: {log}\n{log.read_text()[-12000:]}")
    contents = log.read_text()
    assert returncode == 0, f"{log}\n{contents[-12000:]}"
    reports = [
        json.loads(line.split("=", 1)[1])
        for line in contents.splitlines()
        if line.startswith("NANO_EP_MOE_OK=")
    ]
    assert len(reports) == nproc and {r["rank"] for r in reports} == set(
        range(nproc)
    ), contents
