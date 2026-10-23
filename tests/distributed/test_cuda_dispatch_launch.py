"""Validate public dispatch independently of combine execution."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nano_ep.backends._build import load_extension
from nano_ep.testing.launcher import run_logged

pytestmark = [pytest.mark.distributed]


def test_public_cuda_dispatch(request):
    if not request.config.getoption("--run-distributed"):
        pytest.skip("opt in with --run-distributed on allocated GPUs")
    nproc = request.config.getoption("--nproc")
    if nproc not in (1, 2, 4, 8) or torch.cuda.device_count() < nproc:
        pytest.fail(
            "public dispatch acceptance needs --nproc 1/2/4/8 and that many GPUs"
        )
    load_extension()
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["NANO_EP_EXPECTED_WORLD"] = str(nproc)
    env.setdefault("OMP_NUM_THREADS", "1")
    logs = Path(env.get("NANO_EP_LOG_DIR", repo / ".nano_ep_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"dispatch-torchrun-{nproc}.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        str(Path(__file__).with_name("dispatch_worker.py")),
    ]
    try:
        status = run_logged(command, cwd=repo, env=env, log=log, timeout=180)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"public dispatch torchrun timed out: {log}\n{log.read_text()[-12000:]}"
        )
    contents = log.read_text()
    assert status == 0, contents
    for rank in range(nproc):
        assert f"NANO_EP_PUBLIC_DISPATCH_RANK_OK={rank}" in contents, contents
        for mode in (False, True):
            assert f"NANO_EP_PUBLIC_DISPATCH_MODE_OK={rank}:{mode}" in contents, (
                contents
            )
