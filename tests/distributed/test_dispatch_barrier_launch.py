"""Run fused dispatch entry/exit checks on 2/4/8 independent GPUs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nano_ep.backends._build import (
    load_extension,
    prepare_dispatch_epilogue,
    prepare_dispatch_main,
)
from nano_ep.testing.launcher import run_logged

pytestmark = [pytest.mark.distributed]


def test_cooperative_dispatch_barriers(request):
    if not request.config.getoption("--run-distributed"):
        pytest.skip("opt in with --run-distributed --nproc 2/4/8 on allocated GPUs")
    nproc = request.config.getoption("--nproc")
    if nproc == 1:
        pytest.skip("cross-GPU dispatch barriers require at least two peers")
    if nproc not in (2, 4, 8) or torch.cuda.device_count() < nproc:
        pytest.fail("cooperative dispatch acceptance requires 2/4/8 visible GPUs")
    from peer_worker import build_probe

    load_extension()
    probe = build_probe()
    prepare_dispatch_main(256, 3, nproc, 3, 2153, 4)
    for mode in (False, True):
        prepare_dispatch_epilogue(256, 3, 3, cached_mode=mode)
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["NANO_EP_PEER_PROBE_PATH"] = probe.__file__
    env.setdefault("OMP_NUM_THREADS", "1")
    logs = Path(env.get("NANO_EP_LOG_DIR", repo / ".nano_ep_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"dispatch-barrier-torchrun-{nproc}.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        str(Path(__file__).with_name("dispatch_barrier_worker.py")),
    ]
    try:
        status = run_logged(command, cwd=repo, env=env, log=log, timeout=600)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"cooperative dispatch torchrun timed out: {log}\n{log.read_text()[-12000:]}"
        )
    contents = log.read_text()
    assert status == 0, contents
    for rank in range(nproc):
        assert f"NANO_EP_DISPATCH_BARRIER_RANK_OK={rank}:24" in contents, contents
