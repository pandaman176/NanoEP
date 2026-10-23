import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nano_ep.testing.launcher import run_logged

pytestmark = [pytest.mark.distributed]


def test_single_node_torchrun(request):
    if not request.config.getoption("--run-distributed"):
        pytest.skip("opt in with --run-distributed --nproc N on an allocated GPU node")
    nproc = request.config.getoption("--nproc")
    if nproc not in (1, 2, 4, 8) or torch.cuda.device_count() < nproc:
        pytest.fail(
            f"requested {nproc} ranks; need nproc 1/2/4/8 and that many visible CUDA GPUs"
        )
    from nano_ep.backends._build import load_extension

    load_extension()  # Build once before rank startup; a compiler failure stays a failure.
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["NANO_EP_EXPECTED_WORLD"] = str(nproc)
    env.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        str(Path(__file__).with_name("worker.py")),
    ]
    log_dir = Path(env.get("NANO_EP_LOG_DIR", repo / ".nano_ep_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"torchrun-{nproc}.log"
    try:
        returncode = run_logged(command, cwd=repo, env=env, log=log, timeout=240)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"torchrun timed out (possible protocol deadlock): {log}\n{log.read_text()[-12000:]}"
        )
    contents = log.read_text()
    assert returncode == 0, contents
    assert contents.count("NANO_EP_RANK_OK=") == nproc, contents
    for rank in range(nproc):
        assert f"NANO_EP_PUBLIC_COMBINE_RANK_OK={rank}" in contents, contents
