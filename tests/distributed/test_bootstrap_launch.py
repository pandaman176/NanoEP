import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nano_ep.backends._build import load_extension
from nano_ep.testing.launcher import run_logged

pytestmark = [pytest.mark.distributed]


def test_cuda_buffer_bootstrap(request):
    if not request.config.getoption("--run-distributed"):
        pytest.skip("opt in with --run-distributed on allocated GPUs")
    nproc = request.config.getoption("--nproc")
    if nproc < 1 or torch.cuda.device_count() < nproc:
        pytest.fail(f"bootstrap acceptance needs {nproc} visible CUDA GPUs")
    load_extension()
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["NANO_EP_EXPECTED_WORLD"] = str(nproc)
    env.setdefault("OMP_NUM_THREADS", "1")
    logs = Path(env.get("NANO_EP_LOG_DIR", repo / ".nano_ep_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"bootstrap-torchrun-{nproc}.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        str(Path(__file__).with_name("bootstrap_worker.py")),
    ]
    try:
        status = run_logged(command, cwd=repo, env=env, log=log, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail(f"bootstrap torchrun timed out: {log}\n{log.read_text()[-12000:]}")
    contents = log.read_text()
    assert status == 0, contents
    for rank in range(nproc):
        for failure in ("allocate", "export", "import"):
            assert f"NANO_EP_BOOTSTRAP_ROLLBACK_OK={rank}:{failure}" in contents, (
                contents
            )
        for lifetime in range(2):
            assert f"NANO_EP_BOOTSTRAP_LIFETIME_OK={rank}:{lifetime}" in contents, (
                contents
            )
        assert f"NANO_EP_BOOTSTRAP_RANK_OK={rank}" in contents, contents
