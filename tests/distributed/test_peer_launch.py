import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nano_ep.backends._build import load_extension
from nano_ep.testing.launcher import run_logged

pytestmark = [pytest.mark.distributed]


def test_peer_runtime_two_ranks(request):
    if not request.config.getoption("--run-distributed"):
        pytest.skip("opt in with --run-distributed on two allocated GPUs")
    if request.config.getoption("--nproc") != 2:
        pytest.skip("independent PeerRuntime acceptance targets --nproc 2")
    if torch.cuda.device_count() < 2:
        pytest.fail("PeerRuntime acceptance requires two visible CUDA GPUs")
    from peer_worker import load_probe

    load_extension()
    load_probe()  # Build before the worker timeout starts.
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    logs = Path(env.get("NANO_EP_LOG_DIR", repo / ".nano_ep_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "peer-torchrun-2.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(Path(__file__).with_name("peer_worker.py")),
    ]
    try:
        status = run_logged(command, cwd=repo, env=env, log=log, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"PeerRuntime torchrun timed out: {log}\n{log.read_text()[-12000:]}"
        )
    contents = log.read_text()
    assert status == 0, contents
    assert contents.count("NANO_EP_PEER_LIFETIME_OK=") == 4, contents
    for rank in range(2):
        assert f"NANO_EP_PEER_RANK_OK={rank}" in contents, contents
        assert f"NANO_EP_NOTIFY_RANK_OK={rank}" in contents, contents
        assert f"NANO_EP_PAYLOAD_RANK_OK={rank}" in contents, contents
        assert f"NANO_EP_DISPATCH_MAIN_RANK_OK={rank}" in contents, contents
        assert f"NANO_EP_DISPATCH_COPY_RANK_OK={rank}" in contents, contents
        assert f"NANO_EP_COMBINE_MAIN_RANK_OK={rank}" in contents, contents
        assert f"NANO_EP_COMBINE_EPILOGUE_RANK_OK={rank}" in contents, contents
