# Testing

Install development dependencies with `python -m pip install -e '.[dev]'`. Run from this checkout. No cluster scheduler is assumed; obtain GPU resources using your environment's normal mechanism before enabling CUDA tests.

```bash
python -m pytest tests/unit
python -m pytest tests/native --run-cuda
python -m pytest tests/distributed --run-distributed --nproc 2
```

Unit tests cover independent CPU references, configuration, C++ window arithmetic, buffer initialization/rollback, dispatch/combine orchestration and stream ownership using test doubles. CUDA tests compile the actual extension and check operators, metadata, main/epilogue numerical output, padding and window lifetime. Device-assert negative cases use separate processes.

Distributed tests launch bounded torchrun process groups. They cover real IPC, delayed peers, barriers, resource cleanup, cached routing, shared-arena reuse and public dispatch/combine. Run the independent peer-runtime test with two GPUs; use the public API launcher for other process counts:

```bash
python -m pytest tests/distributed/test_peer_launch.py --run-distributed --nproc 2
python -m pytest tests/distributed/test_launch.py --run-distributed --nproc 8
```

Larger forward-example profiles are opt-in:

```bash
python -m pytest tests/distributed/test_acceptance.py --run-distributed --run-acceptance --nproc 2
```

CUDA and distributed tests are skipped unless explicitly enabled. Enabled tests fail if the required hardware is unavailable. The peer-runtime launcher skips process counts other than two, and the larger example profiles remain skipped without `--run-acceptance`. Review skip counts and per-rank completion markers, not just the exit code. Logs default to `.nano_ep_logs/`; set `NANO_EP_LOG_DIR` to choose a location. Extensions compile before worker timeouts start. A logical-peer probe on one GPU tests addressing and arithmetic; only multi-process IPC tests establish peer communication behavior.

Source and wheel distributions include all CUDA sources. `python -m nano_ep.build` compiles the installed package on a CUDA host; it never looks for another checkout. Run the example from outside the repository to validate an installed wheel:

```bash
python -m pip install dist/nano_ep-*.whl
cd /tmp
torchrun --standalone --nproc-per-node=2 -m nano_ep.examples.moe --async-op --cached
```

With development dependencies installed, `python tools/check_package.py` builds the wheel from its sdist, verifies every bundled native source, and imports the installed wheel outside the checkout. Add `--compile` on a GPU host to compile those bundled sources.
