# NanoEP

Single-node expert-parallel dispatch and combine for PyTorch, using CUDA IPC and peer GPU memory access.

- FP32/BF16 payloads, compact and expert-major expanded layouts.
- Cached routing handles, optional weight payloads, and asynchronous operations.
- Minimal by design: readable CUDA kernels without NVSHMEM or NCCL Gin.

## Install and build

Requires Linux, Python 3.11+, CUDA-enabled PyTorch 2.10+, Ninja, a C++20 compiler, and a CUDA toolkit compatible with PyTorch. Target: Hopper/SM90 GPUs with pairwise CUDA peer access. Multi-node communication and automatic autograd are currently not supported.

Install a CUDA build of PyTorch appropriate for your system, then run from this checkout:

```bash
python -m pip install .
TORCH_CUDA_ARCH_LIST=9.0 python -m nano_ep.build
```

Installation packages the CUDA sources. The extension compiles on the first GPU operation, or explicitly with the command above; compiled output is cached by PyTorch. DeepEP, NVSHMEM and NCCL Gin are not required.

To build distributable source and wheel packages:

```bash
python -m pip install build
python -m build
```

## Python API

Save the following as `example.py`, then run `torchrun --standalone --nproc-per-node=2 example.py` (one process per GPU):

```python
import os
import torch
import torch.distributed as dist
from nano_ep import CudaBuffer, EPConfig

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("gloo")
try:
    rank, size = dist.get_rank(), dist.get_world_size()
    config = EPConfig(num_experts=2 * size, num_topk=2, hidden=128, num_max_tokens_per_rank=16)
    x = torch.randn(16, 128, device="cuda")
    indices = torch.tensor([[2 * rank, 2 * rank + 1]], device="cuda", dtype=torch.int64).repeat(16, 1)
    weights = torch.full((16, 2), 0.5, device="cuda")

    with CudaBuffer(dist.group.WORLD, config) as buffer:
        received = buffer.dispatch(x, topk_idx=indices, weights=weights, do_expand=True, zero_padding=True)
        # Example identity experts: apply each gate weight exactly once.
        expert_x = received.x * received.topk_weights[:, None]
        result = buffer.combine(expert_x, received.handle)
        torch.testing.assert_close(result.x, x)
finally:
    dist.destroy_process_group()
```

Use global expert IDs, contiguous expert placement per rank, and `-1` for masked choices. Call collectives in the same order on all ranks. `combine` sums expert outputs; its optional `weights` argument returns a separate payload and does not weight `x`.

For asynchronous calls, pass `async_op=True` and call `result.event.current_stream_wait()` on the consumer stream before using the result. Dispatch may still synchronize for exact receive counts; CUDA Graph capture is not supported.

See [API](docs/api.md) for layouts, cached dispatch, numerical behavior and stream ownership. A runnable forward MoE example is included:

```bash
torchrun --standalone --nproc-per-node=2 -m nano_ep.examples.moe --async-op --cached
```

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest tests/unit
python -m pytest tests/native --run-cuda
python -m pytest tests/distributed --run-distributed --nproc 2
```

See [testing](docs/testing.md) for more details.

## Future Plans

- **Deeper Kernel Fusion** — Fuse more dispatch/combine stages to reduce kernel launches and intermediate memory traffic.
- **More CUDA Magic** — Explore TMA, warp specialization, and better communication/computation overlap.
- **Shape-specialized JIT Kernels** — Extend the existing JIT build with specialization for common hidden sizes and routing configurations.

## Acknowledgements

Thanks to the [DeepEP](https://github.com/deepseek-ai/DeepEP) authors and contributors for open-sourcing their work. NanoEP's design draws inspiration from DeepEP's expert-parallel dispatch/combine abstractions.

## License

MIT. See [LICENSE](LICENSE).
