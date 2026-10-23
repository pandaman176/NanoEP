import pytest
import torch

from nano_ep.backends import ops

pytestmark = [pytest.mark.cuda]


@pytest.fixture(autouse=True)
def require_cuda(request):
    if not request.config.getoption("--run-cuda"):
        pytest.skip("opt in with --run-cuda; native tests are unverified on CPU")
    if not torch.cuda.is_available():
        pytest.fail("--run-cuda requested but NVIDIA CUDA is unavailable")
    from nano_ep.backends._build import load_extension

    load_extension()


def test_route_counts():
    idx = torch.tensor([[0, 1, 5], [-1, 3, 4], [-1, -1, -1]], device="cuda")
    ranks, experts = ops.route_counts(idx, 6, 2)
    torch.testing.assert_close(ranks.cpu(), torch.tensor([1, 2]))
    torch.testing.assert_close(experts.cpu(), torch.tensor([1, 1, 0, 1, 1, 1]))
    assert ranks.dtype == experts.dtype == torch.int64
    assert ranks.is_cuda and experts.is_cuda


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_pack(dtype, empty):
    x = torch.arange(5 * 128, device="cuda").reshape(5, 128).to(dtype)
    idx = torch.tensor([] if empty else [3, 0, 3, 4], device="cuda", dtype=torch.int64)
    got = ops.pack_rows(x, idx)
    torch.testing.assert_close(got, x[idx], rtol=0, atol=0)
    assert got.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_expand_reduce_ignore_padding(dtype):
    x = torch.arange(3 * 128, device="cuda").reshape(3, 128).to(dtype)
    x[2] = float("nan")
    slots = torch.tensor([[0, 4, -1], [1, -1, 5], [-1, -1, -1]], device="cuda")
    expanded = ops.expand_rows(x, slots, 8)
    expected = torch.zeros((8, 128), device="cuda", dtype=dtype)
    expected[[0, 4]] = x[0]
    expected[[1, 5]] = x[1]
    torch.testing.assert_close(expanded, expected, rtol=0, atol=0)
    expanded[[2, 3, 6, 7]] = float("nan")
    got = ops.reduce_rows(expanded, slots)
    wanted = torch.stack(
        [x[0].float() * 2, x[1].float() * 2, torch.zeros(128, device="cuda")]
    ).to(dtype)
    torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_reduce_accumulation_precision(dtype):
    if dtype == torch.bfloat16:
        values = [256.0, 1.0, -256.0]
        expected = 1.0
    else:
        values = [1.0, 2.0**-40, -1.0]
        expected = 2.0**-40
    x = torch.tensor(values, device="cuda", dtype=dtype).reshape(3, 1)
    slots = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int64)

    got = ops.reduce_rows(x, slots)

    torch.testing.assert_close(
        got, torch.tensor([[expected]], device="cuda", dtype=dtype), rtol=0, atol=0
    )


def test_empty_native_operations():
    idx = torch.empty((0, 3), device="cuda", dtype=torch.int64)
    r, e = ops.route_counts(idx, 6, 2)
    assert r.sum().item() == e.sum().item() == 0
    x = torch.empty((0, 128), device="cuda")
    expanded = ops.expand_rows(x, idx, 0)
    reduced = ops.reduce_rows(expanded, idx)
    assert expanded.shape == reduced.shape == (0, 128)


def test_current_stream_launch():
    stream = torch.cuda.Stream()
    x = torch.zeros((8, 128), device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        x.fill_(7)
        y = ops.pack_rows(x, torch.arange(7, -1, -1, device="cuda"))
        y = y + 1
    stream.synchronize()
    assert torch.all(y == 8)
