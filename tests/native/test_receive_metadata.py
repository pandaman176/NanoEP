import pytest
import torch

from nano_ep.backends._build import load_extension

pytestmark = [pytest.mark.cuda]


@pytest.fixture
def native(request):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda or --run-distributed")
    if not torch.cuda.is_available():
        pytest.fail("CUDA tests requested but NVIDIA CUDA is unavailable")
    return load_extension()


def make_records(rank, kind, alignment):
    local_experts = 40 if kind == "wide" else 5
    topk = 32 if kind == "wide" else 3
    start = rank * local_experts
    foreign = local_experts if rank == 0 else 0
    if kind == "empty":
        indices = torch.empty((0, topk), dtype=torch.int64)
    elif kind == "wide":
        pattern = [
            -1 if k % 4 == 1 else foreign + k if k % 4 == 2 else start + k
            for k in range(topk)
        ]
        indices = torch.tensor([pattern] * 257)
    elif kind == "skew":
        indices = torch.tensor([[start, -1, foreign]] * 257)
    else:
        indices = torch.tensor(
            [
                [start, start + 1, -1],
                [foreign, start + 1, start + local_experts - 1],
                [-1, start, foreign],
            ]
            * 5
        )

    rows = indices.shape[0]
    # Keep provenance deliberately different from physical receive-row order.
    src_ranks = torch.arange(rows, dtype=torch.int64) % 4
    src_tokens = torch.arange(rows - 1, -1, -1, dtype=torch.int64) * 3 + 7
    hidden = 7
    row_bytes = (16 + 12 * topk + 4 * hidden + 15) // 16 * 16
    stage = torch.full((rows, row_bytes), 165, dtype=torch.uint8)
    for row in range(rows):
        header = torch.cat(
            (src_ranks[row : row + 1], src_tokens[row : row + 1], indices[row])
        )
        stage[row, : 16 + 8 * topk] = header.view(torch.uint8)

    local = torch.where(
        (indices >= start) & (indices < start + local_experts), indices - start, -1
    )
    counts = torch.bincount(local[local >= 0], minlength=local_experts)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64),
            ((counts + alignment - 1) // alignment * alignment).cumsum(0),
        )
    )
    return stage, offsets, src_ranks, src_tokens, local, counts


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("kind", ["empty", "mixed", "skew", "wide"])
@pytest.mark.parametrize("alignment", [1, 4, 16])
def test_receive_metadata(native, rank, kind, alignment):
    stage, offsets, src_ranks, src_tokens, local, counts = make_records(
        rank, kind, alignment
    )
    local_experts, topk = counts.numel(), local.shape[1]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device_stage, device_offsets = stage.cuda(), offsets.cuda()
        results = [
            native.build_receive_metadata(
                device_stage, device_offsets, rank, local_experts, topk
            )
            for _ in range(2)
        ]
        torch.testing.assert_close(device_stage.cpu(), stage)
        torch.testing.assert_close(device_offsets.cpu(), offsets)
        # The returned tensors must own metadata after the communication window is reused.
        device_stage.fill_(0)
        cpu_results = [[value.cpu() for value in result] for result in results]

    for device_result, result in zip(results, cpu_results, strict=True):
        assert len(result) == 4
        for tensor in device_result:
            assert tensor.dtype == torch.int64 and tensor.device == device_stage.device
            assert tensor.is_contiguous()
        actual_ranks, actual_tokens, actual_local, slots = result
        torch.testing.assert_close(actual_ranks, src_ranks)
        torch.testing.assert_close(actual_tokens, src_tokens)
        torch.testing.assert_close(actual_local, local)
        assert slots.shape == local.shape
        assert (slots[local < 0] == -1).all()
        for expert, count in enumerate(counts.tolist()):
            # Assignment order may vary, but each expert's actual interval must be filled exactly once.
            expected = torch.arange(
                offsets[expert].item(), offsets[expert].item() + count
            )
            torch.testing.assert_close(slots[local == expert].sort().values, expected)


@pytest.mark.parametrize(
    "kind", ["cpu", "dtype", "stride", "alignment", "row_bytes", "offsets", "topk"]
)
def test_invalid_receive_metadata_storage(native, kind):
    stage, offsets, _, _, local, counts = make_records(1, "mixed", 4)
    stage, offsets = stage.cuda(), offsets.cuda()
    topk = local.shape[1]
    if kind == "cpu":
        stage = stage.cpu()
    elif kind == "dtype":
        stage = stage.float()
    elif kind == "stride":
        stage = stage[:, ::2]
    elif kind == "alignment":
        storage = torch.empty(stage.numel() + 1, dtype=torch.uint8, device="cuda")
        stage = storage[1:].view(stage.shape)
    elif kind == "row_bytes":
        stage = stage[:, :16].contiguous()
    elif kind == "offsets":
        offsets = offsets[:-1]
    else:
        topk = 33
    with pytest.raises(RuntimeError, match="stage_buffer|expert_offsets|num_topk"):
        native.build_receive_metadata(stage, offsets, 1, counts.numel(), topk)
