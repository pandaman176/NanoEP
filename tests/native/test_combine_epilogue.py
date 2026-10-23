"""Native source reduction with independently encoded, poisoned return packets."""

import pytest
import torch
from torch.utils import deterministic

from nano_ep.backends._build import load_extension
from nano_ep.testing.native_combine import (
    assert_combined_output,
    expert_inputs,
    make_return_case,
    make_return_packet,
)

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


@pytest.fixture(autouse=True)
def poison_empty_outputs(native):
    """Require explicit writes for masked weights and every output coordinate."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        deterministic.fill_uninitialized_memory = fill
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def inputs(*, mode=False, dtype=torch.float32, tokens=2):
    ranks, experts, capacity, hidden, topk = 2, 6, 5, 33, 3
    partitions = min(ranks, topk) if mode else topk
    row_bytes = (4 * hidden + 4 * topk + 15) // 16 * 16
    return {
        "stage_buffer": torch.full(
            (partitions, capacity, row_bytes), 165, dtype=torch.uint8, device="cuda"
        ),
        "source_topk_snapshot": torch.full(
            (tokens, topk), -1, dtype=torch.int64, device="cuda"
        ),
        "hidden": hidden,
        "x_dtype": dtype,
        "num_experts": experts,
        "world_size": ranks,
        "max_tokens": capacity,
        "bias": None,
        "with_weights": False,
        "expand": False,
        "allow_multiple_reduction": mode,
    }


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_source(native, mode, expand, weighted, dtype):
    args = inputs(mode=mode, dtype=dtype, tokens=0)
    args.update(
        expand=expand,
        with_weights=weighted,
        bias=torch.empty((0, args["hidden"]), dtype=dtype, device="cuda"),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x, weights = native.combine_epilogue(**args)
    stream.synchronize()
    assert x.shape == (0, args["hidden"]) and x.dtype == dtype
    assert weights.shape == ((0, 3) if weighted else (0,))
    assert weights.dtype == torch.float32
    assert x.device == weights.device == args["stage_buffer"].device
    assert (args["stage_buffer"] == 165).all().item()


@pytest.mark.parametrize(
    "kind",
    [
        "stage_device",
        "stage_dtype",
        "stage_rank",
        "stage_stride",
        "stage_partitions",
        "stage_capacity",
        "stage_row_bytes",
        "stage_alignment",
        "snapshot_device",
        "snapshot_dtype",
        "snapshot_rank",
        "snapshot_stride",
        "output_dtype",
        "world_size",
        "experts",
        "hidden",
        "topk",
        "capacity",
        "source_capacity",
        "bias_device",
        "bias_dtype",
        "bias_shape",
        "bias_stride",
    ],
)
def test_invalid_storage_precedes_launch(native, kind):
    args = inputs()
    stage = args["stage_buffer"]
    before = stage.clone()
    if kind == "stage_device":
        args["stage_buffer"] = stage.cpu()
    elif kind == "stage_dtype":
        args["stage_buffer"] = stage.float()
    elif kind == "stage_rank":
        args["stage_buffer"] = stage.flatten()
    elif kind == "stage_stride":
        args["stage_buffer"] = stage.transpose(0, 1)
    elif kind.startswith("stage_") and kind != "stage_alignment":
        shape = list(stage.shape)
        dim = {"stage_partitions": 0, "stage_capacity": 1, "stage_row_bytes": 2}[kind]
        shape[dim] += 1
        args["stage_buffer"] = torch.empty(shape, dtype=torch.uint8, device="cuda")
    elif kind == "stage_alignment":
        args["stage_buffer"] = torch.empty(
            stage.numel() + 1, dtype=torch.uint8, device="cuda"
        )[1:].view(stage.shape)
    elif kind == "snapshot_device":
        args["source_topk_snapshot"] = args["source_topk_snapshot"].cpu()
    elif kind == "snapshot_dtype":
        args["source_topk_snapshot"] = args["source_topk_snapshot"].int()
    elif kind == "snapshot_rank":
        args["source_topk_snapshot"] = args["source_topk_snapshot"].flatten()
    elif kind == "snapshot_stride":
        args["source_topk_snapshot"] = torch.empty(
            (2, 6), dtype=torch.int64, device="cuda"
        )[:, ::2]
    elif kind == "output_dtype":
        args["x_dtype"] = torch.float64
    elif kind == "world_size":
        args["world_size"] = 0
    elif kind == "experts":
        args["num_experts"] = 5
    elif kind == "hidden":
        args["hidden"] = 0
    elif kind == "topk":
        args["source_topk_snapshot"] = torch.empty(
            (2, 0), dtype=torch.int64, device="cuda"
        )
    elif kind == "capacity":
        args["max_tokens"] = 0
    elif kind == "source_capacity":
        args["source_topk_snapshot"] = torch.empty(
            (6, 3), dtype=torch.int64, device="cuda"
        )
    elif kind == "bias_device":
        args["bias"] = torch.empty((2, 33))
    elif kind == "bias_dtype":
        args["bias"] = torch.empty((2, 33), dtype=torch.bfloat16, device="cuda")
    elif kind == "bias_shape":
        args["bias"] = torch.empty((33,), device="cuda")
    else:
        args["bias"] = torch.empty((2, 66), device="cuda")[:, ::2]
    with pytest.raises(RuntimeError):
        native.combine_epilogue(**args)
    torch.testing.assert_close(stage, before, rtol=0, atol=0)


def check_epilogue(native, case, source, expand, mode, weighted, bias):
    packet = make_return_packet(case, source, expand, mode, weighted)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stage = packet.cuda()
        snapshot = case.indices[source].cuda()
        device_bias = bias.cuda() if bias is not None else None
        outputs = native.combine_epilogue(
            stage,
            snapshot,
            case.hidden,
            case.values[source].dtype,
            case.num_experts,
            len(case.indices),
            case.capacity,
            device_bias,
            weighted,
            expand,
            mode,
        )
    stream.synchronize()
    assert all(output.device == stage.device for output in outputs)
    assert_combined_output(outputs, case, source, expand, mode, weighted, bias)
    torch.testing.assert_close(stage.cpu(), packet, rtol=0, atol=0)
    torch.testing.assert_close(snapshot.cpu(), case.indices[source], rtol=0, atol=0)
    if bias is not None:
        torch.testing.assert_close(
            device_bias.cpu(), bias, rtol=0, atol=0, equal_nan=True
        )

    # Outputs own their storage after the communication window is overwritten.
    with torch.cuda.stream(stream):
        stage.zero_()
    stream.synchronize()
    assert_combined_output(outputs, case, source, expand, mode, weighted, bias)
    return outputs


@pytest.mark.parametrize(
    "ranks,topk,hidden",
    [(1, 3, 4), (2, 3, 33), (2, 2, 65), (4, 2, 1), (2, 32, 1)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("biased", [False, True])
def test_source_reduction(
    native, ranks, topk, hidden, dtype, expand, mode, weighted, biased
):
    case = make_return_case(ranks, topk, hidden, dtype, tokens=11)
    case.capacity += 7
    for source in sorted({0, ranks - 1}):
        masked = case.indices[source] < 0
        case.values[source][masked] = float("nan")
        case.weights[source][masked] = float("nan")
        bias = (
            (torch.arange(11 * hidden).reshape(11, hidden) % 17 - 8).to(dtype) / 8
            if biased
            else None
        )
        check_epilogue(native, case, source, expand, mode, weighted, bias)


@pytest.mark.parametrize(
    "kind,tokens,ranks,topk,hidden",
    [
        ("mixed", 0, 2, 3, 65),
        ("empty", 5, 2, 3, 33),
        ("masked", 11, 4, 2, 65),
        ("nonfinite", 11, 2, 3, 33),
        ("mixed", 3, 32, 32, 1),
        ("mixed", 19, 1, 1, 129),
        ("mixed", 19, 2, 32, 33),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_source_reduction_edges(
    native, kind, tokens, ranks, topk, hidden, dtype, expand, mode, weighted
):
    case = make_return_case(ranks, topk, hidden, dtype, kind=kind, tokens=tokens)
    case.capacity += 7
    source = ranks - 1
    bias = torch.full((len(case.indices[source]), hidden), 0.5, dtype=dtype)
    check_epilogue(native, case, source, expand, mode, weighted, bias)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_fp32_accumulation_and_bias_before_final_cast(native, expand, mode):
    case = make_return_case(3, 3, 65, torch.bfloat16, tokens=1)
    case.capacity += 3
    case.indices[0][0] = torch.tensor([0, 3, -1])
    case.values[0][0, 0].fill_(256)
    case.values[0][0, 1].fill_(1)
    case.values[0][0, 2].fill_(float("nan"))
    case.weights[0][0] = torch.tensor([0.2, -0.7, float("nan")])
    bias = torch.full((1, 65), -256, dtype=torch.bfloat16)
    x, _ = check_epilogue(native, case, 0, expand, mode, True, bias)
    torch.testing.assert_close(
        x.cpu(), torch.ones((1, 65), dtype=torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
def test_compact_client_rounding_differs_from_expanded(native, expand, mode):
    case = make_return_case(2, 3, 33, torch.bfloat16, tokens=1)
    case.capacity += 3
    case.indices[0][0] = torch.tensor([0, 1, 3])
    case.values[0][0, 0].fill_(256)
    case.values[0][0, 1].fill_(1)
    case.values[0][0, 2].fill_(-256)
    x, _ = check_epilogue(native, case, 0, expand, mode, False, None)
    expected = torch.full((1, 33), 1 if expand else 0, dtype=torch.bfloat16)
    torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_main_to_epilogue_self_reuses_window(native, dtype, expand, mode, weighted):
    case = make_return_case(1, 3, 65, dtype, tokens=11)
    case.capacity += 7
    layout = native.window_layout(
        1, case.capacity, case.hidden, case.topk, case.num_experts, mode
    )
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    previous, saved, returned = None, None, None
    try:
        with torch.cuda.stream(stream):
            _, _, metadata = expert_inputs(case, 0, expand)
            device_metadata = [tensor.cuda() for tensor in metadata]
            for step in range(2):
                if step:
                    case.values = [value + 0.25 for value in case.values]
                    case.weights = [weight - 0.5 for weight in case.weights]
                x, weights, _ = expert_inputs(case, 0, expand)
                bias = torch.full((11, 65), 0.5, dtype=dtype) if step else None
                runtime.local_window()[layout[3] :].fill_(165)
                returned = native.combine_main(
                    runtime,
                    x.cuda(),
                    device_metadata,
                    weights.cuda() if weighted else None,
                    case.capacity,
                    expand,
                    mode,
                )
                outputs = native.combine_epilogue(
                    returned,
                    device_metadata[8],
                    case.hidden,
                    dtype,
                    case.num_experts,
                    1,
                    case.capacity,
                    bias.cuda() if bias is not None else None,
                    weighted,
                    expand,
                    mode,
                )
                assert_combined_output(outputs, case, 0, expand, mode, weighted, bias)
                if previous is not None:
                    for actual, expected in zip(previous, saved, strict=True):
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )
                previous, saved = outputs, [output.cpu().clone() for output in outputs]
                for actual, expected in zip(device_metadata, metadata, strict=True):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        stream.synchronize()
    finally:
        torch.cuda.synchronize()
        runtime.close_peer_mappings()
        del returned
        runtime.close()
