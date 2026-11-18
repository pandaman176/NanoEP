"""Check native combine results and their ownership across arena reuse."""

import pytest
import torch

from nano_ep.backends._build import load_extension
from nano_ep.testing.native_combine import (
    assert_combined_output,
    expert_inputs,
    make_return_case,
)

pytestmark = [pytest.mark.cuda]


@pytest.fixture
def cuda_device(request):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda or --run-distributed")
    if not torch.cuda.is_available():
        pytest.fail("CUDA tests requested but NVIDIA CUDA is unavailable")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def native(cuda_device):
    return load_extension()


def test_native_combine_preserves_invalid_input_error(native):
    case = make_return_case(1, 3, 256, torch.bfloat16, tokens=1)
    layout = native.window_layout(
        1, case.capacity, case.hidden, case.topk, case.num_experts, False
    )
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    try:
        x, _, metadata = expert_inputs(case, 0, False)
        with pytest.raises(RuntimeError, match="combine x must be contiguous CUDA"):
            native.combine(
                runtime,
                x,
                [tensor.cuda() for tensor in metadata],
                None,
                None,
                case.capacity,
                False,
                False,
                0,
                torch.cuda.current_stream().cuda_stream,
            )
    finally:
        torch.cuda.synchronize()
        runtime.close_peer_mappings()
        runtime.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_native_combine_reuses_window(native, dtype, expand, mode, weighted):
    case = make_return_case(1, 3, 256, dtype, tokens=11)
    case.capacity += 7
    layout = native.window_layout(
        1, case.capacity, case.hidden, case.topk, case.num_experts, mode
    )
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    previous, saved = None, None
    try:
        with torch.cuda.stream(stream):
            _, _, metadata = expert_inputs(case, 0, expand)
            device_metadata = [tensor.cuda() for tensor in metadata]
            for step in range(2):
                if step:
                    case.values = [value + 0.25 for value in case.values]
                    case.weights = [weight - 0.5 for weight in case.weights]
                x, weights, _ = expert_inputs(case, 0, expand)
                bias = torch.full((11, 256), 0.5, dtype=dtype) if step else None
                runtime.local_window()[layout[3] :].fill_(165)
                outputs, _ = native.combine(
                    runtime,
                    x.cuda(),
                    device_metadata,
                    weights.cuda() if weighted else None,
                    bias.cuda() if bias is not None else None,
                    case.capacity,
                    expand,
                    mode,
                    0,
                    stream.cuda_stream,
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
        runtime.close()
