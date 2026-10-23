import pytest

from nano_ep.backends._build import load_extension

pytestmark = [pytest.mark.cuda]


@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
@pytest.mark.parametrize("ranks,topk", [(1, 4), (2, 8), (4, 4), (8, 2)])
def test_window_layout_binding_and_allocation(
    request, ranks, topk, allow_multiple_reduction
):
    if not (
        request.config.getoption("--run-cuda")
        or request.config.getoption("--run-distributed")
    ):
        pytest.skip("opt in with --run-cuda on an allocated GPU")
    native = load_extension()
    tokens, hidden, experts = 19, 33, ranks * 8
    layout = native.window_layout(
        ranks, tokens, hidden, topk, experts, allow_multiple_reduction
    )

    def align(size):
        return (size + 255) // 256 * 256

    payload = align(256 + (ranks + experts + 1) * 8)
    returned = payload
    partitions = min(ranks, topk) if allow_multiple_reduction else topk
    total = align(
        payload
        + max(
            ranks * tokens * ((16 + 12 * topk + 4 * hidden + 15) // 16 * 16),
            partitions * tokens * ((4 * hidden + 4 * topk + 15) // 16 * 16),
        )
    )
    assert layout == [0, 256, payload, returned, total]
    assert native.window_layout(2, 3, 4, 2, 4) == [0, 256, 512, 512, 1024]
    with pytest.raises(ValueError, match="WindowLayout"):
        native.window_layout(0, 3, 4, 2, 4)
    with pytest.raises(OverflowError, match="overflow"):
        native.window_layout(32, (1 << 63) - 1, 4, 2, 64)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    assert window.numel() == layout[-1]
    assert window.data_ptr() % 256 == 0
    for start, end in zip(layout[:-1], layout[1:], strict=True):
        assert window[start:end].numel() == end - start
    del window
    runtime.close_peer_mappings()
    runtime.close()
