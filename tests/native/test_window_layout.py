import pytest

from nano_ep.backends._build import load_extension
from nano_ep.testing.wire import align32, dispatch_row_bytes, return_row_bytes

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

    payload = align32(align32(ranks * 8) + (ranks + experts + 1) * 8)
    returned = payload
    partitions = min(ranks, topk) if allow_multiple_reduction else topk
    total = align32(
        payload
        + max(
            ranks * tokens * dispatch_row_bytes(hidden, topk),
            partitions * tokens * return_row_bytes(hidden, topk),
        )
    )
    assert layout == [0, align32(ranks * 8), payload, returned, total]
    assert native.window_layout(2, 3, 4, 2, 4) == [0, 32, 96, 96, 864]
    with pytest.raises(ValueError, match="WindowLayout"):
        native.window_layout(0, 3, 4, 2, 4)
    with pytest.raises(OverflowError, match="overflow"):
        native.window_layout(32, (1 << 63) - 1, 4, 2, 64)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    assert window.numel() == layout[-1]
    assert window.data_ptr() % 32 == 0
    for start, end in zip(layout[:-1], layout[1:], strict=True):
        assert window[start:end].numel() == end - start
    del window
    runtime.close_peer_mappings()
    runtime.close()
