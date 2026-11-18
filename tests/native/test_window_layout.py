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

    assert len(layout) == 5
    assert layout[0] == 0
    assert all(a <= b for a, b in zip(layout[:-1], layout[1:], strict=True))
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
