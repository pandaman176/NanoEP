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
        pytest.skip(
            "opt in with --run-cuda or --run-distributed; IPC lifecycle is unverified on CPU"
        )
    if not torch.cuda.is_available():
        pytest.fail("--run-cuda requested but NVIDIA CUDA is unavailable")
    return load_extension()


def test_two_phase_close(native):
    runtime = native.PeerRuntime(0, 1, 4096)
    handle = runtime.export_local_handle()
    runtime.import_peer_handles([handle])
    window = runtime.local_window()
    window.fill_(23)
    torch.cuda.synchronize()

    with pytest.raises(RuntimeError, match="close_peer_mappings"):
        runtime.close()
    runtime.close_peer_mappings()
    runtime.close_peer_mappings()
    # Unmapping peers does not free the source-owned allocation.
    assert torch.all(window == 23).item()
    for method in (
        runtime.export_local_handle,
        runtime.peer_addresses,
        runtime.local_window,
    ):
        with pytest.raises(RuntimeError, match="closing or closed"):
            method()
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.barrier(torch.cuda.current_stream().cuda_stream, 1)

    torch.cuda.synchronize()
    del window
    # One rank has no remote importers; the host group barrier is vacuous.
    runtime.close()
    runtime.close()
    runtime.close_peer_mappings()
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.local_window()


@pytest.mark.parametrize("experts", [4, 260])
def test_automatic_barrier_epochs_and_notify_reuse(native, experts):
    layout = native.window_layout(1, 8, 4, 2, experts)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    window = runtime.local_window()
    count_offset = layout[1]
    counts = window[count_offset : count_offset + (1 + experts) * 8].view(torch.int64)
    counts.fill_(-1)
    torch.cuda.synchronize()
    runtime.import_peer_handles([runtime.export_local_handle()])
    stream = torch.cuda.current_stream()
    try:
        runtime.barrier(stream.cuda_stream, 5)
        runtime.barrier(stream.cuda_stream)
        assert window.view(torch.int64)[0].item() == 6
        for value in (0, 7, 0, 2):
            send = torch.tensor([value], dtype=torch.int64, device="cuda")
            send_experts = torch.zeros(experts, dtype=torch.int64, device="cuda")
            send_experts[0] = value
            send_experts[-1] = (value + 1) // 2
            runtime.barrier(stream.cuda_stream)
            received, received_experts = native.notify_counts(
                runtime, send, send_experts, count_offset
            )
            assert received.tolist() == [value]
            torch.testing.assert_close(received_experts, send_experts)
            assert counts.tolist() == [-1] * (1 + experts)
            runtime.barrier(stream.cuda_stream)
        # Four entry/exit pairs advance only the barrier signal.
        assert window.view(torch.int64)[0].item() == 14
        assert (
            window[count_offset + (1 + experts) * 8 : count_offset + (2 + experts) * 8]
            .view(torch.int64)
            .item()
            == 0
        )
    finally:
        torch.cuda.synchronize()
        del counts, window
        runtime.close_peer_mappings()
        runtime.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 11])
def test_reserve_and_put_payload(native, dtype, weighted, tokens):
    hidden, topk, capacity = 37, 3, 11
    layout = native.window_layout(1, capacity, hidden, topk, 4)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    window = runtime.local_window()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            x = (
                torch.arange(tokens * hidden, device="cuda").reshape(tokens, hidden) / 8
            ).to(dtype)
            pattern = torch.tensor(
                [[0, 1, 2], [-1, -1, -1], [3, -1, 0], [-1, 2, -1]], device="cuda"
            )
            indices = pattern.repeat(((tokens + 3) // 4, 1))[:tokens].contiguous()
            weights = (
                torch.arange(tokens * topk, device="cuda", dtype=torch.float32).reshape(
                    tokens, topk
                )
                / 7
                if weighted
                else None
            )
            runtime.barrier(stream.cuda_stream)
            slots = native.reserve_and_put(runtime, x, indices, weights, 4, capacity)
            runtime.barrier(stream.cuda_stream)
            # Read on the producer stream; no default-stream ordering is assumed.
            slots_cpu, indices_cpu, x_cpu = slots.cpu(), indices.cpu(), x.float().cpu()
            weights_cpu = weights.cpu() if weighted else None
            received = window.cpu()

        assert slots_cpu.shape == (tokens, 1)
        assert slots_cpu.dtype == torch.int64
        valid_tokens = [
            token for token in range(tokens) if (indices_cpu[token] >= 0).any()
        ]
        assert received[layout[1] + 40 : layout[1] + 48].view(
            torch.int64
        ).item() == len(valid_tokens)
        assert sorted(slots_cpu[valid_tokens, 0].tolist()) == list(
            range(len(valid_tokens))
        )
        row_bytes = (16 + 12 * topk + 4 * hidden + 15) // 16 * 16
        for token in range(tokens):
            slot = slots_cpu[token, 0].item()
            if token not in valid_tokens:
                assert slot == -1
                continue
            row = received[
                layout[2] + slot * row_bytes : layout[2] + (slot + 1) * row_bytes
            ]
            assert row[:16].view(torch.int64).tolist() == [0, token]
            torch.testing.assert_close(
                row[16 : 16 + 8 * topk].view(torch.int64), indices_cpu[token]
            )
            if weighted:
                torch.testing.assert_close(
                    row[16 + 8 * topk : 16 + 12 * topk].view(torch.float32),
                    weights_cpu[token],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                row[16 + 12 * topk : 16 + 12 * topk + 4 * hidden].view(torch.float32),
                x_cpu[token],
                rtol=0,
                atol=0,
            )
    finally:
        torch.cuda.synchronize()
        del window
        runtime.close_peer_mappings()
        runtime.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 11])
def test_cached_payload_preserves_mapping_and_counter(native, dtype, tokens):
    hidden, topk, capacity = 37, 3, 11
    layout = native.window_layout(1, capacity, hidden, topk, 4)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    window = runtime.local_window()
    recv_next = window[layout[1] + 40 : layout[1] + 48].view(torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    row_bytes = (16 + 12 * topk + 4 * hidden + 15) // 16 * 16
    try:
        with torch.cuda.stream(stream):
            pattern = torch.tensor(
                [[0, 1, 2], [-1, -1, -1], [3, -1, 0], [-1, 2, -1]], device="cuda"
            )
            indices = pattern.repeat(((tokens + 3) // 4, 1))[:tokens].contiguous()
            base_x = (
                torch.arange(tokens * hidden, device="cuda").reshape(tokens, hidden) / 8
            ).to(dtype)
            slots = None
            previous_slots = None
            for phase in range(4):
                cached = phase in (1, 2)
                if phase == 3:
                    indices = torch.full_like(indices, -1)
                    indices[::2, 0] = 3
                x = base_x + phase * 64
                weights = (
                    torch.full(
                        (tokens, topk), phase + 0.25, device="cuda", dtype=torch.float32
                    )
                    if phase != 2
                    else None
                )
                if not cached:
                    recv_next.zero_()
                runtime.barrier(stream.cuda_stream)
                old_slots, old_counter = slots, recv_next.item()
                slots = native.reserve_and_put(
                    runtime,
                    x,
                    indices,
                    weights,
                    4,
                    capacity,
                    cached_dst_slots=slots if cached else None,
                )
                runtime.barrier(stream.cuda_stream)
                slots_cpu, received, indices_cpu = (
                    slots.cpu(),
                    window.cpu(),
                    indices.cpu(),
                )
                x_cpu = x.float().cpu()
                expected_count = int((indices_cpu >= 0).any(dim=1).sum())
                assert recv_next.item() == expected_count
                if cached:
                    assert slots.data_ptr() == old_slots.data_ptr()
                    torch.testing.assert_close(slots_cpu, previous_slots)
                    assert recv_next.item() == old_counter
                previous_slots = slots_cpu.clone()
                assert sorted(slots_cpu[slots_cpu >= 0].tolist()) == list(
                    range(expected_count)
                )
                for token in range(tokens):
                    slot = slots_cpu[token, 0].item()
                    if not (indices_cpu[token] >= 0).any():
                        assert slot == -1
                        continue
                    row = received[
                        layout[2] + slot * row_bytes : layout[2]
                        + (slot + 1) * row_bytes
                    ]
                    assert row[:16].view(torch.int64).tolist() == [0, token]
                    torch.testing.assert_close(
                        row[16 : 16 + 8 * topk].view(torch.int64), indices_cpu[token]
                    )
                    torch.testing.assert_close(
                        row[16 + 12 * topk : 16 + 12 * topk + 4 * hidden].view(
                            torch.float32
                        ),
                        x_cpu[token],
                        rtol=0,
                        atol=0,
                    )
                    if weights is not None:
                        torch.testing.assert_close(
                            row[16 + 8 * topk : 16 + 12 * topk].view(torch.float32),
                            weights[token].cpu(),
                            rtol=0,
                            atol=0,
                        )
    finally:
        torch.cuda.synchronize()
        del recv_next, window
        runtime.close_peer_mappings()
        runtime.close()


@pytest.mark.parametrize("kind", ["shape", "dtype", "device", "stride", "empty_shape"])
def test_invalid_cached_slot_storage(native, kind):
    tokens = 0 if kind == "empty_shape" else 3
    layout = native.window_layout(1, 3, 4, 2, 4)
    runtime = native.PeerRuntime(0, 1, layout[-1])
    runtime.import_peer_handles([runtime.export_local_handle()])
    x = torch.zeros((tokens, 4), device="cuda")
    indices = torch.zeros((tokens, 2), device="cuda", dtype=torch.int64)
    slots = torch.zeros((tokens, 1), device="cuda", dtype=torch.int64)
    if kind in ("shape", "empty_shape"):
        slots = torch.zeros((tokens, 2), device="cuda", dtype=torch.int64)
    elif kind == "dtype":
        slots = slots.float()
    elif kind == "device":
        slots = slots.cpu()
    elif kind == "stride":
        slots = torch.zeros((tokens, 2), device="cuda", dtype=torch.int64)[:, :1]
    try:
        with pytest.raises(RuntimeError, match="cached_dst_slots"):
            native.reserve_and_put(runtime, x, indices, None, 4, 3, slots)
    finally:
        torch.cuda.synchronize()
        runtime.close_peer_mappings()
        runtime.close()
