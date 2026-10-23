"""Exercise the portable C++ window layout without a CUDA extension."""

from pathlib import Path
import shutil
import subprocess

import pytest



@pytest.fixture(scope="module")
def layout_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("WindowLayout host tests require a C++20 compiler")
    directory = tmp_path_factory.mktemp("window-layout")
    source = directory / "check.cpp"
    source.write_text(
        '#include "layout.hpp"\n'
        "#include <iostream>\n"
        "#include <string>\n"
        "int main(int argc, char** argv) {\n"
        "  try {\n"
        "    const auto x = nano_ep::WindowLayout::make(std::stoi(argv[1]), std::stoll(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoll(argv[5]), std::stoi(argv[6]));\n"
        "    if (argc > 7) {\n"
        "      std::cout << x.return_rank_layout << ' ' << x.return_partitions << ' ' << x.return_capacity << ' '\n"
        "        << nano_ep::WindowLayout::return_row_bytes(std::stoi(argv[3]), std::stoi(argv[4]));\n"
        "    } else {\n"
        "      std::cout << x.signal_offset << ' ' << x.count_offset << ' ' << x.payload_offset << ' ' << x.return_offset << ' ' << x.num_bytes << ' ' << x.recv_next_offset;\n"
        "    }\n"
        "    return 0;\n"
        "  } catch (const std::exception& error) {\n"
        "    std::cerr << error.what(); return 2;\n"
        "  }\n"
        "}\n"
    )
    binary = directory / "check"
    repo = Path(__file__).resolve().parents[2]
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-I",
            str(repo / "csrc"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    return binary


def run_layout(
    binary,
    ranks,
    tokens,
    hidden,
    topk,
    experts,
    allow_multiple_reduction=False,
    *,
    details=False,
):
    return subprocess.run(
        [
            str(binary),
            *map(
                str,
                (ranks, tokens, hidden, topk, experts, int(allow_multiple_reduction)),
            ),
            *(["details"] if details else []),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_hand_calculated_layout(layout_binary):
    # R=2,T=3,H=4,K=2: 6 dispatch rows (64 B), 6 return rows (32 B).
    result = run_layout(layout_binary, 2, 3, 4, 2, 4)
    assert result.returncode == 0, result.stderr
    assert list(map(int, result.stdout.split())) == [0, 256, 512, 512, 1024, 304]


@pytest.mark.parametrize(
    "ranks,tokens,hidden,topk,experts",
    [
        (1, 1, 1, 1, 1),
        (2, 3, 4, 2, 4),
        (2, 17, 33, 8, 16),
        (4, 17, 127, 3, 12),
        (8, 2048, 4096, 8, 256),
        (32, 2, 129, 32, 1024),
    ],
)
@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
def test_shared_arena_covers_both_phases_without_overlapping_control(
    layout_binary, ranks, tokens, hidden, topk, experts, allow_multiple_reduction
):
    result = run_layout(
        layout_binary, ranks, tokens, hidden, topk, experts, allow_multiple_reduction
    )
    assert result.returncode == 0, result.stderr
    signal, count, payload, returned, total, recv_next = map(int, result.stdout.split())
    assert signal == 0
    assert all(
        offset % 256 == 0 for offset in (signal, count, payload, returned, total)
    )
    assert signal < count < payload == returned < total
    assert signal + ranks * 8 <= count
    assert recv_next == count + (ranks + experts) * 8
    assert 0 <= payload - (recv_next + 8) < 256
    payload_stride = (16 + 12 * topk + 4 * hidden + 15) // 16 * 16
    return_stride = (4 * hidden + 4 * topk + 15) // 16 * 16
    partitions = min(ranks, topk) if allow_multiple_reduction else topk
    payload_bytes = ranks * tokens * payload_stride
    return_bytes = partitions * tokens * return_stride
    assert total == (payload + max(payload_bytes, return_bytes) + 255) // 256 * 256
    previous_total = (
        ((payload + payload_bytes + 255) // 256 * 256 + return_bytes + 255) // 256 * 256
    )
    assert previous_total - total == min(
        (payload_bytes + 255) // 256 * 256, (return_bytes + 255) // 256 * 256
    )


@pytest.mark.parametrize(
    "ranks,topk", [(1, 1), (1, 4), (2, 8), (4, 4), (8, 2), (32, 32)]
)
@pytest.mark.parametrize("allow_multiple_reduction", [False, True])
def test_return_partition_policy(layout_binary, ranks, topk, allow_multiple_reduction):
    tokens, hidden = 19, 33
    result = run_layout(
        layout_binary,
        ranks,
        tokens,
        hidden,
        topk,
        ranks * 32,
        allow_multiple_reduction,
        details=True,
    )
    assert result.returncode == 0, result.stderr
    rank_layout, partitions, capacity, stride = map(int, result.stdout.split())
    assert bool(rank_layout) == (allow_multiple_reduction and ranks <= topk)
    assert partitions == (min(ranks, topk) if allow_multiple_reduction else topk)
    assert capacity == partitions * tokens
    assert stride == (4 * hidden + 4 * topk + 15) // 16 * 16
    # Partition-major addressing spans exactly the reserved rows.
    slots = {
        partition * tokens + token
        for partition in range(partitions)
        for token in range(tokens)
    }
    assert slots == set(range(capacity))


def test_modes_preserve_dispatch_regions_and_reduce_return_storage(layout_binary):
    offsets = []
    for mode in (False, True):
        result = run_layout(layout_binary, 2, 128, 33, 8, 16, mode)
        assert result.returncode == 0, result.stderr
        offsets.append(list(map(int, result.stdout.split())))
    assert offsets[0][:4] == offsets[1][:4]
    assert offsets[0][5] == offsets[1][5]
    # Single reduction needs 8 return partitions; multiple fits in dispatch's arena.
    assert offsets[0][4] == 512 + 8 * 128 * 176
    assert offsets[1][4] == 512 + 2 * 128 * 256


def test_maximum_replaces_sum_without_rejecting_representable_layout(layout_binary):
    # Each phase fits int64, but summing their bytes would overflow.
    tokens = ((1 << 63) - 1) // 40
    result = run_layout(layout_binary, 1, tokens, 1, 1, 1)
    assert result.returncode == 0, result.stderr
    offsets = list(map(int, result.stdout.split()))
    assert offsets[2] == offsets[3] == 512
    assert offsets[4] == (512 + tokens * 32 + 255) // 256 * 256


def test_count_region_has_no_epoch_fields(layout_binary):
    # 32 rank counts + 64 expert counts + recv_next occupy 776 B.
    result = run_layout(layout_binary, 32, 1, 1, 1, 64)
    assert result.returncode == 0, result.stderr
    assert list(map(int, result.stdout.split())) == [0, 256, 1280, 1280, 2304, 1024]


def test_expert_count_capacity_changes_offsets(layout_binary):
    small = run_layout(layout_binary, 2, 1, 1, 1, 4)
    large = run_layout(layout_binary, 2, 1, 1, 1, 128)
    assert small.returncode == large.returncode == 0
    small_offsets = list(map(int, small.stdout.split()))
    large_offsets = list(map(int, large.stdout.split()))
    assert small_offsets[2] == 512
    assert large_offsets[2] == 1536
    assert large_offsets[5] == 1296


@pytest.mark.parametrize(
    "args",
    [
        (0, 1, 1, 1, 4),
        (33, 1, 1, 1, 66),
        (1, 0, 1, 1, 4),
        (1, -1, 1, 1, 4),
        (1, 1, 0, 1, 4),
        (1, 1, -1, 1, 4),
        (1, 1, 1, 0, 4),
        (1, 1, 1, 33, 4),
        (2, 1, 1, 1, 0),
        (2, 1, 1, 1, -2),
        (2, 1, 1, 1, 3),
    ],
)
def test_invalid_dimensions(layout_binary, args):
    result = run_layout(layout_binary, *args)
    assert result.returncode == 2
    assert "WindowLayout" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        (32, (1 << 63) - 1, 1, 1, 64),
        (1, ((1 << 63) - 1) // 32, 1, 1, 4),
        (1, ((1 << 63) - 1) // 128, 1, 32, 32),
        (1, ((1 << 63) - 1) // 4608 + 1, 1, 32, 32),
        (1, 1, 1, 1, (1 << 63) - 1),
    ],
)
def test_checked_size_overflow(layout_binary, args):
    result = run_layout(layout_binary, *args)
    assert result.returncode == 2
    assert "overflow" in result.stderr
