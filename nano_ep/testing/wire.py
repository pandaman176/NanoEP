"""Independent byte-layout specification for BF16 EP communication tests."""


def align32(size: int) -> int:
    return (size + 31) // 32 * 32


def dispatch_offsets(topk: int) -> tuple[int, int, int]:
    """Return byte offsets for int64 top-k, FP32 weights and BF16 x."""
    topk_offset = 32
    weights_offset = align32(topk_offset + 8 * topk)
    x_offset = align32(weights_offset + 4 * topk)
    return topk_offset, weights_offset, x_offset


def dispatch_row_bytes(hidden: int, topk: int) -> int:
    return align32(dispatch_offsets(topk)[2] + 2 * hidden)


def return_weights_offset(hidden: int) -> int:
    return align32(2 * hidden)


def return_row_bytes(hidden: int, topk: int) -> int:
    return align32(return_weights_offset(hidden) + 4 * topk)
