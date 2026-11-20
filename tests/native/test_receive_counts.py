"""Check the fused notify finalizer against an independent int64 count oracle."""

import pytest
import torch

pytestmark = [pytest.mark.cuda]


@pytest.mark.parametrize(
    "ranks,experts,alignment,kind",
    [
        pytest.param(1, 31, 1, "random", id="single-rank-unaligned"),
        pytest.param(2, 31, 4, "random", id="two-ranks-aligned"),
        pytest.param(8, 32, 16, "random", id="eight-ranks-aligned"),
        pytest.param(8, 33, 4, "zero", id="all-masked"),
    ],
)
def test_fused_receive_prefixes(payload_probe, ranks, experts, alignment, kind):
    generator = torch.Generator().manual_seed(71)
    incoming = torch.randint(0, 19, (ranks + ranks * experts,), generator=generator, dtype=torch.int64)
    if kind == "zero":
        incoming.zero_()
    else:
        # Empty experts and empty source ranks remain part of the prefix domain.
        incoming[ranks:].view(ranks, experts)[:, ::3] = 0
        if ranks > 1:
            incoming[ranks:].view(ranks, experts)[0].zero_()
        incoming[:ranks] = incoming[ranks:].view(ranks, experts).sum(1)
    expected_counts = incoming[ranks:].view(ranks, experts).sum(0)
    padded = (expected_counts + alignment - 1) // alignment * alignment
    expected_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), padded.cumsum(0)))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = payload_probe.run_finalize_receive_counts(incoming.cuda(), ranks, alignment)
        outputs = [tensor.cpu() for tensor in actual]
    torch.testing.assert_close(outputs[0], incoming[:ranks].cumsum(0), rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected_offsets, rtol=0, atol=0)
    torch.testing.assert_close(outputs[4], expected_offsets[:-1], rtol=0, atol=0)
    torch.testing.assert_close(
        outputs[3],
        torch.stack((incoming[:ranks].sum(), expected_offsets[-1])),
        rtol=0,
        atol=0,
    )
