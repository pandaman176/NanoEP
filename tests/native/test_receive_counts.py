"""Check the fused notify finalizer against an independent int64 count oracle."""

import pytest
import torch

pytestmark = [pytest.mark.cuda]


@pytest.mark.parametrize("ranks", [1, 2, 8, 32])
@pytest.mark.parametrize("experts", [1, 31, 32, 33, 129, 260])
@pytest.mark.parametrize("alignment", [1, 3, 4, 16])
@pytest.mark.parametrize("kind", ["random", "zero", "wide"])
def test_fused_receive_prefixes(payload_probe, ranks, experts, alignment, kind):
    generator = torch.Generator().manual_seed(71)
    incoming = torch.randint(
        0, 19, (ranks + ranks * experts,), generator=generator, dtype=torch.int64
    )
    if kind == "zero":
        incoming.zero_()
    elif kind == "wide":
        incoming.fill_(2**32 - 1)
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
        actual = payload_probe.run_finalize_receive_counts(
            incoming.cuda(), ranks, alignment
        )
        outputs = [tensor.cpu() for tensor in actual]
    torch.testing.assert_close(outputs[0], incoming[:ranks].cumsum(0), rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected_offsets, rtol=0, atol=0)
