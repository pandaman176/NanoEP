from dataclasses import dataclass

import torch


@dataclass
class Case:
    xs: list[torch.Tensor]
    indices: list[torch.Tensor]
    weights: list[torch.Tensor]
    expert_matrices: torch.Tensor
    num_experts: int
    num_topk: int
    hidden: int
    capacity: int


def make_case(
    world_size: int = 2,
    *,
    seed: int = 0,
    kind: str = "random",
    dtype: torch.dtype = torch.float32,
) -> Case:
    """Small reproducible inputs, with weights intentionally nonzero at -1 slots."""
    if kind not in {"random", "empty", "masked", "skew", "mixed"}:
        raise ValueError(f"unknown case: {kind}")
    generator = torch.Generator().manual_seed(seed)
    num_experts, num_topk, hidden, capacity = world_size * 3, 3, 4, 5
    xs, indices, weights = [], [], []
    for rank in range(world_size):
        count = 5 - rank % 4
        if kind == "empty" and rank == world_size - 1:
            count = 0
        x = torch.randn(count, hidden, generator=generator, dtype=dtype)
        idx = torch.empty((count, num_topk), dtype=torch.int64)
        for token in range(count):
            if kind == "skew":
                selected = torch.tensor([2, 0, 1], dtype=torch.int64)
            elif kind == "mixed" and token % 2 == 0:
                base = (token % world_size) * 3
                selected = torch.tensor([base + 2, base, base + 1], dtype=torch.int64)
            else:
                selected = torch.randperm(num_experts, generator=generator)[:num_topk]
            if kind == "masked" or (kind in {"random", "mixed"} and token == 0):
                selected = selected.clone()
                selected[1 if kind != "masked" else slice(None)] = -1
            idx[token] = selected
        weight_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        w = torch.randn(count, num_topk, generator=generator, dtype=weight_dtype) + 0.3
        xs.append(x)
        indices.append(idx)
        weights.append(w)
    matrices = torch.randn(num_experts, hidden, hidden, generator=generator, dtype=dtype) * 0.2
    return Case(xs, indices, weights, matrices, num_experts, num_topk, hidden, capacity)
