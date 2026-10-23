from dataclasses import dataclass


@dataclass(frozen=True)
class EPConfig:
    """
    Configuration for the Expert Parallelism (EP) system.

    num_experts: Number of experts in the system.

    num_topk: Number of top-k experts to consider for each token.

    hidden: Hidden dimension of the input tensors.

    num_max_tokens_per_rank: Maximum number of tokens per rank.
    """

    num_experts: int
    num_topk: int
    hidden: int
    num_max_tokens_per_rank: int

    def __post_init__(self):
        if (
            min(
                self.num_experts,
                self.num_topk,
                self.hidden,
                self.num_max_tokens_per_rank,
            )
            <= 0
        ):
            raise ValueError("EP dimensions and token capacity must be positive")
        if self.num_topk > min(32, self.num_experts):
            raise ValueError("num_topk must be <= min(32, num_experts)")
