from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PairwiseRankingLoss(nn.Module):
    def __init__(self, pairs_per_node: int) -> None:
        super().__init__()
        self.pairs_per_node = pairs_per_node

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_nodes = targets.shape[0]
        if num_nodes < 2:
            return scores.new_tensor(0.0)

        sample_size = max(num_nodes, self.pairs_per_node * num_nodes)
        left_idx = torch.randint(0, num_nodes, (sample_size,), device=targets.device)
        right_idx = torch.randint(0, num_nodes, (sample_size,), device=targets.device)

        target_sign = torch.sign(targets[left_idx] - targets[right_idx])
        valid = target_sign != 0
        if not torch.any(valid):
            return scores.new_tensor(0.0)

        margin = scores[left_idx] - scores[right_idx]
        return F.softplus(-target_sign[valid] * margin[valid]).mean()
