from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNNodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        convs = [GCNConv(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = x
        for conv in self.convs:
            hidden = conv(hidden, edge_index)
            hidden = torch.relu(hidden)
            hidden = self.dropout(hidden)
        return self.output_layer(hidden).squeeze(-1)
