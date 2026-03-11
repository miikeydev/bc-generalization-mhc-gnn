from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import EdgeConv, GATConv, GINConv, SAGEConv


class SAGENodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        convs = [SAGEConv(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden_dim, hidden_dim))
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


class GATNodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        head_dim = max(1, hidden_dim // num_heads)
        convs = [GATConv(input_dim, head_dim, heads=num_heads, dropout=dropout, concat=True)]
        for _ in range(num_layers - 1):
            convs.append(GATConv(head_dim * num_heads, head_dim, heads=num_heads, dropout=dropout, concat=True))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(head_dim * num_heads, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = x
        for conv in self.convs:
            hidden = conv(hidden, edge_index)
            hidden = torch.relu(hidden)
            hidden = self.dropout(hidden)
        return self.output_layer(hidden).squeeze(-1)


class GINNodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        def make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        convs = [GINConv(make_mlp(input_dim, hidden_dim))]
        for _ in range(num_layers - 1):
            convs.append(GINConv(make_mlp(hidden_dim, hidden_dim)))
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


def _edgeconv_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim * 2, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class EdgeConvNodeRegressor(nn.Module):
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

        convs = [EdgeConv(_edgeconv_mlp(input_dim, hidden_dim))]
        for _ in range(num_layers - 1):
            convs.append(EdgeConv(_edgeconv_mlp(hidden_dim, hidden_dim)))
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


class EdgeConvResidualNodeRegressor(nn.Module):
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

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(EdgeConv(_edgeconv_mlp(hidden_dim, hidden_dim)) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        for conv in self.convs:
            residual = hidden
            hidden = conv(hidden, edge_index)
            hidden = torch.relu(hidden)
            hidden = self.dropout(hidden)
            hidden = hidden + residual
        return self.output_layer(hidden).squeeze(-1)
