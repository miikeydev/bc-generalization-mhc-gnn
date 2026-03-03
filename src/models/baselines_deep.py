from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import APPNP as APPNPConv
from torch_geometric.nn import GCN2Conv, GCNConv, JumpingKnowledge


class GCNIINodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        alpha: float = 0.1,
        theta: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i + 1, shared_weights=True)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = torch.relu(self.input_proj(x))
        hidden = x0
        for conv in self.convs:
            hidden = self.dropout(hidden)
            hidden = conv(hidden, x0, edge_index)
            hidden = torch.relu(hidden)
        return self.output_layer(hidden).squeeze(-1)


class APPNPNodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        alpha: float = 0.1,
        K: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*layers)
        self.prop = APPNPConv(K=K, alpha=alpha)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.mlp(x)
        hidden = self.prop(hidden, edge_index)
        return self.output_layer(hidden).squeeze(-1)


class JKNetNodeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        mode: str = "max",
    ) -> None:
        super().__init__()
        convs = [GCNConv(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(convs)
        if mode == "lstm":
            self.jk = JumpingKnowledge(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        else:
            self.jk = JumpingKnowledge(mode=mode)
        jk_out_dim = hidden_dim * num_layers if mode == "cat" else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(jk_out_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = x
        representations: list[torch.Tensor] = []
        for conv in self.convs:
            hidden = conv(hidden, edge_index)
            hidden = torch.relu(hidden)
            hidden = self.dropout(hidden)
            representations.append(hidden)
        out = self.jk(representations)
        return self.output_layer(out).squeeze(-1)
