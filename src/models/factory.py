from __future__ import annotations

from torch import nn

from .gcn_baseline import GCNNodeRegressor


def build_model(config: dict, input_dim: int) -> nn.Module:
    model_cfg = config["model"]
    model_name = model_cfg["name"].lower()

    if model_name == "gcn":
        return GCNNodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
