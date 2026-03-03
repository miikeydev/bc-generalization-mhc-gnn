from __future__ import annotations

from torch import nn

from .gcn_baseline import GCNNodeRegressor
from .hyper_connection_gnn import HyperConnectionGNNRegressor


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

    if model_name in {"hc_gnn", "mhc_gnn", "mhc_lite_gnn", "hc", "mhc", "mhc_lite"}:
        variant_map = {
            "hc_gnn": "hc",
            "hc": "hc",
            "mhc_gnn": "mhc",
            "mhc": "mhc",
            "mhc_lite_gnn": "mhc_lite",
            "mhc_lite": "mhc_lite",
        }
        variant = variant_map[model_name]

        return HyperConnectionGNNRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            n_streams=int(model_cfg.get("n_streams", 4)),
            variant=variant,
            gnn_type=str(model_cfg.get("gnn_type", "gcn")),
            use_dynamic=bool(model_cfg.get("use_dynamic", True)),
            use_static=bool(model_cfg.get("use_static", True)),
            init_alpha=float(model_cfg.get("mapping_init_alpha", 0.01)),
            sinkhorn_tau=float(model_cfg.get("sinkhorn_tau", 0.1)),
            sinkhorn_iters=int(model_cfg.get("sinkhorn_iters", 20)),
            mhc_lite_max_permutations=(
                int(model_cfg["mhc_lite_max_permutations"])
                if model_cfg.get("mhc_lite_max_permutations") is not None
                else None
            ),
            mhc_lite_permutation_seed=int(model_cfg.get("mhc_lite_permutation_seed", 0)),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
