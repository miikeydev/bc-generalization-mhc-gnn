from __future__ import annotations

from torch import nn

from .baselines_deep import APPNPNodeRegressor, GCNIINodeRegressor, JKNetNodeRegressor
from .baselines_shallow import GATNodeRegressor, GINNodeRegressor, SAGENodeRegressor
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

    if model_name == "sage":
        return SAGENodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
        )

    if model_name == "gat":
        return GATNodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            num_heads=int(model_cfg.get("num_heads", 4)),
        )

    if model_name == "gin":
        return GINNodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
        )

    if model_name == "gcnii":
        return GCNIINodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            alpha=float(model_cfg.get("alpha", 0.1)),
            theta=float(model_cfg.get("theta", 0.5)),
        )

    if model_name == "appnp":
        return APPNPNodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            alpha=float(model_cfg.get("alpha", 0.1)),
            K=int(model_cfg.get("K", 10)),
        )

    if model_name == "jknet":
        return JKNetNodeRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            mode=str(model_cfg.get("mode", "max")),
        )

    raise ValueError(f"Unsupported model name: {model_name}")