from __future__ import annotations

import json
from typing import Any


def normalize_feature_mode(mode: str) -> str:
    mode_lower = str(mode).lower().strip()
    if mode_lower == "structural_only":
        return "structural_only"
    if mode_lower in {"degree_only", "degree"}:
        return "degree_only"
    if mode_lower in {"degree_plus_rwpe", "degree_rwpe", "random_walk", "rwpe"}:
        return "degree_plus_rwpe"
    if mode_lower in {"degree_plus_ppr", "degree_ppr", "ppr"}:
        return "degree_plus_ppr"
    if mode_lower in {"random", "gaussian"}:
        return "random"
    if mode_lower in {"none", "constant", "ones"}:
        return "none"
    return mode_lower


def normalize_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(data_cfg)

    cfg["train_families"] = list(cfg.get("train_families", ["er", "ba", "sbm"]))
    cfg["ood_families"] = list(cfg.get("ood_families", ["ws", "rgg"]))
    cfg["train_graphs"] = int(cfg.get("train_graphs", 160))
    cfg["val_graphs"] = int(cfg.get("val_graphs", 30))
    cfg["test_graphs_id"] = int(cfg.get("test_graphs_id", 30))
    cfg["test_graphs_ood"] = int(cfg.get("test_graphs_ood", 30))
    cfg["train_num_nodes"] = int(cfg.get("train_num_nodes", 100))
    cfg["test_num_nodes"] = int(cfg.get("test_num_nodes", 500))
    cfg["average_degree"] = int(cfg.get("average_degree", 8))
    cfg["feature_mode"] = normalize_feature_mode(str(cfg.get("feature_mode", "structural_only")))
    cfg["lap_pe_dim"] = int(cfg.get("lap_pe_dim", 8))
    cfg["random_feature_dim"] = int(cfg.get("random_feature_dim", 16))
    cfg["bc_backend"] = str(cfg.get("bc_backend", "networkx")).lower()
    cfg["bc_mode"] = str(cfg.get("bc_mode", "exact")).lower()
    cfg["bc_approximation_k"] = cfg.get("bc_approximation_k")
    cfg["train_sign_flip_lappe"] = bool(cfg.get("train_sign_flip_lappe", False))
    cfg["feature_config"] = dict(cfg.get("feature_config", {}))

    if cfg["bc_mode"] not in {"exact", "approx"}:
        raise ValueError(f"Unsupported bc_mode: {cfg['bc_mode']}")
    if cfg["bc_backend"] not in {"networkx", "networkit", "auto"}:
        raise ValueError(f"Unsupported bc_backend: {cfg['bc_backend']}")

    return cfg


def infer_input_dim_from_data_config(data_cfg: dict[str, Any]) -> int:
    normalized = normalize_data_config(data_cfg)
    feature_mode = normalized["feature_mode"]
    feature_config = normalized["feature_config"]

    if feature_mode == "degree_only":
        return 2
    if feature_mode == "degree_plus_rwpe":
        return 2 + int(feature_config.get("rwpe_dim", 8))
    if feature_mode == "degree_plus_ppr":
        return 2 + int(feature_config.get("ppr_dim", 8))
    if feature_mode == "structural_only":
        return 2 + int(normalized["lap_pe_dim"])
    if feature_mode == "random":
        return int(normalized["random_feature_dim"])
    if feature_mode == "none":
        return 1
    raise ValueError(f"Cannot infer input_dim for feature_mode={feature_mode}")


def make_data_cache_signature(data_cfg: dict[str, Any]) -> tuple[Any, ...]:
    normalized = normalize_data_config(data_cfg)
    return (
        normalized["feature_mode"],
        normalized["lap_pe_dim"],
        normalized["random_feature_dim"],
        normalized["bc_backend"],
        normalized["bc_mode"],
        normalized["bc_approximation_k"],
        json.dumps(normalized["feature_config"], sort_keys=True, separators=(",", ":")),
    )


def compute_size_bucket(num_nodes: int) -> str:
    if num_nodes < 200:
        return "small"
    if num_nodes < 1000:
        return "medium"
    return "large"
