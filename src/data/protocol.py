from __future__ import annotations

import json
from typing import Any


def normalize_feature_mode(mode: str) -> str:
    mode_lower = str(mode).lower().strip()
    if mode_lower == "structural_only":
        return "structural_only"
    if mode_lower in {
        "degree_random_walk_ppr",
        "degree_rw_ppr",
        "degree_random_walk_ppr_combined",
        "combined",
        "combined_structural",
    }:
        return "degree_random_walk_ppr"
    if mode_lower in {"degree_only", "degree"}:
        return "degree"
    if mode_lower in {"degree_plus_rwpe", "degree_rwpe", "random_walk", "rwpe"}:
        return "random_walk"
    if mode_lower in {"degree_plus_ppr", "degree_ppr", "ppr"}:
        return "ppr"
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
    cfg["train_num_nodes"] = normalize_int_sampling_spec(cfg.get("train_num_nodes", 100), default=100, minimum=8)
    cfg["test_num_nodes"] = normalize_int_sampling_spec(cfg.get("test_num_nodes", 500), default=500, minimum=8)
    cfg["average_degree"] = normalize_int_sampling_spec(cfg.get("average_degree", 8), default=8, minimum=1)
    cfg["train_average_degree"] = normalize_int_sampling_spec(
        cfg.get("train_average_degree", cfg["average_degree"]),
        default=8,
        minimum=1,
    )
    cfg["test_average_degree"] = normalize_int_sampling_spec(
        cfg.get("test_average_degree", cfg["average_degree"]),
        default=8,
        minimum=1,
    )
    cfg["feature_mode"] = normalize_feature_mode(str(cfg.get("feature_mode", "structural_only")))
    cfg["lap_pe_dim"] = int(cfg.get("lap_pe_dim", 8))
    cfg["random_feature_dim"] = int(cfg.get("random_feature_dim", 16))
    cfg["bc_backend"] = str(cfg.get("bc_backend", "networkit")).lower()
    cfg["bc_mode"] = "exact"
    cfg["train_sign_flip_lappe"] = bool(cfg.get("train_sign_flip_lappe", False))
    cfg["feature_config"] = dict(cfg.get("feature_config", {}))

    if "bc_mode" in data_cfg and str(data_cfg["bc_mode"]).lower() != "exact":
        raise ValueError("Only exact betweenness centrality is supported")
    if "bc_approximation_k" in data_cfg and data_cfg["bc_approximation_k"] is not None:
        raise ValueError("bc_approximation_k is not supported: betweenness is exact-only")
    if cfg["bc_backend"] != "networkit":
        raise ValueError(f"Unsupported bc_backend: {cfg['bc_backend']}")

    return cfg


def normalize_int_sampling_spec(value: Any, default: int, minimum: int) -> dict[str, Any]:
    raw_value = default if value is None else value

    if isinstance(raw_value, bool):
        raw_value = int(raw_value)

    if isinstance(raw_value, int):
        return {"kind": "constant", "value": max(minimum, int(raw_value))}

    if isinstance(raw_value, (list, tuple)):
        choices = [max(minimum, int(item)) for item in raw_value]
        if not choices:
            raise ValueError("Sampling choices cannot be empty")
        return {"kind": "choice", "values": choices}

    if not isinstance(raw_value, dict):
        return {"kind": "constant", "value": max(minimum, int(raw_value))}

    if "choices" in raw_value:
        choices = [max(minimum, int(item)) for item in raw_value.get("choices", [])]
        if not choices:
            raise ValueError("Sampling choices cannot be empty")
        return {"kind": "choice", "values": choices}

    lower = raw_value.get("min", raw_value.get("low", raw_value.get("start")))
    upper = raw_value.get("max", raw_value.get("high", raw_value.get("stop")))

    if lower is not None or upper is not None:
        lower_int = max(minimum, int(lower if lower is not None else default))
        upper_int = max(lower_int, int(upper if upper is not None else lower_int))
        step = max(1, int(raw_value.get("step", 1)))
        mode = str(raw_value.get("mode", "uniform")).lower()
        if mode not in {"uniform", "linspace"}:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        return {
            "kind": "range",
            "min": lower_int,
            "max": upper_int,
            "step": step,
            "mode": mode,
        }

    if "value" in raw_value:
        return {"kind": "constant", "value": max(minimum, int(raw_value["value"]))}

    raise ValueError(f"Unsupported sampling spec: {raw_value}")


def sample_int_from_spec(spec: dict[str, Any], rng: Any) -> int:
    kind = spec["kind"]

    if kind == "constant":
        return int(spec["value"])

    if kind == "choice":
        values = spec["values"]
        return int(values[int(rng.integers(0, len(values)))])

    if kind == "range":
        low = int(spec["min"])
        high = int(spec["max"])
        step = int(spec.get("step", 1))
        if high <= low:
            return low
        count = ((high - low) // step) + 1
        return int(low + step * int(rng.integers(0, count)))

    raise ValueError(f"Unsupported sampling spec kind: {kind}")


def infer_input_dim_from_data_config(data_cfg: dict[str, Any]) -> int:
    normalized = normalize_data_config(data_cfg)
    feature_mode = normalized["feature_mode"]
    feature_config = normalized["feature_config"]

    if feature_mode == "degree":
        return 2
    if feature_mode == "degree_random_walk_ppr":
        return 2 + int(feature_config.get("rwpe_dim", 8)) + int(feature_config.get("ppr_dim", 8))
    if feature_mode == "random_walk":
        return int(feature_config.get("rwpe_dim", 8))
    if feature_mode == "ppr":
        return int(feature_config.get("ppr_dim", 8))
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
        json.dumps(normalized["feature_config"], sort_keys=True, separators=(",", ":")),
    )


def compute_size_bucket(num_nodes: int) -> str:
    if num_nodes < 200:
        return "small"
    if num_nodes < 1000:
        return "medium"
    return "large"
