from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch

from src.models.hyper_connection_gnn import build_permutation_bank


def _mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "n": int(tensor.numel()),
    }


def _split_metric_suffix(key: str) -> tuple[str, str] | None:
    if key.endswith("_node_std_mean"):
        return key[: -len("_node_std_mean")], "node_std_mean"
    if key.endswith("_node_std_std"):
        return key[: -len("_node_std_std")], "node_std_std"
    if key.endswith("_mean"):
        return key[: -len("_mean")], "mean"
    if key.endswith("_std"):
        return key[: -len("_std")], "std"
    return None


def _normalized_row_abs_entropy(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weights = matrix.abs()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
    denom = math.log(matrix.shape[-1]) if matrix.shape[-1] > 1 else 1.0
    return entropy / max(denom, eps)


def _identity_distance(matrix: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    return torch.linalg.norm(matrix - identity, dim=(-2, -1)) / math.sqrt(matrix.shape[-1])


def _row_col_sum_error(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_error = (matrix.sum(dim=-1) - 1.0).abs().mean(dim=-1)
    col_error = (matrix.sum(dim=-2) - 1.0).abs().mean(dim=-1)
    return row_error, col_error


def _nearest_permutation_distance(matrix: torch.Tensor, permutation_bank: torch.Tensor) -> torch.Tensor:
    diffs = matrix.unsqueeze(1) - permutation_bank.unsqueeze(0)
    distances = torch.linalg.norm(diffs, dim=(-2, -1)) / math.sqrt(matrix.shape[-1])
    return distances.min(dim=1).values


def summarize_layer_mapping(layer_mapping: dict[str, torch.Tensor | int]) -> dict[str, float | int]:
    h_res = layer_mapping["h_res"]
    if not isinstance(h_res, torch.Tensor):
        raise ValueError("layer_mapping['h_res'] must be a tensor")

    h_res_cpu = h_res.detach().cpu().float()
    stats: dict[str, float | int] = {
        "layer_index": int(layer_mapping["layer_index"]),
        "n_streams": int(h_res_cpu.shape[-1]),
        "identity_distance_mean": float(_identity_distance(h_res_cpu).mean().item()),
        "identity_distance_std": float(_identity_distance(h_res_cpu).std(unbiased=False).item()),
        "row_abs_entropy_mean": float(_normalized_row_abs_entropy(h_res_cpu).mean().item()),
        "row_abs_entropy_std": float(_normalized_row_abs_entropy(h_res_cpu).std(unbiased=False).item()),
    }

    row_error, col_error = _row_col_sum_error(h_res_cpu)
    stats["row_sum_error_mean"] = float(row_error.mean().item())
    stats["row_sum_error_std"] = float(row_error.std(unbiased=False).item())
    stats["col_sum_error_mean"] = float(col_error.mean().item())
    stats["col_sum_error_std"] = float(col_error.std(unbiased=False).item())

    permutation_bank = layer_mapping.get("permutation_bank")
    if isinstance(permutation_bank, torch.Tensor):
        bank_cpu = permutation_bank.detach().cpu().float()
    else:
        bank_cpu = build_permutation_bank(
            n_streams=int(h_res_cpu.shape[-1]),
            max_permutations=None,
            seed=0,
        ).float()

    perm_dist = _nearest_permutation_distance(h_res_cpu, bank_cpu)
    stats["nearest_permutation_distance_mean"] = float(perm_dist.mean().item())
    stats["nearest_permutation_distance_std"] = float(perm_dist.std(unbiased=False).item())

    weights = layer_mapping.get("weights")
    if isinstance(weights, torch.Tensor):
        weights_cpu = weights.detach().cpu().float()
        max_coeff = weights_cpu.max(dim=-1).values
        coeff_entropy = -(weights_cpu * weights_cpu.clamp_min(1e-8).log()).sum(dim=-1)
        denom = math.log(weights_cpu.shape[-1]) if weights_cpu.shape[-1] > 1 else 1.0
        coeff_entropy = coeff_entropy / max(denom, 1e-8)
        stats["max_permutation_coeff_mean"] = float(max_coeff.mean().item())
        stats["max_permutation_coeff_std"] = float(max_coeff.std(unbiased=False).item())
        stats["permutation_coeff_entropy_mean"] = float(coeff_entropy.mean().item())
        stats["permutation_coeff_entropy_std"] = float(coeff_entropy.std(unbiased=False).item())

    raw_res = layer_mapping.get("raw_res")
    if isinstance(raw_res, torch.Tensor):
        raw_cpu = raw_res.detach().cpu().float()
        raw_perm_dist = _nearest_permutation_distance(raw_cpu, bank_cpu)
        stats["raw_identity_distance_mean"] = float(_identity_distance(raw_cpu).mean().item())
        stats["raw_identity_distance_std"] = float(_identity_distance(raw_cpu).std(unbiased=False).item())
        stats["raw_nearest_permutation_distance_mean"] = float(raw_perm_dist.mean().item())
        stats["raw_nearest_permutation_distance_std"] = float(raw_perm_dist.std(unbiased=False).item())

    return stats


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["regime"]), int(row["layer_index"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (regime, layer_index), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary: dict[str, Any] = {
            "regime": regime,
            "layer_index": layer_index,
            "num_graphs": len(bucket),
        }
        numeric_keys = [
            key
            for key in bucket[0].keys()
            if key not in {"model", "seed", "regime", "graph_index", "graph_label", "layer_index"}
        ]
        for key in numeric_keys:
            values = [float(row[key]) for row in bucket]
            stats = _mean_std(values)
            split = _split_metric_suffix(key)
            if split is None:
                summary[f"{key}_mean"] = stats["mean"]
                summary[f"{key}_std"] = stats["std"]
                continue
            metric_name, stat_kind = split
            if stat_kind == "mean":
                summary[f"{metric_name}_mean"] = stats["mean"]
                summary[f"{metric_name}_std"] = stats["std"]
            elif stat_kind == "std":
                summary[f"{metric_name}_node_std_mean"] = stats["mean"]
                summary[f"{metric_name}_node_std_std"] = stats["std"]
            else:
                raise ValueError(f"Unexpected stat kind in graph aggregation: {stat_kind}")
        summaries.append(summary)
    return summaries


def aggregate_regime_summaries(seed_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in seed_summaries:
        key = (str(row["regime"]), int(row["layer_index"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (regime, layer_index), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary: dict[str, Any] = {
            "regime": regime,
            "layer_index": layer_index,
            "num_seeds": len(bucket),
        }
        numeric_keys = [
            key
            for key in bucket[0].keys()
            if key not in {"model", "seed", "regime", "layer_index", "num_graphs"}
        ]
        for key in numeric_keys:
            stats = _mean_std([float(row[key]) for row in bucket])
            split = _split_metric_suffix(key)
            if split is None:
                summary[f"{key}_mean"] = stats["mean"]
                summary[f"{key}_std"] = stats["std"]
                continue
            metric_name, stat_kind = split
            if stat_kind == "mean":
                summary[f"{metric_name}_mean"] = stats["mean"]
                summary[f"{metric_name}_std"] = stats["std"]
            elif stat_kind == "std":
                summary[f"{metric_name}_seed_std_mean"] = stats["mean"]
                summary[f"{metric_name}_seed_std_std"] = stats["std"]
            elif stat_kind == "node_std_mean":
                summary[f"{metric_name}_node_std_mean"] = stats["mean"]
                summary[f"{metric_name}_node_std_std"] = stats["std"]
            elif stat_kind == "node_std_std":
                summary[f"{metric_name}_node_std_seed_std_mean"] = stats["mean"]
                summary[f"{metric_name}_node_std_seed_std_std"] = stats["std"]
            else:
                raise ValueError(f"Unexpected stat kind in seed aggregation: {stat_kind}")
        summaries.append(summary)
    return summaries


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
