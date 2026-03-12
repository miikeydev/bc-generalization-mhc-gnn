from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from src.analysis.model_budget import benchmark_training_runtime, count_trainable_parameters, write_budget_payload
from src.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/model_budget.yaml")
    return parser.parse_args()


def _resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    relative_to_config = config_path.parent / path
    if relative_to_config.exists():
        return relative_to_config
    return path


def _load_experiment_config(source_dir: Path, benchmark_seed: int | None) -> dict[str, Any]:
    direct_config_path = source_dir / "resolved_config.json"
    if direct_config_path.exists():
        config = load_config(direct_config_path)
    else:
        seed_dir = source_dir / f"seed_{0 if benchmark_seed is None else benchmark_seed}"
        config = load_config(seed_dir / "resolved_config.json")
    if benchmark_seed is not None:
        config = copy.deepcopy(config)
        config["experiment"]["seed"] = int(benchmark_seed)
    return config


def _load_aggregated_metrics(source_dir: Path) -> dict[str, Any]:
    metrics_path = source_dir / "aggregated_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    single_run_metrics_path = source_dir / "metrics.json"
    if single_run_metrics_path.exists():
        return json.loads(single_run_metrics_path.read_text(encoding="utf-8"))
    return {}


def _extract_mean(metrics: dict[str, Any], key: str) -> float | None:
    payload = metrics.get(key)
    if isinstance(payload, dict) and "mean" in payload:
        return float(payload["mean"])
    if isinstance(payload, (int, float)):
        return float(payload)
    return None


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_root = ensure_dir(config["output_root"])
    params_only = bool(config.get("params_only", False))
    rows: list[dict[str, Any]] = []

    for experiment in config["experiments"]:
        name = str(experiment["name"])
        source_dir = _resolve_path(str(experiment["source_dir"]), config_path)
        benchmark_seed = experiment.get("benchmark_seed")
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        resolved_config = _load_experiment_config(source_dir, benchmark_seed if benchmark_seed is not None else 0)
        param_count = count_trainable_parameters(resolved_config)
        print(f"  parameters={param_count}")
        aggregated_metrics = _load_aggregated_metrics(source_dir)

        row = {
            "name": name,
            "source_dir": str(source_dir),
            "benchmark_seed": int(resolved_config["experiment"]["seed"]),
            "parameter_count": param_count,
        }
        if not params_only:
            train_seconds, benchmark_summary = benchmark_training_runtime(resolved_config)
            print(f"  train_seconds={train_seconds:.2f}")
            row.update(
                {
                    "train_seconds": train_seconds,
                    "best_epoch": int(benchmark_summary["best_epoch"]),
                    "best_val_kendall_single_run": float(benchmark_summary["best_val_kendall"]),
                    "test_id_kendall_single_run": float(benchmark_summary["test_id"].get("kendall", 0.0)),
                    "test_ood_kendall_single_run": float(benchmark_summary["test_ood"].get("kendall", 0.0)),
                }
            )
        row.update(
            {
                "test_ood_kendall_mean": _extract_mean(aggregated_metrics, "test_ood_kendall"),
                "test_ood_spearman_mean": _extract_mean(aggregated_metrics, "test_ood_spearman"),
                "test_ood_precision_at_10_mean": _extract_mean(aggregated_metrics, "test_ood_precision_at_10"),
                "test_ood_ndcg_at_10_mean": _extract_mean(aggregated_metrics, "test_ood_ndcg_at_10"),
            }
        )
        rows.append(row)

    write_budget_payload(output_root, rows)
    print(f"\nBudget analysis done. Results under {output_root}/")


if __name__ == "__main__":
    main()
