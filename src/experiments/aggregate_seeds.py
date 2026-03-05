from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.utils import write_json


DUEL_METRICS = ("test_id_kendall", "test_ood_kendall", "test_id_spearman", "test_ood_spearman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", help="Aggregate a single model dir")
    group.add_argument("--output-root", help="Aggregate all model dirs and print duel comparison")
    return parser.parse_args()


def _load_results(results_dir: Path) -> list[dict]:
    index = results_dir / "per_seed_results.json"
    if index.exists():
        with index.open() as f:
            return json.load(f)
    rows: list[dict] = []
    for metrics_path in sorted(results_dir.glob("seed_*/metrics.json")):
        seed = int(metrics_path.parent.name.split("_")[1])
        with metrics_path.open() as f:
            data = json.load(f)
        row: dict = {
            "seed": seed,
            "best_epoch": data.get("best_epoch", 0),
            "best_val_kendall": data.get("best_val_kendall", 0.0),
        }
        for split in ("val", "test_id", "test_ood"):
            for k, v in data.get(split, {}).items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    numeric_keys = [
        k for k in rows[0]
        if k not in {"seed", "output_dir", "model"} and isinstance(rows[0][k], (int, float))
    ]
    aggregated: dict = {}
    for key in numeric_keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        aggregated[key] = {"mean": mean, "std": math.sqrt(variance), "n": n}
    return aggregated


def aggregate_model_dir(results_dir: Path) -> dict:
    rows = _load_results(results_dir)
    if not rows:
        return {}
    aggregated = _aggregate(rows)
    write_json(results_dir / "aggregated_metrics.json", aggregated)
    print(f"Aggregated {len(rows)} seeds → {results_dir}/aggregated_metrics.json")
    for key in DUEL_METRICS:
        if key in aggregated:
            m = aggregated[key]
            print(f"  {key}: {m['mean']:.4f} ± {m['std']:.4f}")
    return aggregated


def _build_comparison(output_root: Path) -> dict:
    comparison: dict = {}
    for model_dir in sorted(output_root.iterdir()):
        if not model_dir.is_dir():
            continue
        agg_path = model_dir / "aggregated_metrics.json"
        if not agg_path.exists():
            rows = _load_results(model_dir)
            if not rows:
                continue
            agg = _aggregate(rows)
            write_json(agg_path, agg)
        else:
            with agg_path.open() as f:
                agg = json.load(f)
        comparison[model_dir.name] = {k: agg[k] for k in DUEL_METRICS if k in agg}
    return comparison


def main() -> None:
    args = parse_args()

    if args.results_dir:
        aggregate_model_dir(Path(args.results_dir))
        return

    output_root = Path(args.output_root)
    comparison = _build_comparison(output_root)
    if not comparison:
        print(f"No aggregated results found under {output_root}")
        return

    write_json(output_root / "duel_comparison.json", comparison)
    print(f"\nDuel comparison ({output_root}/duel_comparison.json):")
    header = f"{'model':<30}" + "".join(f"  {k:<28}" for k in DUEL_METRICS)
    print(header)
    print("-" * len(header))
    for model, metrics in comparison.items():
        row_str = f"{model:<30}"
        for key in DUEL_METRICS:
            if key in metrics:
                m = metrics[key]
                row_str += f"  {m['mean']:.4f}±{m['std']:.4f}{'':>16}"
            else:
                row_str += f"  {'N/A':<28}"
        print(row_str)


if __name__ == "__main__":
    main()
