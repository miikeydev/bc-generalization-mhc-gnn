from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-index", default="outputs/depth_sweep/sweep_index.json")
    parser.add_argument("--output-csv", default="outputs/depth_sweep/depth_results.csv")
    return parser.parse_args()


def load_from_index(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def load_from_outputs(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        parts = metrics_path.parts
        try:
            model = parts[-3]
            depth = int(parts[-2].lstrip("L"))
        except (IndexError, ValueError):
            continue
        with metrics_path.open() as f:
            data = json.load(f)
        row: dict = {"model": model, "depth": depth, "best_epoch": data.get("best_epoch", 0), "best_val_kendall": data.get("best_val_kendall", 0.0)}
        for split in ("val", "test_id", "test_ood"):
            for k, v in data.get(split, {}).items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    index_path = Path(args.sweep_index)
    csv_path = Path(args.output_csv)

    if index_path.exists():
        rows = load_from_index(index_path)
    else:
        print(f"{index_path} not found, scanning {index_path.parent}/")
        rows = load_from_outputs(index_path.parent)

    if not rows:
        print("No results found. Run: python -m src.experiments.run_depth_sweep first.")
        return

    all_keys: dict[str, None] = {}
    for row in rows:
        all_keys.update(dict.fromkeys(row.keys()))
    fieldnames = ["model", "depth"] + [k for k in all_keys if k not in {"model", "depth"}]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.get("model", ""), r.get("depth", 0))):
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Saved {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
