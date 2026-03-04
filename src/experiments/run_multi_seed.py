from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.experiments.aggregate_seeds import aggregate_model_dir
from src.train import train_from_config
from src.utils import ensure_dir, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", default="configs/multi_seed_duels.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = load_config(args.sweep_config)

    seeds = [int(s) for s in sweep["seeds"]]
    base_configs = sweep["base_configs"]
    output_root = Path(sweep["output_root"])

    all_results: list[dict] = []

    for base_path in base_configs:
        base_cfg = load_config(base_path)
        model_name = base_cfg["experiment"]["name"]
        model_dir = output_root / model_name
        per_seed_results: list[dict] = []

        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["experiment"]["seed"] = seed
            cfg["experiment"]["output_dir"] = str(model_dir / f"seed_{seed}")

            print(f"\n{'='*60}\n{model_name}  seed={seed}\n{'='*60}")
            summary = train_from_config(cfg)

            row: dict = {
                "model": model_name,
                "seed": seed,
                "output_dir": cfg["experiment"]["output_dir"],
                "best_epoch": summary["best_epoch"],
                "best_val_kendall": summary["best_val_kendall"],
            }
            for split in ("val", "test_id", "test_ood"):
                for k, v in summary[split].items():
                    row[f"{split}_{k}"] = v

            per_seed_results.append(row)
            print(
                f"  val_kendall={row.get('val_kendall', 0):.4f}"
                f"  test_id_kendall={row.get('test_id_kendall', 0):.4f}"
                f"  test_ood_kendall={row.get('test_ood_kendall', 0):.4f}"
            )

        ensure_dir(model_dir)
        write_json(model_dir / "per_seed_results.json", per_seed_results)
        aggregate_model_dir(model_dir)
        all_results.extend(per_seed_results)

    write_json(ensure_dir(output_root) / "all_results.json", all_results)
    print(f"\nSweep done. All results under {output_root}/")


if __name__ == "__main__":
    main()
