from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.train import train_from_config
from src.utils import ensure_dir, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", default="configs/depth_sweep.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = load_config(args.sweep_config)

    depths = [int(d) for d in sweep["depths"]]
    base_configs = sweep["base_configs"]
    output_root = sweep["output_root"]

    results: list[dict] = []

    for base_path in base_configs:
        base_cfg = load_config(base_path)
        model_name = base_cfg["model"]["name"]

        for depth in depths:
            cfg = copy.deepcopy(base_cfg)
            cfg["model"]["num_layers"] = depth
            cfg["experiment"]["name"] = f"{model_name}_L{depth}"
            cfg["experiment"]["output_dir"] = f"{output_root}/{model_name}/L{depth}"

            print(f"\n{'='*60}\n{model_name}  L={depth}\n{'='*60}")
            summary = train_from_config(cfg)

            row: dict = {
                "model": model_name,
                "depth": depth,
                "output_dir": cfg["experiment"]["output_dir"],
                "best_epoch": summary["best_epoch"],
                "best_val_kendall": summary["best_val_kendall"],
            }
            for split in ("val", "test_id", "test_ood"):
                for k, v in summary[split].items():
                    row[f"{split}_{k}"] = v

            results.append(row)
            print(f"  val_kendall={row.get('val_kendall', 0):.4f}  test_id_kendall={row.get('test_id_kendall', 0):.4f}  test_ood_kendall={row.get('test_ood_kendall', 0):.4f}")

    out_dir = ensure_dir(output_root)
    write_json(out_dir / "sweep_index.json", results)
    print(f"\nSweep done. Index saved to {output_root}/sweep_index.json")


if __name__ == "__main__":
    main()
