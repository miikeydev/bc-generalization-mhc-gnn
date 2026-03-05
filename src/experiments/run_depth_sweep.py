from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.train import train_from_config
from src.utils import ensure_dir, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", default="configs/sweeps/depth_sweep.yaml")
    return parser.parse_args()


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_path(path_value: str, sweep_config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    relative_to_sweep = sweep_config_path.parent / path
    if relative_to_sweep.exists():
        return relative_to_sweep
    return path


def _build_from_catalog(sweep: dict, sweep_config_path: Path) -> list[dict]:
    catalog_path_value = sweep.get("catalog")
    model_ids = sweep.get("model_ids")
    if catalog_path_value is None or model_ids is None:
        return []

    catalog = load_config(_resolve_path(str(catalog_path_value), sweep_config_path))
    common = catalog.get("common", {})
    models = catalog.get("models", {})
    common_overrides = sweep.get("common_overrides", {})
    overrides_by_model = sweep.get("overrides_by_model", {})

    configs: list[dict] = []
    for model_id in model_ids:
        if model_id not in models:
            raise ValueError(f"Unknown model_id '{model_id}' in sweep config")
        cfg = _merge_dicts(common, models[model_id])
        cfg = _merge_dicts(cfg, common_overrides)
        model_overrides = overrides_by_model.get(model_id, {})
        cfg = _merge_dicts(cfg, model_overrides)
        if "model" not in cfg or "name" not in cfg["model"]:
            raise ValueError(f"Missing model.name for model_id '{model_id}'")
        if "experiment" not in cfg or "name" not in cfg["experiment"]:
            raise ValueError(f"Missing experiment.name for model_id '{model_id}'")
        configs.append(cfg)
    return configs


def _build_from_base_configs(sweep: dict, sweep_config_path: Path) -> list[dict]:
    base_paths = sweep.get("base_configs", [])
    return [load_config(_resolve_path(str(path), sweep_config_path)) for path in base_paths]


def _load_base_configs(sweep: dict, sweep_config_path: Path) -> list[dict]:
    from_catalog = _build_from_catalog(sweep, sweep_config_path)
    if from_catalog:
        return from_catalog
    from_base = _build_from_base_configs(sweep, sweep_config_path)
    if from_base:
        return from_base
    raise ValueError("Sweep config must define either (catalog + model_ids) or base_configs")


def main() -> None:
    args = parse_args()
    sweep_config_path = Path(args.sweep_config)
    sweep = load_config(sweep_config_path)

    depths = [int(d) for d in sweep["depths"]]
    output_root = sweep["output_root"]
    base_configs = _load_base_configs(sweep, sweep_config_path)

    results: list[dict] = []

    for base_cfg in base_configs:
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
