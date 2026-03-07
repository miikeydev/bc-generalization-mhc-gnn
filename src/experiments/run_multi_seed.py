from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.experiments.aggregate_seeds import aggregate_model_dir
from src.train import train_from_config
from src.utils import ensure_dir, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", default="configs/multi_seed/duels/multi_seed_duels.yaml")
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
        if "experiment" not in cfg or "name" not in cfg["experiment"]:
            raise ValueError(f"Missing experiment.name for model_id '{model_id}'")
        configs.append(cfg)
    return configs


def _build_from_experiments(sweep: dict, sweep_config_path: Path) -> list[dict]:
    catalog_path_value = sweep.get("catalog")
    experiments = sweep.get("experiments")
    if catalog_path_value is None or experiments is None:
        return []

    catalog = load_config(_resolve_path(str(catalog_path_value), sweep_config_path))
    common = catalog.get("common", {})
    models = catalog.get("models", {})
    common_overrides = sweep.get("common_overrides", {})

    configs: list[dict] = []
    for experiment in experiments:
        model_id = str(experiment["model_id"])
        if model_id not in models:
            raise ValueError(f"Unknown model_id '{model_id}' in sweep config")
        cfg = _merge_dicts(common, models[model_id])
        cfg = _merge_dicts(cfg, common_overrides)
        cfg = _merge_dicts(cfg, experiment.get("overrides", {}))
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = str(experiment["name"])
        configs.append(cfg)
    return configs


def _build_from_base_configs(sweep: dict, sweep_config_path: Path) -> list[dict]:
    base_paths = sweep.get("base_configs", [])
    return [load_config(_resolve_path(str(path), sweep_config_path)) for path in base_paths]


def _load_base_configs(sweep: dict, sweep_config_path: Path) -> list[dict]:
    from_experiments = _build_from_experiments(sweep, sweep_config_path)
    if from_experiments:
        return from_experiments
    from_catalog = _build_from_catalog(sweep, sweep_config_path)
    if from_catalog:
        return from_catalog
    from_base_configs = _build_from_base_configs(sweep, sweep_config_path)
    if from_base_configs:
        return from_base_configs
    raise ValueError("Sweep config must define either experiments, (catalog + model_ids), or base_configs")


def main() -> None:
    args = parse_args()
    sweep_config_path = Path(args.sweep_config)
    sweep = load_config(sweep_config_path)

    seeds = [int(s) for s in sweep["seeds"]]
    output_root = Path(sweep["output_root"])
    base_configs = _load_base_configs(sweep, sweep_config_path)

    all_results: list[dict] = []

    for base_cfg in base_configs:
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
