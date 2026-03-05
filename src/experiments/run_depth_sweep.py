from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.train import train_from_config
from src.utils import ensure_dir, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", default="configs/sweeps/depth_sweep.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
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


def _resolve_seeds(sweep: dict, cli_seeds: list[int] | None) -> list[int]:
    if cli_seeds is not None and len(cli_seeds) > 0:
        return [int(s) for s in cli_seeds]
    if "seeds" in sweep:
        return [int(s) for s in sweep["seeds"]]
    return [int(sweep.get("seed", 0))]


def _seed_output_root(base_output_root: str, seed: int, is_multi_seed: bool) -> str:
    if "{seed}" in base_output_root:
        return base_output_root.format(seed=seed)
    if is_multi_seed:
        return f"{base_output_root}/seed_{seed}"
    return base_output_root


def _run_label(cfg: dict) -> str:
    experiment_name = cfg.get("experiment", {}).get("name")
    if experiment_name:
        return str(experiment_name)
    return str(cfg["model"]["name"])


def main() -> None:
    args = parse_args()
    sweep_config_path = Path(args.sweep_config)
    sweep = load_config(sweep_config_path)

    depths = [int(d) for d in sweep["depths"]]
    seeds = _resolve_seeds(sweep, args.seeds)
    output_root = str(sweep["output_root"])
    base_configs = _load_base_configs(sweep, sweep_config_path)
    is_multi_seed = len(seeds) > 1

    labels = [_run_label(cfg) for cfg in base_configs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            "Duplicate run labels detected in sweep config: "
            + ", ".join(duplicates)
            + ". Ensure experiment.name is unique per model variant."
        )

    all_results: list[dict] = []

    for sweep_seed in seeds:
        seed_output_root = _seed_output_root(output_root, sweep_seed, is_multi_seed)
        results: list[dict] = []

        for base_cfg in base_configs:
            model_name = base_cfg["model"]["name"]
            run_label = _run_label(base_cfg)

            for depth in depths:
                cfg = copy.deepcopy(base_cfg)
                cfg["model"]["num_layers"] = depth
                cfg.setdefault("experiment", {})
                cfg["experiment"]["seed"] = sweep_seed
                cfg["experiment"]["name"] = f"{run_label}_L{depth}"
                cfg["experiment"]["output_dir"] = f"{seed_output_root}/{run_label}/L{depth}"

                print(f"\n{'='*60}\nseed={sweep_seed}  {run_label}  L={depth}\n{'='*60}")
                summary = train_from_config(cfg)

                row: dict = {
                    "model": run_label,
                    "model_name": model_name,
                    "depth": depth,
                    "output_dir": cfg["experiment"]["output_dir"],
                    "seed": sweep_seed,
                    "best_epoch": summary["best_epoch"],
                    "best_val_kendall": summary["best_val_kendall"],
                }
                for split in ("val", "test_id", "test_ood"):
                    for k, v in summary[split].items():
                        row[f"{split}_{k}"] = v

                results.append(row)
                all_results.append(row)
                print(
                    f"  val_kendall={row.get('val_kendall', 0):.4f}"
                    f"  test_id_kendall={row.get('test_id_kendall', 0):.4f}"
                    f"  test_ood_kendall={row.get('test_ood_kendall', 0):.4f}"
                )

        out_dir = ensure_dir(seed_output_root)
        write_json(out_dir / "sweep_index.json", results)
        print(f"\nSeed {sweep_seed} done. Index saved to {seed_output_root}/sweep_index.json")

    if is_multi_seed:
        aggregate_root = output_root.replace("{seed}", "all") if "{seed}" in output_root else output_root
        out_dir = ensure_dir(aggregate_root)
        write_json(out_dir / "sweep_index_all_seeds.json", all_results)
        print(f"\nMulti-seed sweep done. Index saved to {aggregate_root}/sweep_index_all_seeds.json")


if __name__ == "__main__":
    main()
