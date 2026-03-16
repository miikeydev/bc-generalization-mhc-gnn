from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from src.data.real_graphs import build_real_graph_data
from src.data.protocol import infer_input_dim_from_data_config, make_data_cache_signature
from src.eval import compute_graph_metrics
from src.experiments.aggregate_seeds import aggregate_model_dir
from src.models import build_model
from src.utils import ensure_dir, load_config, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_graph_transfer/gcn_family_cora.yaml")
    return parser.parse_args()


def _resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    relative_to_config = config_path.parent / path
    if relative_to_config.exists():
        return relative_to_config
    return path


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def _load_seed_model(seed_dir: Path) -> tuple[dict, torch.nn.Module, torch.device]:
    resolved_config = load_config(seed_dir / "resolved_config.json")
    input_dim = infer_input_dim_from_data_config(resolved_config.get("data", {}))
    model = build_model(config=resolved_config, input_dim=input_dim)
    state_dict = torch.load(seed_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = _resolve_device(resolved_config.get("device", "auto"))
    model = model.to(device)
    model.eval()
    return resolved_config, model, device


def _evaluate_seed(
    seed: int,
    dataset_name: str,
    dataset_root: str,
    seed_dir: Path,
    topk_values: list[int],
    topk_ratios: list[float],
    data_cache: dict[tuple, object],
) -> dict:
    resolved_config, model, device = _load_seed_model(seed_dir)
    set_global_seed(seed)
    data_cfg = resolved_config.get("data", {})
    cache_key = (
        dataset_name.lower(),
        dataset_root,
        seed,
        *make_data_cache_signature(data_cfg),
    )
    if cache_key not in data_cache:
        data_cache[cache_key] = build_real_graph_data(
            dataset_name=dataset_name,
            feature_mode=str(data_cfg.get("feature_mode", "structural_only")),
            lap_pe_dim=int(data_cfg.get("lap_pe_dim", 8)),
            random_feature_dim=int(data_cfg.get("random_feature_dim", 16)),
            rng_seed=seed,
            root=dataset_root,
            feature_config=data_cfg.get("feature_config", {}),
            bc_backend=data_cfg.get("bc_backend", "networkit"),
        )
    real_data = data_cache[cache_key].clone()

    with torch.no_grad():
        predictions = model(real_data.x.to(device), real_data.edge_index.to(device)).detach().cpu().numpy()

    metrics = compute_graph_metrics(
        y_pred=predictions,
        y_true=real_data.y_raw.numpy(),
        topk_values=topk_values,
        topk_ratios=topk_ratios,
    )

    return {
        "seed": seed,
        "dataset": dataset_name.lower(),
        "num_nodes": int(real_data.num_nodes),
        "num_edges_undirected": int(real_data.num_edges_undirected),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    dataset_name = str(config["dataset"]["name"])
    dataset_root = str(config["dataset"].get("root", "data/real_graphs"))
    topk_values = [int(v) for v in config["evaluation"]["topk_values"]]
    topk_ratios = [float(v) for v in config["evaluation"]["topk_ratios"]]
    output_root = ensure_dir(config["output_root"])
    all_results: list[dict] = []
    data_cache: dict[tuple, object] = {}

    for experiment in config["experiments"]:
        name = str(experiment["name"])
        source_dir = _resolve_path(str(experiment["source_dir"]), config_path)
        seeds = [int(v) for v in experiment.get("seeds", config.get("seeds", []))]
        model_dir = ensure_dir(output_root / name)
        per_seed_results: list[dict] = []

        for seed in seeds:
            seed_dir = source_dir / f"seed_{seed}"
            print(f"\n{'='*60}\n{name}  seed={seed}\n{'='*60}")
            row = _evaluate_seed(
                seed=seed,
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                seed_dir=seed_dir,
                topk_values=topk_values,
                topk_ratios=topk_ratios,
                data_cache=data_cache,
            )
            row["model"] = name
            row["source_dir"] = str(seed_dir)
            row["output_dir"] = str(model_dir / f"seed_{seed}")
            per_seed_results.append(row)
            all_results.append(copy.deepcopy(row))
            ensure_dir(model_dir / f"seed_{seed}")
            write_json(model_dir / f"seed_{seed}" / "metrics.json", row)
            print(
                f"  kendall={row.get('kendall', 0):.4f}"
                f"  spearman={row.get('spearman', 0):.4f}"
                f"  precision_at_10={row.get('precision_at_10', 0):.4f}"
            )

        write_json(model_dir / "per_seed_results.json", per_seed_results)
        aggregate_model_dir(model_dir)

    write_json(output_root / "all_results.json", all_results)
    print(f"\nReal-graph transfer done. Results under {output_root}/")


if __name__ == "__main__":
    main()
