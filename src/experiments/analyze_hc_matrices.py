from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch

from src.analysis import aggregate_regime_summaries, aggregate_seed_rows, summarize_layer_mapping, write_csv
from src.data import build_inductive_datasets, build_real_graph_data
from src.models import build_model
from src.utils import ensure_dir, load_config, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/hc_matrices_gcn.yaml")
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


def _load_seed_model(seed_dir: Path) -> tuple[dict[str, Any], torch.nn.Module, torch.device]:
    resolved_config = load_config(seed_dir / "resolved_config.json")
    
    input_dim = _infer_input_dim_from_config(resolved_config)
    
    model = build_model(config=resolved_config, input_dim=input_dim)
    state_dict = torch.load(seed_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = _resolve_device(resolved_config.get("device", "auto"))
    model = model.to(device)
    model.eval()
    return resolved_config, model, device


def _infer_input_dim_from_config(config: dict) -> int:
    data_cfg = config.get("data", {})
    feature_mode = str(data_cfg.get("feature_mode", "structural_only"))
    
    if feature_mode in {"degree_only", "degree"}:
        return 2
    
    if feature_mode in {"degree_plus_rwpe", "degree_rwpe"}:
        rwpe_dim = data_cfg.get("feature_config", {}).get("rwpe_dim", 8)
        return 2 + rwpe_dim
    
    if feature_mode in {"degree_plus_ppr", "degree_ppr"}:
        ppr_dim = data_cfg.get("feature_config", {}).get("ppr_dim", 8)
        return 2 + ppr_dim
    
    if feature_mode == "structural_only":
        lap_pe_dim = int(data_cfg.get("lap_pe_dim", 8))
        return 2 + lap_pe_dim
    
    if feature_mode in {"random", "gaussian"}:
        return int(data_cfg.get("random_feature_dim", 16))
    
    if feature_mode in {"none", "constant", "ones"}:
        return 1
    
    raise ValueError(f"Cannot infer input_dim for feature_mode={feature_mode}")


def _select_synthetic_graphs(bundle: Any, max_id_graphs: int, max_ood_graphs: int) -> list[tuple[str, int, Any, str]]:
    selected: list[tuple[str, int, Any, str]] = []
    for idx in range(min(max_id_graphs, len(bundle.test_id))):
        data = bundle.test_id[idx]
        selected.append(("synthetic_id", idx, data, str(getattr(data, "family", "unknown"))))
    for idx in range(min(max_ood_graphs, len(bundle.test_ood))):
        data = bundle.test_ood[idx]
        selected.append(("synthetic_ood", idx, data, str(getattr(data, "family", "unknown"))))
    return selected


def _evaluate_graph(model: torch.nn.Module, device: torch.device, graph_data: Any) -> list[dict[str, Any]]:
    with torch.no_grad():
        _, mappings = model(
            graph_data.x.to(device),
            graph_data.edge_index.to(device),
            return_mappings=True,
        )
    return [summarize_layer_mapping(layer_mapping) for layer_mapping in mappings]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    output_root = ensure_dir(config["output_root"])
    synthetic_cfg = config.get("synthetic", {})
    real_cfg = config.get("real", {})
    synthetic_enabled = bool(synthetic_cfg.get("enabled", True))
    max_id_graphs = int(synthetic_cfg.get("max_id_graphs", 3))
    max_ood_graphs = int(synthetic_cfg.get("max_ood_graphs", 3))
    real_datasets = real_cfg.get("datasets", [])

    synthetic_cache: dict[tuple[str, int], Any] = {}
    real_cache: dict[tuple[str, str, str, int, int, int], Any] = {}
    all_experiment_summaries: list[dict[str, Any]] = []

    for experiment in config["experiments"]:
        name = str(experiment["name"])
        source_dir = _resolve_path(str(experiment["source_dir"]), config_path)
        seeds = [int(v) for v in experiment.get("seeds", config.get("seeds", []))]
        model_dir = ensure_dir(output_root / name)
        graph_rows: list[dict[str, Any]] = []
        seed_rows: list[dict[str, Any]] = []

        for seed in seeds:
            seed_dir = source_dir / f"seed_{seed}"
            print(f"\n{'='*60}\n{name}  seed={seed}\n{'='*60}")
            resolved_config, model, device = _load_seed_model(seed_dir)
            set_global_seed(seed)

            if synthetic_enabled:
                synth_key = (str(source_dir), seed)
                if synth_key not in synthetic_cache:
                    synthetic_cache[synth_key] = build_inductive_datasets(resolved_config, seed)
                bundle = synthetic_cache[synth_key]
                for regime, graph_index, graph_data, graph_label in _select_synthetic_graphs(
                    bundle,
                    max_id_graphs=max_id_graphs,
                    max_ood_graphs=max_ood_graphs,
                ):
                    layer_summaries = _evaluate_graph(model=model, device=device, graph_data=graph_data)
                    for summary in layer_summaries:
                        row = {
                            "model": name,
                            "seed": seed,
                            "regime": regime,
                            "graph_index": graph_index,
                            "graph_label": graph_label,
                            **summary,
                        }
                        graph_rows.append(row)

            for dataset_cfg in real_datasets:
                dataset_name = str(dataset_cfg["name"])
                dataset_root = str(dataset_cfg.get("root", "data/real_graphs"))
                data_cfg = resolved_config.get("data", {})
                real_key = (
                    dataset_name.lower(),
                    dataset_root,
                    str(data_cfg.get("feature_mode", "structural_only")),
                    int(data_cfg.get("lap_pe_dim", 8)),
                    int(data_cfg.get("random_feature_dim", 16)),
                    str(data_cfg.get("bc_backend", "networkx")),
                    seed,
                )
                if real_key not in real_cache:
                    real_cache[real_key] = build_real_graph_data(
                        dataset_name=dataset_name,
                        feature_mode=str(data_cfg.get("feature_mode", "structural_only")),
                        lap_pe_dim=int(data_cfg.get("lap_pe_dim", 8)),
                        random_feature_dim=int(data_cfg.get("random_feature_dim", 16)),
                        rng_seed=seed,
                        root=dataset_root,
                        feature_config=data_cfg.get("feature_config", {}),
                        bc_backend=data_cfg.get("bc_backend", "networkx"),
                        bc_mode=data_cfg.get("bc_mode", "exact"),
                    )
                graph_data = real_cache[real_key].clone()
                layer_summaries = _evaluate_graph(model=model, device=device, graph_data=graph_data)
                for summary in layer_summaries:
                    row = {
                        "model": name,
                        "seed": seed,
                        "regime": dataset_name.lower(),
                        "graph_index": 0,
                        "graph_label": dataset_name.lower(),
                        **summary,
                    }
                    graph_rows.append(row)

            current_seed_rows = aggregate_seed_rows([row for row in graph_rows if int(row["seed"]) == seed])
            for row in current_seed_rows:
                row["model"] = name
                row["seed"] = seed
                seed_rows.append(row)
            print(f"  collected {len(current_seed_rows)} layer summaries for seed {seed}")

        regime_rows = aggregate_regime_summaries(seed_rows)
        for row in regime_rows:
            row["model"] = name
        write_csv(model_dir / "graph_stats.csv", graph_rows)
        write_csv(model_dir / "seed_stats.csv", seed_rows)
        write_csv(model_dir / "regime_stats.csv", regime_rows)
        write_json(model_dir / "graph_stats.json", {"rows": graph_rows})
        write_json(model_dir / "seed_stats.json", {"rows": seed_rows})
        write_json(model_dir / "regime_stats.json", {"rows": regime_rows})
        all_experiment_summaries.extend(copy.deepcopy(regime_rows))

    write_csv(output_root / "all_regime_stats.csv", all_experiment_summaries)
    write_json(output_root / "all_regime_stats.json", {"rows": all_experiment_summaries})
    print(f"\nHC matrix analysis done. Results under {output_root}/")


if __name__ == "__main__":
    main()
