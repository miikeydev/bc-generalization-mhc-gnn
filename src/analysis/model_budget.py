from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from torch_geometric.loader import DataLoader

from src.data import build_inductive_datasets
from src.models import build_model
from src.train import train_from_config
from src.utils import write_json


def count_trainable_parameters(config: dict[str, Any], seed: int | None = None) -> int:
    resolved_seed = int(config["experiment"]["seed"] if seed is None else seed)
    datasets = build_inductive_datasets(config=config, seed=resolved_seed)
    loader = DataLoader(datasets.train, batch_size=1, shuffle=False)
    first_batch = next(iter(loader))
    input_dim = int(first_batch.x.shape[1])
    model = build_model(config=config, input_dim=input_dim)
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def benchmark_training_runtime(config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    benchmark_config = _prepare_benchmark_config(config)
    start = time.perf_counter()
    summary = train_from_config(benchmark_config)
    duration_seconds = time.perf_counter() - start
    return float(duration_seconds), summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path_obj = Path(path)
    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_budget_payload(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_csv(output_path / "budget_table.csv", rows)
    write_json(output_path / "budget_table.json", {"rows": rows})


def _prepare_benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    benchmark_config = {
        **config,
        "experiment": {
            **config["experiment"],
            "output_dir": str(Path("outputs") / "budget_tmp" / str(config["experiment"]["name"])),
            "save_artifacts": False,
        },
    }
    return benchmark_config
