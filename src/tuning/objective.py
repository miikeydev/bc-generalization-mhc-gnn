from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from src.train import train_from_config
from src.utils import ensure_dir, load_config, write_json

from .search_spaces import sample_search_space


@dataclass
class TuningObjective:
    manifest: dict[str, Any]
    manifest_path: Path
    output_dir: Path

    def __post_init__(self) -> None:
        self.catalog = load_config(self._resolve_path(str(self.manifest["catalog"])))
        self.study_cfg = self.manifest["study"]
        self.objective_cfg = self.manifest["objective"]
        self.proxy_overrides = self.manifest.get("proxy_overrides", {})
        self.search_space = self.manifest["search_space"]
        self.metric_name = str(self.objective_cfg.get("metric", "best_val_kendall"))
        self.model_ids = [str(model_id) for model_id in self.objective_cfg["model_ids"]]
        self.depths = [int(depth) for depth in self.objective_cfg["depths"]]
        self.trial_seed_base = int(self.study_cfg.get("trial_seed_base", 1000))
        self.save_trial_artifacts = bool(self.objective_cfg.get("save_trial_artifacts", False))
        self.trials_dir = ensure_dir(self.output_dir / "trials")

    def __call__(self, trial: optuna.Trial) -> float:
        sampled_params = sample_search_space(trial, self.search_space)
        trial_dir = self.trials_dir / f"trial_{trial.number:04d}"
        if self.save_trial_artifacts:
            ensure_dir(trial_dir)
        records: list[dict[str, Any]] = []
        scores: list[float] = []
        step = 0

        for model_id in self.model_ids:
            for depth in self.depths:
                cfg = self._build_trial_config(
                    model_id=model_id,
                    depth=depth,
                    trial_number=trial.number,
                    trial_dir=trial_dir,
                    sampled_params=sampled_params,
                )
                summary = train_from_config(cfg)
                metric_value = float(summary[self.metric_name])
                scores.append(metric_value)
                record = {
                    "model_id": model_id,
                    "depth": depth,
                    "best_epoch": int(summary["best_epoch"]),
                    "best_val_kendall": float(summary["best_val_kendall"]),
                    "test_id_kendall": float(summary["test_id"].get("kendall", 0.0)),
                    "test_ood_kendall": float(summary["test_ood"].get("kendall", 0.0)),
                }
                records.append(record)
                trial.report(float(sum(scores) / len(scores)), step=step)
                if trial.should_prune():
                    trial.set_user_attr("sampled_params", sampled_params)
                    trial.set_user_attr("partial_runs", records)
                    if self.save_trial_artifacts:
                        write_json(trial_dir / "partial_summary.json", {"score": float(sum(scores) / len(scores)), "runs": records})
                    raise optuna.TrialPruned()
                step += 1

        objective_value = float(sum(scores) / len(scores)) if scores else float("-inf")
        summary_payload = {
            "objective": objective_value,
            "metric": self.metric_name,
            "sampled_params": sampled_params,
            "runs": records,
        }
        trial.set_user_attr("sampled_params", sampled_params)
        trial.set_user_attr("runs", records)
        trial.set_user_attr("objective", objective_value)
        if self.save_trial_artifacts:
            write_json(trial_dir / "summary.json", summary_payload)
        return objective_value

    def _build_trial_config(
        self,
        model_id: str,
        depth: int,
        trial_number: int,
        trial_dir: Path,
        sampled_params: dict[str, Any],
    ) -> dict[str, Any]:
        base_cfg = self._build_config_from_catalog(model_id)
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["seed"] = self.trial_seed_base + trial_number
        cfg["experiment"]["save_artifacts"] = self.save_trial_artifacts
        cfg["experiment"]["output_dir"] = str(trial_dir / model_id / f"L{depth}")
        cfg["model"]["num_layers"] = depth
        self._deep_update(cfg, self.proxy_overrides)
        for path, value in sampled_params.items():
            self._set_by_path(cfg, path, value)
        return cfg

    def _build_config_from_catalog(self, model_id: str) -> dict[str, Any]:
        try:
            model_cfg = self.catalog["models"][model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model_id '{model_id}' in tuning manifest") from exc
        cfg = copy.deepcopy(self.catalog["common"])
        self._deep_update(cfg, model_cfg)
        return cfg

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        relative_to_manifest = self.manifest_path.parent / path
        if relative_to_manifest.exists():
            return relative_to_manifest
        return path

    def _deep_update(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_update(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def _set_by_path(self, payload: dict[str, Any], path: str, value: Any) -> None:
        keys = path.split(".")
        current = payload
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


def sanitize_trial_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, list):
        return [sanitize_trial_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_trial_value(val) for key, val in value.items()}
    return value
