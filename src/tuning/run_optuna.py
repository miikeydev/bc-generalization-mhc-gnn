from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import optuna
import yaml

from src.utils import ensure_dir, load_config, write_json

from .objective import TuningObjective, sanitize_trial_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hp_search/gcn_proxy.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    return parser.parse_args()


def cli_main() -> None:
    args = parse_args()
    manifest_path = Path(args.config)
    manifest = load_config(manifest_path)
    output_dir = ensure_dir(manifest["study"]["output_dir"])
    objective = TuningObjective(manifest=manifest, manifest_path=manifest_path, output_dir=output_dir)
    study = optuna.create_study(
        study_name=str(manifest["study"]["name"]),
        direction=str(manifest["study"].get("direction", "maximize")),
        sampler=optuna.samplers.TPESampler(seed=int(manifest["study"].get("sampler_seed", 0))),
        pruner=_build_pruner(manifest["study"].get("pruner", {})),
    )
    n_trials = int(args.n_trials) if args.n_trials is not None else int(manifest["study"]["n_trials"])
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    _write_outputs(study=study, manifest=manifest, output_dir=output_dir)
    print(
        {
            "study_name": manifest["study"]["name"],
            "best_value": study.best_value if study.best_trial is not None else None,
            "best_trial": study.best_trial.number if study.best_trial is not None else None,
            "output_dir": str(output_dir),
        }
    )


def _build_pruner(pruner_cfg: dict[str, Any]):
    pruner_name = str(pruner_cfg.get("name", "median")).lower()
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 2)),
        )
    raise ValueError(f"Unsupported pruner name: {pruner_name}")


def _write_outputs(study: optuna.Study, manifest: dict[str, Any], output_dir: Path) -> None:
    trials_payload = [_serialize_trial(trial) for trial in study.trials]
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial if complete_trials else None
    best_params = sanitize_trial_value(best_trial.user_attrs.get("sampled_params", {}) if best_trial is not None else {})
    best_summary = {
        "study_name": manifest["study"]["name"],
        "metric": manifest["objective"].get("metric", "best_val_kendall"),
        "best_value": best_trial.value if best_trial is not None else None,
        "best_trial": best_trial.number if best_trial is not None else None,
        "best_params": best_params,
        "best_runs": sanitize_trial_value(best_trial.user_attrs.get("runs", []) if best_trial is not None else []),
    }
    study_summary = {
        "study_name": manifest["study"]["name"],
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "n_complete": len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]),
        "best_value": best_trial.value if best_trial is not None else None,
        "best_trial": best_trial.number if best_trial is not None else None,
    }
    top_trials = sorted(
        [trial for trial in trials_payload if trial["state"] == "COMPLETE" and trial["value"] is not None],
        key=lambda trial: trial["value"],
        reverse=str(manifest["study"].get("direction", "maximize")).lower() == "maximize",
    )[:5]

    write_json(output_dir / "study_summary.json", study_summary)
    write_json(output_dir / "best_params.json", best_summary)
    write_json(output_dir / "top_trials.json", {"trials": top_trials})
    _write_trials_csv(output_dir / "trials.csv", trials_payload)
    with (output_dir / "best_overrides.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_nest_paths(best_params), handle, sort_keys=False)


def _serialize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": sanitize_trial_value(trial.value),
        "params": sanitize_trial_value(trial.params),
        "sampled_params": sanitize_trial_value(trial.user_attrs.get("sampled_params", {})),
        "runs": sanitize_trial_value(trial.user_attrs.get("runs", trial.user_attrs.get("partial_runs", []))),
    }


def _write_trials_csv(path: Path, trials_payload: list[dict[str, Any]]) -> None:
    rows = []
    param_keys = sorted({key for trial in trials_payload for key in trial["params"].keys()})
    for trial in trials_payload:
        row = {
            "number": trial["number"],
            "state": trial["state"],
            "value": trial["value"],
        }
        for key in param_keys:
            row[key] = trial["params"].get(key)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["number", "state", "value", *param_keys])
        writer.writeheader()
        writer.writerows(rows)


def _nest_paths(flattened: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in flattened.items():
        current = nested
        keys = str(path).split(".")
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    return nested


if __name__ == "__main__":
    cli_main()
