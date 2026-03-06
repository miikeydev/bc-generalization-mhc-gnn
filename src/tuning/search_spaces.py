from __future__ import annotations

from typing import Any

import optuna


def sample_search_space(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for path, spec in search_space.items():
        sampled[path] = _sample_value(trial, path, spec)
    return sampled


def _sample_value(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    param_type = str(spec["type"]).lower()
    if param_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if param_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if param_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=float(spec["step"]) if spec.get("step") is not None else None,
            log=bool(spec.get("log", False)),
        )
    if param_type == "float_log":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if param_type == "int_log":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=True)
    raise ValueError(f"Unsupported search space type: {param_type}")
