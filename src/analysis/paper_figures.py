from __future__ import annotations

import json
import numbers
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from src.utils import ensure_dir, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["gcn", "gcn_residual", "hc_gcn", "mhc_gcn", "mhc_lite_gcn"]
MODEL_LABELS = {
    "gcn": "GCN",
    "gcn_residual": "GCN+Res",
    "hc_gcn": "HC-GCN",
    "mhc_gcn": "mHC-GCN",
    "mhc_lite_gcn": "mHC-lite-GCN",
}
MODEL_COLORS = {
    "gcn": "#6B7280",
    "gcn_residual": "#C96A3D",
    "hc_gcn": "#0F2F5A",
    "mhc_gcn": "#2B6FCF",
    "mhc_lite_gcn": "#9DCCF6",
}
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "actor": "Actor",
    "wisconsin": "Wisconsin",
    "amazoncomputers": "Amazon Computers",
    "coauthorcs": "Coauthor CS",
    "coauthorphysics": "Coauthor Physics",
}
SPLIT_LABELS = {
    "test_id": "IID",
    "test_ood": "OOD",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#C7CED8",
            "axes.labelcolor": "#122240",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#445063",
            "ytick.color": "#445063",
            "grid.color": "#D8DEE7",
            "grid.linestyle": "-",
            "grid.alpha": 0.65,
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_shared_legend(
    fig: plt.Figure,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int,
    bbox_to_anchor: tuple[float, float] = (0.5, 1.03),
) -> None:
    deduped: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        deduped.setdefault(label, handle)
    fig.legend(
        list(deduped.values()),
        list(deduped.keys()),
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        columnspacing=1.4,
        handlelength=2.4,
    )


def finalize_figure(
    fig: plt.Figure,
    *,
    title: str,
    title_y: float = 0.985,
    legend_handles: list[Any] | None = None,
    legend_labels: list[str] | None = None,
    legend_ncol: int = 1,
    legend_y: float = 0.855,
    layout_top: float = 0.82,
    layout_left: float = 0.055,
    layout_bottom: float = 0.04,
) -> None:
    fig.suptitle(title, y=title_y, fontsize=13, fontweight="bold")
    if legend_handles and legend_labels:
        add_shared_legend(fig, legend_handles, legend_labels, ncol=legend_ncol, bbox_to_anchor=(0.5, legend_y))
    fig.tight_layout(rect=[layout_left, layout_bottom, 1, layout_top])


def save_figure(fig: plt.Figure, output_base: Path) -> list[str]:
    saved_paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        saved_paths.append(str(path))
    plt.close(fig)
    return saved_paths


def write_table_bundle(df: pd.DataFrame, output_base: Path, float_digits: int = 4) -> list[str]:
    saved_paths: list[str] = []
    csv_path = output_base.with_suffix(".csv")
    tex_path = output_base.with_suffix(".tex")
    json_path = output_base.with_suffix(".json")

    df.to_csv(csv_path, index=False)
    saved_paths.append(str(csv_path))

    latex_df = df.copy()
    latex_df.to_latex(
        tex_path,
        index=False,
        escape=False,
        float_format=lambda value: (
            f"{float(value):.{float_digits}f}"
            if isinstance(value, numbers.Real) and not isinstance(value, bool)
            else str(value)
        ),
    )
    saved_paths.append(str(tex_path))

    write_json(json_path, {"rows": df.to_dict(orient="records")})
    saved_paths.append(str(json_path))
    return saved_paths


def _resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _extract_depth(run_name: str) -> tuple[str, int] | None:
    match = re.match(r"(.+)_l(\d+)$", run_name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_legacy_aggregate_root(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agg_path in sorted(root.glob("*/aggregated_metrics.json")):
        parsed = _extract_depth(agg_path.parent.name)
        if parsed is None:
            continue
        model, depth = parsed
        payload = _load_json(agg_path)
        row: dict[str, Any] = {"run_name": agg_path.parent.name, "model": model, "depth": depth}
        for key, value in payload.items():
            if isinstance(value, dict) and {"mean", "std", "n"} <= set(value.keys()):
                row[f"{key}_mean"] = float(value["mean"])
                row[f"{key}_std"] = float(value["std"])
                row[f"{key}_n"] = int(value["n"])
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_model_aggregate_root(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agg_path in sorted(root.glob("*/aggregated_metrics.json")):
        model = agg_path.parent.name
        payload = _load_json(agg_path)
        row: dict[str, Any] = {"run_name": model, "model": model}
        for key, value in payload.items():
            if isinstance(value, dict) and {"mean", "std", "n"} <= set(value.keys()):
                row[f"{key}_mean"] = float(value["mean"])
                row[f"{key}_std"] = float(value["std"])
                row[f"{key}_n"] = int(value["n"])
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_size_generalization_eval_details(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed_path in sorted(root.glob("*/seed_*/eval_details.json")):
        model = seed_path.parent.parent.name
        seed = int(seed_path.parent.name.split("_")[-1])
        payload = _load_json(seed_path)
        for split, split_rows in payload.items():
            for row in split_rows:
                rows.append({"model": model, "seed": seed, "split": split, **row})
    return pd.DataFrame(rows)


def load_real_transfer_results(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for all_results_path in sorted(root.glob("*/all_results.json")):
        payload = _load_json(all_results_path)
        for row in payload:
            rows.append(dict(row))
    return pd.DataFrame(rows)


def load_matrix_regime_stats(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stats_path in sorted(root.glob("*/regime_stats.json")):
        model = stats_path.parent.name
        payload = _load_json(stats_path)
        for row in payload["rows"]:
            rows.append({"model": model, **row})
    return pd.DataFrame(rows)


def build_backbone_funnel_table(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    rows = config["tables"]["backbone_funnel"]["rows"]
    df = pd.DataFrame(rows)
    return write_table_bundle(df, output_dir / "table_1_backbone_funnel", float_digits=4)


def plot_controlled_anchor(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    figure_cfg = config["figures"]["controlled_anchor"]
    root = _resolve_path(figure_cfg["root"], config_path)
    metric = figure_cfg.get("metric", "test_ood_kendall")
    depths = [int(value) for value in figure_cfg.get("depths", [4, 8])]
    models = figure_cfg.get("models", MODEL_ORDER)

    df = load_legacy_aggregate_root(root)
    df = df[df["model"].isin(models) & df["depth"].isin(depths)].copy()
    df["label"] = df["model"].map(MODEL_LABELS)

    configure_plot_style()
    fig, axes = plt.subplots(1, len(depths), figsize=(10.4, 4.8), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    y_positions = np.arange(len(models))
    max_value = float(df[f"{metric}_mean"].max() + df[f"{metric}_std"].max() + 0.07)
    for ax, depth in zip(axes, depths):
        depth_df = df[df["depth"] == depth].set_index("model").reindex(models)
        means = depth_df[f"{metric}_mean"].to_numpy(dtype=float)
        stds = depth_df[f"{metric}_std"].to_numpy(dtype=float)
        edgecolors = ["#0F172A" if model == "hc_gcn" and depth == 8 else "none" for model in models]
        linewidths = [1.2 if edge != "none" else 0.0 for edge in edgecolors]
        ax.barh(
            y_positions,
            means,
            xerr=stds,
            color=[MODEL_COLORS[model] for model in models],
            edgecolor=edgecolors,
            linewidth=linewidths,
            capsize=3,
            height=0.64,
        )
        for idx, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 0.012, idx, f"{mean:.3f}", va="center", ha="left", fontsize=8.5, color="#223042")
        ax.set_title(f"L{depth}", pad=10)
        ax.set_xlabel("OOD Kendall")
        ax.set_xlim(0.0, max_value)
        ax.grid(axis="x")
        ax.set_axisbelow(True)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([MODEL_LABELS[model] for model in models])
    axes[0].invert_yaxis()
    if len(axes) > 1:
        axes[1].tick_params(axis="y", length=0)
    finalize_figure(fig, title="Controlled Synthetic Anchor", layout_top=0.9)
    return save_figure(fig, output_dir / "figure_1_controlled_anchor")


def build_legacy_support_table(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    rows: list[dict[str, Any]] = []
    for row_cfg in config["tables"]["legacy_support"]["rows"]:
        root = _resolve_path(row_cfg["root"], config_path)
        run_name = row_cfg["run"]
        payload = _load_json(root / run_name / "aggregated_metrics.json")
        rows.append(
            {
                "group": row_cfg["group"],
                "variant": row_cfg["label"],
                "depth": int(row_cfg["depth"]),
                "ood_kendall": float(payload["test_ood_kendall"]["mean"]),
                "ood_spearman": float(payload["test_ood_spearman"]["mean"]),
                "takeaway": row_cfg["takeaway"],
            }
        )
    df = pd.DataFrame(rows)
    return write_table_bundle(df, output_dir / "table_2_legacy_support", float_digits=4)


def plot_matrix_stats(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    figure_cfg = config["figures"]["matrix_stats"]
    root = _resolve_path(figure_cfg["root"], config_path)
    metrics = figure_cfg.get(
        "metrics",
        ["identity_distance_mean", "row_abs_entropy_mean", "nearest_permutation_distance_mean"],
    )
    metric_titles = figure_cfg.get(
        "metric_titles",
        ["Identity distance", "Row entropy", "Nearest-permutation distance"],
    )
    models = figure_cfg.get("models", ["hc_gcn_l8", "mhc_gcn_l8", "mhc_lite_gcn_l8"])

    df = load_matrix_regime_stats(root)
    df = df[df["model"].isin(models)].copy()
    agg_df = (
        df.groupby(["model", "layer_index"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["model", "layer_index"])
    )

    configure_plot_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 3.7), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, title in zip(axes, metrics, metric_titles):
        for model in models:
            model_df = agg_df[agg_df["model"] == model].sort_values("layer_index")
            base_model = model.replace("_l8", "")
            ax.plot(
                model_df["layer_index"] + 1,
                model_df[metric],
                color=MODEL_COLORS[base_model],
                marker="o",
                linewidth=2.0,
                label=MODEL_LABELS.get(base_model, model),
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(True)
    axes[0].set_ylabel("Mean value across analyzed regimes")
    handles, labels = axes[0].get_legend_handles_labels()
    finalize_figure(
        fig,
        title="Routing-Matrix Derived Statistics",
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=3,
        legend_y=0.845,
        layout_top=0.81,
        layout_left=0.07,
    )
    return save_figure(fig, output_dir / "figure_2_matrix_stats")


def plot_size_generalization_curves(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    figure_cfg = config["figures"]["size_generalization_curves"]
    root = _resolve_path(figure_cfg["root"], config_path)
    metric = figure_cfg.get("metric", "kendall")
    split_order = figure_cfg.get("split_order", ["test_id", "test_ood"])
    size_order = [int(value) for value in figure_cfg.get("size_order", [1000, 2000, 5000, 10000])]
    models = figure_cfg.get("models", MODEL_ORDER)

    df = load_size_generalization_eval_details(root)
    df = df[df["split"].isin(split_order) & df["model"].isin(models)].copy()
    df["target_num_nodes"] = df["target_num_nodes"].astype(int)

    grouped = df.groupby(["split", "model", "target_num_nodes"], as_index=False)[metric].agg(mean="mean", std="std")

    configure_plot_style()
    fig, axes = plt.subplots(1, len(split_order), figsize=(11.4, 4.1), sharey=True)
    if len(split_order) == 1:
        axes = [axes]
    for ax, split in zip(axes, split_order):
        split_df = grouped[grouped["split"] == split]
        for model in models:
            model_df = split_df[split_df["model"] == model].set_index("target_num_nodes").reindex(size_order)
            means = model_df["mean"].to_numpy(dtype=float)
            stds = model_df["std"].to_numpy(dtype=float)
            x = np.arange(len(size_order))
            ax.plot(x, means, color=MODEL_COLORS[model], marker="o", linewidth=2.0, label=MODEL_LABELS[model])
            ax.fill_between(x, means - stds, means + stds, color=MODEL_COLORS[model], alpha=0.12)
        ax.set_xticks(np.arange(len(size_order)))
        ax.set_xticklabels([str(size) for size in size_order])
        ax.set_title(SPLIT_LABELS.get(split, split))
        ax.set_xlabel("Target graph size")
        ax.grid(True)
    axes[0].set_ylabel("Kendall")
    handles, labels = axes[0].get_legend_handles_labels()
    finalize_figure(
        fig,
        title="Size Generalization under Large Graph Shift",
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=5,
        legend_y=0.845,
        layout_top=0.81,
    )
    return save_figure(fig, output_dir / "figure_3_size_generalization_curves")


def build_family_delta_summary_table(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    table_cfg = config["tables"]["family_delta_summary"]
    root = _resolve_path(table_cfg["root"], config_path)
    metric = table_cfg.get("metric", "kendall")
    size_small = int(table_cfg.get("small_size", 1000))
    size_large = int(table_cfg.get("large_size", 10000))
    models = table_cfg.get("models", MODEL_ORDER)
    family_specs = table_cfg["families"]

    df = load_size_generalization_eval_details(root)
    df["target_num_nodes"] = df["target_num_nodes"].astype(int)
    grouped = (
        df.groupby(["split", "family", "model", "target_num_nodes"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value"})
    )

    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {"model": MODEL_LABELS[model]}
        for family_spec in family_specs:
            split = family_spec["split"]
            family = family_spec["family"]
            label = family_spec["label"]
            small = grouped[
                (grouped["split"] == split)
                & (grouped["family"] == family)
                & (grouped["model"] == model)
                & (grouped["target_num_nodes"] == size_small)
            ]["value"]
            large = grouped[
                (grouped["split"] == split)
                & (grouped["family"] == family)
                & (grouped["model"] == model)
                & (grouped["target_num_nodes"] == size_large)
            ]["value"]
            row[label] = float(large.iloc[0] - small.iloc[0]) if not small.empty and not large.empty else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    return write_table_bundle(df, output_dir / "table_4_family_delta_summary", float_digits=4)


def build_input_depth_followup_table(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    table_cfg = config["tables"]["input_depth_followups"]
    models = table_cfg.get("models", MODEL_ORDER)
    rows: list[dict[str, Any]] = []

    for input_spec in table_cfg["inputs"]:
        root = _resolve_path(input_spec["root"], config_path)
        df = load_model_aggregate_root(root)
        for model in models:
            model_payload = df[df["run_name"] == model]
            if model_payload.empty:
                continue
            row = model_payload.iloc[0]
            rows.append(
                {
                    "panel": "Input ablation",
                    "setting": input_spec["label"],
                    "model": MODEL_LABELS[model],
                    "id_kendall": float(row["test_id_kendall_mean"]),
                    "ood_kendall": float(row["test_ood_kendall_mean"]),
                }
            )

    l8_root = _resolve_path(table_cfg["depth_followup"]["l8_root"], config_path)
    l16_root = _resolve_path(table_cfg["depth_followup"]["l16_root"], config_path)
    l8_df = load_model_aggregate_root(l8_root)
    l16_df = load_model_aggregate_root(l16_root)
    for model in models:
        l8_payload = l8_df[l8_df["model"] == model]
        l16_payload = l16_df[l16_df["model"] == model]
        if l8_payload.empty or l16_payload.empty:
            continue
        l8_row = l8_payload.iloc[0]
        l16_row = l16_payload.iloc[0]
        rows.append(
            {
                "panel": "Depth follow-up",
                "setting": "L8",
                "model": MODEL_LABELS[model],
                "id_kendall": float(l8_row["test_id_kendall_mean"]),
                "ood_kendall": float(l8_row["test_ood_kendall_mean"]),
            }
        )
        rows.append(
            {
                "panel": "Depth follow-up",
                "setting": "L16",
                "model": MODEL_LABELS[model],
                "id_kendall": float(l16_row["test_id_kendall_mean"]),
                "ood_kendall": float(l16_row["test_ood_kendall_mean"]),
            }
        )

    df = pd.DataFrame(rows)
    return write_table_bundle(df, output_dir / "table_3_input_depth_followups", float_digits=4)


def plot_real_transfer_summary(config: dict[str, Any], config_path: Path, output_dir: Path) -> list[str]:
    figure_cfg = config["figures"]["real_transfer_summary"]
    root = _resolve_path(figure_cfg["root"], config_path)
    metric = figure_cfg.get("metric", "kendall")
    dataset_order = figure_cfg.get("dataset_order", list(DATASET_LABELS.keys()))
    models = figure_cfg.get("models", MODEL_ORDER)

    df = load_real_transfer_results(root)
    grouped = df.groupby(["dataset", "model"], as_index=False)[metric].mean()
    pivot = grouped.pivot(index="dataset", columns="model", values=metric).reindex(index=dataset_order, columns=models)
    mean_series = pivot.mean(axis=0)

    configure_plot_style()
    fig, (ax_main, ax_mean) = plt.subplots(1, 2, figsize=(10.9, 5.2), gridspec_kw={"width_ratios": [4.8, 1.7]})
    dataset_positions = np.arange(len(dataset_order))
    offsets = np.linspace(-0.24, 0.24, len(models))
    x_min = float(np.nanmin(pivot.to_numpy()) - 0.02)
    x_max = float(np.nanmax(pivot.to_numpy()) + 0.04)

    for offset, model in zip(offsets, models):
        values = pivot[model].to_numpy(dtype=float)
        ax_main.scatter(
            values,
            dataset_positions + offset,
            s=52,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            alpha=0.62,
            zorder=3,
        )

    for row_idx, dataset in enumerate(dataset_order):
        row_values = pivot.loc[dataset].to_numpy(dtype=float)
        best_idx = int(np.nanargmax(row_values))
        best_model = models[best_idx]
        best_value = row_values[best_idx]
        ax_main.scatter(
            [best_value],
            [dataset_positions[row_idx] + offsets[best_idx]],
            s=52,
            facecolors=MODEL_COLORS[best_model],
            edgecolors="#0F172A",
            linewidths=1.7,
            alpha=1.0,
            zorder=4,
        )

    for boundary in np.arange(len(dataset_order) - 1) + 0.5:
        ax_main.axhline(boundary, color="#E6EBF1", linewidth=0.8, zorder=1)

    ax_main.set_yticks(dataset_positions)
    ax_main.set_yticklabels([DATASET_LABELS.get(dataset, dataset) for dataset in dataset_order])
    ax_main.set_xlabel("Mean Kendall")
    ax_main.set_title("Per-dataset ranking")
    ax_main.set_xlim(x_min, x_max)
    ax_main.grid(axis="x")
    ax_main.set_axisbelow(True)
    ax_main.invert_yaxis()

    model_positions = np.arange(len(models))
    mean_values = np.array([mean_series[model] for model in models], dtype=float)
    ax_mean.barh(
        model_positions,
        mean_values,
        color=[MODEL_COLORS[model] for model in models],
        height=0.62,
    )
    for idx, value in enumerate(mean_values):
        ax_mean.text(value + 0.005, idx, f"{value:.3f}", va="center", ha="left", fontsize=8.5, color="#223042")
    ax_mean.set_yticks(model_positions)
    ax_mean.set_yticklabels([MODEL_LABELS[model] for model in models])
    ax_mean.set_xlabel("Mean")
    ax_mean.set_title("Across datasets")
    ax_mean.grid(axis="x")
    ax_mean.set_xlim(x_min, x_max)
    ax_mean.invert_yaxis()

    handles, labels = ax_main.get_legend_handles_labels()
    finalize_figure(
        fig,
        title="Zero-Shot Real-Graph Transfer from Size-Generalization L8 Checkpoints",
        legend_handles=handles,
        legend_labels=labels,
        legend_ncol=5,
        legend_y=0.845,
        layout_top=0.81,
    )
    return save_figure(fig, output_dir / "figure_5_real_transfer_summary")


def generate_paper_figure_assets(config: dict[str, Any], config_path: Path) -> dict[str, list[str]]:
    output_root = ensure_dir(_resolve_path(config["output_root"], config_path))
    figures_dir = ensure_dir(output_root / "figures")
    tables_dir = ensure_dir(output_root / "tables")

    manifest: dict[str, list[str]] = {}
    manifest["table_1_backbone_funnel"] = build_backbone_funnel_table(config, config_path, tables_dir)
    manifest["figure_1_controlled_anchor"] = plot_controlled_anchor(config, config_path, figures_dir)
    manifest["table_2_legacy_support"] = build_legacy_support_table(config, config_path, tables_dir)
    manifest["figure_2_matrix_stats"] = plot_matrix_stats(config, config_path, figures_dir)
    manifest["figure_3_size_generalization_curves"] = plot_size_generalization_curves(config, config_path, figures_dir)
    manifest["table_4_family_delta_summary"] = build_family_delta_summary_table(config, config_path, tables_dir)
    manifest["table_3_input_depth_followups"] = build_input_depth_followup_table(config, config_path, tables_dir)
    manifest["figure_5_real_transfer_summary"] = plot_real_transfer_summary(config, config_path, figures_dir)

    write_json(output_root / "manifest.json", manifest)
    return manifest
