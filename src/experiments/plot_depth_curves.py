from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_STYLE: dict[str, dict] = {
    "gcn":          {"color": "#4878cf", "linestyle": "-",  "marker": "o"},
    "gcnii":        {"color": "#ff7f0e", "linestyle": "-",  "marker": "D"},
    "appnp":        {"color": "#8c564b", "linestyle": "-",  "marker": "P"},
    "jknet":        {"color": "#17becf", "linestyle": "-",  "marker": "X"},
    "hc_gnn":       {"color": "#1f77b4", "linestyle": ":",  "marker": "o"},
    "mhc_gnn":      {"color": "#d62728", "linestyle": ":",  "marker": "s"},
    "mhc_lite_gnn": {"color": "#2ca02c", "linestyle": ":",  "marker": "^"},
}

SPLIT_LABEL = {"val": "Validation", "test_id": "Test ID", "test_ood": "Test OOD"}

METRICS = [
    ("kendall",        "Kendall τ"),
    ("spearman",       "Spearman ρ"),
    ("ndcg_at_10",     "NDCG@10"),
    ("ndcg_at_50",     "NDCG@50"),
    ("precision_at_10", "Precision@10"),
    ("precision_at_50", "Precision@50"),
]

SPLITS = ["val", "test_id", "test_ood"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/depth_sweep/depth_results.csv")
    parser.add_argument("--figures-dir", default="outputs/depth_sweep/figures")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def group(rows: list[dict]) -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        try:
            depth = int(row["depth"])
        except (KeyError, ValueError):
            continue
        result[row["model"]][depth] = row
    return result


def get_values(depth_rows: dict[int, dict], key: str) -> tuple[list[int], list[float]]:
    depths = sorted(depth_rows.keys())
    values = []
    for d in depths:
        try:
            values.append(float(depth_rows[d].get(key, "nan") or "nan"))
        except ValueError:
            values.append(float("nan"))
    return depths, values


def plot_single(rows: list[dict], split: str, metric: str, ylabel: str, out_dir: Path) -> None:
    key = f"{split}_{metric}"
    grouped = group(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, depth_rows in sorted(grouped.items()):
        style = MODEL_STYLE.get(model, {"color": "gray", "linestyle": "-", "marker": "x"})
        depths, values = get_values(depth_rows, key)
        ax.plot(depths, values, label=model, linewidth=1.8, markersize=6, **style)
    ax.set_xlabel("Depth L")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{SPLIT_LABEL.get(split, split)} — {ylabel}")
    ax.set_xticks(sorted({int(r["depth"]) for r in rows if r.get("depth")}))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_{metric}.png", dpi=150)
    plt.close(fig)


def plot_summary(rows: list[dict], out_dir: Path) -> None:
    summary_metrics = [("kendall", "Kendall τ"), ("ndcg_at_10", "NDCG@10"), ("precision_at_10", "Precision@10")]
    grouped = group(rows)
    fig, axes = plt.subplots(len(summary_metrics), len(SPLITS), figsize=(13, 9), sharex=True)

    for col, split in enumerate(SPLITS):
        for row_idx, (metric, ylabel) in enumerate(summary_metrics):
            ax = axes[row_idx][col]
            key = f"{split}_{metric}"
            for model, depth_rows in sorted(grouped.items()):
                style = MODEL_STYLE.get(model, {"color": "gray", "linestyle": "-", "marker": "x"})
                depths, values = get_values(depth_rows, key)
                ax.plot(depths, values, label=model, linewidth=1.5, markersize=5, **style)
            if row_idx == 0:
                ax.set_title(SPLIT_LABEL.get(split, split), fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_idx == len(summary_metrics) - 1:
                ax.set_xlabel("Depth L", fontsize=9)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_cols = max(1, min(4, len(labels)))
    fig.legend(handles, labels, loc="lower center", ncol=legend_cols, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Depth Scaling Summary", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "depth_scaling_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    figures_dir = Path(args.figures_dir)

    if not csv_path.exists():
        print(f"{csv_path} not found. Run: python -m src.experiments.collect_results first.")
        return

    rows = load_csv(csv_path)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for metric, ylabel in METRICS:
            plot_single(rows, split, metric, ylabel, figures_dir)

    plot_summary(rows, figures_dir)
    print(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
