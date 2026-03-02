from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    ranking_loss,
    topk_values: list[int],
    topk_ratios: list[float],
) -> dict[str, float]:
    model.eval()
    graph_losses: list[float] = []
    aggregated: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index)
            batch_index = _batch_index(batch)

            for graph_id in torch.unique(batch_index):
                graph_mask = batch_index == graph_id
                y_pred = predictions[graph_mask]
                y_true_norm = batch.y[graph_mask]
                y_true_raw = batch.y_raw[graph_mask]

                loss_value = ranking_loss(y_pred, y_true_norm).item()
                graph_losses.append(loss_value)

                graph_metrics = compute_graph_metrics(
                    y_pred=y_pred.detach().cpu().numpy(),
                    y_true=y_true_raw.detach().cpu().numpy(),
                    topk_values=topk_values,
                    topk_ratios=topk_ratios,
                )
                for key, value in graph_metrics.items():
                    aggregated[key].append(value)

    output: dict[str, float] = {
        "loss": float(np.mean(graph_losses)) if graph_losses else 0.0,
    }
    for key, values in aggregated.items():
        output[key] = float(np.mean(values)) if values else 0.0
    return output


def compute_graph_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    topk_values: list[int],
    topk_ratios: list[float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["spearman"] = _safe_spearman(y_true, y_pred)
    metrics["kendall"] = _safe_kendall(y_true, y_pred)

    num_nodes = y_true.shape[0]
    for k in _resolve_topk_values(num_nodes, topk_values, topk_ratios):
        metrics[f"precision_at_{k}"] = precision_at_k(y_pred, y_true, k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(y_pred, y_true, k)

    for ratio in topk_ratios:
        k = max(1, int(round(ratio * num_nodes)))
        suffix = _ratio_suffix(ratio)
        metrics[f"precision_at_{suffix}"] = precision_at_k(y_pred, y_true, k)
        metrics[f"ndcg_at_{suffix}"] = ndcg_at_k(y_pred, y_true, k)

    return metrics


def precision_at_k(y_pred: np.ndarray, y_true: np.ndarray, k: int) -> float:
    k = max(1, min(k, y_true.shape[0]))
    pred_top = np.argsort(-y_pred)[:k]
    true_top = np.argsort(-y_true)[:k]
    overlap = np.intersect1d(pred_top, true_top).shape[0]
    return float(overlap / k)


def ndcg_at_k(y_pred: np.ndarray, y_true: np.ndarray, k: int) -> float:
    k = max(1, min(k, y_true.shape[0]))
    pred_order = np.argsort(-y_pred)[:k]
    ideal_order = np.argsort(-y_true)[:k]

    gains_pred = np.maximum(y_true[pred_order], 0.0)
    gains_ideal = np.maximum(y_true[ideal_order], 0.0)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((2.0**gains_pred - 1.0) * discounts))
    idcg = float(np.sum((2.0**gains_ideal - 1.0) * discounts))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = spearmanr(y_true, y_pred).correlation
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def _safe_kendall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = kendalltau(y_true, y_pred).correlation
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def _resolve_topk_values(num_nodes: int, topk_values: list[int], topk_ratios: list[float]) -> list[int]:
    absolute = [max(1, min(num_nodes, int(k))) for k in topk_values]
    ratio_based = [max(1, min(num_nodes, int(round(r * num_nodes)))) for r in topk_ratios]
    return sorted(set(absolute + ratio_based))


def _ratio_suffix(ratio: float) -> str:
    pct = ratio * 100.0
    if pct.is_integer():
        return f"top_{int(pct)}pct"
    compact = str(pct).replace(".", "p")
    return f"top_{compact}pct"


def _batch_index(batch) -> torch.Tensor:
    if hasattr(batch, "batch") and batch.batch is not None:
        return batch.batch
    return torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)
