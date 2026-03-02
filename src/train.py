from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.data import build_inductive_datasets
from src.eval import evaluate_loader
from src.losses import PairwiseRankingLoss
from src.models import build_model
from src.utils import ensure_dir, load_config, set_global_seed, write_json


def train_from_config(config: dict) -> dict:
    seed = int(config["experiment"]["seed"])
    set_global_seed(seed)

    output_dir = ensure_dir(config["experiment"]["output_dir"])
    write_json(output_dir / "resolved_config.json", config)

    datasets = build_inductive_datasets(config=config, seed=seed)
    batch_size = int(config["training"]["batch_size"])

    train_loader = DataLoader(datasets.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets.val, batch_size=batch_size, shuffle=False)
    test_id_loader = DataLoader(datasets.test_id, batch_size=batch_size, shuffle=False)
    test_ood_loader = DataLoader(datasets.test_ood, batch_size=batch_size, shuffle=False)

    first_batch = next(iter(train_loader))
    input_dim = int(first_batch.x.shape[1])

    model = build_model(config=config, input_dim=input_dim)
    device = _resolve_device(config.get("device", "auto"))
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    ranking_loss = PairwiseRankingLoss(pairs_per_node=int(config["training"]["ranking_pairs_per_node"]))

    topk_values = [int(k) for k in config["evaluation"]["topk_values"]]
    topk_ratios = [float(r) for r in config["evaluation"]["topk_ratios"]]

    max_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])

    best_state = None
    best_val_kendall = float("-inf")
    best_epoch = 0
    patience_count = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ranking_loss=ranking_loss,
            device=device,
        )

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            ranking_loss=ranking_loss,
            topk_values=topk_values,
            topk_ratios=topk_ratios,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss", 0.0),
            "val_spearman": val_metrics.get("spearman", 0.0),
            "val_kendall": val_metrics.get("kendall", 0.0),
        }
        history.append(epoch_record)

        current_kendall = val_metrics.get("kendall", 0.0)
        if current_kendall > best_val_kendall:
            best_val_kendall = current_kendall
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_model.pt")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = evaluate_loader(
        model=model,
        loader=val_loader,
        device=device,
        ranking_loss=ranking_loss,
        topk_values=topk_values,
        topk_ratios=topk_ratios,
    )
    final_test_id_metrics = evaluate_loader(
        model=model,
        loader=test_id_loader,
        device=device,
        ranking_loss=ranking_loss,
        topk_values=topk_values,
        topk_ratios=topk_ratios,
    )
    final_test_ood_metrics = evaluate_loader(
        model=model,
        loader=test_ood_loader,
        device=device,
        ranking_loss=ranking_loss,
        topk_values=topk_values,
        topk_ratios=topk_ratios,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_kendall": best_val_kendall,
        "train_label_mean": datasets.train_label_mean,
        "train_label_std": datasets.train_label_std,
        "val": final_val_metrics,
        "test_id": final_test_id_metrics,
        "test_ood": final_test_ood_metrics,
        "history": history,
    }

    write_json(output_dir / "metrics.json", summary)
    return summary


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    ranking_loss: PairwiseRankingLoss,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch.x, batch.edge_index)
        batch_index = _batch_index(batch)

        graph_losses = []
        for graph_id in torch.unique(batch_index):
            graph_mask = batch_index == graph_id
            loss_value = ranking_loss(predictions[graph_mask], batch.y[graph_mask])
            graph_losses.append(loss_value)

        if len(graph_losses) == 0:
            continue

        total_loss = torch.stack(graph_losses).mean()
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.item()))

    return float(sum(losses) / len(losses)) if losses else 0.0


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def _batch_index(batch) -> torch.Tensor:
    if hasattr(batch, "batch") and batch.batch is not None:
        return batch.batch
    return torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_gcn.yaml")
    return parser.parse_args()


def cli_main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    summary = train_from_config(config)
    print(
        {
            "best_epoch": summary["best_epoch"],
            "best_val_kendall": summary["best_val_kendall"],
            "test_id_kendall": summary["test_id"].get("kendall", 0.0),
            "test_ood_kendall": summary["test_ood"].get("kendall", 0.0),
        }
    )


if __name__ == "__main__":
    cli_main()
