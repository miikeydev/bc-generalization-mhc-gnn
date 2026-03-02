# bc-generalization-mhc-gnn
Benchmarking depth scaling and Top-K retrieval for betweenness centrality using deep GNNs and manifold-constrained hyper-connections.

## Architecture 
- `src/data`: graph generation (ER/BA/SBM/WS/RGG), node features, BC labels, inductive splits (`train/val/test_id/test_ood`).
- `src/models`: model definitions (`gcn_baseline`) + `factory.py` to select models from config.
- `src/losses`: pairwise ranking loss for BC ordering.
- `src/eval.py`: Spearman, Kendall, Precision@K, NDCG@K.
- `src/train.py`: end-to-end training loop, early stopping, checkpoint + metrics export.

Pipeline: `data -> model -> train -> eval`.

## Run
```bash
uv run python -m src.train --config configs/smoke.yaml
uv run python -m src.train --config configs/baseline_gcn.yaml
```

Outputs are written to `outputs/<experiment_name>/`:
- `best_model.pt`
- `metrics.json`
- `resolved_config.json`
