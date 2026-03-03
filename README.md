# bc-generalization-mhc-gnn
Benchmarking depth scaling and Top-K retrieval for betweenness centrality using deep GNNs and manifold-constrained hyper-connections.

## Architecture 
- `src/data`: graph generation (ER/BA/SBM/WS/RGG), node features, BC labels, inductive splits (`train/val/test_id/test_ood`).
- `src/models`: model definitions (`gcn_baseline`) + `factory.py` to select models from config.
- `src/losses`: pairwise ranking loss for BC ordering.
- `src/eval.py`: Spearman, Kendall, Precision@K, NDCG@K.
- `src/train.py`: end-to-end training loop, early stopping, checkpoint + metrics export.

Pipeline: `data -> model -> train -> eval`.

## Model Variants
- `gcn`: baseline GCN regressor
- `hc_gnn`: multi-stream hyper-connections without manifold constraint
- `mhc_gnn`: manifold-constrained hyper-connections with Sinkhorn projection
- `mhc_lite_gnn`: exact doubly-stochastic residual mixing via convex combination of permutation matrices

## Run
```bash
uv run python -m src.train --config configs/smoke.yaml
uv run python -m src.train --config configs/baseline_gcn.yaml
uv run python -m src.train --config configs/mhc_gcn.yaml
uv run python -m src.train --config configs/hc_gcn.yaml
uv run python -m src.train --config configs/mhc_lite_gcn.yaml
```

## Depth Sweep
```bash
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/depth_sweep.yaml
uv run python -m src.experiments.collect_results --sweep-index outputs/depth_sweep/sweep_index.json --output-csv outputs/depth_sweep/depth_results.csv
uv run python -m src.experiments.plot_depth_curves --csv outputs/depth_sweep/depth_results.csv --figures-dir outputs/depth_sweep/figures
```

Outputs are written to `outputs/<experiment_name>/`:
- `best_model.pt`
- `metrics.json`
- `resolved_config.json`
