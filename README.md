# bc-generalization-mhc-gnn
Benchmarking depth scaling and Top-K retrieval for betweenness centrality using deep GNNs and manifold-constrained hyper-connections.

## Architecture 
- `src/data`: graph generation (ER/BA/SBM/WS/RGG), node features, BC labels, inductive splits (`train/val/test_id/test_ood`).
- `src/models`: model definitions (`gcn_baseline`) + `factory.py` to select models from config.
- `src/losses`: pairwise ranking loss for BC ordering.
- `src/eval.py`: Spearman, Kendall, Precision@K, NDCG@K.
- `src/train.py`: end-to-end training loop, early stopping, checkpoint + metrics export.
- `configs/sweeps`: depth sweep definitions.
- `configs/multi_seed/duels`: targeted multi-seed comparisons.
- `configs/multi_seed/full`: full multi-seed benchmark configs.
- `configs/isolated/smoke`: fast isolated smoke tests.
- `configs/README.md`: config map and run commands.

Pipeline: `data -> model -> train -> eval`.

## Model Variants
- `gcn`: baseline GCN regressor
- `sage` / `gat` / `gin`: shallow message-passing baselines
- `gcnii` / `appnp` / `jknet`: deep baseline families
- `hc_gnn`: multi-stream hyper-connections without manifold constraint
- `mhc_gnn`: manifold-constrained hyper-connections with Sinkhorn projection
- `mhc_lite_gnn`: exact doubly-stochastic residual mixing via convex combination of permutation matrices

## Run
```bash
uv run python -m src.train --config configs/isolated/single/train_default.yaml
```

## Depth Sweep
```bash
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep.yaml
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep_gcnii_hc.yaml
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep_appnp_hc.yaml
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep_jknet_hc.yaml
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep_full.yaml
uv run python -m src.experiments.collect_results --sweep-index outputs/depth_sweep/sweep_index.json --output-csv outputs/depth_sweep/depth_results.csv
uv run python -m src.experiments.plot_depth_curves --csv outputs/depth_sweep/depth_results.csv --figures-dir outputs/depth_sweep/figures
```

## Multi-Seed
```bash
uv run python -m src.experiments.run_multi_seed --sweep-config configs/multi_seed/duels/multi_seed_duels.yaml
uv run python -m src.experiments.aggregate_seeds --output-root outputs/multi_seed

uv run python -m src.experiments.run_multi_seed --sweep-config configs/multi_seed/full/multi_seed_full.yaml
uv run python -m src.experiments.aggregate_seeds --output-root outputs/multi_seed_full
```

## Smoke (Isolated)
```bash
uv run python -m src.experiments.run_multi_seed --sweep-config configs/isolated/smoke/multi_seed_smoke_duels.yaml
```

Outputs are written to `outputs/<experiment_name>/`:
- `best_model.pt`
- `metrics.json`
- `resolved_config.json`
