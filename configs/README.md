# Config Layout

## Core single-run config
- `configs/isolated/single/train_default.yaml`

Used with:
```bash
uv run python -m src.train --config <path>
```

## Depth sweeps
- `configs/sweeps/`

Used with:
```bash
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep.yaml
uv run python -m src.experiments.run_depth_sweep --sweep-config configs/sweeps/depth_sweep_full.yaml
```

Notes:
- `seed` runs single-seed depth sweep.
- `seeds: [..]` runs multi-seed depth sweep in one command.
- optional CLI override: `--seeds 0 1 2 3 4`.

## Multi-seed benchmarks
- `configs/multi_seed/catalog.yaml`: shared model catalog + common protocol
- `configs/multi_seed/duels/`: targeted comparisons
- `configs/multi_seed/full/`: full benchmark matrix
- standard seed set: `[0, 1, 2, 3, 4]`

Used with:
```bash
uv run python -m src.experiments.run_multi_seed --sweep-config configs/multi_seed/duels/multi_seed_duels.yaml
uv run python -m src.experiments.aggregate_seeds --output-root outputs/multi_seed

uv run python -m src.experiments.run_multi_seed --sweep-config configs/multi_seed/full/multi_seed_full.yaml
uv run python -m src.experiments.aggregate_seeds --output-root outputs/multi_seed_full
```

## Isolated tests
- `configs/isolated/smoke/`: fast smoke checks only
- `configs/isolated/single/`: single-run config for debugging

Used with:
```bash
uv run python -m src.experiments.run_multi_seed --sweep-config configs/isolated/smoke/multi_seed_smoke_duels.yaml
```

---

## Data Configuration (v1 / v2 Protocol)

### Feature Modes
Configured via `data.feature_mode`:
- `structural_only` (default, legacy): degree + log_degree + LapPE. LapPE dimension set by `data.lap_pe_dim`.
- `degree`: degree + log_degree (2 dimensions, EXTRA: not from paper).
- `degree_random_walk_ppr`: concatenated input with degree + log_degree + scalable random-walk diffusion + scalable approximate PPR diffusion.
- `random_walk`: scalable random-walk diffusion features only. Output dimension and steps configured via `data.feature_config`.
- `ppr`: scalable approximate PPR diffusion features only. Output dimension, alpha, and diffusion steps configured via `data.feature_config`.
- `random`: random Gaussian features. Dimension set by `data.random_feature_dim`.
- `none`: constant features (all ones).

### Betweenness Centrality Backend
Configured via `data.bc_backend` and `data.bc_mode`:
- `bc_backend`: `networkit` only.
- `bc_mode`: `exact` only.

### Feature Config
Additional parameters via `data.feature_config` dict:
```yaml
data:
  feature_mode: degree_random_walk_ppr
  feature_config:
    rwpe_dim: 8
    rwpe_steps: 5
    ppr_dim: 8
    ppr_alpha: 0.15
    ppr_steps: 8
```

For `ppr`:
```yaml
data:
  feature_mode: ppr
  feature_config:
    ppr_dim: 8
    ppr_alpha: 0.15
    ppr_steps: 8
```

### Graph Regime Sampling
`train_num_nodes`, `test_num_nodes`, `train_average_degree`, and `test_average_degree` can be:
- a scalar value
- a list of choices
- a range spec with `min`, `max`, and optional `step`

Examples:
```yaml
data:
  train_num_nodes: [64, 96, 128, 160]
  test_num_nodes:
    min: 256
    max: 1024
    step: 128
  train_average_degree:
    min: 4
    max: 12
    step: 2
  test_average_degree: [8, 12, 16]
```

Legacy configs still work:
```yaml
data:
  train_num_nodes: 100
  test_num_nodes: 500
  average_degree: 8
```
If `train_average_degree` / `test_average_degree` are omitted, they fall back to `average_degree`.

### Graph Metadata
All generated graphs include structural metadata (written to Data objects):
- `target_num_nodes`, `target_average_degree`: sampled regime requested for that graph
- `num_nodes_graph`, `num_edges_graph`: actual graph size
- `avg_degree`, `density`: graph statistics
- `size_bucket`: `small` (<200 nodes) | `medium` (200–1000) | `large` (≥1000)
- `family`: graph family (er, ba, sbm, ws, rgg)

For real graphs, additional metadata:
- `clustering`: average clustering coefficient
- `assortativity`: degree assortativity coefficient

### Backward Compatibility
Old configs using `structural_only`, `lap_pe_dim`, and related keys continue to work unchanged.
The config normalizer auto-maps them to the v2 protocol.
