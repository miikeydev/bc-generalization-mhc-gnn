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
```

## Multi-seed benchmarks
- `configs/multi_seed/catalog.yaml`: shared model catalog + common protocol
- `configs/multi_seed/duels/`: targeted comparisons
- `configs/multi_seed/full/`: full benchmark matrix

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
