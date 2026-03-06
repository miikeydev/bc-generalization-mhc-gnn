# Potential Patches

## 1) Strong Baseline Alignment
Status: `done`  
Label: `paper-aligned`

### Goal
Add strong non-HC baselines for fair comparison.

### Implemented
- Standard baselines integrated: `GCN`, `GraphSAGE`, `GAT`, `GIN`
- Deep baselines integrated: `GCNII`, `APPNP`, `JKNet`
- Model integration completed in `src/models/` + `src/models/factory.py`
- Dedicated configs added in `configs/`
- HC family also extended to deep backbones (`gnn_type = gcnii/appnp/jknet`)

### Evaluation
- Same protocol as HC variants
- `Spearman`, `Kendall`, `Precision@K`, `NDCG@K`

## 2) Depth Scaling Campaign
Status: `done`  
Label: `paper-aligned`

### Goal
Run a real depth sweep instead of isolated runs.

### Implemented
- Sweep pipeline implemented:
  - `src/experiments/run_depth_sweep.py`
  - `src/experiments/collect_results.py`
  - `src/experiments/plot_depth_curves.py`
- Depth sweeps run on:
  - mixed setup: `configs/sweeps/depth_sweep.yaml`
  - deep+HC setups:
    - `configs/sweeps/depth_sweep_gcnii_hc.yaml`
    - `configs/sweeps/depth_sweep_appnp_hc.yaml`
    - `configs/sweeps/depth_sweep_jknet_hc.yaml`
- Outputs saved under `outputs/depth_sweep*`

### Evaluation
- Depth curves on `val`, `test_id`, `test_ood`
- Ranking + Top-K metrics at each depth

## 3) Multi-Seed Statistical Protocol
Status: `done`  
Label: `paper-aligned`

### Goal
Avoid single-seed conclusions.

### Implemented
- Full `5`-seed depth sweep completed on `L={2,4,8,16,32}`
- Per-seed outputs and all-seed sweep index saved under `outputs/depth_sweep_full/`
- Mean/std aggregation and figure synthesis completed
- Campaign snapshot frozen in `docs/campaigns/depth_sweep_full_5seed_2026-03-06/`

### Outcome
- Main depth conclusions are now backed by fixed seeds instead of single-seed inspection
- Current evidence supports HC/mHC gains mainly in shallow/mid-depth regimes, not a strong ultra-deep scaling claim on BC inductive ranking

## 4) Core Ablations for HC/mHC/mHC-lite
Status: `todo`  
Label: `paper-aligned`

### Goal
Isolate which mechanism gives the gain.

### Remaining
- `n_streams`
- `sinkhorn_iters`
- `sinkhorn_tau`
- `use_dynamic` / `use_static`
- `mhc_lite_max_permutations` when needed

### Evaluation
- Effect size on ID/OOD ranking and Top-K
- Sensitivity by depth level

## 5) Fair Budget and Capacity Matching
Status: `todo`  
Label: `paper-aligned`

### Goal
Ensure gains are architectural, not budget artifacts.

### Remaining
- Keep training budget comparable across models
- Track parameter count per model
- Match capacity where possible
- Add compute-aware comparison table

### Evaluation
- Performance with fair compute/capacity settings

## 6) Hyperparameter Search Protocol
Status: `in_progress`  
Label: `EXTRA (not from paper)`

### Goal
Reduce hyperparameter bias without full brute-force sweeps.

### Scope
- Add a lightweight search protocol on validation only
- Tune by backbone family:
  - `GCN`
  - `GCNII`
  - `APPNP` (secondary)
  - `JKNet` (secondary)
- Use a two-stage budget:
  - proxy stage (reduced epochs/data, `1` seed, baseline vs `mHC` only)
  - confirmation stage (full setup on top candidates, `3` seeds)
- Freeze selected hyperparameters before final multi-seed depth sweeps (`5` seeds)

### Implemented
- `Optuna` added to project dependencies

### TODO Checklist
- Define search spaces with priority on:
  - training regime: `epochs`, `patience`, scheduler, `lr`, `weight_decay`, `dropout`
  - HC/mHC knobs: `n_streams`, `sinkhorn_tau`, `sinkhorn_iters`, `mapping_init_alpha`
  - family-specific params when needed (`GCNII alpha/theta`, `APPNP alpha/K`, `JKNet mode`)
- Start with `GCN` and `GCNII` families, baseline vs `mHC` only, then revisit `HC` / `mHC-lite` / other families only if needed
- Add `configs/hp_search/` manifests for each family
- Add runner script with pruning and deterministic trial seeds
- Export `best_params.json` per family
- Update `configs/multi_seed/catalog.yaml` with frozen family-wise hyperparameters

### Evaluation
- Delta versus current defaults on `val/test_id/test_ood`
- Report search budget and selected settings

## 7) Execution Order
Status: `in_progress`  
Label: `paper-aligned`

### Goal
Run the remaining work in a defensible order and avoid scope drift.

### Order
1. Freeze the current `5`-seed depth campaign as the untuned reference point.
2. Run targeted hyperparameter search on `GCN` and `GCNII` families first, baseline vs `mHC`, with a reduced proxy budget (`15` trials, `3` depths).
3. Re-test the tuned `GCN` / `GCNII` families on fixed multi-seed depth runs.
4. If the deep gap remains, run core HC/mHC ablations (`n_streams`, Sinkhorn, dynamic/static).
5. Only then decide whether `APPNP` / `JKNet` deserve family-specific tuning.
6. Freeze tuned hyperparameters in `configs/multi_seed/catalog.yaml`.
7. Re-run final multi-seed duels and final comparison tables with frozen settings.

### Exit Criteria
- Main claims are backed by `mean ± std` over fixed seeds.
- `GCNII + mHC` conclusion is based on tuned and untuned multi-depth evidence.
- Final tables include both ranking (`Kendall/Spearman`) and Top-K (`Precision@K/NDCG@K`).

## Current Findings
Status: `done`  
Label: `paper-aligned`

### Frozen Reference
- Campaign archive: `docs/campaigns/depth_sweep_full_5seed_2026-03-06/`
- Reference commit: `0d025b8`

### What the `5`-Seed Depth Sweep Shows
- `HC/mHC` improve several backbones on BC ranking, especially in `OOD`
- The gain is concentrated at shallow/mid depth (`L2-L8`)
- The current setup does not support a strong “ultra-deep scaling” claim for HC/mHC on this task

### Best OOD Depth by Model Family
- `hc_gcnii`: best overall mean `OOD Kendall` at `L2` (`0.3824 ± 0.0253`)
- `mhc_gcn`: best `mHC` point at `L2` (`0.3812 ± 0.0238`)
- `hc_gcn`: strongest `GCN` family gain at `L4` (`0.3809 ± 0.0386`)
- `appnp`: best baseline deep profile at `L32` (`0.3544 ± 0.0258`)

### Interpretation to Carry Forward
- `GCN` and `GCNII` are the highest-value tuning targets because they test the core research question directly
- `APPNP` and `JKNet` remain useful comparison families, but are secondary for the next tuning round
- The most plausible next lever is protocol adaptation (`epochs`, `patience`, scheduler, regularization) before concluding that HC/mHC fundamentally underperform in deep BC ranking
