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
Status: `todo`  
Label: `paper-aligned`

### Goal
Avoid single-seed conclusions.

### Remaining
- Run multi-seed on depth scaling with a staged protocol:
  - Stage A: keep full depth maps in single-seed (`L={2,4,8,16,32}`)
  - Stage B: run `3` seeds on key depths (`L={2,8,16,32}`)
  - Stage C: run `5` seeds on critical checkpoints (best depth + deepest depth + failure depth)
- Aggregate metrics to `mean ± std`
- Save per-seed and aggregated outputs
- Report confidence/stability in final comparisons

### Current Gap
- Multi-seed exists for duels/full runs, but depth sweeps are not yet statistically consolidated

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
Status: `todo`  
Label: `EXTRA (not from paper)`

### Goal
Reduce hyperparameter bias without full brute-force sweeps.

### Scope
- Add a lightweight search protocol (Optuna or random search) on validation only
- Use a two-stage budget:
  - proxy stage (reduced epochs/data)
  - confirmation stage (full setup on top candidates)
- Freeze selected hyperparameters before final multi-seed depth sweeps

### Evaluation
- Delta versus current defaults on `val/test_id/test_ood`
- Report search budget and selected settings
