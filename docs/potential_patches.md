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
- Tune by backbone family:
  - `GCN`
  - `GCNII`
  - `APPNP`
  - `JKNet`
- Use a two-stage budget:
  - proxy stage (reduced epochs/data, `1` seed)
  - confirmation stage (full setup on top candidates, `3` seeds)
- Freeze selected hyperparameters before final multi-seed depth sweeps (`5` seeds)

### TODO Checklist
- Define search spaces (`lr`, `weight_decay`, `dropout`, `ranking_pairs_per_node` + family-specific params)
- Add `configs/hp_search/` manifests for each family
- Add runner script with pruning and deterministic trial seeds
- Export `best_params.json` per family
- Update `configs/multi_seed/catalog.yaml` with frozen family-wise hyperparameters

### Evaluation
- Delta versus current defaults on `val/test_id/test_ood`
- Report search budget and selected settings

## 7) Execution Order
Status: `todo`  
Label: `paper-aligned`

### Goal
Run the remaining work in a defensible order and avoid scope drift.

### Order
1. Run `5`-seed duels first (`multi_seed_duels.yaml` and `multi_seed_duels_v2.yaml`) and freeze the baseline statistical table (`mean ± std`).
2. Close the `GCNII` question with depth-focused multi-seed runs (`L={4,8,16,32}`), then expand to critical depths with `5` seeds.
3. Execute family-wise hyperparameter search (`GCN`, `GCNII`, `APPNP`, `JKNet`) using proxy budget, then confirmation budget.
4. Freeze tuned hyperparameters in `configs/multi_seed/catalog.yaml`.
5. Re-run final multi-seed duels with frozen hyperparameters.
6. Produce final ranking + Top-K report with stability analysis (ID/OOD).

### Exit Criteria
- Main claims are backed by `mean ± std` over fixed seeds.
- `GCNII + mHC` conclusion is based on multiple depths, not only `L=16`.
- Final tables include both ranking (`Kendall/Spearman`) and Top-K (`Precision@K/NDCG@K`).
