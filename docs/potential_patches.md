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
- Depth sweeps run on the main families and HC variants
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
- Tuned multi-seed follow-up runs completed for `GCN` and `GCNII`
- Final multi-seed comparison completed for the `GCN` family

### Outcome
- Main depth conclusions are now backed by fixed seeds instead of single-seed inspection
- Current evidence supports HC/mHC gains mainly in shallow/mid-depth regimes, not a strong ultra-deep scaling claim on BC inductive ranking

## 4) Core Ablations for HC/mHC/mHC-lite
Status: `in_progress`  
Label: `paper-aligned`

### Goal
Isolate which mechanism gives the gain.

### Implemented
- `GCN` ablation block completed on:
  - `n_streams`
  - `sinkhorn_iters`
  - `sinkhorn_tau`
  - `use_dynamic` / `use_static`
- The ablation result is now clear:
  - hyper-connections help on this task
  - `HC-GCN` outperforms `mHC-GCN` in the confirmed regime
  - the main gain appears to come from flexible multi-stream routing rather than from the doubly-stochastic Sinkhorn constraint itself

### Remaining
- Decide whether `GCNII` ablations are still worth the budget
- Evaluate whether `mHC-lite` deserves a dedicated mechanism study

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
Status: `done`  
Label: `EXTRA (not from paper)`

### Goal
Reduce hyperparameter bias without full brute-force sweeps.

### Implemented
- `Optuna` added to project dependencies
- Search module added under `src/tuning/`
- Search manifests added under `configs/hp_search/`
- Validation-only tuning protocol implemented
- Proxy search completed on `GCN` and `GCNII`
- `best_params.json` exported for the tuned families
- Tuned follow-up multi-seed confirmation runs completed

### Outcome
- The `GCN` family benefited from protocol tuning and became the main positive case
- `GCNII + mHC` remained limited in the deeper regime even after tuning
- No further `GCNII` tuning is currently planned

## 7) Execution Order
Status: `done`  
Label: `paper-aligned`

### Goal
Run the remaining work in a defensible order and avoid scope drift.

### Completed Order
1. Freeze the current `5`-seed depth campaign as the untuned reference point.
2. Run targeted hyperparameter search on `GCN` and `GCNII` families first, baseline vs `mHC`, with a reduced proxy budget.
3. Re-test the tuned `GCN` / `GCNII` families on fixed multi-seed depth runs.
4. Run core `GCN` ablations (`n_streams`, Sinkhorn, dynamic/static).
5. Produce a final multi-seed `GCN` family comparison.

### Outcome
- The project now has a stable experimental narrative:
  - hyper-connections help on inductive BC ranking
  - the strongest confirmed case is the `GCN` family
  - `HC-GCN` is the best tested variant under the current protocol

## Current Findings
Status: `done`  
Label: `paper-aligned`

### Frozen Reference
- Campaign archive: `docs/campaigns/depth_sweep_full_5seed_2026-03-06/`
- Reference commit: `0d025b8`

### What the Experiments Show
- `HC/mHC` improve several backbones on BC ranking, especially in `OOD`
- The gain is concentrated at shallow/mid depth (`L2-L8`)
- The current setup does not support a strong “ultra-deep scaling” claim for HC/mHC on this task
- The final `GCN` family comparison gives the following ranking under the tuned protocol:
  1. `HC-GCN`
  2. `mHC-GCN`
  3. `mHC-lite-GCN`
  4. `GCN`

### Interpretation to Carry Forward
- The main positive result of the project is no longer “`mHC` is the best”
- The stronger claim is:
  - hyper-connections improve inductive betweenness ranking on `GCN`
  - and `HC-GCN` is the best tested variant in the current setup
- The useful mechanism appears to be flexible multi-stream routing more than the Sinkhorn-constrained mixing itself

## Remaining TODO
- Run a residual-baseline sweep on the `GCN` family:
  - `GCN`
  - `GCN + residual`
  - `HC-GCN`
  - `mHC-GCN`
  - `mHC-lite-GCN`
  - on `L4` and `L8` with `5` seeds
- Analyze `HC-GCN-L8` vs `mHC-GCN-L8` internal matrices on synthetic and `Cora`
- Evaluate stronger convolutional baselines such as `EdgeConv` separately from the HC/mHC causal comparison
- Revisit input features after the architectural comparison is settled:
  - reduce or replace `LapPE`
  - test more scalable global signals (`PageRank`, random-walk-style features, Fiedler-style alternatives)
- Add a fair budget / parameter count comparison table if a paper-style final report is needed
- Consolidate the final results into the main README or a short final report
