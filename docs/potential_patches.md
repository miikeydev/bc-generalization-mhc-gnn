# Potential Patches

## 1) External-style Features Ablation
Status: `todo`  
Label: `EXTRA (not from paper)`

### Goal
Add an optional feature mode to reproduce the external reference style:
- `OneHotDegree`
- `RandomWalkPE`

This is for ablation/comparison only, not part of the core paper-aligned scope.

### Why
- Compare current structural features (`degree + log_degree + LapPE`) vs external-style features.
- Quantify impact on ranking and Top-K metrics in the same training/eval pipeline.

## 2) Strong Baseline Alignment (Paper Scope)
Status: `todo`  
Label: `paper-aligned`

### Goal
Upgrade baseline coverage to match the core research question more rigorously:
- Standard single-stream baselines: `GCN`, `GraphSAGE`, `GAT`, `GIN`
- Deep baselines: `GCNII`, `APPNP` or `GPRGNN`, `JKNet`

### Why
- Current baseline is centered on `GCN`, which is useful but not sufficient for a strong claim.
- The implementation plan explicitly calls for strong deep-GNN baselines when evaluating depth scaling.
- Fair comparison requires matching each HC/mHC/mHC-lite variant against a strong non-HC counterpart.

### Proposed patch scope
- Add baseline model files under `src/models/` for missing standard/deep variants.
- Extend `src/models/factory.py` to select each baseline from config.
- Add dedicated configs under `configs/` for each baseline and depth sweep setup.
- Keep training/eval pipeline unchanged (`src/train.py`, `src/eval.py`) for apples-to-apples comparison.

### Evaluation
- Run identical protocol for all baselines and HC variants:
  - Same graph families and ID/OOD split
  - Same seed policy
  - Same depth sweep (`L = 2, 4, 8, 16, 32`)
- Report:
  - `Spearman`, `Kendall`
  - `Precision@K`, `NDCG@K`
  - Mean and std across seeds

## 3) Depth Scaling Campaign (Missing)
Status: `todo`  
Label: `paper-aligned`

### Problem
No real depth-scaling campaign has been run yet.

### Why
- The training code is valid for single runs, but not yet used for a systematic sweep.
- We still need a depth sweep `L = {2, 4, 8, 16, 32, ...}` with multi-seed reporting.

### References
- Training entrypoint: [train.py](/home/mvayssieres/dev/bc-generalization-mhc-gnn/src/train.py)
- Implementation plan: section `Ablation axes`

### Proposed patch scope
- Add a reproducible runner for depth sweeps across models and seeds.
- Store per-run outputs and aggregate `mean ± std` tables.

### Evaluation
- Compare depth trends for `gcn`, `hc_gnn`, `mhc_gnn`, `mhc_lite_gnn`.
- Report ID/OOD ranking + Top-K metrics at each depth.
