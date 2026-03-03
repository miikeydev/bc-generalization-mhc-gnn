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
