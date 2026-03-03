# Potential Patches

## 1) Strong Baseline Alignment
Status: `todo`  
Label: `paper-aligned`

### Goal
Add strong non-HC baselines for fair comparison.

### Scope
- Standard baselines: `GCN`, `GraphSAGE`, `GAT`, `GIN`
- Deep baselines: `GCNII`, `APPNP` or `GPRGNN`, `JKNet`
- Integrate in `src/models/` and `src/models/factory.py`
- Add dedicated configs in `configs/`

### Evaluation
- Same protocol as HC variants
- Report `Spearman`, `Kendall`, `Precision@K`, `NDCG@K`

## 2) Depth Scaling Campaign
Status: `todo`  
Label: `paper-aligned`

### Goal
Run a real depth sweep instead of isolated runs.

### Scope
- Sweep `L = {2, 4, 8, 16, 32, ...}`
- Compare `gcn`, `hc_gnn`, `mhc_gnn`, `mhc_lite_gnn`
- Use the existing training entrypoint: [train.py](/home/mvayssieres/dev/bc-generalization-mhc-gnn/src/train.py)
- Align with Implementation Plan section `Ablation axes`

### Evaluation
- Depth curves on `val`, `test_id`, `test_ood`
- Ranking + Top-K metrics at each depth

## 3) Multi-Seed Statistical Protocol
Status: `todo`  
Label: `paper-aligned`

### Goal
Avoid single-seed conclusions.

### Scope
- Run each setup with `3-5` seeds
- Aggregate metrics to `mean ± std`
- Save per-seed and aggregated outputs

### Evaluation
- Stability and variance across seeds
- Robust comparison between model families

## 4) Core Ablations for HC/mHC/mHC-lite
Status: `todo`  
Label: `paper-aligned`

### Goal
Isolate which mechanism gives the gain.

### Scope
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

### Scope
- Keep training budget comparable across models
- Track parameter count per model
- Match capacity where possible

### Evaluation
- Performance with fair compute/capacity settings

## 6) Reporting Pack (Paper-Style)
Status: `todo`  
Label: `paper-aligned`

### Goal
Produce publishable final artifacts.

### Scope
- Consolidated tables for all model families
- Depth-scaling figures (ID/OOD)
- Top-K comparison figures

### Evaluation
- One final report with all core claims traceable

## 7) External-Style Features Ablation
Status: `todo`  
Label: `EXTRA (not from paper)`

### Goal
Optional ablation with external-style features.

### Scope
- `OneHotDegree`
- `RandomWalkPE`

### Evaluation
- Delta versus `degree + log_degree + LapPE`
