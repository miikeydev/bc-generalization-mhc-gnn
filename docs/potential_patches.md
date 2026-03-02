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
