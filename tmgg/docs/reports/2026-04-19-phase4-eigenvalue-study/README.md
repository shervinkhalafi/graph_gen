# 2026-04-19 — Phase 4 Eigenvalue Study

Final eigenvalue-study sweep across all target datasets. Applies the Phase 3 v2 surrogate infrastructure (Fréchet-mean frame, permutation-null control, four estimator variants) to the real-data benchmarks and the synthetic diversity sweep.

For the theoretical justification of the Fréchet-frame choice and the permutation-null bias offset — with citations — see [`theory.md`](theory.md).

## Setup

- Seeds: [42, 123, 2024, 7, 11] (n = 5).
- Noise levels: [0.01, 0.05, 0.1, 0.15, 0.2].
- Noise types: ['gaussian', 'digress'].
- k values: [4, 8, 16, 32] (per-dataset filtered to ``k ≤ n``).
- Frame modes: ['frechet', 'per_graph'] (frechet is the headline).
- Estimators: ['knn_top_k', 'knn_1d', 'bin_1d', 'invariants_knn'].
- Permutation null: reported for every cell.

### Datasets

| Dataset | Category | Filter band | Notes |
|---|---|---|---|
| sbm_d0.00 | synthetic_sbm | fixed n=50 | diversity=0.0 |
| sbm_d0.33 | synthetic_sbm | fixed n=50 | diversity=0.33 |
| sbm_d0.67 | synthetic_sbm | fixed n=50 | diversity=0.67 |
| sbm_d1.00 | synthetic_sbm | fixed n=50 | diversity=1.0 |
| spectre_sbm | spectre_sbm | (30, 80) | upstream DiGress fixture |
| enzymes | tu | (30, 60) | community-structured TU benchmark (enzymes) |
| proteins | tu | (30, 80) | community-structured TU benchmark (proteins) |
| collab | tu | (30, 80) | community-structured TU benchmark (collab) |

### Retained graph counts (at each seed)

| Dataset | n | N retained | note |
|---|---|---|---|
| sbm_d0.00 | 50 | 200 | invariant across seeds for fixed datasets (per-seed redraw) |
| sbm_d0.33 | 50 | 200 | invariant across seeds for fixed datasets (per-seed redraw) |
| sbm_d0.67 | 50 | 200 | invariant across seeds for fixed datasets (per-seed redraw) |
| sbm_d1.00 | 50 | 200 | invariant across seeds for fixed datasets (per-seed redraw) |
| spectre_sbm | 80 | 68 | invariant across seeds for fixed datasets |
| enzymes | 60 | 305 | invariant across seeds for fixed datasets |
| proteins | 80 | 383 | invariant across seeds for fixed datasets |
| collab | 80 | 3715 | invariant across seeds for fixed datasets |

## Headline: dataset ordering at (frame=frechet, estimator=knn_top_k, k=8, noise=gaussian, ε=0.1)

Calibrated margin = mean(FVE_real) − mean(FVE_null), where FVE = `Var(E[B | Λ̃_k]) / Var(B)` is the fraction of variance explained of the projected clean adjacency `B` by the noisy top-k eigenvalues. Success criterion: margin ≥ 0.10 with null below 0.30.

| Dataset | real FVE (mean) | null FVE (mean) | calibrated margin | pass? |
|---|---|---|---|---|
| sbm_d0.00 | 0.139 | 0.097 | 0.043 | ✗ |
| sbm_d0.33 | 0.302 | 0.097 | 0.205 | ✓ |
| sbm_d0.67 | 0.561 | 0.102 | 0.459 | ✓ |
| sbm_d1.00 | 0.718 | 0.090 | 0.629 | ✓ |
| spectre_sbm | 0.353 | 0.081 | 0.272 | ✓ |
| enzymes | 0.197 | 0.097 | 0.100 | ✓ |
| proteins | 0.284 | 0.100 | 0.184 | ✓ |
| collab | 0.645 | 0.101 | 0.545 | ✓ |

### Sorted ranking (best margin first)

1. **sbm_d1.00** — margin 0.629
2. **collab** — margin 0.545
3. **sbm_d0.67** — margin 0.459
4. **spectre_sbm** — margin 0.272
5. **sbm_d0.33** — margin 0.205
6. **proteins** — margin 0.184
7. **enzymes** — margin 0.100
8. **sbm_d0.00** — margin 0.043

### Ranking stability

Across all (2 × 5 × 4 × 4) = 160 (noise_type × noise_level × estimator × k) cells:

- **top-1** matches the headline in 20/160 cells (12 %).
- **top-2 set** matches in 160/160 cells (100 %).
- **pass/fail set** (margin ≥ 0.10 AND null < 0.30) matches in 6/160 cells (4 %).
- Full-ranking exact match in 10/160 cells (6 %) — included for completeness; datasets with close margins swap ranks within tolerance so this metric understates true stability.

## Noise-level sensitivity (headline cell)

For each dataset at (knn_top_k, frechet, gaussian, k=8), report FVE vs. ε series (mean±std) and whether the calibrated margin stays positive across ε.

| Dataset | FVE series | null series | margin at each ε |
|---|---|---|---|
| sbm_d0.00 | 0.14±0.01, 0.14±0.01, 0.14±0.00, 0.14±0.00, 0.13±0.00 | 0.10±0.00, 0.10±0.00, 0.10±0.00, 0.09±0.00, 0.10±0.00 | 0.05, 0.05, 0.04, 0.04, 0.03 |
| sbm_d0.33 | 0.31±0.02, 0.31±0.02, 0.30±0.01, 0.30±0.02, 0.28±0.01 | 0.10±0.01, 0.10±0.01, 0.10±0.01, 0.10±0.01, 0.10±0.01 | 0.22, 0.21, 0.20, 0.20, 0.18 |
| sbm_d0.67 | 0.57±0.02, 0.57±0.02, 0.56±0.03, 0.55±0.02, 0.54±0.03 | 0.10±0.01, 0.10±0.01, 0.10±0.01, 0.10±0.00, 0.10±0.00 | 0.47, 0.47, 0.46, 0.45, 0.44 |
| sbm_d1.00 | 0.73±0.02, 0.72±0.02, 0.72±0.02, 0.71±0.02, 0.70±0.02 | 0.09±0.01, 0.09±0.01, 0.09±0.01, 0.09±0.01, 0.09±0.01 | 0.63, 0.63, 0.63, 0.62, 0.61 |
| spectre_sbm | 0.35±0.00, 0.35±0.00, 0.35±0.01, 0.33±0.01, 0.32±0.01 | 0.08±0.01, 0.08±0.01, 0.08±0.01, 0.09±0.01, 0.09±0.01 | 0.27, 0.27, 0.27, 0.24, 0.23 |
| enzymes | 0.21±0.00, 0.20±0.00, 0.20±0.00, 0.19±0.00, 0.17±0.00 | 0.10±0.00, 0.10±0.00, 0.10±0.00, 0.10±0.00, 0.10±0.00 | 0.11, 0.11, 0.10, 0.09, 0.08 |
| proteins | 0.29±0.00, 0.29±0.00, 0.28±0.00, 0.27±0.00, 0.25±0.00 | 0.10±0.00, 0.10±0.00, 0.10±0.00, 0.10±0.01, 0.10±0.00 | 0.19, 0.19, 0.18, 0.17, 0.15 |
| collab | 0.65±0.00, 0.65±0.00, 0.65±0.00, 0.64±0.00, 0.64±0.00 | 0.10±0.00, 0.10±0.00, 0.10±0.00, 0.10±0.00, 0.10±0.00 | 0.55, 0.55, 0.54, 0.54, 0.54 |

## Synthetic SBM: diversity ↑ (at ε=0.1, k=8, headline cell)

Replicates the Phase 3 v2 finding at a single ε on the real-data grid. Should be monotone across sbm_d{0, 0.33, 0.67, 1.0}.

| Diversity | real FVE | null FVE | margin |
|---|---|---|---|
| 0.00 | 0.139 | 0.097 | 0.043 |
| 0.33 | 0.302 | 0.097 | 0.205 |
| 0.67 | 0.561 | 0.102 | 0.459 |
| 1.00 | 0.718 | 0.090 | 0.629 |

## Conclusion

7/8 datasets pass the calibrated-margin criterion (≥0.10) at the headline cell. Ranking stability: top-1 match 20/160, top-2 set match 160/160, pass/fail set match 6/160.

Raw data: `phase4_sweep.csv` (25600 rows).
