# 2026-04-19 — Diversity-knob validation (Phase 3)

Phase 3 of the improvement-gap plan (`docs/plans/2026-04-18-improvement-gap-surrogate-and-spectrum-diversity.md`). Sweeps the SBM diversity knob at four settings and reports the improvement-gap surrogate ĝ (eq. 18 in the NeurIPS draft) at `k ∈ {4, 8, 16, 32}` with both kNN (10 neighbours) and binning (4 quantile bins) estimators. The knob is validated if ĝ increases monotonically with diversity.

## Setup

- `num_graphs = 200`, `num_nodes = 50`, seed = 42
- Hyperparameter ranges at `diversity > 0`: `num_blocks ∈ [2, 5]`, `p_intra ∈ [0.3, 0.9]`, `p_inter ∈ [0.01, 0.2]`
- Fixed-mode reference (diversity=0) uses the midpoints: `num_blocks = 4`, `p_intra = 0.600`, `p_inter = 0.105`
- Noise: gaussian @ ε = 0.1

## Results

| diversity | k | estimator | ĝ | trace(Cov B) | ratio |
|-----------|---|-----------|-----|-----|-----|
| 0.00 | 4 | knn | 0.3380 | 0.6589 | 0.5131 |
| 0.00 | 4 | bin | 0.1184 | 0.6589 | 0.1796 |
| 0.00 | 8 | knn | 0.3594 | 0.8532 | 0.4212 |
| 0.00 | 8 | bin | 0.1395 | 0.8532 | 0.1634 |
| 0.00 | 16 | knn | 0.3920 | 1.1008 | 0.3561 |
| 0.00 | 16 | bin | 0.1571 | 1.1008 | 0.1427 |
| 0.00 | 32 | knn | 0.4287 | 1.5014 | 0.2855 |
| 0.00 | 32 | bin | 0.1684 | 1.5014 | 0.1122 |
| 0.33 | 4 | knn | 4.9870 | 5.8864 | 0.8472 |
| 0.33 | 4 | bin | 0.8898 | 5.8864 | 0.1512 |
| 0.33 | 8 | knn | 5.0485 | 6.1334 | 0.8231 |
| 0.33 | 8 | bin | 0.9589 | 6.1334 | 0.1563 |
| 0.33 | 16 | knn | 5.1591 | 6.5065 | 0.7929 |
| 0.33 | 16 | bin | 1.0399 | 6.5065 | 0.1598 |
| 0.33 | 32 | knn | 5.2357 | 6.9265 | 0.7559 |
| 0.33 | 32 | bin | 1.0514 | 6.9265 | 0.1518 |
| 0.67 | 4 | knn | 21.3997 | 23.9201 | 0.8946 |
| 0.67 | 4 | bin | 2.0071 | 23.9201 | 0.0839 |
| 0.67 | 8 | knn | 21.7076 | 24.7752 | 0.8762 |
| 0.67 | 8 | bin | 2.3408 | 24.7752 | 0.0945 |
| 0.67 | 16 | knn | 22.1379 | 25.5723 | 0.8657 |
| 0.67 | 16 | bin | 2.6791 | 25.5723 | 0.1048 |
| 0.67 | 32 | knn | 22.3919 | 26.2754 | 0.8522 |
| 0.67 | 32 | bin | 2.7448 | 26.2754 | 0.1045 |
| 1.00 | 4 | knn | 29.6420 | 34.2838 | 0.8646 |
| 1.00 | 4 | bin | 4.4956 | 34.2838 | 0.1311 |
| 1.00 | 8 | knn | 30.2107 | 35.9576 | 0.8402 |
| 1.00 | 8 | bin | 5.2437 | 35.9576 | 0.1458 |
| 1.00 | 16 | knn | 31.2402 | 37.5905 | 0.8311 |
| 1.00 | 16 | bin | 6.0208 | 37.5905 | 0.1602 |
| 1.00 | 32 | knn | 31.7233 | 38.6522 | 0.8207 |
| 1.00 | 32 | bin | 6.1434 | 38.6522 | 0.1589 |

## Monotonicity verdict

- `k=4`, `knn`: monotone ↑  (series: 0.3380, 4.9870, 21.3997, 29.6420)
- `k=4`, `bin`: monotone ↑  (series: 0.1184, 0.8898, 2.0071, 4.4956)
- `k=8`, `knn`: monotone ↑  (series: 0.3594, 5.0485, 21.7076, 30.2107)
- `k=8`, `bin`: monotone ↑  (series: 0.1395, 0.9589, 2.3408, 5.2437)
- `k=16`, `knn`: monotone ↑  (series: 0.3920, 5.1591, 22.1379, 31.2402)
- `k=16`, `bin`: monotone ↑  (series: 0.1571, 1.0399, 2.6791, 6.0208)
- `k=32`, `knn`: monotone ↑  (series: 0.4287, 5.2357, 22.3919, 31.7233)
- `k=32`, `bin`: monotone ↑  (series: 0.1684, 1.0514, 2.7448, 6.1434)

## Conclusion

ĝ increases monotonically with diversity across every `(k, estimator)` cell. The knob is validated and Phase 4 (full-dataset study) is unblocked.

Raw data: `diversity_sweep.csv`
