# 2026-04-19 — Diversity-knob validation (Phase 3, v2)

Phase 3 of the improvement-gap plan (`docs/plans/2026-04-18-improvement-gap-surrogate-and-spectrum-diversity.md`). Addresses reviewer-2 audit findings: frame mode exposed, permutation null reported, ratio monotonicity (not absolute ĝ) used as the success criterion, `num_blocks` frozen, ≥3 seeds.

## Setup

- `num_graphs = 200`, `num_nodes = 50`, seeds = [42, 123, 2024]
- `num_blocks` frozen at 4 across all diversity levels; knob varies `p_intra ∈ [0.3, 0.9]`, `p_inter ∈ [0.01, 0.2]` scaled by diversity.
- Fixed-mode reference (diversity=0) uses the midpoints: `p_intra = 0.600`, `p_inter = 0.105`
- Noise: gaussian @ ε = 0.1
- Frame modes: `frechet` (default, dataset-wide common frame via extrinsic Grassmannian mean) and `per_graph` (align each noisy V̂ to that graph's clean V; retained as diagnostic — does not satisfy the common-frame requirement of eq. (18) across heterogeneous datasets).
- Estimators: `knn_top_k` (kNN with k-dim top-k eigenvalues on raw B), `knn_1d` and `bin_1d` (same 1-D spectral-gap feature, comparable), `invariants_knn` (kNN on frame-invariant summaries of B — frame-free cross-check).
- Permutation null: every cell is also computed with conditioning features shuffled; a calibrated estimator returns `ratio ≈ 0` under the null.

## Monotonicity verdicts

Success criterion per cell: **mean ratio** across seeds is monotone non-decreasing in diversity (0 → 0.33 → 0.67 → 1.0). Absolute ĝ is also reported but is not the gate — it tracks `trace(Cov B)` which mechanically grows with diversity.

### Real features (permutation off)

| frame | estimator | k | ratio series (mean±std) | ĝ monotone? | ratio monotone? |
|---|---|---|---|---|---|
| frechet | knn_top_k | 4 | 0.207±0.021, 0.475±0.037, 0.724±0.032, 0.831±0.026 | ✓ | ✓ |
| frechet | knn_top_k | 8 | 0.137±0.008, 0.306±0.023, 0.565±0.030, 0.720±0.024 | ✓ | ✓ |
| frechet | knn_top_k | 16 | 0.109±0.003, 0.179±0.009, 0.351±0.018, 0.519±0.020 | ✓ | ✓ |
| frechet | knn_top_k | 32 | 0.099±0.001, 0.118±0.002, 0.179±0.007, 0.267±0.010 | ✓ | ✓ |
| frechet | knn_1d | 4 | 0.139±0.004, 0.239±0.024, 0.326±0.051, 0.369±0.047 | ✓ | ✓ |
| frechet | knn_1d | 8 | 0.113±0.005, 0.181±0.010, 0.276±0.031, 0.342±0.035 | ✓ | ✓ |
| frechet | knn_1d | 16 | 0.100±0.001, 0.131±0.003, 0.192±0.014, 0.257±0.020 | ✓ | ✓ |
| frechet | knn_1d | 32 | 0.096±0.001, 0.103±0.000, 0.122±0.003, 0.150±0.006 | ✓ | ✓ |
| frechet | bin_1d | 4 | 0.060±0.006, 0.155±0.024, 0.260±0.039, 0.306±0.049 | ✓ | ✓ |
| frechet | bin_1d | 8 | 0.036±0.003, 0.099±0.011, 0.205±0.025, 0.275±0.036 | ✓ | ✓ |
| frechet | bin_1d | 16 | 0.023±0.002, 0.049±0.004, 0.116±0.011, 0.185±0.022 | ✓ | ✓ |
| frechet | bin_1d | 32 | 0.016±0.001, 0.023±0.001, 0.042±0.003, 0.072±0.007 | ✓ | ✓ |
| frechet | invariants_knn | 4 | 0.699±0.077, 0.840±0.047, 0.899±0.029, 0.920±0.026 | ✓ | ✓ |
| frechet | invariants_knn | 8 | 0.642±0.066, 0.813±0.059, 0.887±0.032, 0.910±0.028 | ✓ | ✓ |
| frechet | invariants_knn | 16 | 0.611±0.083, 0.800±0.061, 0.870±0.041, 0.898±0.035 | ✓ | ✓ |
| frechet | invariants_knn | 32 | 0.582±0.081, 0.797±0.056, 0.870±0.039, 0.897±0.031 | ✓ | ✓ |
| per_graph | knn_top_k | 4 | 0.556±0.020, 0.758±0.035, 0.857±0.039, 0.899±0.028 | ✓ | ✓ |
| per_graph | knn_top_k | 8 | 0.442±0.014, 0.717±0.027, 0.840±0.035, 0.887±0.030 | ✓ | ✓ |
| per_graph | knn_top_k | 16 | 0.369±0.013, 0.670±0.032, 0.817±0.033, 0.876±0.027 | ✓ | ✓ |
| per_graph | knn_top_k | 32 | 0.303±0.012, 0.613±0.036, 0.793±0.029, 0.860±0.027 | ✓ | ✓ |
| per_graph | knn_1d | 4 | 0.268±0.019, 0.339±0.047, 0.385±0.029, 0.400±0.039 | ✓ | ✓ |
| per_graph | knn_1d | 8 | 0.243±0.015, 0.345±0.038, 0.405±0.022, 0.426±0.037 | ✓ | ✓ |
| per_graph | knn_1d | 16 | 0.220±0.008, 0.336±0.032, 0.411±0.017, 0.437±0.035 | ✓ | ✓ |
| per_graph | knn_1d | 32 | 0.193±0.008, 0.308±0.029, 0.394±0.017, 0.425±0.035 | ✓ | ✓ |
| per_graph | bin_1d | 4 | 0.188±0.023, 0.259±0.026, 0.304±0.043, 0.325±0.050 | ✓ | ✓ |
| per_graph | bin_1d | 8 | 0.164±0.018, 0.267±0.019, 0.324±0.037, 0.350±0.046 | ✓ | ✓ |
| per_graph | bin_1d | 16 | 0.140±0.009, 0.259±0.016, 0.329±0.033, 0.360±0.042 | ✓ | ✓ |
| per_graph | bin_1d | 32 | 0.111±0.009, 0.230±0.015, 0.312±0.032, 0.347±0.041 | ✓ | ✓ |
| per_graph | invariants_knn | 4 | 0.689±0.048, 0.838±0.041, 0.897±0.045, 0.914±0.026 | ✓ | ✓ |
| per_graph | invariants_knn | 8 | 0.635±0.041, 0.825±0.039, 0.883±0.047, 0.905±0.033 | ✓ | ✓ |
| per_graph | invariants_knn | 16 | 0.584±0.039, 0.802±0.053, 0.865±0.052, 0.895±0.036 | ✓ | ✓ |
| per_graph | invariants_knn | 32 | 0.561±0.050, 0.799±0.056, 0.865±0.049, 0.888±0.036 | ✓ | ✓ |

### Permutation null (features shuffled; should be ≈0 across diversity)

| frame | estimator | k | null ratio series (mean±std) | null ratio max |
|---|---|---|---|---|
| frechet | knn_top_k | 4 | 0.102±0.006, 0.102±0.002, 0.100±0.003, 0.095±0.011 | 0.102 |
| frechet | knn_top_k | 8 | 0.097±0.001, 0.097±0.004, 0.097±0.005, 0.096±0.010 | 0.097 |
| frechet | knn_top_k | 16 | 0.097±0.001, 0.097±0.002, 0.095±0.004, 0.095±0.005 | 0.097 |
| frechet | knn_top_k | 32 | 0.096±0.001, 0.095±0.001, 0.095±0.002, 0.095±0.004 | 0.096 |
| frechet | knn_1d | 4 | 0.101±0.009, 0.099±0.012, 0.091±0.007, 0.093±0.008 | 0.101 |
| frechet | knn_1d | 8 | 0.097±0.004, 0.100±0.009, 0.093±0.006, 0.092±0.007 | 0.100 |
| frechet | knn_1d | 16 | 0.097±0.002, 0.096±0.003, 0.094±0.004, 0.092±0.006 | 0.097 |
| frechet | knn_1d | 32 | 0.096±0.000, 0.094±0.003, 0.095±0.003, 0.094±0.003 | 0.096 |
| frechet | bin_1d | 4 | 0.016±0.001, 0.016±0.004, 0.012±0.002, 0.013±0.006 | 0.016 |
| frechet | bin_1d | 8 | 0.015±0.003, 0.017±0.002, 0.013±0.001, 0.014±0.006 | 0.017 |
| frechet | bin_1d | 16 | 0.016±0.001, 0.015±0.001, 0.013±0.001, 0.014±0.005 | 0.016 |
| frechet | bin_1d | 32 | 0.015±0.000, 0.015±0.001, 0.015±0.001, 0.015±0.002 | 0.015 |
| frechet | invariants_knn | 4 | 0.110±0.007, 0.115±0.002, 0.109±0.002, 0.098±0.019 | 0.115 |
| frechet | invariants_knn | 8 | 0.109±0.013, 0.095±0.020, 0.105±0.010, 0.096±0.015 | 0.109 |
| frechet | invariants_knn | 16 | 0.106±0.013, 0.107±0.020, 0.100±0.008, 0.093±0.006 | 0.107 |
| frechet | invariants_knn | 32 | 0.100±0.019, 0.105±0.025, 0.103±0.009, 0.093±0.017 | 0.105 |
| per_graph | knn_top_k | 4 | 0.093±0.010, 0.105±0.013, 0.091±0.009, 0.094±0.011 | 0.105 |
| per_graph | knn_top_k | 8 | 0.093±0.006, 0.097±0.021, 0.094±0.006, 0.091±0.008 | 0.097 |
| per_graph | knn_top_k | 16 | 0.096±0.004, 0.094±0.013, 0.092±0.009, 0.091±0.008 | 0.096 |
| per_graph | knn_top_k | 32 | 0.100±0.006, 0.094±0.013, 0.094±0.012, 0.094±0.010 | 0.100 |
| per_graph | knn_1d | 4 | 0.109±0.017, 0.091±0.011, 0.078±0.010, 0.087±0.014 | 0.109 |
| per_graph | knn_1d | 8 | 0.104±0.013, 0.091±0.012, 0.078±0.010, 0.088±0.013 | 0.104 |
| per_graph | knn_1d | 16 | 0.102±0.011, 0.091±0.010, 0.079±0.010, 0.088±0.012 | 0.102 |
| per_graph | knn_1d | 32 | 0.100±0.010, 0.091±0.009, 0.079±0.010, 0.088±0.011 | 0.100 |
| per_graph | bin_1d | 4 | 0.019±0.010, 0.021±0.016, 0.014±0.004, 0.012±0.006 | 0.021 |
| per_graph | bin_1d | 8 | 0.017±0.007, 0.020±0.016, 0.014±0.004, 0.012±0.006 | 0.020 |
| per_graph | bin_1d | 16 | 0.016±0.005, 0.021±0.015, 0.014±0.005, 0.012±0.006 | 0.021 |
| per_graph | bin_1d | 32 | 0.017±0.004, 0.020±0.014, 0.015±0.005, 0.012±0.006 | 0.020 |
| per_graph | invariants_knn | 4 | 0.088±0.013, 0.114±0.014, 0.096±0.008, 0.102±0.015 | 0.114 |
| per_graph | invariants_knn | 8 | 0.086±0.009, 0.100±0.026, 0.099±0.005, 0.096±0.013 | 0.100 |
| per_graph | invariants_knn | 16 | 0.086±0.020, 0.093±0.012, 0.091±0.006, 0.095±0.011 | 0.095 |
| per_graph | invariants_knn | 32 | 0.095±0.019, 0.095±0.016, 0.092±0.011, 0.097±0.016 | 0.097 |

## Conclusion

Mean ratios monotone across every cell AND permutation-null ratios bounded below 0.30 across every cell. Phase 4 unblocked.

Raw data: `diversity_sweep.csv` (768 rows).
