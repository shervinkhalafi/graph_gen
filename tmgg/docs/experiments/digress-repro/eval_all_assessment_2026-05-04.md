# DiGress repro eval-all aggregation — 2026-05-04

Pulled from Modal volume `tmgg-outputs` and aggregated into
`eval_all_aggregate_2026-05-04.csv` (41 rows). Trajectory plots:
`{qm9,moses,guacamol}_trajectory_2026-05-04.png`.

## Coverage

| dataset  | ckpts evaluated | configured max_steps | actually trained to | wall-clock |
|---       |---              |---                   |---                  |---         |
| QM9      | 25 (steps 1k–25k) | 50,000             | 25,000 (50%)        | 76 min     |
| MOSES    | 10 (steps 5k–100k) | 200,000           | 100,000 (50%)       | 6h 28m     |
| GuacaMol | 6 (steps 10k–250k) | 500,000           | 250,000 (50%)       | 10h 17m    |

QM9 ckpts only carry `validity / uniqueness / novelty` because the
QM9 eval-all dump predates the `_size_distribution` patch
(commit `4229fe3e`); the val-pass leg crashed and only the cheap
sampler-side metrics survived. The 20 `failed` rows in QM9
`index.jsonl` are also that bug. Re-running QM9 eval-all on the
post-patch CLI would fill in the molecular-distribution columns.

## Final-checkpoint comparison vs DiGress paper

### QM9 — step 25k of configured 50k

| metric | ours | DiGress paper | gap |
|---|---|---|---|
| validity | **1.000** | 0.990 | saturated |
| uniqueness | 0.998 | 0.962 | matched (slightly above) |
| novelty | 0.996 | (not reported) | — |

Validity, uniqueness, and novelty are not diagnostic on QM9 — the
model class can satisfy QM9 atom/bond constraints trivially, and
ours saturates from step 1k onwards.

### MOSES — step 100k of configured 200k

| metric | ours | DiGress paper | gap |
|---|---|---|---|
| validity | 1.000 | 0.857 | saturated |
| uniqueness | 1.000 | 1.000 | matched |
| novelty | 1.000 | 0.950 | matched |
| **FCD** ↓ | **20.80** | **1.19** | **17.5× worse (raw FCD)** |
| SNN ↑ | 0.272 | 0.520 | 1.9× worse |
| filters ↑ | 0.565 | 0.920 | half fail PAINS/MCF |
| IntDiv ↑ | 0.873 | 0.882 | matched |
| scaffold_novelty | 0.995 | (sign-flipped, not directly comparable) | — |

FCD is dropping monotonically across training (33.47 → 20.80) and
shows no plateau, so the gap to paper would shrink with more
training. Per-step trajectory: 5k=33.47, 25k=24.01, 70k=21.15,
100k=20.80.

### GuacaMol — step 250k of configured 500k

| metric | ours | DiGress paper | gap |
|---|---|---|---|
| validity | 1.000 | 0.852 | saturated |
| uniqueness | 1.000 | 1.000 | matched |
| novelty | 1.000 | 0.999 | matched |
| **FCD-ChEMBL score** ↑ (=`exp(-raw/5)`) | **0.076** | **0.680** | **9× worse** |
| KL Div ↑ | 0.834 | 0.929 | 11% short |
| FCD-ChEMBL raw ↓ | 12.87 | (paper reports score 0.68 → raw≈1.93) | 6.7× higher |

FCD raw is dropping sharply (23.11 → 12.87 over steps 10k → 250k);
KL_div is climbing (0.65 → 0.83). Both still mid-trajectory.

## Interpretation

1. **Validity / uniqueness / novelty saturate fast and are not
   diagnostic** — ours hits ceiling on all three datasets by step 1k.
2. **FCD, SNN, KL_div show the real story.** Distributionally far
   from paper at the truncated half-config endpoint — but in every
   metric our trajectory is still actively descending toward the
   paper anchor with no plateau visible. There is no evidence the
   model has converged; it stopped because something cut training
   early.
3. **The 50%-cutoff is consistent across all three datasets**, which
   makes a single root cause more likely than three coincident
   timeouts. Modal per-call timeout was already at 24 h ceiling
   (`DEFAULT_TIMEOUTS["fast"] = 86400` in `app.py`), so that's not
   it. The next-most-likely cause is `max_epochs` hitting before
   `max_steps` — the runtimes (76 min QM9, 6h 28m MOSES, 10h 17m
   GuacaMol) imply per-step throughput that, at the configured
   batch size and dataset size, would deplete `max_epochs` exactly
   halfway through `max_steps`.
4. **Two open follow-ups:**
   - (a) Compare configured `max_steps` and `max_epochs` against the
     upstream `cvignac/DiGress` schedule. If `max_epochs` is the
     binding constraint, raise it (or remove it) and relaunch.
   - (b) Re-run QM9 eval-all on the post-`4229fe3e` CLI to fill in
     the molecular-distribution columns. With QM9 only trained to
     25k of 50k, the FCD/SNN values would still be diagnostic for
     "how close are we" the same way MOSES/GuacaMol are.

## Files

- `eval_all_aggregate_2026-05-04.csv` — long-format DataFrame: one
  row per (dataset, step) with all eight reported metric columns
  (NaN where the dataset family doesn't report that metric).
- `qm9_trajectory_2026-05-04.png` — validity/uniqueness/novelty
  vs step.
- `moses_trajectory_2026-05-04.png` — FCD, SNN, filters, IntDiv vs
  step (4-panel) with paper anchors as red dashed lines.
- `guacamol_trajectory_2026-05-04.png` — FCD-ChEMBL score, KL_div
  vs step with paper anchors.
