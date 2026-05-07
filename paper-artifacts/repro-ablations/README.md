# Repro-ablations bundle

Self-contained data + context bundle for the eight Table 2 cells in
the paper (4 variants × 2 datasets, single seed 666).

## What's here

| path | what's in it |
|------|--------------|
| `README.md` | this file |
| `context/` | how to interpret the numbers — baselines, anchors, MMD units |
| `context/mmd_baselines/` | train↔test MMD² JSONs (denominator for ratios) |
| `configs/digress_{sbm,enzymes}.yaml` | DiGress baseline hydra config per dataset |
| `configs/digress_pearl_{sbm,enzymes}.yaml` | + R-PEARL features |
| `configs/digress_pearl_spectral_{sbm,enzymes}.yaml` | + R-PEARL + Spectral attention |
| `configs/digress_pearl_gcat_{sbm,enzymes}.yaml` | + R-PEARL + GCAT attention |
| `data/runs_index.csv` | one row per run, identity + status + final step |
| `data/per_run_history/<run_id>.parquet` | full wandb history per run, wide |
| `data/DATA-DICTIONARY.md` | column docs + controlled vocabularies |
| `media/per_run/<run_id>/` | latest wandb-rendered adjacency / graph sample images |

## Units and conventions — MMD values are squared

Every value under the `mmd` metric namespace (and inside the
`gen-val/*_mmd` keys in the parquets) is a **squared MMD² value** —
the V-statistic biased estimator. Do not square-root before comparing
to DiGress paper Table 1 ratios or HiGen Table 1 raw values; both
publish MMD² too.

Full unit semantics, kernel choice, and bandwidth rationale:
`context/mmd-units-and-protocol.md`.

## Quick start

```python
import pandas as pd

idx = pd.read_csv("data/runs_index.csv")
print(idx[["run_id", "config_name", "final_state", "final_step"]])

# Per-run wandb history (training metrics, MMD evals, gradient health):
hist = pd.read_parquet("data/per_run_history/cgfv3f85.parquet")
mmd_cols = [c for c in hist.columns if c.startswith("gen-val/") and c.endswith("_mmd")]
print(hist[mmd_cols].dropna(how="all").tail())
```

## Caveats

1. **Single seed (666)** per cell, matching the paper's protocol.
2. **Eval cadence.** Each run logs `gen-val/*_mmd` at the start of every
   75 k-step eval cycle. Pivot the parquets on `trainer/global_step`.
3. **Orbit MMD is volatile cycle-to-cycle** (~3× swings within a run).
   Headline reports use the min or mean over multiple cycles.

## Cross-links

- Top-level reviewer README: `../../README.md`
- MMD units / protocol: `context/mmd-units-and-protocol.md`
- Paper anchors + ratio comparison: `context/ANCHORS.md`
