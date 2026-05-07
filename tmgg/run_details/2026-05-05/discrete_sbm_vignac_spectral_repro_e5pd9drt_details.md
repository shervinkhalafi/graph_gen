# `discrete_sbm_vignac_spectral_repro` / `e5pd9drt`

**Launched:** 2026-05-05 21:22 UTC
**Status:** running (heartbeat 2026-05-06 11:56 UTC; runtime 52407s ≈ 14.6h at last query). **First eval cycle landed at step 75k.**

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_spectral_repro` |
| wandb_project | `discrete-sbm-vignac-spectral-repro` |
| run_id | `e5pd9drt` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_spectral_repro/e5pd9drt/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-sbm-vignac-spectral-repro/runs/e5pd9drt> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below; no parquet/checkpoint download yet.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 52407s, mid-training. **First eval cycle landed at step 75000.**

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.2251 | _highest_ on SBM panel at step 75k (vignac=0.200, pearl=0.185) |
| clustering MMD² | 0.1300 | matches SBM panel floor (~0.13) |
| orbit MMD² | 0.0842 | _best_ on SBM panel at step 75k (vignac=0.108, pearl=0.107) |
| spectral MMD² | 0.2125 | matches panel band ~0.21 |
| MMD ratios | _pending — needs `data/eval/mmd_baselines/spectre_sbm.json`_ | |
| train_loss_epoch | 0.780 | |
| val_NLL | 5388.46 | |
| mean_step_kl | 0.00850 | |
| grad_norm_total | 0.107 | healthy |
| effective_lr | 2.56e-07 | healthy |
| epoch | 9074 | |
| global_step | 90749 | |
| step_time_s | **0.563** | **~3× slower than the Linear-Q/K/V variants (~0.18s)** |

**Health:** ✓ stable; throughput-disadvantaged.

### Per-step MMD trajectory

| step | degree | clustering | orbit | spectral |
|-----:|-------:|-----------:|------:|---------:|
| 75k  | 0.2251 | 0.1300 | 0.0842 | 0.2125 |

## Visuals

- none yet (no completed eval cycles)

## Notes

Vignac extra-features baseline + `SpectralProjectionLayer` Q/K/V (k=16, 3 polynomial terms, normalised eigenvalues). Tests whether spectral attention is additive to Vignac's hand-crafted spectral extra features.

**Confirmed launched** (the file mtime alone wasn't enough; this run resolves the question raised in earlier runlog versions).

**Throughput consequence:** `SpectralProjectionLayer` recomputes the Laplacian eigendecomposition every step. At ~0.56s/step vs ~0.18s, 24h yields ≈ 153k steps here vs ≈ 430k for the Linear variants. **Cross-variant comparisons must control for step count, not wall-clock** — at 24h wall-clock this run will be ~3× under-trained relative to the others.

The val_NLL 3416 is suspiciously good given mid-training, but with no eval cycle yet to corroborate. Hold judgment until first MMDs land.
