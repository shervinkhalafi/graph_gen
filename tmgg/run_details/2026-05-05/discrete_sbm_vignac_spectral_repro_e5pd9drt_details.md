# `discrete_sbm_vignac_spectral_repro` / `e5pd9drt`

**Launched:** 2026-05-05 21:22 UTC
**Status:** running (heartbeat 2026-05-06 07:12 UTC; runtime 35731s ≈ 9.9h at last query)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_spectral_repro` |
| wandb_project | `discrete-sbm-vignac-spectral-repro` |
| run_id | `e5pd9drt` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_spectral_repro/e5pd9drt/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-vignac-spectral-repro/runs/e5pd9drt> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below; no parquet/checkpoint download yet.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 35731s, mid-training. **No eval cycle has run yet** — `gen-val/*_mmd` are all null at the snapshot moment.

| metric | value | comment |
|--------|------:|---------|
| MMDs | — | first eval cycle hasn't fired yet |
| MMD ratios | N/A | |
| train_loss_epoch | 0.895 | |
| val_NLL | 3416.16 | best val_NLL seen on SBM panel — note caveat below |
| mean_step_kl | null | |
| grad_norm_total | 0.261 | healthy |
| effective_lr | 2.67e-07 | healthy |
| epoch | 6287 | |
| global_step | 62879 | |
| step_time_s | **0.565** | **~3× slower than the Linear-Q/K/V variants (~0.18s)** |

**Health:** ✓ stable, but ✗ throughput-disadvantaged.

## Visuals

- none yet (no completed eval cycles)

## Notes

Vignac extra-features baseline + `SpectralProjectionLayer` Q/K/V (k=16, 3 polynomial terms, normalised eigenvalues). Tests whether spectral attention is additive to Vignac's hand-crafted spectral extra features.

**Confirmed launched** (the file mtime alone wasn't enough; this run resolves the question raised in earlier runlog versions).

**Throughput consequence:** `SpectralProjectionLayer` recomputes the Laplacian eigendecomposition every step. At ~0.56s/step vs ~0.18s, 24h yields ≈ 153k steps here vs ≈ 430k for the Linear variants. **Cross-variant comparisons must control for step count, not wall-clock** — at 24h wall-clock this run will be ~3× under-trained relative to the others.

The val_NLL 3416 is suspiciously good given mid-training, but with no eval cycle yet to corroborate. Hold judgment until first MMDs land.
