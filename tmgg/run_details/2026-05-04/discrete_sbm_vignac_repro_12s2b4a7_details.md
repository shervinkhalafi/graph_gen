# `discrete_sbm_vignac_repro` / `12s2b4a7`

**Launched:** 2026-05-04 16:23 UTC
**Status:** crashed at 2026-05-05 16:21 UTC (runtime 86245s ≈ 23h57m — likely Modal 24h timeout)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_repro` |
| wandb_project | `discrete-sbm-vignac-repro` |
| run_id | `12s2b4a7` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_repro/12s2b4a7/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-vignac-repro/runs/12s2b4a7> |

## Fetched

- **status:** partial — summary parquet only, no checkpoints
- **local_path:** `wandb_export/sbm-repro-report-2026-05-05/data/vignac/`

## Diagnostics

> Snapshot from `wandb_export/sbm-repro-report-2026-05-05/data/vignac/summary.json`. Snapshot runtime 66579s ≈ 18.5h (so the snapshot is from ~5h before the run actually ended; final step in W&B is 430119, snapshot is at 330619).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² (gen-val) | 0.1848 | DiGress paper ratio reference: 1.6 |
| clustering MMD² | 0.1326 | |
| orbit MMD² | 0.1424 | |
| spectral MMD² | 0.2126 | |
| MMD ratios | _pending_ | needs `data/eval/mmd_baselines/spectre_sbm.json` (PICKUP doc Task 2) |
| train_loss_epoch | 0.7748 | |
| val_NLL | 4418.83 | |
| mean_step_kl | 0.00856 | |
| grad_norm_total | 0.071 | healthy |
| effective_lr | 5.5e-08 | small but stable |
| epoch | 33061 | |
| global_step (snapshot) | 330619 | final 430119 |
| step_time_s | 0.182 | ~5.5 steps/s |

**Health:** ✓ stable.

## Visuals

- `wandb_export/sbm-repro-report-2026-05-05/figures/` — cross-variant comparison panels; this run is the "vignac" column.
- `wandb_export/sbm-repro-report-2026-05-05/report.typ:131` — step-equal tables for val_NLL and the four MMDs at common eval steps (5k–210k for NLL; 75k, 150k for MMDs).

## Notes

Faithful Vignac/GDPO SBM recipe; anchor for the SBM panel — every other variant defaults from this. Earlier debugging iterations (Apr 27 – Apr 30, 13 runs in the same project) include one full 5h finished run `3ftqoz4y` (2026-04-29 22:15 → 2026-04-30 03:15 UTC, step 275000); see W&B project for the rest.
