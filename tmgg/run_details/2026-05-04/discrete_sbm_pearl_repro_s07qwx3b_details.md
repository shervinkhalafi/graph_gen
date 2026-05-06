# `discrete_sbm_pearl_repro` / `s07qwx3b`

**Launched:** 2026-05-04 21:38 UTC
**Status:** crashed at 2026-05-05 21:36 UTC (runtime 86256s ≈ 23h58m — likely 24h timeout)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_repro` |
| wandb_project | `discrete-sbm-pearl-repro` |
| run_id | `s07qwx3b` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_repro/s07qwx3b/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-repro/runs/s07qwx3b> |

## Fetched

- **status:** partial — summary parquet only, no checkpoints
- **local_path:** `wandb_export/sbm-repro-report-2026-05-05/data/pearl/`

## Diagnostics

> Snapshot from `wandb_export/sbm-repro-report-2026-05-05/data/pearl/summary.json`. Snapshot runtime 47708s ≈ 13.2h (final W&B step 513549, snapshot at 282509).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1856 | within noise of vignac |
| clustering MMD² | 0.1339 | |
| orbit MMD² | 0.0946 | better than vignac (0.142) — possible PEARL benefit |
| spectral MMD² | 0.2115 | |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.8394 | |
| val_NLL | 4285.47 | best val_NLL in the SBM panel |
| mean_step_kl | 0.00853 | |
| grad_norm_total | 0.089 | healthy |
| effective_lr | 1.06e-07 | healthy |
| epoch | 28250 | |
| global_step (snapshot) | 282509 | final 513549 |
| step_time_s | 0.214 | |

**Health:** ✓ stable.

## Visuals

- `wandb_export/sbm-repro-report-2026-05-05/figures/` — "pearl" column.
- `wandb_export/sbm-repro-report-2026-05-05/report.typ:131` — step-equal tables.

## Notes

R-PEARL `ExtraFeatures` swap (no eigh, GNN-on-random-features), plain Linear Q/K/V. Cleanest improvement signal in the SBM panel: orbit MMD ~0.10 vs vignac's ~0.14 at the same wall-clock; spectral / clustering / degree are all within noise.

One earlier crash in the same project on 2026-05-04 21:27 UTC: `gzyithje`, ran 1 minute, no W&B summary. Likely an init failure; not analysed.
