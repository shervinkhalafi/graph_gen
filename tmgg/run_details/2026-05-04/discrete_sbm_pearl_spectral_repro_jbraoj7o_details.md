# `discrete_sbm_pearl_spectral_repro` / `jbraoj7o`

**Launched:** 2026-05-04 21:52 UTC
**Status:** crashed at 2026-05-05 21:49 UTC (runtime 86214s ≈ 23h57m — likely 24h timeout)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_spectral_repro` |
| wandb_project | `discrete-sbm-pearl-spectral-repro` |
| run_id | `jbraoj7o` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_spectral_repro/jbraoj7o/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-spectral-repro/runs/jbraoj7o> |

## Fetched

- **status:** partial — summary parquet only, no checkpoints
- **local_path:** `wandb_export/sbm-repro-report-2026-05-05/data/pearl-spec/`

## Diagnostics

> Snapshot from `wandb_export/sbm-repro-report-2026-05-05/data/pearl-spec/summary.json`. Snapshot runtime 46884s ≈ 13.0h (final W&B step 402759, snapshot at 219099).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1867 | |
| clustering MMD² | 0.1342 | |
| orbit MMD² | 0.0989 | similar to plain pearl (0.095) |
| spectral MMD² | 0.2072 | |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.8949 | |
| val_NLL | 4613.35 | |
| mean_step_kl | 0.00852 | |
| grad_norm_total | 0.080 | healthy |
| effective_lr | 2.20e-07 | healthy |
| epoch | 21909 | |
| global_step (snapshot) | 219099 | final 402759 |
| step_time_s | 0.194 | |

**Health:** ✓ stable.

## Visuals

- `wandb_export/sbm-repro-report-2026-05-05/figures/` — "pearl-spec" column.
- `wandb_export/sbm-repro-report-2026-05-05/report.typ:131` — step-equal tables.

## Notes

R-PEARL features + `SpectralProjectionLayer` Q/K/V. The only PEARL variant that brings eigh back, but in attention rather than features. **Naming caveat:** "spectral" here is the attention projection, not the features.

Reading: spectral attention does *not* improve over plain-Linear PEARL on SBM — MMDs are within noise of `pearl_repro`. Suggests the eigh cost is not buying anything on SBM block structure that PEARL features alone don't already provide.

One earlier crash in the same project on 2026-05-04 21:07 UTC: `1gd2i9t4`, ran 2 minutes, no summary. Likely init failure; not analysed.
