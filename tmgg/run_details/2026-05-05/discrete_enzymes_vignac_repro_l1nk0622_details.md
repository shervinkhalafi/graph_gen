# `discrete_enzymes_vignac_repro` / `l1nk0622`

**Launched:** 2026-05-05 15:42 UTC
**Status:** running (heartbeat 2026-05-06 07:12 UTC; runtime 56178s ≈ 15.6h at last query)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_vignac_repro` |
| wandb_project | `discrete-enzymes-vignac-repro` |
| run_id | `l1nk0622` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_vignac_repro/l1nk0622/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-vignac-repro/runs/l1nk0622> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below; no parquet/checkpoint download yet.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 56178s.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1868 | HiGen-reported DiGress = 0.004 → ~10⁴× gap |
| clustering MMD² | 0.1108 | |
| orbit MMD² | 0.1188 | |
| spectral MMD² | 0.2048 | |
| MMD ratios | _pending — needs `data/eval/mmd_baselines/pyg_enzymes.json`_ | resolves whether the gap is undertraining or kernel/sigma mismatch (PICKUP doc Task 2/3) |
| train_loss_epoch | 0.975 | |
| val_NLL | 422.78 | |
| mean_step_kl | 0.0101 | |
| grad_norm_total | 0.399 | healthy |
| effective_lr | 3.06e-07 | healthy |
| epoch | 126 | small dataset → many steps per epoch; epochs not directly comparable to SBM |
| global_step | 442699 | |
| step_time_s | 0.105 | fastest in panel (small graphs) |

**Health:** ✓ stable.

## Visuals

- none yet — no enzyme repro report exists; this would be the place to start.

## Notes

Vignac baseline on enzymes. **ENZYMES has no upstream DiGress config** — config is our best-effort reconstruction; hyperparameters are not paper-anchored.

The 10⁴× MMD² gap vs HiGen's reported numbers is the central open question: either we are massively undertrained or the kernel/sigma differs from HiGen's. PICKUP doc Task 2 (compute-baseline + ratio scoring) and Task 3 (kernel/sigma audit) will resolve.
