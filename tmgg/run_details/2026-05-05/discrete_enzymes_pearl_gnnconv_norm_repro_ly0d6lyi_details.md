# `discrete_enzymes_pearl_gnnconv_norm_repro` / `ly0d6lyi` (failed, attempt 2/3)

**Launched:** 2026-05-05 20:06 UTC
**Status:** failed at 2026-05-06 07:04 UTC (runtime 39503s ≈ 11h, final step 250749)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_norm_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-norm-repro` |
| run_id | `ly0d6lyi` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_norm_repro/ly0d6lyi/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-gnnconv-norm-repro/runs/ly0d6lyi> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 39503s (= total run; this is the at-failure snapshot).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1839 | within enzymes panel norm |
| clustering MMD² | 0.1103 | |
| orbit MMD² | 0.1477 | |
| spectral MMD² | 0.2031 | |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.982 | |
| val_NLL | 430.79 | matches healthy enzyme runs |
| mean_step_kl | 0.0102 | |
| grad_norm_total | 0.294 | healthy |
| effective_lr | 2.44e-07 | healthy |
| epoch | 71 | |
| global_step | 250749 | |
| step_time_s | 0.152 | |

**Health:** ✓ stable at the point of failure — diagnostics show no numerical issue.

## Visuals

- none.

## Notes

Failure was **not** a numerical blow-up — gradients and lr healthy, MMDs in line with the rest of the enzymes panel. Cause must be something else: preempt, OOM at eval, Modal volume hiccup, or other infra-side issue. **Pull `modal app logs tmgg-spectral` for `ly0d6lyi` to confirm before relying on the latest restart `zyawhwrx`.**

This is attempt 2 of 3 in the `discrete-enzymes-pearl-gnnconv-norm-repro` project.

See sibling files:
- `run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_txfr1vms_details.md` — first failed attempt (4.4h)
- `run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_zyawhwrx_details.md` — third (running) attempt
