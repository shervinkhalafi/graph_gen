# `discrete_enzymes_vignac_repro_exact` / `2026-05-06-enzymes-vignac-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of ENZYMES vanilla DiGress baseline. There is no
upstream Vignac ENZYMES experiment, so "exact" here means the same
model-side parity choices as the SBM exact variant applied on top of
the existing ENZYMES data + cadence pipeline.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_vignac_repro_exact` |
| wandb_project | `discrete-enzymes-vignac-repro-exact` |
| run_id | `2026-05-06-enzymes-vignac-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_vignac_repro_exact/2026-05-06-enzymes-vignac-1/` |
| modal_function_call_id | `fc-01KQYRPC8G81QSZ9QQ8E05RCPC` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-enzymes-vignac-1` in <https://wandb.ai/graph_denoise_team/discrete-enzymes-vignac-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=75000` per
> ENZYMES panel cadence.

## Notes

Re-launched after the `mask_zero_diag` fix. The buggy panel run for
this variant was `l1nk0622` (the only finished ENZYMES run; orbit
r=311 best in buggy panel). The post-fix run replaces it as the
architecture-clean baseline. Preempt-resume enabled.
