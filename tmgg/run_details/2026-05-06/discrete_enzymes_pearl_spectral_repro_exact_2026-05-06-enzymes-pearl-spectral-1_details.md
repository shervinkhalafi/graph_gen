# `discrete_enzymes_pearl_spectral_repro_exact` / `2026-05-06-enzymes-pearl-spectral-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of ENZYMES PEARL features + spectral attention arm.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_spectral_repro_exact` |
| wandb_project | `discrete-enzymes-pearl-spectral-repro-exact` |
| run_id | `2026-05-06-enzymes-pearl-spectral-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_spectral_repro_exact/2026-05-06-enzymes-pearl-spectral-1/` |
| modal_function_call_id | `fc-01KQYRPSDQAY1P5BAW1Y1ABYFD` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-enzymes-pearl-spectral-1` in <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-spectral-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=75000`.
> Spectral Q/K/V costs ~3× per step.

## Notes

Re-launched after the `mask_zero_diag` fix. The buggy panel run for
this variant was `4n28svrj` (best orbit on enzymes panel: r=390 best
at 75k). Post-fix run validates whether the spectral-attention orbit
edge survives the architecture correction. Preempt-resume enabled.
