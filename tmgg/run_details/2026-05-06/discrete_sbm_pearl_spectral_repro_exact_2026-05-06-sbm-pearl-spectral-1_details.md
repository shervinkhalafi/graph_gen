# `discrete_sbm_pearl_spectral_repro_exact` / `2026-05-06-sbm-pearl-spectral-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of SBM PEARL features + spectral attention arm.
Stacks spectral Q/K/V projections on top of the pearl exact config.

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_spectral_repro_exact` |
| wandb_project | `discrete-sbm-pearl-spectral-repro-exact` |
| run_id | `2026-05-06-sbm-pearl-spectral-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_spectral_repro_exact/2026-05-06-sbm-pearl-spectral-1/` |
| modal_function_call_id | `fc-01KQYRNZRZ0MD33VYHM6TDCB7T` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-sbm-pearl-spectral-1` in <https://wandb.ai/<TEAM-ENTITY>/discrete-sbm-pearl-spectral-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=4400`.
> Spectral Q/K/V costs ~3× per step (eigh per diffusion step inside
> attention), so eval cadence is expensive on top of training. Plan to
> override `model.eval_every_n_steps=75000` on a future re-launch if
> total wall-time budget binds.

## Notes

Re-launched after the `mask_zero_diag` fix. The buggy panel run for
this variant was `jbraoj7o` (SBM panel: best orbit r=2.47 at 375k —
arm with the strongest signal in the buggy panel). The post-fix run
will tell us whether the orbit edge holds under correct architecture.
Preempt-resume enabled.
