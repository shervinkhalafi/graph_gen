# `discrete_sbm_pearl_repro_exact` / `2026-05-06-sbm-pearl-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of SBM PEARL features arm. Inherits the four
upstream-parity deltas from `discrete_sbm_vignac_repro_exact`
(`dim_ffy=2048`, `amsgrad=false`, `seed=0`,
`use_upstream_hidden_edge_diagonal=true`).

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_repro_exact` |
| wandb_project | `discrete-sbm-pearl-repro-exact` |
| run_id | `2026-05-06-sbm-pearl-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_repro_exact/2026-05-06-sbm-pearl-1/` |
| modal_function_call_id | `fc-01KQYRNSEK23MRX1ATF9T3BMYZ` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-sbm-pearl-1` in <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=4400`.

## Notes

Re-launched after the `mask_zero_diag` fix. The buggy panel run for
this variant was `s07qwx3b` (SBM panel: orbit r=2.72 best at 150k);
this run is the architecture-clean replacement under the post-fix
forward path. Preempt-resume enabled.
