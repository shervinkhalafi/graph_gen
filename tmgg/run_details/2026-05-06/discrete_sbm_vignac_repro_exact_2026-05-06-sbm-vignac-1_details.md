# `discrete_sbm_vignac_repro_exact` / `2026-05-06-sbm-vignac-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-`mask_zero_diag`-fix re-launch of the SBM vanilla DiGress paper-anchor
parity reference. Replaces the killed `lptjvfbe` run.

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_repro_exact` |
| wandb_project | `discrete-sbm-vignac-repro-exact` |
| run_id | `2026-05-06-sbm-vignac-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_repro_exact/2026-05-06-sbm-vignac-1/` |
| modal_function_call_id | `fc-01KQYRNJ9S0STFDZBEPFBY5WZQ` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-sbm-vignac-1` in <https://wandb.ai/graph_denoise_team/discrete-sbm-vignac-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=4400` per the
> exact-cadence inheritance from `discrete_sbm_vignac_repro_exact`.

## Notes

Re-launched under the post-fix architecture (commit `f4f9665a` enabled
via `use_upstream_hidden_edge_diagonal=true` in the inherited config),
with the named-resume mechanism (commit `44c1d5fd`): preempted restarts
of this exact command will resume from `last.ckpt` and append to the
same W&B run via `wandb_run_id.txt` sidecar.

Cancellation of the prior `lptjvfbe` was driven by the diagonal-mask
divergence; see [`run_details/2026-05-06/discrete_sbm_vignac_repro_exact_lptjvfbe_details.md`](discrete_sbm_vignac_repro_exact_lptjvfbe_details.md).
