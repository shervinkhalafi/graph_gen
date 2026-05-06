# `discrete_enzymes_pearl_gnnconv_norm_repro_exact` / `2026-05-06-enzymes-pearl-gnnconv-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of ENZYMES PEARL features + BareGraphConv
(normalized adjacency) attention arm.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_norm_repro_exact` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-norm-repro-exact` |
| run_id | `2026-05-06-enzymes-pearl-gnnconv-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_norm_repro_exact/2026-05-06-enzymes-pearl-gnnconv-1/` |
| modal_function_call_id | `fc-01KQYRPZKRW6PJHP1QDMKKVDAY` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-enzymes-pearl-gnnconv-1` in <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-gnnconv-norm-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=75000`.

## Notes

Re-launched after the `mask_zero_diag` fix. Buggy panel runs:
`txfr1vms` and `ly0d6lyi` (both infra-failed) plus `zyawhwrx` (only
75k cycle, never matured). Post-fix run is the first real chance at
a mature ENZYMES gnnconv-norm trajectory. Preempt-resume enabled.
