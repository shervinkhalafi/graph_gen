# `discrete_enzymes_pearl_repro_exact` / `2026-05-06-enzymes-pearl-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of ENZYMES PEARL features arm.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_repro_exact` |
| wandb_project | `discrete-enzymes-pearl-repro-exact` |
| run_id | `2026-05-06-enzymes-pearl-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_repro_exact/2026-05-06-enzymes-pearl-1/` |
| modal_function_call_id | `fc-01KQYRPK5RP70YGEZ795ASDM7K` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-enzymes-pearl-1` in <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=75000`.

## Notes

Re-launched after the `mask_zero_diag` fix. Buggy panel runs:
`ge461v1o` (original; killed as collateral) and `vejeny0f` (resume of
ge461v1o; killed under the bug fix work). The post-fix run is a clean
re-start under the corrected architecture; preempt-resume enabled.
