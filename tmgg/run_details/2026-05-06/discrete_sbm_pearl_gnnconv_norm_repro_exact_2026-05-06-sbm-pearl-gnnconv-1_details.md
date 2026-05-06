# `discrete_sbm_pearl_gnnconv_norm_repro_exact` / `2026-05-06-sbm-pearl-gnnconv-1`

**Launched:** 2026-05-06 13:46 UTC
**Status:** running (preempt-resume enabled via `force_fresh=false`).

Post-fix re-launch of SBM PEARL features + BareGraphConv (normalized
adjacency) attention arm. Same expressive class as the spectral
variant but no eigh during attention.

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_gnnconv_norm_repro_exact` |
| wandb_project | `discrete-sbm-pearl-gnnconv-norm-repro-exact` |
| run_id | `2026-05-06-sbm-pearl-gnnconv-1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_gnnconv_norm_repro_exact/2026-05-06-sbm-pearl-gnnconv-1/` |
| modal_function_call_id | `fc-01KQYRP5Z0SB93YEFW4Y6JSMZE` |
| gpu_tier | `fast` |
| W&B URL | search `2026-05-06-sbm-pearl-gnnconv-1` in <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-gnnconv-norm-repro-exact> |

## Fetched

- **status:** no — just launched.
- **local_path:** —

## Diagnostics

> Pending — first eval cycle at `eval_every_n_steps=4400`.

## Notes

Re-launched after the `mask_zero_diag` fix. The buggy panel run for
this variant was `rarihsee` (SBM panel: indistinguishable from
vanilla DiGress on degree/cluster/orbit/spectral within ~2% — strong
"matches vanilla" candidate). The post-fix run validates whether the
parity claim holds under correct architecture. Preempt-resume enabled.
