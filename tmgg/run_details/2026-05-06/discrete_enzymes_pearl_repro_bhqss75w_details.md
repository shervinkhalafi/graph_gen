# `discrete_enzymes_pearl_repro` / `bhqss75w` (collateral force-fresh restart, killed)

**Launched:** 2026-05-06 10:01 UTC (Modal auto-reassign after `ge461v1o` was killed as collateral)
**Ended:** 2026-05-06 10:09 UTC (killed; runtime 0.13h ≈ 8 min, step 3599, no eval cycle)
**Status:** failed (killed by user via container stop on Modal web UI; one-step path was wrong — `force_fresh=true` was discarding `ge461v1o`'s 18.3h of progress)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_repro` |
| wandb_project | `discrete-enzymes-pearl-repro` |
| run_id | `bhqss75w` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_repro/discrete_enzymes_pearl_repro_DiffusionModule_dGraphDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260505T154023/` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-repro/runs/bhqss75w> |

## Notes

This run is the **un-wanted side effect** of stopping the `_gnnconv_raw` containers. `ge461v1o` (a healthy run, started 2026-05-05 15:42 UTC) and the divergent `dt0ux9zh` had containers at the same start time slot in `modal container list`; the user's `modal container stop` on the divergent containers caught `ge461v1o` in the same SIGINT pass. Modal then auto-reassigned `ge461v1o`'s function call input to a fresh container, which started this run.

Because `force_fresh=True` is the launcher default, the reassign discarded `ge461v1o`'s `last.ckpt` (step 395k) and started from step 0 — wasting 18.3h of training. The user noticed and killed `bhqss75w` after 8 min.

Subsequently relaunched as [`vejeny0f`](discrete_enzymes_pearl_repro_vejeny0f_details.md) with `force_fresh=false` and `+run_id=<original-ge461v1o-name>`, which correctly resumed from `last.ckpt`.

**Lesson recorded:** when stopping a Modal container, the function-call input is reassigned by default. If the original launcher used `force_fresh=true`, the reassign starts from scratch — losing the killed run's progress. To stop a run cleanly without losing checkpoints, cancel at the function-call level (Modal web UI → app → function → call → cancel), not the container level.
