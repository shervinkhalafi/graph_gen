# `discrete_enzymes_pearl_repro` / `vejeny0f` (resumed from `ge461v1o` checkpoint)

**Launched:** 2026-05-06 10:17 UTC
**Status:** running (heartbeat 2026-05-06 10:23 UTC; runtime 0.10h ≈ 6 min, step 396899)
**Resumes:** [`ge461v1o`](../../2026-05-05/discrete_enzymes_pearl_repro_ge461v1o_details.md) — picked up from `last.ckpt` at step ~395k. Effectively continues the 18.3h of training that was almost lost when `bhqss75w` started fresh.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_repro` |
| wandb_project | `discrete-enzymes-pearl-repro` |
| run_id | `vejeny0f` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_repro/discrete_enzymes_pearl_repro_DiffusionModule_dGraphDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260505T154023/` |
| modal_function_call_id | `fc-01KQYCHAQKH30SEY7PDPSWA9G8` |
| modal_container_id (initial) | `ta-01KQYCHB9P1Q697RZ8PXFK973E` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-repro/runs/vejeny0f> |

## Launch invocation

```
PRECISION=bf16-mixed scripts/run-digress-repro-modal.zsh enzymes-pearl \
  force_fresh=false \
  +run_id=discrete_enzymes_pearl_repro_DiffusionModule_dGraphDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260505T154023
```

Two specific overrides matter:

- `force_fresh=false` (override of the project default `True`) — stops Hydra from appending a new `_fresh_<UTC>` suffix to the run_id and re-creating an empty directory.
- `+run_id=<the original ge461v1o run name>` — the `+` prefix is required because `run_id` is not in the base config; Hydra needs to *append* it. Forces the output directory to exactly the existing one with `last.ckpt` inside.

Together these route Lightning's `_find_last_checkpoint(checkpoint_dir)` at
`run_experiment.py:424` to the existing `last.ckpt`, and pass it through to
`trainer.fit(model, data_module, ckpt_path=last_ckpt)` at line 472.

Confirmed working from the container's startup log:

```
2026-05-06 10:17:18.887 | INFO | tmgg.training.orchestration.run_experiment:run_experiment:467 - Resuming from checkpoint: /data/outputs/discrete_enzymes_pearl_repro/discrete_enzymes_pearl_repro_DiffusionModule_dGraphDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260505T154023/checkpoints/last.ckpt
```

W&B step at first log: 396899 — exactly where `ge461v1o` left off (step 398249 was the last logged W&B value, but the latest checkpoint was at step 395k; resuming to step ~397k after 6 min of additional training).

## Diagnostics

> Will fill in once a full eval cycle has fired post-resume. The
> 75000-step `eval_every_n_steps` cadence inherited from
> `discrete_sbm_official` means the next eval lands at step 450000
> (≈8h after resume).

| metric | value | comment |
|--------|------:|---------|
| MMDs | _pre-resume snapshot in [`ge461v1o`'s detail file](../../2026-05-05/discrete_enzymes_pearl_repro_ge461v1o_details.md)_ | next refresh at step 450k |
| Loss / gradient health | _pre-resume snapshot in `ge461v1o`_ | |
| Step counts | epoch≈100, global_step=396899 (resumed) | |

**Health:** ✓ resumed cleanly. The W&B run is brand-new (different internal id than `ge461v1o`) but logs continue from step ~395k, so the W&B step axis has a discontinuity around that point (visible as a fresh run starting near step 395k rather than 0).

## Anchor comparison

> _Pending_ — anchor comparison from this run will look identical to `ge461v1o`'s until 75k more steps land. See [`ge461v1o`'s Anchor comparison](../../2026-05-05/discrete_enzymes_pearl_repro_ge461v1o_details.md#anchor-comparison) for the pre-resume baseline.

## Visuals

- none yet.

## Notes

This run rescued 18.3h of training that was nearly discarded by an
inadvertent `force_fresh=True` reassign (see
[`bhqss75w`'s notes](discrete_enzymes_pearl_repro_bhqss75w_details.md))
after a Modal container stop on the `_gnnconv_raw` divergent containers
caught `ge461v1o`'s container in the same SIGINT pass.

The relaunch is a paper-clean continuation of the same training trajectory:
same hyperparameters, same data, same seed, same checkpoint state.
Only the W&B run id and the Modal function call id have changed.
