# `discrete_sbm_vignac_repro_exact` / `lptjvfbe`

**Launched:** 2026-05-06 09:39:58 UTC
**Cancelled:** 2026-05-06 (Modal call `fc-01KQYAH4XVTE4F505EEC0DCMNK` cancelled via `scripts.sweep.kill_call`).
**Status:** cancelled — invalidated by `_GraphTransformer.forward` divergence from upstream DiGress.

## Kill reason

`src/tmgg/models/digress/transformer_model.py:834` calls `.mask_zero_diag()` on the post-`mlp_in_E` hidden activation, zeroing the hidden E diagonal at the entry to the transformer stack. Upstream DiGress (`digress-upstream-readonly/src/models/transformer_model.py:268`) calls `.mask(node_mask)` at this point — padding-only, hidden diagonal preserved until the final output mask. The MLP's bias term puts non-zero values on the hidden diagonal which upstream carries through every `XEyTransformerLayer` (consumed by `e_mul(E)`, `e_add(E)`, `Etoy(E)`, FiLM y→E); tmgg zeroes them, so every layer's E-derived computation diverges from upstream even with identical weights. State-dict comparison at `dim_ffy=2048` measured max diffs `X=0.0087496, E=0.0010284`; monkey-patching `mask_zero_diag` → upstream-style `mask` made outputs exactly equal.

The `_repro_exact` config aims for byte-for-byte parity with Vignac's published SBM run, which the divergence prevents. Continuing the run would burn compute on a model that is not numerically equivalent to upstream DiGress. Re-launch after the one-line fix at `transformer_model.py:834` lands.

## Pre-cancel snapshot

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_repro_exact` |
| wandb_project | `discrete-sbm-vignac-repro-exact` |
| run_id | `lptjvfbe` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_repro_exact/discrete_sbm_vignac_repro_exact_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s0_fresh_20260506T093958/` |
| modal_function_call_id | `fc-01KQYAH4XVTE4F505EEC0DCMNK` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-sbm-vignac-repro-exact/runs/lptjvfbe> |

## Fetched

- **status:** no — run just started; nothing to pull yet.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 8101s. **No eval cycle has logged `gen-val/*_mmd` yet despite step 15249 ≫ 4400** — investigate whether `check_val_every_n_epoch=100` is gating eval (1524 epochs / 100 = 15 evals expected, but maybe eval triggers val/NLL only and sample-generation/MMD eval rides a different cadence). val/NLL has logged: epoch_NLL=5106.65, val/loss=0.808.

| metric | value | comment |
|--------|------:|---------|
| MMDs | _none logged yet_ | step 15249 — investigate gating |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.935 | |
| val_NLL | 5106.65 | val cycles ARE running — just no MMD logged yet |
| grad_norm_total | 0.0364 | healthy (lowest on SBM panel) |
| effective_lr | 2.24e-07 | healthy |
| epoch | 1524 | |
| global_step | 15249 | |
| step_time_s | 0.514 | _slower than expected_ — typical Linear-Q/K/V SBM step is ~0.18s |

**Health:** ✓ healthy training; ⚠ MMD eval cadence not yet observed.

## Anchor comparison

> _Pending_ — fill in once gen-val MMDs land. The expected outcome:
> SBM clustering and orbit converge to within ~2× of DiGress paper
> r=1.5/1.7 (matching `discrete_sbm_vignac_repro` panel), and degree
> drops from ~540× towards paper's r=1.6 if the dim_ffy=2048 →
> dim_ffy=256 + amsgrad=true → false changes were the reason for the
> SBM degree blow-up.

## Visuals

- none yet.

## Notes

This is the **paper-anchor parity** variant of the Vignac SBM
reproduction. Differs from `discrete_sbm_vignac_repro` (the
GDPO-aligned variant) on five documented items:

- `dim_ffy=256` (was 2048; matched `gdpo_sbm.ckpt`'s actual trained shape).
- `seed=0` (was 666; GDPO's seed).
- `amsgrad=false` (was true; PyTorch default applies, matching upstream
  `train_default.yaml` which doesn't set `amsgrad`).
- `check_val_every_n_epoch=100` and `eval_every_n_steps=4400` (was
  step-based 5000 / 75000; matches upstream `experiment/sbm.yaml`'s
  `check_val_every_n_epochs=100` and `sample_every_val=4`).
- Run completion target: 50000 epochs (≈550000 steps) — same as the
  GDPO-aligned variant. Need to lift the 24h Modal timeout to actually
  hit this number; current runs cap at ~430k steps.

**Known platform-side deviation from upstream:** `trainer.precision=
bf16-mixed` is the launcher default
(`scripts/run-digress-repro-modal.zsh`). Upstream DiGress uses fp32
(no precision setting in `train_default.yaml`). To rerun in true fp32,
relaunch with `PRECISION=32-true scripts/run-digress-repro-modal.zsh
sbm-vignac-exact`. The bf16-mixed deviation applies to *every* repro
run in the panel; it is not specific to this `_exact` variant.

See [`docs/eval/2026-05-06-mmd-ratio-analysis.md`](../../docs/eval/2026-05-06-mmd-ratio-analysis.md)
"GDPO reference" subsection for the historical context on the
`_repro` vs `_repro_exact` split.
