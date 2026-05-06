# `discrete_sbm_vignac_repro_exact` / `lptjvfbe`

**Launched:** 2026-05-06 09:39:58 UTC
**Status:** running (just spawned; first eval cycle ≈4400 steps away)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_repro_exact` |
| wandb_project | `discrete-sbm-vignac-repro-exact` |
| run_id | `lptjvfbe` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_repro_exact/discrete_sbm_vignac_repro_exact_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s0_fresh_20260506T093958/` |
| modal_function_call_id | `fc-01KQYAH4XVTE4F505EEC0DCMNK` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-vignac-repro-exact/runs/lptjvfbe> |

## Fetched

- **status:** no — run just started; nothing to pull yet.
- **local_path:** —

## Diagnostics

> Run just started; no eval cycle yet. First MMD snapshot expected at
> step 4400 (≈14 minutes of training at the SBM panel's typical 0.18 s
> step time). Re-snapshot diagnostics here once the first eval lands.

| metric | value | comment |
|--------|------:|---------|
| MMDs | _pending_ | first eval at step 4400 |
| MMD ratios | _pending_ | needs MMDs first |
| Loss / gradient health | _pending_ | |
| Step counts | epoch=0, global_step≪4400 | |

**Health:** ✓ healthy at startup; will revisit once eval lands.

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
