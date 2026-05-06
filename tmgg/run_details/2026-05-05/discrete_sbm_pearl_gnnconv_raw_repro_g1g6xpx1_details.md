# `discrete_sbm_pearl_gnnconv_raw_repro` / `g1g6xpx1` ⚠ blew up

**Launched:** 2026-05-05 18:56 UTC
**Status:** running but numerically diverged (heartbeat 2026-05-06 07:12 UTC; runtime 44533s ≈ 12.4h at last query)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_gnnconv_raw_repro` |
| wandb_project | `discrete-sbm-pearl-gnnconv-raw-repro` |
| run_id | `g1g6xpx1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_gnnconv_raw_repro/g1g6xpx1/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-gnnconv-raw-repro/runs/g1g6xpx1> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 44533s.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | **0.4329** | far worse than panel norm (~0.184) |
| clustering MMD² | 0.1400 | |
| orbit MMD² | 0.1419 | |
| spectral MMD² | 0.2698 | |
| MMD ratios | _pending — irrelevant for divergent runs_ | |
| train_loss_epoch | 1.072 | |
| val_NLL | 6577.79 | second-worst on panel after `qao36vwu` |
| mean_step_kl | 0.00892 | |
| grad_norm_total | **2.16e+16** | ⚠ |
| effective_lr | **2.48e+9** | ⚠ optimizer state corrupted |
| epoch | 22811 | |
| global_step | 228119 | |
| step_time_s | 0.208 | |

**Health:** ✗ blew up — same divergence signature as the first run `qao36vwu` of this variant.

## Visuals

- none — would not be diagnostic on a divergent run.

## Notes

Second attempt at the `_raw_` SBM variant. First attempt `qao36vwu` (2026-05-04 22:10 → 2026-05-05 18:51 UTC) also blew up; this relaunch reproduces the same failure mode and on a worse trajectory (degree MMD 0.43 vs 0.30 on the first run).

**Together with the enzymes-side `dt0ux9zh`** — which also shows huge `grad_norm_total` (7600) and out-of-band `effective_lr` (1.10e-3) on the same `_raw_` config — the variant is unviable as configured. The `_norm_` sister variant (normalised adjacency) trains cleanly on both datasets; the issue is specifically the un-normalised Q/K/V projection.

**Action:** kill, fix (normalisation + gradient clipping + lower lr), or document-as-known-broken. See `runlog.md` "Backfill checklist" item 8.
