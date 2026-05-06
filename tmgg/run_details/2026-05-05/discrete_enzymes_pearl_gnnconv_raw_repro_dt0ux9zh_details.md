# `discrete_enzymes_pearl_gnnconv_raw_repro` / `dt0ux9zh` ⚠ blew up

**Launched:** 2026-05-05 15:42 UTC
**Status:** running but numerically diverged (heartbeat 2026-05-06 07:12 UTC; runtime 56145s ≈ 15.6h at last query)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_raw_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-raw-repro` |
| run_id | `dt0ux9zh` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_raw_repro/dt0ux9zh/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-gnnconv-raw-repro/runs/dt0ux9zh> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 56145s.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.2113 | worse than the four healthy enzyme variants (~0.187) |
| clustering MMD² | 0.0942 | best in enzymes panel — but on a divergent run |
| orbit MMD² | **0.5380** | ~5× worse than the healthy variants |
| spectral MMD² | 0.1954 | |
| MMD ratios | _irrelevant for divergent runs_ | |
| train_loss_epoch | 1.061 | |
| val_NLL | **1204.10** | ~3× worse than the healthy enzyme runs (~430) |
| mean_step_kl | 0.0106 | |
| grad_norm_total | **7600** | ⚠ orders of magnitude above the healthy ~0.2 |
| effective_lr | **1.10e-3** | ⚠ orders of magnitude above the healthy ~3e-7 |
| epoch | 109 | |
| global_step | 381599 | |
| step_time_s | 0.129 | |

**Health:** ✗ blew up — same divergence signature as the SBM-side runs `qao36vwu` and `g1g6xpx1`.

## Visuals

- none — would not be diagnostic on a divergent run.

## Notes

R-PEARL features + `BareGraphConv` Q/K/V (raw, un-normalised adjacency).

**Confirms the cross-dataset finding from the SBM side:** un-normalised `BareGraphConv` Q/K/V is numerically unstable. Both SBM runs (`qao36vwu`, `g1g6xpx1`) and this enzymes run all show grad_norm in the 1e3–1e18 range and effective_lr at 1e-3 or higher. The `_norm_` sister variant uses normalised adjacency and stays stable on both datasets.

The clustering MMD 0.094 looks superficially good, but is meaningless on a run that has clearly diverged on the other metrics — do not cite.

**Action:** kill, fix (normalisation + gradient clipping + lower lr), or document-as-known-broken. See `runlog.md` "Backfill checklist" item 8.
