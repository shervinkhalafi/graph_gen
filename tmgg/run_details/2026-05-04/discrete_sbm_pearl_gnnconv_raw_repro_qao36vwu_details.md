# `discrete_sbm_pearl_gnnconv_raw_repro` / `qao36vwu` ⚠ blew up

**Launched:** 2026-05-04 22:10 UTC
**Status:** crashed at 2026-05-05 18:51 UTC (runtime 74505s ≈ 20.7h — *not* the 24h timeout signature)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_gnnconv_raw_repro` |
| wandb_project | `discrete-sbm-pearl-gnnconv-raw-repro` |
| run_id | `qao36vwu` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_gnnconv_raw_repro/qao36vwu/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-gnnconv-raw-repro/runs/qao36vwu> |

## Fetched

- **status:** partial — summary parquet only, no checkpoints. Identity ambiguous: `_runtime` 45839s in the local export does *not* match either run's W&B-reported runtime (`qao36vwu`: 74505s; `g1g6xpx1`: 44190s+). May be from `g1g6xpx1` mid-flight, or from this run captured before crash. Verify from manifest before citing in any report.
- **local_path:** `wandb_export/sbm-repro-report-2026-05-05/data/pearl-gnnconv-raw/`

## Diagnostics

> Snapshot source ambiguous (see above). Treating as `qao36vwu`-end-state for the purpose of this entry, but flag.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.2991 | far worse than the rest of the panel (≈0.184) |
| clustering MMD² | 0.1355 | |
| orbit MMD² | 0.1426 | |
| spectral MMD² | 0.2265 | |
| MMD ratios | _pending — but irrelevant for divergent runs_ | |
| train_loss_epoch | 0.986 | worst in panel |
| val_NLL | 6886.35 | worst in panel |
| mean_step_kl | 0.00895 | |
| grad_norm_total | **2.43e+18** | ⚠ ~22 orders of magnitude above healthy |
| effective_lr | **Infinity** | ⚠ optimizer state corrupted |
| epoch | 23134 | |
| global_step (snapshot) | 231349 | final 375639 |
| step_time_s | 0.170 | |

**Health:** ✗ blew up. Together with the second run `g1g6xpx1` (also blew up) and the enzymes-side `dt0ux9zh` (also blew up), the entire `_raw_` family is unviable as currently configured.

## Visuals

- `wandb_export/sbm-repro-report-2026-05-05/figures/` — "pearl-gnnconv-raw" column. Caveat: figures may show pre-divergence state, not end-of-run; treat as illustrative only.

## Notes

R-PEARL features + `BareGraphConv` Q/K/V (raw, un-normalised adjacency). The hypothesis for the blow-up: degree-mismatched eigenvalue blow-up → unbounded grad → NaN/Inf in optimizer state. Compare to the `_norm_` sister variant `rarihsee` which trained cleanly.

**Do not include this run in cross-variant comparisons.** Either fix the variant (add normalisation, gradient clipping, lower lr) or drop it — see `runlog.md` "Open questions" and "Backfill checklist" item 8.
