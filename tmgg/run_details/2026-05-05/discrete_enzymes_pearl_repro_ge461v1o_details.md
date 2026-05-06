# `discrete_enzymes_pearl_repro` / `ge461v1o`

**Launched:** 2026-05-05 15:42 UTC
**Status:** running (heartbeat 2026-05-06 07:12 UTC; runtime 56133s ≈ 15.6h at last query)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_repro` |
| wandb_project | `discrete-enzymes-pearl-repro` |
| run_id | `ge461v1o` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_repro/ge461v1o/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-repro/runs/ge461v1o> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 56133s.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1857 | within noise of vignac (0.187) |
| clustering MMD² | 0.1084 | |
| orbit MMD² | **0.1980** | **worse than vignac (0.119) — opposite sign to SBM** |
| spectral MMD² | 0.2004 | |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.980 | |
| val_NLL | 431.02 | |
| mean_step_kl | 0.0101 | |
| grad_norm_total | 0.189 | healthy |
| effective_lr | 2.99e-07 | healthy |
| epoch | 97 | |
| global_step | 340299 | |
| step_time_s | 0.147 | |

**Health:** ✓ stable.

## Visuals

- none yet.

## Notes

R-PEARL extra features, plain Linear Q/K/V.

**Cross-dataset finding:** PEARL's orbit-MMD benefit on SBM (0.095 vs vignac's 0.142) does *not* transfer to enzymes (0.198 vs vignac's 0.119 — regression). Hypothesis: PEARL's GNN-on-random-features encoding captures SBM's block-mixing pattern but not enzyme tertiary-structure motifs. Spectral attention recovers the enzyme orbit gain — see `discrete_enzymes_pearl_spectral_repro_4n28svrj` (orbit 0.097).

This is the first concrete signal that PEARL is dataset-dependent. Worth re-checking once eval cycles converge and once MMD ratios are computed.
