# `discrete_enzymes_pearl_repro` / `ge461v1o`

**Launched:** 2026-05-05 15:42 UTC
**Ended:** 2026-05-06 10:00 UTC (collateral kill — runtime 18.29h, step 398249)
**Status:** crashed (container caught up in the `_gnnconv_raw` cancellation; W&B heartbeat timed out shortly after; final state `crashed`)
**Resumed by:** [`vejeny0f`](../../run_details/2026-05-06/discrete_enzymes_pearl_repro_vejeny0f_details.md) — fresh container picks up from `last.ckpt` at step ~395k via `force_fresh=false`.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_repro` |
| wandb_project | `discrete-enzymes-pearl-repro` |
| run_id | `ge461v1o` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_repro/ge461v1o/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-enzymes-pearl-repro/runs/ge461v1o> |

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

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/pyg_enzymes.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. ENZYMES: paper has no ENZYMES — column blank.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.1857 | 2.9976e-04 | 619.49 | 13.34 | n/a |
| clustering | 0.1084 | 1.0443e-02 | 10.38 | 7.95 | n/a |
| orbit      | 0.1980 | 1.7318e-04 | 1143.31 | 11.55 | n/a |
| spectral   | 0.2004 | 2.8479e-03 | 70.37 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- none yet.

## Notes

R-PEARL extra features, plain Linear Q/K/V.

**Cross-dataset finding:** PEARL's orbit-MMD benefit on SBM (0.095 vs vignac's 0.142) does *not* transfer to enzymes (0.198 vs vignac's 0.119 — regression). Hypothesis: PEARL's GNN-on-random-features encoding captures SBM's block-mixing pattern but not enzyme tertiary-structure motifs. Spectral attention recovers the enzyme orbit gain — see `discrete_enzymes_pearl_spectral_repro_4n28svrj` (orbit 0.097).

This is the first concrete signal that PEARL is dataset-dependent. Worth re-checking once eval cycles converge and once MMD ratios are computed.

**Collateral kill 2026-05-06 ≈09:35 UTC.** The `_gnnconv_raw` enzymes container (`dt0ux9zh`) and this run's container were both at 17:40 CEST in `modal container list` — when the user stopped the divergent `_gnnconv_raw` containers, this run's container was caught in the same SIGINT pass. W&B continued reporting `state=running` for ≈25 min until the heartbeat finally timed out. Modal then reassigned the function call's input to a fresh container, which produced [`bhqss75w`](../../run_details/2026-05-06/discrete_enzymes_pearl_repro_bhqss75w_details.md) — a `force_fresh=true` restart that *discarded* this run's 18.3h of progress. `bhqss75w` was killed shortly after; the call was finally relaunched with `force_fresh=false` and the original `run_id`, producing [`vejeny0f`](../../run_details/2026-05-06/discrete_enzymes_pearl_repro_vejeny0f_details.md), which is properly resuming from this run's `last.ckpt`.
