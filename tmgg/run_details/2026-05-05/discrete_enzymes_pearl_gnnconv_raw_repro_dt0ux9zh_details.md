# `discrete_enzymes_pearl_gnnconv_raw_repro` / `dt0ux9zh` ⚠ blew up — killed

**Launched:** 2026-05-05 15:42 UTC
**Ended:** 2026-05-06 09:36 UTC (killed — runtime 17.90h, step 438799)
**Status:** failed (container stopped via `modal container stop` 2026-05-06 ≈09:35 UTC; final W&B state `failed`)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_raw_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-raw-repro` |
| run_id | `dt0ux9zh` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_raw_repro/dt0ux9zh/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-enzymes-pearl-gnnconv-raw-repro/runs/dt0ux9zh> |

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

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/pyg_enzymes.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. ENZYMES: paper has no ENZYMES — column blank.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.2113 | 2.9976e-04 | 704.89 | 13.34 | n/a |
| clustering | 0.0942 | 1.0443e-02 | 9.02 | 7.95 | n/a |
| orbit      | 0.5380 | 1.7318e-04 | 3106.56 | 11.55 | n/a |
| spectral   | 0.1954 | 2.8479e-03 | 68.61 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- none — would not be diagnostic on a divergent run.

## Notes

R-PEARL features + `BareGraphConv` Q/K/V (raw, un-normalised adjacency).

**Confirms the cross-dataset finding from the SBM side:** un-normalised `BareGraphConv` Q/K/V is numerically unstable. Both SBM runs (`qao36vwu`, `g1g6xpx1`) and this enzymes run all show grad_norm in the 1e3–1e18 range and effective_lr at 1e-3 or higher. The `_norm_` sister variant uses normalised adjacency and stays stable on both datasets.

The clustering MMD 0.094 looks superficially good, but is meaningless on a run that has clearly diverged on the other metrics — do not cite.

**Action:** kill, fix (normalisation + gradient clipping + lower lr), or document-as-known-broken. See `runlog.md` "Backfill checklist" item 8.

**Killed via `modal container stop` 2026-05-06 09:36 UTC** to free Modal capacity. The function call's input was reassigned once to `b7lqqac8` (failed at step 1849 in 6 min) before the call was cancelled at the function-call level via the Modal web UI. Unlike the SBM-side `_raw_` chain (which spawned three reassigns before the call was cancelled), the enzymes side stopped cleanly after one reassign.
