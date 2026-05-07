# `discrete_sbm_vignac_repro` / `12s2b4a7`

**Launched:** 2026-05-04 16:23 UTC
**Status:** crashed at 2026-05-05 16:21 UTC (runtime 86245s ≈ 23h57m — likely Modal 24h timeout)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_vignac_repro` |
| wandb_project | `discrete-sbm-vignac-repro` |
| run_id | `12s2b4a7` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_vignac_repro/12s2b4a7/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-sbm-vignac-repro/runs/12s2b4a7> |

## Fetched

- **status:** partial — summary parquet only, no checkpoints
- **local_path:** `wandb_export/sbm-repro-report-2026-05-05/data/vignac/`

## Diagnostics

> Snapshot from `wandb_export/sbm-repro-report-2026-05-05/data/vignac/summary.json`. Snapshot runtime 66579s ≈ 18.5h (so the snapshot is from ~5h before the run actually ended; final step in W&B is 430119, snapshot is at 330619).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² (gen-val) | 0.1848 | DiGress paper ratio reference: 1.6 |
| clustering MMD² | 0.1326 | |
| orbit MMD² | 0.1424 | |
| spectral MMD² | 0.2126 | |
| MMD ratios | _pending_ | needs `data/eval/mmd_baselines/spectre_sbm.json` (PICKUP doc Task 2) |
| train_loss_epoch | 0.7748 | |
| val_NLL | 4418.83 | |
| mean_step_kl | 0.00856 | |
| grad_norm_total | 0.071 | healthy |
| effective_lr | 5.5e-08 | small but stable |
| epoch | 33061 | |
| global_step (snapshot) | 330619 | final 430119 |
| step_time_s | 0.182 | ~5.5 steps/s |

**Health:** ✓ stable.

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/spectre_sbm.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. Verbatim from arXiv:2209.14734v3 Table 1 SBM row.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.1848 | 3.4125e-04 | 541.53 | 3.81 | 1.60 |
| clustering | 0.1326 | 3.3118e-02 | 4.00 | 1.50 | 1.50 |
| orbit      | 0.1424 | 3.0991e-02 | 4.59 | 1.40 | 1.70 |
| spectral   | 0.2126 | 2.8182e-03 | 75.44 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- `wandb_export/sbm-repro-report-2026-05-05/figures/` — cross-variant comparison panels; this run is the "vignac" column.
- `wandb_export/sbm-repro-report-2026-05-05/report.typ:131` — step-equal tables for val_NLL and the four MMDs at common eval steps (5k–210k for NLL; 75k, 150k for MMDs).

## Notes

Faithful Vignac/GDPO SBM recipe; anchor for the SBM panel — every other variant defaults from this. Earlier debugging iterations (Apr 27 – Apr 30, 13 runs in the same project) include one full 5h finished run `3ftqoz4y` (2026-04-29 22:15 → 2026-04-30 03:15 UTC, step 275000); see W&B project for the rest.
