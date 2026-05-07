# `discrete_enzymes_vignac_repro` / `l1nk0622`

**Launched:** 2026-05-05 15:42 UTC
**Ended:** 2026-05-06 11:10 UTC
**Status:** **finished** — hit target `trainer/global_step=550000`. Runtime 70085s ≈ 19.47h. First completed enzymes panel run.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_vignac_repro` |
| wandb_project | `discrete-enzymes-vignac-repro` |
| run_id | `l1nk0622` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_vignac_repro/l1nk0622/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-enzymes-vignac-repro/runs/l1nk0622> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 11:56 UTC (run finished); no parquet/checkpoint download yet.
- **local_path:** —

## Diagnostics

> Live W&B summary at completion, runtime 70085s. Run hit `trainer/global_step=550000` cleanly. Final eval at step 524999.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1901 | HiGen-reported DiGress = 0.004 → ~10⁴× gap |
| clustering MMD² | 0.1175 | |
| orbit MMD² | 0.0896 | |
| spectral MMD² | 0.2078 | |
| MMD ratios | _pending — needs `data/eval/mmd_baselines/pyg_enzymes.json`_ | resolves whether the gap is undertraining or kernel/sigma mismatch (PICKUP doc Task 2/3) |
| train_loss_epoch | 0.985 | |
| val_NLL | 419.48 | |
| mean_step_kl | 0.0101 | |
| grad_norm_total | 0.212 | healthy |
| effective_lr | 2.42e-07 | healthy |
| epoch | 158 | small dataset → many steps per epoch; epochs not directly comparable to SBM |
| global_step | 550000 | hit target |
| step_time_s | 0.162 | small graphs let fast variant cruise |

**Health:** ✓ stable; reached target step count.

### Per-step MMD trajectory

> History pulled via `get_run_history_tool` 2026-05-06 11:56 UTC. Eval cadence is every 75k steps. Trajectory is essentially flat — variant plateaus at the panel-floor MMDs by the very first eval cycle.

| step | degree | clustering | orbit | spectral |
|-----:|-------:|-----------:|------:|---------:|
| 75k  | 0.1842 | 0.1198 | 0.0743 | 0.1990 |
| 150k | 0.1797 | 0.1076 | 0.1152 | 0.1932 |
| 225k | 0.1899 | 0.1168 | 0.0538 | 0.2028 |
| 300k | 0.1822 | 0.1029 | 0.0946 | 0.2012 |
| 375k | 0.1868 | 0.1108 | 0.1188 | 0.2048 |
| 450k | 0.1857 | 0.1344 | 0.0818 | 0.2084 |
| 525k | 0.1901 | 0.1175 | 0.0896 | 0.2078 |

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/pyg_enzymes.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. ENZYMES: paper has no ENZYMES — column blank.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.1868 | 2.9976e-04 | 623.16 | 13.34 | n/a |
| clustering | 0.1108 | 1.0443e-02 | 10.61 | 7.95 | n/a |
| orbit      | 0.1188 | 1.7318e-04 | 685.98 | 11.55 | n/a |
| spectral   | 0.2048 | 2.8479e-03 | 71.91 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- none yet — no enzyme repro report exists; this would be the place to start.

## Notes

Vignac baseline on enzymes. **ENZYMES has no upstream DiGress config** — config is our best-effort reconstruction; hyperparameters are not paper-anchored.

The 10⁴× MMD² gap vs HiGen's reported numbers is the central open question: either we are massively undertrained or the kernel/sigma differs from HiGen's. PICKUP doc Task 2 (compute-baseline + ratio scoring) and Task 3 (kernel/sigma audit) will resolve.
