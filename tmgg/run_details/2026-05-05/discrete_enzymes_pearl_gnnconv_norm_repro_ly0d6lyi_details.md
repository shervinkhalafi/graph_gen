# `discrete_enzymes_pearl_gnnconv_norm_repro` / `ly0d6lyi` (failed, attempt 2/3)

**Launched:** 2026-05-05 20:06 UTC
**Status:** failed at 2026-05-06 07:04 UTC (runtime 39503s ≈ 11h, final step 250749)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_norm_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-norm-repro` |
| run_id | `ly0d6lyi` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_norm_repro/ly0d6lyi/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-enzymes-pearl-gnnconv-norm-repro/runs/ly0d6lyi> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 39503s (= total run; this is the at-failure snapshot).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1839 | within enzymes panel norm |
| clustering MMD² | 0.1103 | |
| orbit MMD² | 0.1477 | |
| spectral MMD² | 0.2031 | |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.982 | |
| val_NLL | 430.79 | matches healthy enzyme runs |
| mean_step_kl | 0.0102 | |
| grad_norm_total | 0.294 | healthy |
| effective_lr | 2.44e-07 | healthy |
| epoch | 71 | |
| global_step | 250749 | |
| step_time_s | 0.152 | |

**Health:** ✓ stable at the point of failure — diagnostics show no numerical issue.

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/pyg_enzymes.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. ENZYMES: paper has no ENZYMES — column blank.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.1839 | 2.9976e-04 | 613.48 | 13.34 | n/a |
| clustering | 0.1103 | 1.0443e-02 | 10.56 | 7.95 | n/a |
| orbit      | 0.1477 | 1.7318e-04 | 852.86 | 11.55 | n/a |
| spectral   | 0.2031 | 2.8479e-03 | 71.32 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- none.

## Notes

Failure was **not** a numerical blow-up — gradients and lr healthy, MMDs in line with the rest of the enzymes panel. Cause must be something else: preempt, OOM at eval, Modal volume hiccup, or other infra-side issue. **Pull `modal app logs tmgg-spectral` for `ly0d6lyi` to confirm before relying on the latest restart `zyawhwrx`.**

This is attempt 2 of 3 in the `discrete-enzymes-pearl-gnnconv-norm-repro` project.

See sibling files:
- `run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_txfr1vms_details.md` — first failed attempt (4.4h)
- `run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_zyawhwrx_details.md` — third (running) attempt
