# `discrete_sbm_pearl_gnnconv_norm_repro` / `rarihsee`

**Launched:** 2026-05-04 22:09 UTC
**Status:** crashed at 2026-05-05 22:05 UTC (runtime 86150s ≈ 23h56m — likely 24h timeout)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_gnnconv_norm_repro` |
| wandb_project | `discrete-sbm-pearl-gnnconv-norm-repro` |
| run_id | `rarihsee` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_gnnconv_norm_repro/rarihsee/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-gnnconv-norm-repro/runs/rarihsee> |

## Fetched

- **status:** partial — summary parquet only, no checkpoints
- **local_path:** `wandb_export/sbm-repro-report-2026-05-05/data/pearl-gnnconv-norm/`

## Diagnostics

> Snapshot from `wandb_export/sbm-repro-report-2026-05-05/data/pearl-gnnconv-norm/summary.json`. Snapshot runtime 45860s ≈ 12.7h (final W&B step 443079, snapshot at 234799).

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1838 | within noise of vignac/pearl |
| clustering MMD² | 0.1318 | best in panel |
| orbit MMD² | 0.1186 | between vignac (0.142) and pearl (0.095) |
| spectral MMD² | 0.2170 | |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.776 | |
| val_NLL | 5278.17 | worst non-divergent NLL in panel |
| mean_step_kl | 0.00851 | |
| grad_norm_total | **1.489** | **~20× larger than the other healthy variants** |
| effective_lr | 2.47e-07 | within healthy range |
| epoch | 23479 | |
| global_step (snapshot) | 234799 | final 443079 |
| step_time_s | 0.188 | |

**Health:** ⚠ training succeeded end-to-end but grad_norm is elevated (1.49 vs ~0.08 elsewhere). Worth probing per-layer `grad_norm/*` and `grad_cosine/*` from W&B history before declaring this variant fully clean.

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/spectre_sbm.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. Verbatim from arXiv:2209.14734v3 Table 1 SBM row.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.1838 | 3.4125e-04 | 538.60 | 3.81 | 1.60 |
| clustering | 0.1318 | 3.3118e-02 | 3.98 | 1.50 | 1.50 |
| orbit      | 0.1186 | 3.0991e-02 | 3.83 | 1.40 | 1.70 |
| spectral   | 0.2170 | 2.8182e-03 | 77.00 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- `wandb_export/sbm-repro-report-2026-05-05/figures/` — "pearl-gnnconv-norm" column.
- `wandb_export/sbm-repro-report-2026-05-05/report.typ:131` — step-equal tables.

## Notes

R-PEARL features + `BareGraphConv` Q/K/V (normalised adjacency). Distinct from the `_raw_` sister variant, which is numerically unstable. Despite the elevated grad_norm, MMDs remain in line with the other healthy variants — convergent and informative; flag, not abort.

The higher val_NLL (5278) versus similar MMDs (0.184 / 0.132 / 0.119 / 0.217) is interesting: model fits the noise schedule worse but generates samples of similar quality. Worth probing whether NLL is an honest progress signal here or whether `BareGraphConv` Q/K/V changes the variational lower bound's tightness.
