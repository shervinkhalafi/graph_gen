# `discrete_enzymes_pearl_spectral_repro` / `4n28svrj`

**Launched:** 2026-05-05 15:37 UTC
**Status:** running (heartbeat 2026-05-06 11:56 UTC; runtime 73129s ≈ 20.3h at last query). 6 eval cycles logged through step 450k.

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_spectral_repro` |
| wandb_project | `discrete-enzymes-pearl-spectral-repro` |
| run_id | `4n28svrj` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_spectral_repro/4n28svrj/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-spectral-repro/runs/4n28svrj> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 73129s. **Latest eval at step 450k shows orbit MMD² regressed from 0.097 → 0.176** — orbit is volatile across cycles (see trajectory below). Use min-over-cycles, not last-cycle, for headline.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1881 | matches panel floor |
| clustering MMD² | 0.1152 | |
| orbit MMD² | 0.1757 (latest) / **0.0674 (min @ 75k)** | volatile cycle-to-cycle (range 0.067–0.186) |
| spectral MMD² | 0.1872 | best on enzymes panel |
| train_loss_epoch | 0.980 | |
| val_NLL | 453.36 | |
| mean_step_kl | 0.0101 | |
| grad_norm_total | 0.150 | healthy |
| effective_lr | 1.70e-07 | healthy |
| epoch | 132 | |
| global_step | 462549 | |
| step_time_s | 0.155 | not as slow as the SBM-side spectral variant — likely smaller graph size lets eigh stay cheap |

**Health:** ✓ stable; orbit volatility is a measurement-noise concern, not a training-stability one.

### Per-step MMD trajectory

| step | degree | clustering | orbit | spectral |
|-----:|-------:|-----------:|------:|---------:|
| 75k  | 0.1818 | 0.1021 | 0.0674 | 0.1821 |
| 150k | 0.1801 | 0.1159 | 0.0741 | 0.1835 |
| 225k | 0.1794 | 0.1104 | 0.1858 | 0.1924 |
| 300k | 0.1874 | 0.1082 | 0.0968 | 0.1965 |
| 375k | 0.1846 | 0.1123 | 0.0850 | 0.1858 |
| 450k | 0.1881 | 0.1152 | 0.1757 | 0.1872 |

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/pyg_enzymes.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. ENZYMES: paper has no ENZYMES — column blank.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.1874 | 2.9976e-04 | 625.16 | 13.34 | n/a |
| clustering | 0.1082 | 1.0443e-02 | 10.36 | 7.95 | n/a |
| orbit      | 0.0968 | 1.7318e-04 | 558.95 | 11.55 | n/a |
| spectral   | 0.1965 | 2.8479e-03 | 69.00 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- none yet.

## Notes

R-PEARL features + `SpectralProjectionLayer` Q/K/V. **"Spectral" here is the attention projection, not the features** — same naming caveat as the SBM-side variant.

**Most interesting finding so far:** spectral attention recovers the orbit-MMD gain on enzymes that plain PEARL loses. Plain `pearl_repro` regresses orbit on enzymes (0.198 vs vignac 0.119); adding `SpectralProjectionLayer` Q/K/V brings orbit back to 0.097 — better than vignac. Suggests the spectral *attention* signal is what matters on enzymes, not the eigh-based features per se.

The smaller-graph environment (enzymes graphs are ~5-50 nodes vs SBM's ~150) may be why the spectral-attention overhead is tolerable here (step_time 0.139s vs SBM's 0.565s).

Hold final judgement until eval cycles converge.
