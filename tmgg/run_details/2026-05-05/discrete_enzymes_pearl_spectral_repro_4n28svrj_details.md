# `discrete_enzymes_pearl_spectral_repro` / `4n28svrj`

**Launched:** 2026-05-05 15:37 UTC
**Status:** running (heartbeat 2026-05-06 07:12 UTC; runtime 56444s ≈ 15.7h at last query)

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

> Live W&B summary, runtime 56444s.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1874 | |
| clustering MMD² | 0.1082 | |
| orbit MMD² | **0.0968** | best in enzymes panel; beats vignac (0.119) and pearl-only (0.198) |
| spectral MMD² | 0.1965 | best on enzymes panel |
| MMD ratios | _pending_ | |
| train_loss_epoch | 0.985 | |
| val_NLL | 455.69 | |
| mean_step_kl | 0.0101 | |
| grad_norm_total | 0.173 | healthy |
| effective_lr | 1.73e-07 | healthy |
| epoch | 102 | |
| global_step | 358299 | |
| step_time_s | 0.139 | not as slow as the SBM-side spectral variant — likely smaller graph size lets eigh stay cheap |

**Health:** ✓ stable.

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
