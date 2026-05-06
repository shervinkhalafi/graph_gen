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

## Visuals

- none yet.

## Notes

R-PEARL features + `SpectralProjectionLayer` Q/K/V. **"Spectral" here is the attention projection, not the features** — same naming caveat as the SBM-side variant.

**Most interesting finding so far:** spectral attention recovers the orbit-MMD gain on enzymes that plain PEARL loses. Plain `pearl_repro` regresses orbit on enzymes (0.198 vs vignac 0.119); adding `SpectralProjectionLayer` Q/K/V brings orbit back to 0.097 — better than vignac. Suggests the spectral *attention* signal is what matters on enzymes, not the eigh-based features per se.

The smaller-graph environment (enzymes graphs are ~5-50 nodes vs SBM's ~150) may be why the spectral-attention overhead is tolerable here (step_time 0.139s vs SBM's 0.565s).

Hold final judgement until eval cycles converge.
