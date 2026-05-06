# `discrete_enzymes_pearl_gnnconv_norm_repro` / `zyawhwrx` (running, attempt 3/3)

**Launched:** 2026-05-06 07:07 UTC
**Status:** running (heartbeat 2026-05-06 07:12 UTC; runtime 654s ≈ 11 min at last query)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_norm_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-norm-repro` |
| run_id | `zyawhwrx` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_norm_repro/zyawhwrx/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-enzymes-pearl-gnnconv-norm-repro/runs/zyawhwrx> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 654s. Too fresh — no eval cycle, no full epoch yet.

| metric | value | comment |
|--------|------:|---------|
| MMDs | — | not logged yet |
| Loss | train_epoch=null, val_NLL=null | not logged yet |
| mean_step_kl | null | |
| grad_norm_total | 0.316 | healthy |
| effective_lr | **1.50e-06** | ~10× higher than the other healthy enzyme runs (~3e-7); could be warmup, or a config tweak between attempts. **Verify.** |
| epoch | 0 | |
| global_step | 3299 | |
| step_time_s | 0.168 | |

**Health:** ✓ healthy at startup, but the elevated effective_lr is worth checking against `discrete_enzymes_pearl_gnnconv_norm_repro.yaml` to confirm intent.

## Visuals

- none yet.

## Notes

Third attempt at `discrete_enzymes_pearl_gnnconv_norm_repro` — relaunched ~3 minutes after the second attempt `ly0d6lyi` failed. **Diagnostics from `ly0d6lyi` show no numerical instability at the failure point, so the cause is infra-side.** Without pulling `modal app logs tmgg-spectral` for the previous failures, this restart may hit the same issue.

**Pre-flight check before relying on this run:**
1. Pull `modal app logs tmgg-spectral` for `txfr1vms` (4.4h) and `ly0d6lyi` (11h) to identify the failure cause.
2. Verify `effective_lr=1.50e-6` is intentional vs. `~3e-7` elsewhere.

See sibling files:
- `run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_txfr1vms_details.md`
- `run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_ly0d6lyi_details.md`
