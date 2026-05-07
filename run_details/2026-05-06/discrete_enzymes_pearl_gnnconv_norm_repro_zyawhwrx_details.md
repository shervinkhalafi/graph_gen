# `discrete_enzymes_pearl_gnnconv_norm_repro` / `zyawhwrx` (running, attempt 3/3)

**Launched:** 2026-05-06 07:07 UTC
**Status:** running (heartbeat 2026-05-06 11:56 UTC; runtime 17337s ≈ 4.8h at last query). **First eval cycle landed at step 75k.**

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_norm_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-norm-repro` |
| run_id | `zyawhwrx` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_norm_repro/zyawhwrx/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-enzymes-pearl-gnnconv-norm-repro/runs/zyawhwrx> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 17337s. **First eval cycle landed at step 75k.** The earlier-snapshot elevated effective_lr (1.50e-6) was warmup; now back to ~3e-7 as expected.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | 0.1810 | best _initial_ degree on enzymes panel (vignac=0.184, pearl-spectral=0.182, pearl=0.183) |
| clustering MMD² | 0.1040 | best _initial_ clustering on enzymes panel |
| orbit MMD² | 0.0760 | _equal-best initial_ orbit (pearl-spectral=0.067 at same step) |
| spectral MMD² | 0.1874 | mid-pack |
| train_loss_epoch | 0.982 | |
| val_NLL | 446.92 | |
| mean_step_kl | 0.0100 | |
| grad_norm_total | 0.472 | healthy (slightly elevated vs panel ~0.2 but no divergence) |
| effective_lr | 2.94e-07 | healthy — back in the panel band after warmup |
| epoch | 27 | |
| global_step | 96299 | |
| step_time_s | 0.154 | |

**Health:** ✓ stable; warmup-elevated lr resolved.

### Per-step MMD trajectory

| step | degree | clustering | orbit | spectral |
|-----:|-------:|-----------:|------:|---------:|
| 75k  | 0.1810 | 0.1040 | 0.0760 | 0.1874 |

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
