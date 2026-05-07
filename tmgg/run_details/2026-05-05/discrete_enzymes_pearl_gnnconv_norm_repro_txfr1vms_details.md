# `discrete_enzymes_pearl_gnnconv_norm_repro` / `txfr1vms` (failed, attempt 1/3)

**Launched:** 2026-05-05 15:42 UTC
**Status:** failed at 2026-05-05 20:04 UTC (runtime 15710s ≈ 4.4h, final step 103399)

## Identity

| field | value |
|-------|-------|
| config | `discrete_enzymes_pearl_gnnconv_norm_repro` |
| wandb_project | `discrete-enzymes-pearl-gnnconv-norm-repro` |
| run_id | `txfr1vms` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_enzymes_pearl_gnnconv_norm_repro/txfr1vms/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/<TEAM-ENTITY>/discrete-enzymes-pearl-gnnconv-norm-repro/runs/txfr1vms> |

## Fetched

- **status:** no — diagnostics not pulled (live W&B query was scoped to the most recent failed run `ly0d6lyi` and the latest restart `zyawhwrx`).
- **local_path:** —

## Diagnostics

> _Not pulled yet — only the W&B inventory metadata below is available. To get diagnostics, query `api.run("<TEAM-ENTITY>/discrete-enzymes-pearl-gnnconv-norm-repro/txfr1vms").summary` and merge into this file._

| metric | value |
|--------|------:|
| MMDs | _not pulled_ |
| Loss | _not pulled_ |
| Gradient health | _not pulled_ |
| epoch | _not pulled_ |
| global_step (final) | 103399 |
| runtime_s | 15710 (≈ 4.4h) |

**Health:** unknown — need to pull summary.

## Visuals

- none.

## Notes

First of three attempts at `discrete_enzymes_pearl_gnnconv_norm_repro` on 2026-05-05 / 06. The second attempt `ly0d6lyi` (~11h) also failed but with healthy gradients at the point of failure — strongly suggests an infra-side cause (preempt? OOM at eval? Modal volume hiccup?) rather than numerical instability. **Pull `modal app logs tmgg-spectral` for both `txfr1vms` and `ly0d6lyi` before drawing any conclusions or relying on the latest restart `zyawhwrx`.**

See sibling files:
- `run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_ly0d6lyi_details.md` — second failed attempt
- `run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_zyawhwrx_details.md` — third (running) attempt
