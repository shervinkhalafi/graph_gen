# `discrete_sbm_pearl_gnnconv_raw_repro` / `g6y8ubfg` (auto-reassign #3, ≪10min — chain end)

**Launched:** 2026-05-06 09:52 UTC (Modal auto-reassign after `bepjqwqz` was stopped)
**Ended:** 2026-05-06 09:59 UTC (crashed; runtime 0.11h ≈ 7 min, step 1219, no eval cycle)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_gnnconv_raw_repro` |
| wandb_project | `discrete-sbm-pearl-gnnconv-raw-repro` |
| run_id | `g6y8ubfg` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_gnnconv_raw_repro/discrete_sbm_pearl_gnnconv_raw_repro_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260504T220549/` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-gnnconv-raw-repro/runs/g6y8ubfg> |

## Notes

Third (and final) link in the SBM `_gnnconv_raw` auto-reassign chain after `bepjqwqz`. The user cancelled the underlying Modal function call via the web UI shortly after this container started, terminating the chain. W&B state flipped from `running` to `crashed` once the heartbeat timed out. No further reassigns.

Chain trail:
1. [`qao36vwu`](../../2026-05-04/discrete_sbm_pearl_gnnconv_raw_repro_qao36vwu_details.md) — original launch 2026-05-04 22:10 UTC, blew up.
2. [`g1g6xpx1`](../../2026-05-05/discrete_sbm_pearl_gnnconv_raw_repro_g1g6xpx1_details.md) — relaunch 2026-05-05 18:56 UTC, blew up, killed 2026-05-06 09:36 UTC.
3. `uuifd9v3` (this dir) — auto-reassign #1.
4. `bepjqwqz` (this dir) — auto-reassign #2.
5. `g6y8ubfg` (this entry) — auto-reassign #3, chain ended via call cancellation.
