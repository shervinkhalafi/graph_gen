# Bug Report: 19 Stage 1 Experiments Fail Before W&B Logging

**Date**: 2026-01-05
**Severity**: Low (data exists, only entity misrouting)
**Status**: Resolved - runs found in wrong entity (2026-01-05 15:30 CET)

## Summary

19 out of 108 Stage 1 experiments appeared to be missing from `graph_denoise_team/spectral_denoising`. Investigation revealed all 19 runs completed successfully but logged to `igorkraw/spectral_denoising` instead due to entity routing issues during the original batch launch.

## Affected Configurations

All 19 failures share common characteristics:

```
stage1_filter_bank_lr1e-3_wd1e-2_k16_s{1,2,3}
stage1_filter_bank_lr1e-3_wd1e-2_k8_s{1,2,3}
stage1_filter_bank_lr1e-3_wd1e-3_k16_s{1,2,3}
stage1_filter_bank_lr1e-3_wd1e-3_k8_s{1,2,3}
stage1_filter_bank_lr1e-4_wd1e-2_k16_s{1,2,3}
stage1_filter_bank_lr1e-4_wd1e-2_k8_s1
stage1_linear_pe_lr1e-4_wd1e-2_k8_s1
stage1_self_attention_lr1e-4_wd1e-2_k8_s{1,2}
```

### Pattern Analysis

| Category | Configs | Notes |
|----------|---------|-------|
| `filter_bank` + `lr1e-3` | 12 | Complete failure of this combination |
| `filter_bank` + `lr1e-4_wd1e-2` | 4 | Partial failure |
| `linear_pe` + `lr1e-4_wd1e-2_k8_s1` | 1 | Single failure |
| `self_attention` + `lr1e-4_wd1e-2_k8` | 2 | Seeds 1 and 2 only |

## Reproduction

1. Generate configs: `uv run python src/tmgg/modal/cli/generate_configs.py --stage stage1`
2. Launch all: `doppler run -- uv run python src/tmgg/modal/cli/launch_sweep.py --config-dir configs/stage1/2026-01-05 --gpu debug --wandb-entity graph_denoise_team --wandb-project spectral_denoising`
3. After completion, compare config count vs W&B run count

## Investigation Steps

1. Check Modal function logs for the specific task IDs listed below
2. Look for early termination causes (OOM, config parsing errors, model initialization failures)
3. Compare working vs failing configs for differences beyond the hyperparameters
4. Test one failing config locally with verbose logging

## Task IDs from Latest Relaunch Attempt

```
fc-01KE6XS5WB1GVJENEXWTZ3ZAZE  stage1_filter_bank_lr1e-3_wd1e-2_k16_s1
fc-01KE6XS7CZWSHBMZA2PHP56579  stage1_filter_bank_lr1e-3_wd1e-2_k16_s2
fc-01KE6XS97MFN5V6T9SW3PSN8MD  stage1_filter_bank_lr1e-3_wd1e-2_k16_s3
fc-01KE6XSAPZVMN9Q5BAKKBQ2YSM  stage1_filter_bank_lr1e-3_wd1e-2_k8_s1
fc-01KE6XSCBGVF0GMVCFFXZJHCCG  stage1_filter_bank_lr1e-3_wd1e-2_k8_s2
fc-01KE6XSDR1MP7CP59BZ2CM965K  stage1_filter_bank_lr1e-3_wd1e-2_k8_s3
fc-01KE6XSFGF04NHET8VW4ANYWFY  stage1_filter_bank_lr1e-3_wd1e-3_k16_s1
fc-01KE6XSH36G6CY1Q92873K68PB  stage1_filter_bank_lr1e-3_wd1e-3_k16_s2
fc-01KE6XSJZCD4TQBJTNA0FQPVCC  stage1_filter_bank_lr1e-3_wd1e-3_k16_s3
fc-01KE6XSMXJ42JNZSRH2NFMKZD5  stage1_filter_bank_lr1e-3_wd1e-3_k8_s1
fc-01KE6XSPE00V1V6BCG039Q74VX  stage1_filter_bank_lr1e-3_wd1e-3_k8_s2
fc-01KE6XSR2YXJBCC8GPAMWYFM14  stage1_filter_bank_lr1e-3_wd1e-3_k8_s3
fc-01KE6XSSRWHDZMZXRYZBM118PQ  stage1_filter_bank_lr1e-4_wd1e-2_k16_s1
fc-01KE6XSV3H34CBX1NFAMD89NX9  stage1_filter_bank_lr1e-4_wd1e-2_k16_s2
fc-01KE6XSWWKXT2QZWHEJSKPH4AS  stage1_filter_bank_lr1e-4_wd1e-2_k16_s3
fc-01KE6XSYXY0WX60X85MBTCYRS8  stage1_filter_bank_lr1e-4_wd1e-2_k8_s1
fc-01KE6XT0ES6Z69RPMA976GE3PM  stage1_linear_pe_lr1e-4_wd1e-2_k8_s1
fc-01KE6XT1Z65EZT3BK1KP3K2MVT  stage1_self_attention_lr1e-4_wd1e-2_k8_s1
fc-01KE6XT3DFJ7YT3K3STNM80EXX  stage1_self_attention_lr1e-4_wd1e-2_k8_s2
```

## Hypotheses

1. **Resource contention**: Too many concurrent spawns overwhelmed Modal's scheduler
2. **Config-specific bug**: Something about `filter_bank` PE type or `lr1e-3`/`lr1e-4` causes early crash
3. **Image caching**: First batch warmed cache, subsequent spawns hit cold start race condition
4. **W&B rate limiting**: Concurrent init calls got throttled

## Related Files

- `src/tmgg/modal/cli/spawn_single.py` - Single experiment spawner
- `src/tmgg/modal/cli/launch_sweep.py` - Batch launcher
- `src/tmgg/modal/runner.py` - Modal function definitions
- `configs/stage1/2026-01-05/` - Generated configs

## Resolution

### Root Cause: Entity Routing Race Condition

The 19 runs were NOT missing - they completed successfully but logged to the wrong W&B entity. All runs exist in `igorkraw/spectral_denoising` instead of `graph_denoise_team/spectral_denoising`.

**Why this happened**: During the original batch launch of 108 experiments, some runs started before `_wandb_config` injection completed, causing them to fall back to the WANDB_API_KEY owner's default entity.

### Verification via W&B API (2026-01-05 15:30 CET)

All 19 runs confirmed present in `igorkraw/spectral_denoising` with `state: finished`:

| Category | Expected | Found | Status |
|----------|----------|-------|--------|
| filter_bank_lr1e-3_wd1e-2 | 6 | 6 | ✓ |
| filter_bank_lr1e-3_wd1e-3 | 6 | 6 | ✓ |
| filter_bank_lr1e-4_wd1e-2 | 4 | 4 | ✓ |
| linear_pe_lr1e-4_wd1e-2_k8_s1 | 1 | 1 | ✓ |
| self_attention_lr1e-4_wd1e-2_k8 | 2 | 2 | ✓ |
| **Total** | **19** | **19** | **All found** |

### Data Location

For analysis, use runs from BOTH projects:
- **Primary**: `graph_denoise_team/spectral_denoising` (89 runs)
- **Secondary**: `igorkraw/spectral_denoising` (19 runs from original batch)

### Prevention

The `spawn_single.py` script was updated to accept `--wandb-entity` and `--wandb-project` flags to avoid similar issues in future single-experiment spawns.
