# Pickup: Missing Grid Experiments (2026-01-12)

## Summary

Launched 648 missing experiments on Modal to complete the eigenstructure study grid.

| Stage | Count | Description |
|-------|-------|-------------|
| stage2c_spectral | 288 | filter_bank, linear_pe, self_attention on SBM datasets |
| stage3_pyg | 240 | PyG dataset experiments (ENZYMES, PROTEINS, etc.) |
| stage3_roc | 120 | Ring-of-cliques experiments |

## What Was Done

### 1. Fixed Modal deployment issues

The Modal app was failing to deploy because it tried to upload the entire project directory (~400+ files including configs/, data/, checkpoints/, results/).

**Fix applied in `src/tmgg/modal/image.py`:**
- Changed `add_local_dir` to mount only `src/` instead of project root
- Added separate mount for `pyproject.toml` and `README.md` (needed for editable install)
- Simplified ignore patterns since large directories are no longer in mount scope

### 2. Launched experiments

```bash
# Deployed updated Modal app
doppler run -- uv run modal deploy src/tmgg/modal/run_single.py

# Launched all three stages
doppler run -- uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/missing_grid/stage2c_spectral/ \
    --gpu standard \
    --wandb-entity graph_denoise_team \
    --wandb-project spectral_denoising \
    --skip-existing

doppler run -- uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/missing_grid/stage3_pyg/ \
    --gpu standard \
    --wandb-entity graph_denoise_team \
    --wandb-project spectral_denoising \
    --skip-existing

doppler run -- uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/missing_grid/stage3_roc/ \
    --gpu standard \
    --wandb-entity graph_denoise_team \
    --wandb-project spectral_denoising \
    --skip-existing
```

## Monitoring Progress

- **Modal dashboard**: https://modal.com/apps (check tmgg-spectral app)
- **W&B project**: https://wandb.ai/graph_denoise_team/spectral_denoising

## Once Experiments Complete

### 1. Verify completion

Check that all runs completed successfully:

```bash
# Fetch latest W&B runs
doppler run -- uv run python scripts/fetch_wandb_runs.py \
    --entity graph_denoise_team \
    --project spectral_denoising \
    --output eigenstructure_results_full/wandb_runs_updated.json
```

### 2. Re-run analyses

The analysis scripts in `scripts/` need to be re-run with the complete dataset:

```bash
# Full analysis with all runs
mise run analysis:full

# Or manually:
uv run python scripts/unified_arch_comparison.py \
    --input eigenstructure_results_full/experiment_summary.json \
    --output eigenstructure_results_full/unified_architecture_comparison.md
```

### 3. Update experiment summary

If the experiment summary needs regeneration:

```bash
# This aggregates W&B data into the summary format used by analysis scripts
uv run python scripts/aggregate_wandb_results.py \
    --input eigenstructure_results_full/wandb_runs_updated.json \
    --output eigenstructure_results_full/experiment_summary.json
```

## Files Modified

- `src/tmgg/modal/image.py` - Fixed to mount only src/ directory
- `.modalignore` - Updated with comprehensive ignore patterns (though less critical now)

## Config Locations

- Pre-generated configs: `configs/missing_grid/`
- Stage definitions: `src/tmgg/modal/stage_definitions/stage2c_spectral.yaml`, `stage3_pyg.yaml`, `stage3_roc.yaml`
