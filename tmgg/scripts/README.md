# TMGG Experiment Scripts

This directory contains bash scripts for running graph denoising experiments using the TMGG framework.

## Legacy Replication Scripts

### `tmgg_train_attention.sh` 
Clean TMGG-based replica of `../denoising/train_attention.sh`. Replicates exact same experimental setup using modern tmgg infrastructure.

**Parameters replicated:**
- Fixed block sizes: [10, 5, 3, 2]  
- Noise types: digress, gaussian, rotation
- Noise level: eps=0.3
- Model: num_heads=8, num_layers=8
- Training: 1000 epochs, 128 samples/epoch, batch_size=32, seed=42

**Usage:**
```bash
./tmgg_train_attention.sh
```

### `tmgg_train_gnn.sh`
Clean TMGG-based replica of `../denoising/train_gnn.sh`. Replicates exact GNN experimental setup using modern tmgg infrastructure.

**Parameters replicated:**
- Fixed block sizes: [10, 5, 3, 2]
- Noise types: gaussian, rotation, digress (matches legacy order)
- Noise level: eps=0.3  
- Model: num_layers=1
- Training: 1000 epochs, 128 samples/epoch, batch_size=32, seed=42

**Usage:**
```bash
./tmgg_train_gnn.sh
```

### `test_legacy_compatibility.sh`
Validation script that runs minimal experiments to ensure parameter mapping works correctly between legacy scripts and tmgg CLI tools.

**Usage:**
```bash
./test_legacy_compatibility.sh
```

## Additional Scripts

### `train_hybrid.sh`
Runs experiments with hybrid models combining GNN embeddings with transformer denoising.

### `test_noise_generators.sh`
Quick test script to verify the noise generators are working correctly with reduced epochs for quick validation.

## Key Features

- **Exact Parameter Matching**: Legacy scripts are replicated with mathematical equivalence
- **Clean Architecture**: Uses tmgg's PyTorch Lightning + Hydra infrastructure
- **Legacy Preservation**: Original `../denoising/` codebase remains untouched
- **Drop-in Replacement**: Scripts can be used as direct replacements for legacy experiments

## Legacy vs TMGG Parameter Mapping

| Legacy Parameter | TMGG Command |
|-----------------|---------------|
| `--model_type MultiLayerAttention --eps 0.3` | `tmgg-attention 'model.noise_levels=[0.3]'` |
| `--model_type GNN --num_layers 1` | `tmgg-gnn model.num_layers=1` |
| `--block_sizes "[10, 5, 3, 2]"` | `data=legacy_match` (built into config) |
| `--num_epochs 1000 --seed 42` | `trainer.max_epochs=1000 seed=42` |

## Output

Results are saved to `./outputs/legacy_replication/` with organized subdirectories:
- `attention_digress_eps0.3/`
- `attention_gaussian_eps0.3/`
- `attention_rotation_eps0.3/`
- `gnn_gaussian_eps0.3/`
- `gnn_rotation_eps0.3/`
- `gnn_digress_eps0.3/`

Each experiment includes:
- Model checkpoints
- Training logs with PyTorch Lightning
- Metrics and visualizations logged to W&B
- Hydra configuration files

## Requirements

Before running the scripts, ensure:
1. TMGG is installed: `uv sync` from the tmgg directory
2. You're in the tmgg directory when running scripts
3. All dependencies are available via uv

## Configuration

The legacy replication uses the `legacy_match.yaml` config which ensures exact parameter compatibility with the original denoising scripts while leveraging tmgg's superior experiment management.