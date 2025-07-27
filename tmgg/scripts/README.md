# TMGG Experiment Scripts

This directory contains bash scripts for running graph denoising experiments using the TMGG framework.

## Scripts Overview

### 1. `replicate_notebook_experiments.sh`
Replicates the exact experiments from the original notebooks:
- Attention model experiments (matching `train_attention.sh`)
- GNN model experiments (matching `train_gnn.sh`)  
- Hybrid model experiment (matching `new_denoiser.ipynb`)

Usage:
```bash
./replicate_notebook_experiments.sh
```

### 2. `replicate_denoising_experiments.sh`
Comprehensive script for running various denoising experiments with different configurations:
- Supports all model types (attention, GNN, hybrid)
- Multiple noise types (Gaussian, Rotation, Digress)
- Multiple noise levels

Usage:
```bash
# Run all experiments
./replicate_denoising_experiments.sh

# Run specific experiment type
./replicate_denoising_experiments.sh attention
./replicate_denoising_experiments.sh gnn
./replicate_denoising_experiments.sh hybrid

# Custom output directory and seed
./replicate_denoising_experiments.sh -d my_outputs -s 123 all
```

### 3. `test_noise_generators.sh`
Quick test script to verify the noise generators are working correctly:
- Tests all three noise types
- Runs with reduced epochs for quick validation
- Tests rotation noise with different k values

Usage:
```bash
./test_noise_generators.sh
```

## Configuration

All scripts use Hydra configuration overrides. You can modify experiments by:

1. Editing the scripts directly
2. Adding additional Hydra overrides to the command lines
3. Creating new configuration files in the experiment config directories

## Output

Results are saved to:
- `outputs/notebook_replication/` - For notebook replication
- `outputs/denoising_experiments/` - For comprehensive experiments  
- `outputs/noise_generator_tests/` - For quick tests

Each experiment creates:
- Model checkpoints
- Training logs
- Visualization plots
- Metrics (if Weights & Biases is configured)

## Requirements

Before running the scripts, ensure:
1. TMGG is installed: `pip install -e .` from the tmgg directory
2. All dependencies are installed
3. You're in the tmgg directory when running scripts

## Noise Types and Parameters

### Rotation Noise
Rotation noise requires the `rotation_k` parameter which specifies the dimension of the skew-symmetric matrix used for rotating eigenvectors. The scripts default to `k=20`.

### All Noise Types
- **Gaussian**: Adds Gaussian noise to adjacency matrix
- **Rotation**: Rotates eigenvectors using a skew-symmetric matrix
- **Digress**: Flips edges with specified probability

## Customization

To customize experiments, modify the following parameters:
- `seed`: Random seed for reproducibility
- `max_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `noise_levels`: List of noise intensities to test
- Model-specific parameters (layers, heads, dimensions, etc.)