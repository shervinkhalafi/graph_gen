# TMGG Experiment Scripts

This directory contains bash scripts for running graph denoising experiments using the TMGG framework.

## Core Training Scripts

### 1. `train_attention.sh`
Runs experiments with multi-head attention transformer models for adjacency matrix denoising.

**Usage modes:**
```bash
# Single experiment
./train_attention.sh --noise-type gaussian --noise-level 0.2 --num-epochs 500

# Component batch mode
./train_attention.sh --component gaussian    # All Gaussian noise levels
./train_attention.sh --component rotation    # All Rotation noise levels
./train_attention.sh --component digress     # All Digress noise levels
./train_attention.sh --component all         # All noise types

# Full replication
./train_attention.sh --replicate
```

### 2. `train_gnn.sh`
Runs experiments with various GNN architectures (standard, symmetric, and NodeVar GNNs).

**Usage modes:**
```bash
# Single experiment
./train_gnn.sh --model-type symmetric_gnn --noise-type rotation --noise-level 0.15

# Component batch mode
./train_gnn.sh --component standard-gaussian  # Standard GNN with Gaussian noise
./train_gnn.sh --component symmetric-rotation # Symmetric GNN with Rotation noise
./train_gnn.sh --component all-nodevar        # All NodeVar GNN experiments
./train_gnn.sh --component all               # Full replication

# Full replication
./train_gnn.sh --replicate
```

### 3. `train_hybrid.sh`
Runs experiments with hybrid models combining GNN embeddings with transformer denoising.

**Usage modes:**
```bash
# Single experiment
./train_hybrid.sh --gnn-layers 3 --transformer-heads 8 --noise-type digress

# Component batch mode
./train_hybrid.sh --component transformer-gaussian  # Hybrid+transformer with Gaussian
./train_hybrid.sh --component gnn-only-digress    # GNN-only hybrid with Digress
./train_hybrid.sh --component variations          # Different layer configurations
./train_hybrid.sh --component all                 # Full replication

# Full replication
./train_hybrid.sh --replicate
```

## Replication and Testing Scripts

### 4. `replicate_notebook_experiments.sh`
Replicates the exact experiments from the original notebooks:
- Attention model experiments (matching `train_attention.sh`)
- GNN model experiments (matching `train_gnn.sh`)  
- Hybrid model experiment (matching `new_denoiser.ipynb`)

Usage:
```bash
./replicate_notebook_experiments.sh
```

### 5. `replicate_denoising_experiments.sh`
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

### 6. `test_noise_generators.sh`
Quick test script to verify the noise generators are working correctly:
- Tests all three noise types
- Runs with reduced epochs for quick validation
- Tests rotation noise with different k values

Usage:
```bash
./test_noise_generators.sh
```

## Noise Types and Levels

### Gaussian Noise
- Adds Gaussian noise to adjacency matrix entries
- Default levels: 0.05, 0.1, 0.2, 0.3, 0.4, 0.5

### Rotation Noise
- Applies rotation perturbations to the eigenspace
- Default levels: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
- Requires `rotation_k` parameter (default: 20) for skew-symmetric matrix dimension

### Digress Noise
- Edge flipping noise following the Digress model
- Default levels: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6

## Model Configurations

### Attention Models
- Number of layers: 8 (default)
- Number of heads: 8 (default)
- Hidden dimension: Configured via Hydra

### GNN Models
- **standard_gnn**: Basic message-passing GNN
- **symmetric_gnn**: Enforces symmetric adjacency matrices
- **nodevar_gnn**: Node-variant GNN for heterogeneous graphs
- Number of layers: 1 (default)

### Hybrid Models
- **hybrid_with_transformer**: GNN embeddings + transformer denoising
- **hybrid_gnn_only**: GNN-only baseline without transformer
- GNN layers: 2 (default)
- Transformer layers: 4 (default)
- Transformer heads: 4 (default)

## Common Parameters

All scripts accept these parameters:
- `--noise-type`: Type of noise (gaussian, rotation, digress)
- `--noise-level`: Noise intensity level (float)
- `--num-epochs`: Number of training epochs
- `--experiment-name`: Custom experiment name for tracking

Model-specific parameters:
- Attention: `--num-layers`, `--num-heads`
- GNN: `--model-type`, `--num-layers`
- Hybrid: `--gnn-layers`, `--transformer-layers`, `--transformer-heads`, `--no-transformer`

## Example Workflows

### Quick Test Run
```bash
# Test single configuration
./train_attention.sh --noise-type gaussian --noise-level 0.1 --num-epochs 50
```

### Noise Type Comparison
```bash
# Compare all noise types for attention model
./train_attention.sh --component gaussian
./train_attention.sh --component rotation
./train_attention.sh --component digress
```

### Model Architecture Comparison
```bash
# Compare different GNN architectures on same noise
./train_gnn.sh --component standard-digress
./train_gnn.sh --component symmetric-digress
./train_gnn.sh --component nodevar-digress
```

### Full Experimental Suite
```bash
# Run complete replication for all models
./train_attention.sh --replicate
./train_gnn.sh --replicate
./train_hybrid.sh --replicate
```

## Configuration

All scripts use Hydra configuration overrides. You can modify experiments by:

1. Editing the scripts directly
2. Adding additional Hydra overrides to the command lines
3. Creating new configuration files in the experiment config directories

## Output

Results are saved according to Hydra configuration:
- Model checkpoints
- Training logs
- Visualization plots
- Metrics (if Weights & Biases is configured)

The exact output location depends on your Hydra configuration and can be customized via the `experiment.name` parameter.

## Requirements

Before running the scripts, ensure:
1. TMGG is installed: `pip install -e .` from the tmgg directory
2. All dependencies are installed
3. You're in the scripts directory when running them

## Customization

To customize experiments, modify the following parameters:
- `seed`: Random seed for reproducibility
- `batch_size`: Training batch size
- Model-specific parameters (layers, heads, dimensions, etc.)
- Additional Hydra overrides for fine-grained control