# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graph denoising research project comparing attention-based transformers and graph neural networks for denoising adjacency matrices. The project focuses on Stochastic Block Model (SBM) graphs with three noise types: Gaussian, Rotation, and Digress (edge flipping).

## Architecture

### Current Structure (denoising/)
- `main.py`: Primary training script with command-line argument parsing
- `src/data/sbm.py`: SBM data generation and noise addition functions
- `src/models/attention.py`: Multi-head attention transformer implementations  
- `src/models/gnn.py`: Graph neural network implementations (GNN, NodeVarGNN)
- `results/`: Generated plots and experimental results
- `train_attention.sh` and `train_gnn.sh`: Experiment runner scripts

### Target Structure (tmgg/)
The goal is to reorganize experiments into a clean Hydra + PyTorch Lightning structure:
- Individual experiment submodules with installable Hydra entrypoints
- Shared `experiment_utils` for plotting and statistical tests
- Shared `models` for algorithmic architectures
- Local configuration management within each experiment submodule

## Development Commands

### Running Experiments
```bash
# From denoising/ directory
python main.py --model_type MultiLayerAttention --noise_type Gaussian --eps 0.1 --num_heads 8 --num_layers 8
python main.py --model_type GNN --noise_type Digress --eps 0.3 --num_layers 1

# Using provided scripts
./train_attention.sh  # Runs attention models across noise types
./train_gnn.sh        # Runs GNN models across noise types
```

### Key Arguments
- `--model_type`: 'MultiLayerAttention', 'GNN', or 'NodeVarGNN'
- `--noise_type`: 'Gaussian', 'Rotation', or 'Digress'  
- `--eps`: Noise level (float)
- `--block_sizes`: SBM block structure (e.g., "[10, 5, 3, 2]")
- `--num_epochs`: Training epochs
- `--batch_size`: Training batch size

### Dependencies
Core dependencies (inferred from imports):
- PyTorch (neural networks)
- SciPy (sparse eigenvalue computations via `eigsh`)
- NumPy (matrix operations)
- Matplotlib (plotting results)
- Weights & Biases (`wandb` in notebooks for experiment tracking)

**Note**: No formal dependency management exists yet. Dependencies need to be installed manually.

## Code Architecture Details

### Models (`src/models/`)
- **MultiHeadAttention**: Standard transformer attention with configurable heads/dimensions
- **MultiLayerAttention**: Stacked attention layers for adjacency matrix denoising
- **GNN**: Traditional graph neural network with message passing
- **NodeVarGNN**: Node-variant GNN for heterogeneous graph processing
- **SequentialDenoisingModel**: Hybrid approach combining GNN embeddings with transformer denoising

### Data Generation (`src/data/sbm.py`)
- `generate_sbm_adjacency()`: Creates stochastic block model graphs
- Noise functions: `add_gaussian_noise()`, `add_rotation_noise()`, `add_digress_noise()`
- `AdjacencyMatrixDataset`: PyTorch dataset wrapper for batch processing

### Evaluation Metrics
- Eigenvalue error computation using `scipy.sparse.linalg.eigsh`
- Subspace distance calculations for denoising quality assessment
- Visualization of results across different noise levels and model types

## Experiment Workflow

1. **Data Generation**: SBM graphs with specified block sizes and connectivity probabilities
2. **Noise Addition**: Apply one of three noise types at specified intensity levels
3. **Model Training**: Train denoising models using specified architecture
4. **Evaluation**: Compute eigenvalue errors and generate result plots
5. **Results Storage**: Save plots to `results/` directory with descriptive filenames

## Git Workflow Notes

- Current branch: `igor` (active development)
- Main branch: `main`
- Consider using `claude/` branch pattern for structured development
- Recent commits show active notebook-based experimentation

## Development Priorities

When working on the tmgg/ reorganization:
1. Maintain existing experiment functionality during migration
2. Implement proper Hydra configuration management
3. Add PyTorch Lightning training loop abstractions
4. Create shared utility modules for common functions
5. Establish proper package structure with installable entrypoints