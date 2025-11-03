# TMGG: Graph Denoising Research Framework

A research framework for graph denoising using attention mechanisms, graph neural networks, and hybrid architectures on Stochastic Block Model (SBM) graphs.

## Installation

```bash
git clone <repository-url>
cd tmgg
uv venv
source .venv/bin/activate
uv pip install -e .
```

For development:
```bash
uv pip install -e ".[test]"
```

## Quick Start

### Running Experiments

```bash
# Attention-based denoising
python -m tmgg.experiments.attention_denoising.runner

# GNN-based denoising  
python -m tmgg.experiments.gnn_denoising.runner

# Digress transformer denoising
python -m tmgg.experiments.digress_denoising.runner

# Hybrid GNN+Transformer denoising
python -m tmgg.experiments.hybrid_denoising.runner
```

### Sanity Checks

```bash
# Run experiment validation without full training
python -m tmgg.experiments.attention_denoising.runner sanity_check=true

# Fast development run (1 batch)
python -m tmgg.experiments.gnn_denoising.runner trainer.fast_dev_run=true
```

### Configuration Overrides

All experiments use Hydra for configuration management:

```bash
# Change model architecture
python -m tmgg.experiments.attention_denoising.runner model.num_layers=16 model.num_heads=16

# Change training parameters
python -m tmgg.experiments.gnn_denoising.runner trainer.max_epochs=500 model.learning_rate=0.01

# Change data parameters
python -m tmgg.experiments.hybrid_denoising.runner data.batch_size=64 data.num_nodes=50

# Run hyperparameter sweep
python -m tmgg.experiments.attention_denoising.runner --multirun model.num_layers=4,8,16 model.learning_rate=0.001,0.005,0.01
```

## Model Architectures

### Attention Models
- **Multi-Layer Attention**: Transformer-based denoising with configurable layers and heads
- **Digress**: Graph transformer using edge-based attention mechanisms

### GNN Models  
- **Standard GNN**: Graph convolution with eigenvalue embeddings
- **NodeVar GNN**: Node-variant GNN for heterogeneous processing
- **Symmetric GNN**: Shared embeddings with symmetric operations

### Hybrid Models
- **Hybrid with Transformer**: GNN node embeddings followed by transformer-based reconstruction
- **Hybrid GNN-only**: GNN embeddings without transformer (baseline comparison)

## Configuration System

The framework uses a centralized configuration system located in `tmgg/src/tmgg/exp_configs/`:

```
exp_configs/
├── base_config_attention.yaml    # Base attention experiment config
├── base_config_gnn.yaml         # Base GNN experiment config  
├── base_config_digress.yaml     # Base digress experiment config
├── base_config_hybrid.yaml      # Base hybrid experiment config
├── base/                         # Shared component configs
│   ├── data/                     # Data module configurations
│   ├── trainer/                  # PyTorch Lightning trainer configs
│   └── logger/                   # Logger configurations
├── models/                       # Model-specific configurations
│   ├── attention/
│   ├── gnn/
│   ├── digress/
│   └── hybrid/
└── experiments/                  # Override-based experiment variations
```

### Configuration Examples

**View current configuration:**
```bash
python -m tmgg.experiments.attention_denoising.runner --cfg job
```

**Override specific parameters:**
```bash
# Use different model variant
python -m tmgg.experiments.gnn_denoising.runner /models/gnn/nodevar_gnn@model

# Switch data configuration
python -m tmgg.experiments.attention_denoising.runner /base/data/nx_square@data

# Custom training configuration
python -m tmgg.experiments.hybrid_denoising.runner trainer.max_epochs=1000 trainer.gradient_clip_val=0.5
```

**Create experiment variations:**

Create `exp_configs/experiments/attention_large.yaml`:
```yaml
# @package _global_
defaults:
  - /base_config_attention
  - _self_

model:
  num_layers: 16
  num_heads: 16
  d_model: 40

trainer:
  max_epochs: 500
```

Run with: `python -m tmgg.experiments.attention_denoising.runner +experiments=attention_large`

## Data and Noise Models

The framework supports Stochastic Block Model (SBM) graph generation with three noise types:

- **Gaussian Noise**: Additive Gaussian noise to adjacency matrices
- **Rotation Noise**: Eigenspace rotation using skew-symmetric matrices
- **Digress Noise**: Edge flipping based on noise probability

Data configurations support various graph structures:
- Standard SBM with configurable block sizes and connectivity
- NetworkX-based regular graphs (square lattice, star graphs)
- Custom graph generation parameters

## Experiment Structure

```
experiments/{experiment_name}/
├── lightning_module.py    # PyTorch Lightning wrapper
└── runner.py             # Hydra entry point

experiment_utils/
├── base_lightningmodule.py  # Shared Lightning base class
├── data/                    # Data generation and loading
├── metrics.py              # Evaluation metrics  
├── plotting.py             # Visualization utilities
└── run_experiment.py       # Experiment orchestration
```

## Evaluation Metrics

- **Eigenvalue Error**: L2 distance between eigenvalue spectra
- **Subspace Distance**: Angular distance between eigenspaces
- **Frobenius Error**: Matrix reconstruction error
- **Reconstruction Metrics**: MAE, MSE, BCE for adjacency matrix reconstruction

## Development

### Adding New Models

1. Implement model class in `tmgg/src/tmgg/models/`
2. Create model configuration in `exp_configs/models/{category}/`
3. Update experiment lightning module if needed
4. Add tests in `tmgg/tests/`

### Adding New Experiments

1. Create lightning module in `experiments/{experiment_name}/`
2. Create base configuration in `exp_configs/base_config_{name}.yaml`
3. Add model configurations in `exp_configs/models/{category}/`
4. Create runner script following existing patterns

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tmgg --cov-report=html

# Test specific experiment
pytest tests/experiments/test_attention_denoising.py
```

## Logging and Monitoring

The framework supports multiple logging backends:

- **TensorBoard**: Default logger for training curves and metrics
- **Weights & Biases**: Advanced experiment tracking and visualization
- **CSV Logger**: Simple CSV-based metric logging

Configure loggers via the configuration system:
```bash
# Use wandb logging
python -m tmgg.experiments.attention_denoising.runner /base/logger/wandb@logger

# Multiple loggers
python -m tmgg.experiments.gnn_denoising.runner /base/logger/multi@logger
```
