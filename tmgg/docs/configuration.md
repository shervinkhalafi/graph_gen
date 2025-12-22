# Configuration

The framework uses Hydra for configuration management. This document covers the config hierarchy, common overrides, and how to create custom configurations.

## Config Hierarchy

```
exp_configs/
├── base_config_attention.yaml     # Top-level for attention experiments
├── base_config_gnn.yaml           # Top-level for GNN experiments
├── base_config_digress.yaml       # Top-level for DiGress experiments
├── base_config_hybrid.yaml        # Top-level for hybrid experiments
├── base_config_spectral.yaml      # Top-level for spectral experiments
├── base/
│   ├── trainer/default.yaml       # PyTorch Lightning trainer settings
│   └── logger/                    # tensorboard.yaml, wandb.yaml, csv.yaml
├── models/
│   ├── attention/                 # multi_layer_attention.yaml
│   ├── gnn/                       # standard_gnn.yaml, nodevar_gnn.yaml, symmetric_gnn.yaml
│   ├── digress/                   # digress_transformer.yaml
│   ├── hybrid/                    # hybrid_with_transformer.yaml
│   └── spectral/                  # filter_bank.yaml, linear_pe.yaml, self_attention.yaml
├── data/
│   ├── sbm_default.yaml           # SBM with n=20
│   ├── legacy_match.yaml          # Legacy denoising replication
│   └── ...
└── stage/
    ├── stage1_poc.yaml            # Proof of concept
    ├── stage1_sanity.yaml         # Constant noise memorization
    ├── stage2_validation.yaml     # Cross-dataset validation
    ├── stage3_diversity.yaml      # Dataset diversity (future)
    ├── stage4_benchmarks.yaml     # Real-world benchmarks (future)
    └── stage5_full.yaml           # Full validation (future)
```

Each base config composes defaults from subdirectories:

```yaml
# base_config_attention.yaml
defaults:
  - models/attention/multi_layer_attention@model
  - data: sbm_default
  - base/trainer/default@trainer
  - base/logger/tensorboard@logger
  - _self_

experiment_name: "attention_denoising"
seed: 42
```

## Hydra Override Syntax

Override parameters from the command line:

```bash
# Simple override
uv run tmgg-attention model.num_layers=16

# Nested override
uv run tmgg-gnn model.scheduler_config.T_0=10

# List override (use quotes)
uv run tmgg-attention 'data.noise_levels=[0.1,0.2,0.3]'

# Switch config group
uv run tmgg-gnn data=legacy_match

# Switch model variant
uv run tmgg-gnn model=gnn/nodevar_gnn
```

## Common Overrides

| What | Override | Example |
|------|----------|---------|
| Training steps | `trainer.max_steps=N` | `trainer.max_steps=50000` |
| Validation frequency | `trainer.val_check_interval=N` | `trainer.val_check_interval=500` |
| Learning rate | `model.learning_rate=X` | `model.learning_rate=0.001` |
| Batch size | `data.batch_size=N` | `data.batch_size=64` |
| Model layers | `model.num_layers=N` | `model.num_layers=8` |
| Attention heads | `model.num_heads=N` | `model.num_heads=8` |
| Eigenvectors (spectral) | `model.k=N` | `model.k=50` |
| Noise levels | `'data.noise_levels=[...]'` | `'data.noise_levels=[0.1,0.2]'` |
| Noise type | `data.noise_type=X` | `data.noise_type=gaussian` |
| Scheduler type | `scheduler_config.type=X` | `scheduler_config.type=none` |
| Random seed | `seed=N` | `seed=123` |
| Output directory | `hydra.run.dir=PATH` | `hydra.run.dir=./my_output` |
| Logger backend | `logger=X` | `logger=wandb` |

## Viewing Configuration

View the resolved configuration without running:

```bash
# Print full config
uv run tmgg-attention --cfg job

# Print specific group
uv run tmgg-attention --cfg job --package model
```

## Multirun (Hyperparameter Sweeps)

Run multiple configurations:

```bash
# Sweep over values
uv run tmgg-attention --multirun model.num_layers=4,8,16

# Multiple parameters
uv run tmgg-attention --multirun model.num_layers=4,8 model.learning_rate=0.001,0.01

# Grid search (all combinations)
uv run tmgg-attention --multirun \
  model.num_layers=4,8,16 \
  model.num_heads=4,8 \
  seed=1,2,3
```

## Model Configuration Reference

### Attention Model

```yaml
# models/attention/multi_layer_attention.yaml
_target_: tmgg.experiments.attention_denoising.lightning_module.AttentionDenoisingLightningModule

d_model: 20           # Model dimension (typically matches num_nodes)
num_heads: 8          # Number of attention heads
num_layers: 8         # Number of transformer layers
d_k: null             # Key dimension (defaults to d_model // num_heads)
d_v: null             # Value dimension (defaults to d_model // num_heads)
dropout: 0.0          # Dropout rate
bias: true            # Use bias in linear layers
learning_rate: 0.001
loss_type: "MSE"      # MSE or BCE
```

### GNN Model

```yaml
# models/gnn/standard_gnn.yaml
_target_: tmgg.experiments.gnn_denoising.lightning_module.GNNDenoisingLightningModule

num_layers: 2         # Number of GCN layers
num_terms: 3          # Polynomial filter terms
feature_dim_in: 20    # Input feature dimension
feature_dim_out: 10   # Output feature dimension
eigenvalue_reg: 0.0   # Eigenvalue regularization (0.001 helps gradient stability)
domain: "standard"    # standard or inv-sigmoid
```

### Data Configuration

```yaml
# data/sbm_default.yaml
dataset_name: sbm
dataset_config:
  num_nodes: 20
  p_intra: 1.0        # Intra-block edge probability
  q_inter: 0.0        # Inter-block edge probability
  min_blocks: 2
  max_blocks: 4

num_samples_per_graph: 1000
batch_size: 100
noise_type: "digress"
noise_levels: [0.005, 0.02, 0.05, 0.1, 0.25, 0.4, 0.5]
```

### Trainer Configuration (Step-Based)

Training is configured in **steps**, not epochs, to decouple from batch size and dataset size.

```yaml
# base/trainer/default.yaml
_target_: pytorch_lightning.Trainer

# Step-based training (no epochs)
max_steps: 10000              # Total training steps
max_epochs: -1                # Disable epoch-based termination

# Validation in steps
val_check_interval: 1000      # Validate every 1000 steps
check_val_every_n_epoch: null # Disable epoch-based validation

# Hardware
accelerator: "auto"
devices: "auto"
precision: 32

# Gradient clipping
gradient_clip_val: 1.0
gradient_clip_algorithm: "norm"

log_every_n_steps: 1
```

### Learning Rate Scheduler

Configured via `scheduler_config` in base Lightning module:

```yaml
# base/lightning_base.yaml
scheduler_config:
  type: "cosine_warmup"       # "cosine_warmup", "cosine", "step", or null
  warmup_fraction: 0.02       # 2% of training for linear warmup
  decay_fraction: 0.8         # LR reaches 0 at 80% of training
```

The `cosine_warmup` scheduler automatically adapts to total training steps. Stages typically disable scheduling (`type: none`) for simpler optimization.

### Callbacks Configuration (Step-Based)

```yaml
# base/callbacks/default.yaml
callbacks:
  early_stopping:
    monitor: val/loss
    mode: min
    patience: 10              # Validation checks (× val_check_interval = steps until stop)
    min_delta: 1e-4
  checkpoint:
    monitor: val/loss
    mode: min
    save_top_k: 3
    save_last: true
    filename: "model-step={step:06d}-val_loss={val/loss:.4f}"
```

### Progress Bar Configuration

```yaml
# base/progress_bar/default.yaml
progress_bar:
  metrics_to_show:
    - train_loss
    - val/loss
  show_epoch: true
  metrics_format: ".4f"
```

## Creating Custom Configs

### New Model Configuration

Create a new file in `exp_configs/models/`:

```yaml
# models/attention/large_attention.yaml
_target_: tmgg.experiments.attention_denoising.lightning_module.AttentionDenoisingLightningModule

d_model: 40
num_heads: 16
num_layers: 16
dropout: 0.1
learning_rate: 0.0005
```

Use it:

```bash
uv run tmgg-attention model=attention/large_attention
```

### New Data Configuration

Create a file in `exp_configs/data/`:

```yaml
# data/sbm_large.yaml
dataset_name: sbm
dataset_config:
  num_nodes: 100
  p_intra: 0.9
  q_inter: 0.1
  min_blocks: 3
  max_blocks: 6

num_samples_per_graph: 500
batch_size: 32
noise_type: "gaussian"
noise_levels: [0.1, 0.2, 0.3]
```

Use it:

```bash
uv run tmgg-gnn data=sbm_large
```

### Experiment Variation

Create a file in `exp_configs/experiments/`:

```yaml
# experiments/attention_long_training.yaml
# @package _global_
defaults:
  - /base_config_attention
  - _self_

model:
  num_layers: 12
  dropout: 0.1

trainer:
  max_epochs: 1000
  gradient_clip_val: 0.5
```

Use it:

```bash
uv run tmgg-attention +experiments=attention_long_training
```

## Environment Variables

Hydra respects standard environment variables:

```bash
# Override output directory
HYDRA_FULL_ERROR=1 uv run tmgg-attention  # Show full stack traces
```
