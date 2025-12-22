# Experiments

This document covers running experiments, interpreting results, and using the stage-based experiment system.

## Running Experiments

### Basic Usage

```bash
# Run with default configuration
uv run tmgg-attention

# Override parameters
uv run tmgg-gnn trainer.max_epochs=100 model.num_layers=4

# Different data configuration
uv run tmgg-attention data=legacy_match
```

### Available Commands

| Command | Model Type | Base Config |
|---------|------------|-------------|
| `tmgg-attention` | Multi-layer attention | `base_config_attention.yaml` |
| `tmgg-gnn` | Standard GNN | `base_config_gnn.yaml` |
| `tmgg-hybrid` | GNN + Transformer | `base_config_hybrid.yaml` |
| `tmgg-digress` | DiGress transformer | `base_config_digress.yaml` |
| `tmgg-spectral` | Spectral PE models | `base_config_spectral.yaml` |

### Sanity Checks

Validate setup without full training:

```bash
# Run sanity check (tests data loading, forward pass, loss computation)
uv run tmgg-attention sanity_check=true

# Fast dev run (one batch only)
uv run tmgg-gnn trainer.fast_dev_run=true
```

## Output Structure

Each run creates a timestamped directory:

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── config.yaml              # Resolved configuration
├── checkpoints/
│   ├── model-epoch=05-val_loss=0.1234.ckpt
│   ├── model-epoch=10-val_loss=0.0987.ckpt
│   └── last.ckpt
├── tensorboard/             # TensorBoard logs (if enabled)
│   └── events.out.tfevents.*
└── .hydra/
    ├── config.yaml          # Original config
    ├── hydra.yaml           # Hydra settings
    └── overrides.yaml       # Command-line overrides
```

### Checkpoints

The framework saves:
- Top 3 models by validation loss
- Last checkpoint

Checkpoint naming: `model-epoch={N}-val_loss={X}.ckpt`

### Loading Checkpoints

```python
from tmgg.experiments.attention_denoising.lightning_module import AttentionDenoisingLightningModule

model = AttentionDenoisingLightningModule.load_from_checkpoint(
    "outputs/.../checkpoints/model-epoch=10-val_loss=0.0987.ckpt"
)
```

## Logging Options

### TensorBoard (Default)

```bash
uv run tmgg-attention logger=tensorboard

# View logs
tensorboard --logdir outputs/
```

### Weights & Biases

```bash
uv run tmgg-attention logger=wandb

# Or with project name
uv run tmgg-attention logger=wandb wandb.project="my-project"
```

### CSV Logger

```bash
uv run tmgg-attention logger=csv
```

### Multiple Loggers

```bash
uv run tmgg-attention logger=multi  # TensorBoard + CSV
```

## Metrics

The framework tracks these metrics:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `val/loss` | Validation loss per epoch |
| `test/loss` | Test loss (final evaluation) |
| `train/loss_epoch` | Average training loss per epoch |

Additional metrics computed in final evaluation:
- Eigenvalue error
- Subspace distance
- Reconstruction MAE/MSE

## Hyperparameter Sweeps

### Multirun

Run multiple configurations sequentially:

```bash
# Single parameter sweep
uv run tmgg-attention --multirun model.num_layers=4,8,16

# Multiple parameters (grid search)
uv run tmgg-attention --multirun \
  model.num_layers=4,8 \
  model.learning_rate=0.001,0.01 \
  seed=1,2,3
```

### Grid Search Command

```bash
uv run tmgg-grid-search  # Uses grid search configuration
```

## Stage-Based Experiments

The stage system runs systematic experiments across architectures and datasets with a fixed compute budget. This allows methodical validation of spectral PE approaches before scaling.

### Stage 1: Proof of Concept

**Budget**: 4.4 GPU-hours

**Purpose**: Validate that spectral PE architectures can denoise graphs on a single-graph SBM protocol (n=50).

**Architectures**:
- Linear PE
- Filter Bank
- Self-Attention
- DiGress (official LR=0.0002)
- DiGress (high LR=1e-2, matching spectral)

**Dataset**: SBM n=50, single-graph protocol (same graph for train/val/test, only noise varies)

**Optimizer**: Adam, no weight decay, no LR scheduling

**Hyperparameter sweep**:
- learning_rate: [1e-3, 1e-2]
- noise_levels: [0.01, 0.05, 0.1, 0.2, 0.3]
- model.k: 50 (full spectrum)
- 4 trials per configuration, 3 seeds

**Success criteria**: ≥15% improvement over Linear PE baseline on val_loss.

```bash
uv run tmgg-experiment +stage=stage1_poc              # Single run
uv run tmgg-experiment +stage=stage1_poc sweep=true   # Full sweep
```

### Stage 1 Sanity: Constant Noise Memorization

**Budget**: <20 GPU-minutes

**Purpose**: Validate model functionality by testing memorization of a fixed noisy→clean mapping. This debugging stage verifies gradient flow, model capacity, and optimizer behavior.

**Architectures**: Linear PE, Filter Bank, Self-Attention

**Key setting**: `fixed_noise_seed: 42` produces identical noise at every training step.

**Hyperparameters** (fixed, no sweep):
- learning_rate: 1e-2
- weight_decay: 0.0
- model.k: 50

**Success criteria**: `val_accuracy >= 0.99`

If this stage fails, something fundamental is broken in the architecture or optimizer.

```bash
uv run tmgg-experiment +stage=stage1_sanity
```

### Stage 2: Cross-Dataset Validation

**Budget**: ~19 GPU-hours

**Purpose**: Validate denoising across diverse graph families, with two protocols controlled by `cross_graph`:

| Protocol | `cross_graph` | Description |
|----------|---------------|-------------|
| Single-graph | `false` (default) | Train/val/test use the same graph, only noise varies |
| Cross-graph | `true` | Train/val/test use different graphs (generalization test) |

**Architectures**:
- Linear PE, Filter Bank, Self-Attention (high LR=1e-2, AMSGrad, weight_decay=1e-12)
- DiGress variants (official and high LR)

**Datasets** (9 graph types):
- Synthetic: Erdős-Rényi, d-regular, tree, ring of cliques, LFR, SBM
- PyG benchmarks: QM9, ENZYMES, PROTEINS

**Hyperparameter sweep**:
- noise_levels: [0.01, 0.1, 0.2]
- 4 trials per configuration, 3 seeds

**Success criteria**:
- reconstruction_error < 0.05
- generalization_gap < 0.15

```bash
uv run tmgg-experiment +stage=stage2_validation                      # Single-graph (default)
uv run tmgg-experiment +stage=stage2_validation cross_graph=true     # Cross-graph
uv run tmgg-experiment +stage=stage2_validation sweep=true           # Full sweep
```

### Stage 3: Dataset Diversity

**Budget**: 400 GPU-hours (future work)

**Purpose**: Validate across all graph families.

**Trigger condition**: Run only if Stage 2 achieves <3% error on multi-graph and matches DiGress.

**Architectures**: Filter Bank, Self-Attention, DiGress variants

**Datasets**:
- SBM: n=50, n=100, n=200
- Erdős-Rényi, d-regular, trees
- LFR benchmark (planted community)
- Ring of cliques

**Hyperparameter sweep**:
- learning_rate: [1e-3, 1e-2]
- model.k: [50, 100, 200]
- noise_levels: [0.01, 0.05, 0.1, 0.2, 0.3]
- 8 trials, 5 seeds

```bash
uv run tmgg-experiment +stage=stage3_diversity sweep=true
```

### Stage 4: Real-World Benchmarks

**Budget**: 300 GPU-hours (future work)

**Purpose**: Validate on PyTorch Geometric benchmark datasets.

**Trigger condition**: Run if Stage 3 validates cross-family generalization.

**Datasets**: PyG QM9, ENZYMES, PROTEINS

**Hyperparameter sweep**:
- model.k: [16, 32, 64] (variable graph sizes)
- 6 trials, 5 seeds
- 60-minute timeout, fast GPU tier

```bash
uv run tmgg-experiment +stage=stage4_benchmarks sweep=true
```

### Stage 5: Full Validation

**Budget**: 1500 GPU-hours (future work, for publication)

**Purpose**: Comprehensive ablations and robustness analysis.

**Architectures**: All (Linear PE, Filter Bank, Self-Attention, DiGress variants)

**Datasets**: All 11 datasets from previous stages

**Ablation studies**:
- Spectral polynomial depth: [3, 5, 8]
- Eigenvector count: [2, 4, 8, 16, 32, 64]
- Attention key dimension: [32, 64, 128]

**Statistical analysis**:
- Wilcoxon signed-rank test
- Holm-Bonferroni correction
- Significance level: 0.05

```bash
uv run tmgg-experiment +stage=stage5_full sweep=true
```

### Stage Configuration

Stage configs are defined in `exp_configs/stage/`. Example structure:

```yaml
# stage1_poc.yaml
# @package _global_
defaults:
  - override /data: sbm_single_graph

stage: stage1_poc

# Optimizer settings (harmonized across stages)
learning_rate: 1e-2
weight_decay: 1e-12
optimizer_type: adamw
amsgrad: true
scheduler_config:
  type: none

model:
  k: 50

noise_levels: [0.1]
eval_noise_levels: [0.1]

# Sweep metadata (used by coordinator when sweep=true)
_sweep_config:
  architectures:
    - models/spectral/linear_pe
    - models/spectral/filter_bank
    - models/spectral/self_attention
    - models/digress/digress_sbm_small
    - models/digress/digress_sbm_small_highlr
  hyperparameter_space:
    learning_rate: [1e-3, 1e-2]
    noise_levels:
      - [0.01]
      - [0.1]
  num_trials: 4
  seeds: [1, 2, 3]
  timeout_seconds: 600
  success_criteria:
    metric: val_loss
    improvement_threshold: 0.15
    baseline: linear_pe
```

## Debugging

### Verbose Logging

```bash
# Show full stack traces
HYDRA_FULL_ERROR=1 uv run tmgg-attention

# Debug mode
uv run tmgg-attention trainer.fast_dev_run=true
```

### Common Issues

**CUDA out of memory**: Reduce batch size
```bash
uv run tmgg-attention data.batch_size=32
```

**NaN gradients**: Enable eigenvalue regularization
```bash
uv run tmgg-gnn model.eigenvalue_reg=0.001
```

**Slow training**: Check GPU availability
```bash
uv run tmgg-attention trainer.accelerator=gpu
```

## Reproducibility

For reproducible experiments:

```bash
# Set seed
uv run tmgg-attention seed=42

# Deterministic mode (slower but reproducible)
uv run tmgg-attention trainer.deterministic=true
```

The full configuration is saved in `outputs/.../config.yaml` for reproduction.
