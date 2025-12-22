# TMGG Configuration Guide

This directory contains Hydra configuration files for TMGG experiments.

## Quick Start

```bash
# Single experiment
tmgg-experiment +stage=stage1_poc

# Single experiment with model override
tmgg-experiment +stage=stage1_poc model=models/spectral/filter_bank

# Sweep (using stage's _sweep_config)
tmgg-experiment +stage=stage1_poc sweep=true

# Multirun via Hydra (local)
tmgg-experiment +stage=stage1_poc --multirun \
  model=models/spectral/linear_pe,models/spectral/filter_bank \
  learning_rate=1e-3,1e-2

# Multirun via Modal cloud
tmgg-experiment +stage=stage1_poc --multirun \
  hydra/launcher=tmgg_modal \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

## Config Precedence

Configuration is composed in the following order (later overrides earlier):

1. `base_config_training.yaml` - Shared training infrastructure (optimizer, scheduler, noise, paths)
2. `base_config_*.yaml` - Experiment-type settings (model defaults, wandb project)
3. `stage/*.yaml` - Stage-specific overrides (via +stage=...)
4. `models/**/*.yaml` - Model architecture (via model=...)
5. `data/*.yaml` - Dataset configuration (via data=...)
6. CLI overrides - Highest precedence

## Directory Structure

```
exp_configs/
├── base_config_training.yaml    # Shared training settings (optimizer, scheduler, noise, paths)
├── base_config_spectral.yaml    # Spectral experiments (inherits from training)
├── base_config_attention.yaml   # Attention experiments (inherits from training)
├── base_config_gnn.yaml         # GNN experiments (inherits from training)
├── base_config_hybrid.yaml      # Hybrid experiments (inherits from training)
├── base_config_digress.yaml     # DiGress experiments (inherits from training)
├── base_dataloader.yaml         # Shared dataloader settings
├── data/                        # Dataset configurations
│   ├── single_graph_base.yaml   # Base for single-graph datasets
│   ├── er_single_graph.yaml     # Erdos-Renyi
│   ├── sbm_single_graph.yaml    # Stochastic Block Model
│   └── pyg_*_single_graph.yaml  # PyG benchmark datasets
├── models/                      # Model architectures
│   ├── spectral/                # Spectral denoising models
│   │   ├── linear_pe.yaml
│   │   ├── filter_bank.yaml
│   │   └── self_attention.yaml
│   └── digress/                 # DiGress baselines
│       └── digress_sbm_small.yaml
├── stage/                       # Stage-specific configs
│   ├── stage1_poc.yaml          # Proof of concept
│   ├── stage1_sanity.yaml       # Sanity check
│   └── stage2_validation.yaml   # Cross-dataset validation
└── hydra/                       # Hydra-specific configs
    └── launcher/                # Custom launchers
        ├── tmgg.yaml            # Local execution
        ├── tmgg_modal.yaml      # Modal cloud execution
        └── tmgg_slurm.yaml      # SLURM cluster execution
```

## Optimizer Settings

All experiments use AdamW with these defaults:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-2
  weight_decay: 1e-12
  amsgrad: true
```

## Available Stages

| Stage | Command | Description |
|-------|---------|-------------|
| stage1_poc | `+stage=stage1_poc` | Initial ER graph training |
| stage1_sanity | `+stage=stage1_sanity` | Constant noise memorization test |
| stage2_validation | `+stage=stage2_validation` | Multi-dataset validation |

For cross-graph validation (generalization test):
```bash
tmgg-experiment +stage=stage2_validation cross_graph=true
```

## Model Architectures

### Spectral Denoising Models

| Config | Description |
|--------|-------------|
| `models/spectral/linear_pe` | Linear positional encoding |
| `models/spectral/filter_bank` | Spectral filter bank |
| `models/spectral/self_attention` | Self-attention baseline |

### DiGress Baselines

| Config | Description |
|--------|-------------|
| `models/digress/digress_sbm_small` | DiGress with official learning rate |
| `models/digress/digress_sbm_small_highlr` | DiGress with high learning rate (ablation) |

## Data Configurations

All single-graph configs inherit from `data/single_graph_base.yaml`:

| Config | Description |
|--------|-------------|
| `data/er_single_graph` | Erdos-Renyi random graphs |
| `data/sbm_single_graph` | Stochastic Block Model |
| `data/tree_single_graph` | Random trees |
| `data/pyg_qm9_single_graph` | QM9 molecular graphs |
| `data/pyg_enzymes_single_graph` | ENZYMES protein structures |
| `data/pyg_proteins_single_graph` | PROTEINS dataset |

## Cloud Execution

### Local Development
```bash
tmgg-experiment +stage=stage1_poc
```

### Modal Cloud
```bash
tmgg-experiment +stage=stage1_poc --multirun \
  hydra/launcher=tmgg_modal \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

### SLURM Cluster

For HPC clusters with SLURM, use the `tmgg_slurm` launcher:

```bash
tmgg-experiment +stage=stage1_poc hydra/launcher=tmgg_slurm
```

Override SLURM parameters on the command line:

```bash
tmgg-experiment +stage=stage1_poc hydra/launcher=tmgg_slurm \
  hydra.launcher.slurm_partition=gpu \
  hydra.launcher.slurm_nodes=4 \
  hydra.launcher.slurm_gpus_per_task=1 \
  hydra.launcher.slurm_time_limit="08:00:00"
```

Configure environment setup for your cluster:

```bash
# In a custom launcher config or via overrides
hydra.launcher.slurm_setup_commands='["module load cuda/12.1","conda activate tmgg"]'
```

See `docs/slurm.md` for detailed configuration and troubleshooting.

### Sweep with Stage Config
```bash
tmgg-experiment +stage=stage1_poc sweep=true run_on_modal=true
```

## Common Overrides

```bash
# Change learning rate
tmgg-experiment +stage=stage1_poc learning_rate=5e-3

# Change training steps
tmgg-experiment +stage=stage1_poc trainer.max_steps=10000

# Change batch size
tmgg-experiment +stage=stage1_poc data.batch_size=32

# Disable WandB logging
tmgg-experiment +stage=stage1_poc ~logger
```
