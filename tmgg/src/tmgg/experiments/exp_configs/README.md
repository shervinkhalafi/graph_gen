# TMGG Configuration Guide

This directory contains Hydra configuration files for TMGG experiments.

## Quick Start

```bash
# Single experiment (local)
tmgg-experiment +stage=stage1_poc

# Single experiment with model override
tmgg-experiment +stage=stage1_poc model=models/spectral/filter_bank

# Multirun via Hydra (local, built-in launcher)
tmgg-experiment +stage=stage1_poc --multirun \
  model=models/spectral/linear_pe,models/spectral/filter_bank \
  learning_rate=1e-3,1e-2

# Single experiment on Modal
tmgg-modal run tmgg-spectral-arch model.k=16 seed=1

# Single experiment on Modal (dry-run: resolve config, print summary, exit)
tmgg-modal run tmgg-spectral-arch --dry-run

# Single experiment on Modal (detached / fire-and-forget)
tmgg-modal run tmgg-spectral-arch seed=2 --detach --gpu fast

# Multirun on Modal via Hydra launcher
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
├── base_config_spectral_arch.yaml  # Spectral architecture denoising (inherits from training)
├── base_config_gnn.yaml            # GNN denoising (inherits from training)
├── base_config_gnn_transformer.yaml # GNN+Transformer denoising (inherits from training)
├── base_config_digress.yaml        # DiGress denoising (inherits from training)
├── base_config_gaussian_diffusion.yaml         # Gaussian diffusion generation (inherits from training)
├── base_config_discrete_diffusion_generative.yaml # Categorical diffusion generation (inherits from training)
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
        └── tmgg_modal.yaml      # Modal cloud execution
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

## Cloud Execution (Modal)

### Single Experiment

`tmgg-modal run` resolves the Hydra config locally via compose API, then dispatches the resolved config to Modal. Output paths are patched to use the Modal volume (`/data/outputs`).

```bash
# Standard GPU (A10G), blocking
tmgg-modal run tmgg-spectral-arch model.k=16 seed=1

# Fast GPU (A100), detached (fire-and-forget)
tmgg-modal run tmgg-spectral-arch seed=2 --detach --gpu fast

# Dry run: resolve config and print summary without dispatching
tmgg-modal run tmgg-spectral-arch --dry-run
```

Available `--gpu` tiers: `debug` (T4), `standard` (A10G, default), `fast` (A100), `multi` (A100x2), `h100` (H100).

### Multirun on Modal

Use Hydra's `--multirun` with the `tmgg_modal` launcher. The launcher patches output paths to the Modal volume (same as `tmgg-modal run`) and dispatches all jobs via `ModalRunner.run_sweep()`.

```bash
tmgg-experiment +stage=stage1_poc --multirun \
  hydra/launcher=tmgg_modal \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

For local sweeps, use Hydra's built-in launcher:
```bash
tmgg-experiment +stage=stage1_poc --multirun hydra/launcher=basic \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

### W&B Logging

W&B logging is required by default. To degrade a missing `WANDB_API_KEY`
to a warning and continue without W&B logging, pass `allow_no_wandb=true`:

```bash
tmgg-experiment +stage=stage1_poc allow_no_wandb=true
```

## Common Overrides

```bash
# Change learning rate
tmgg-experiment +stage=stage1_poc learning_rate=5e-3

# Change training steps
tmgg-experiment +stage=stage1_poc trainer.max_steps=10000

# Change batch size
tmgg-experiment +stage=stage1_poc data.batch_size=32

# Allow training without W&B logging
tmgg-experiment +stage=stage1_poc allow_no_wandb=true
```
