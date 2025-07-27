# Logging Configuration Guide

This guide explains how to configure logging for tmgg experiments using the new flexible logging system.

## Overview

The logging system now supports multiple logger backends simultaneously:
- **TensorBoard** (default) - Supports local and S3 storage
- **Weights & Biases** - Cloud-based experiment tracking
- **CSV** - Lightweight local logging

## Quick Start

### Using TensorBoard (Default)

```bash
# Local logging (default)
python -m tmgg.experiments.attention_denoising

# Results will be saved to outputs/*/tensorboard/
```

### Using TensorBoard with S3

```bash
# Configure AWS credentials first
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Run with S3 logging
python -m tmgg.experiments.attention_denoising logger=tensorboard_s3
```

### Using Weights & Biases

```bash
# Login to wandb first
wandb login

# Run with wandb logging
python -m tmgg.experiments.attention_denoising logger=wandb
```

### Using Multiple Loggers

```bash
# Log to both TensorBoard and CSV
python -m tmgg.experiments.attention_denoising logger=multi
```

### Disabling Legacy wandb Config

To use only the new logger configuration without the legacy wandb config:

```bash
python -m tmgg.experiments.attention_denoising logger=tensorboard wandb=null
```

## Configuration Options

### TensorBoard Logger

```yaml
logger:
  - tensorboard:
      save_dir: ${paths.output_dir}/tensorboard  # or s3://bucket/path
      name: ${experiment_name}
      version: null  # Auto-versioning
      log_graph: false
      default_hp_metric: false
```

### Weights & Biases Logger

```yaml
logger:
  - wandb:
      project: your-project-name
      name: ${experiment_name}
      tags: ["tag1", "tag2"]
      save_dir: ${paths.output_dir}
      entity: your-team  # Optional
      log_model: false
```

### CSV Logger

```yaml
logger:
  - csv:
      save_dir: ${paths.output_dir}/csv
      name: ${experiment_name}
      version: null
      flush_logs_every_n_steps: 100
```

## Viewing Results

### TensorBoard

```bash
# Local logs
tensorboard --logdir outputs/

# S3 logs (requires s3fs)
pip install s3fs
tensorboard --logdir s3://your-bucket/experiments/
```

### Weights & Biases

View results at https://wandb.ai/your-entity/your-project

### CSV Logs

CSV files are saved to the configured directory with metrics in tabular format.

## Implementation Details

The logging system uses PyTorch Lightning's native logger support. Key components:

1. **Logger Factory** (`tmgg.experiment_utils.logging.create_loggers`): Creates logger instances based on configuration
2. **Figure Logging** (`tmgg.experiment_utils.logging.log_figure`): Handles matplotlib figures across different logger types
3. **Configuration**: Hydra-based configuration in `config/logger/` directories

## Backward Compatibility

The system maintains backward compatibility with existing wandb configurations. If you have `wandb` config in your experiment yaml, it will be used unless you override with a logger config.