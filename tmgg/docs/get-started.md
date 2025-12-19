# Get started

This guide walks you through installation and running your first experiment.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installed

## Install

From the repository root:

```bash
uv sync
```

For development and tests:

```bash
uv sync --all-extras
```

## Run your first experiment

```bash
# Attention-based denoising
uv run tmgg-attention
```

Other entry points:

```bash
# GNN-based denoising
uv run tmgg-gnn

# Hybrid GNN + Transformer
uv run tmgg-hybrid

# DiGress transformer
uv run tmgg-digress

# Spectral denoising
uv run tmgg-spectral
```

## Common overrides

TMGG uses Hydra overrides. A few examples:

```bash
# Increase training steps
uv run tmgg-attention trainer.max_steps=50000

# Change model depth
uv run tmgg-gnn model.num_layers=8

# Switch dataset configuration
uv run tmgg-attention data=legacy_match
```

Note: training is configured in **steps**, not epochs.

## Next steps

- Read [Configuration](configuration.md) to understand the Hydra hierarchy.
- See [Experiments](experiments.md) for stage-based workflows and outputs.
- Check [Data](data.md) and [Models](models.md) for dataset and architecture details.
