# TMGG: Graph Denoising Research Framework

A research framework for graph denoising using attention mechanisms, graph neural networks, and hybrid architectures. Supports multiple noise models, reproducible experiments via Hydra configs, and cloud execution.

## Installation

```bash
git clone <repository-url>
cd tmgg
uv sync
```

For development with test dependencies:

```bash
uv sync --all-extras
```

## Quick Start

Run your first experiment:

```bash
# Attention-based denoising
uv run tmgg-attention

# GNN-based denoising with custom training steps
uv run tmgg-gnn trainer.max_steps=50000

# Spectral denoising with specific eigenvector count
uv run tmgg-spectral model.k=50

# Run with Weights & Biases logging
uv run tmgg-attention logger=wandb
```

Note: Training is configured in **steps**, not epochs (see [Configuration](docs/configuration.md)).

## Environment Variables

All environment variables are **optional for local runs**. They configure cloud execution, storage backends, and logging integrations.

### Path Discovery (Modal)

| Variable | Required | Description |
|----------|----------|-------------|
| `TMGG_PATH` | No | Path to tmgg package root (directory containing `src/tmgg/`). Auto-discovered if `modal/` and `tmgg/` are siblings. Only set for non-standard directory layouts. |

**Auto-discovery**: In the standard repo layout where `modal/` and `tmgg/` are siblings, path discovery works automatically:
```
my_project/
├── modal/      # tmgg_modal package
└── tmgg/       # tmgg package (auto-discovered)
```

### S3-Compatible Storage

Used for checkpoint persistence and metrics storage. Required only when using `S3Storage` backend.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TMGG_S3_BUCKET` | Yes* | — | S3 bucket name |
| `TMGG_S3_ENDPOINT` | No | AWS default | Custom endpoint URL (for MinIO, Tigris, etc.) |
| `TMGG_S3_ACCESS_KEY` | Yes* | — | AWS access key ID |
| `TMGG_S3_SECRET_KEY` | Yes* | — | AWS secret access key |
| `TMGG_S3_REGION` | No | `us-east-1` | AWS region |

*Required only when using S3Storage backend.

```bash
export TMGG_S3_BUCKET="my-experiments"
export TMGG_S3_ACCESS_KEY="AKIA..."
export TMGG_S3_SECRET_KEY="..."
```

### Tigris Storage (Modal-native)

S3-compatible storage optimized for Modal. Used by `tmgg_modal` package. Configure as Modal secrets.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TMGG_TIGRIS_BUCKET` | Yes* | — | Tigris bucket name |
| `TMGG_TIGRIS_ENDPOINT` | No | `https://fly.storage.tigris.dev` | Tigris endpoint |
| `TMGG_TIGRIS_ACCESS_KEY` | Yes* | — | Tigris access key |
| `TMGG_TIGRIS_SECRET_KEY` | Yes* | — | Tigris secret key |

*Required only when using TigrisStorage with Modal.

```bash
modal secret create tigris-credentials \
  TMGG_TIGRIS_BUCKET=my-bucket \
  TMGG_TIGRIS_ACCESS_KEY=... \
  TMGG_TIGRIS_SECRET_KEY=...
```

### Logging

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WANDB_API_KEY` | No | — | Weights & Biases API key. Required only when using `logger=wandb`. |

```bash
export WANDB_API_KEY="your-api-key"
uv run tmgg-attention logger=wandb
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tmgg-attention` | Attention-based denoising |
| `tmgg-gnn` | GNN-based denoising |
| `tmgg-hybrid` | GNN + Transformer hybrid |
| `tmgg-digress` | DiGress transformer model |
| `tmgg-spectral` | Spectral positional encoding models |
| `tmgg-stage1` | Stage 1: Proof of concept (4.4 GPU-hours) |
| `tmgg-stage1-sanity` | Stage 1 Sanity: Constant noise memorization |
| `tmgg-stage1-5` | Stage 1.5: Cross-dataset validation |
| `tmgg-stage2` to `tmgg-stage5` | Later stage experiments |
| `tmgg-grid-search` | Hyperparameter grid search |
| `tmgg-wandb-export` | Export W&B metrics to CSV |
| `tmgg-tb-export` | Export TensorBoard metrics |

All commands support Hydra overrides:

```bash
# Override model parameters
uv run tmgg-attention model.num_layers=16 model.num_heads=8

# Override training steps and learning rate
uv run tmgg-gnn trainer.max_steps=50000 model.learning_rate=0.001

# Hyperparameter sweep
uv run tmgg-attention --multirun model.num_layers=4,8,16
```

## Project Structure

```
tmgg/
├── src/tmgg/
│   ├── models/              # Neural network architectures
│   │   ├── attention/       # Transformer attention models
│   │   ├── gnn/             # Graph neural networks
│   │   ├── hybrid/          # GNN + Transformer combinations
│   │   ├── layers/          # Shared layers (GCN, MHA, Eigen)
│   │   └── spectral_denoisers/
│   ├── experiments/         # Experiment runners
│   │   ├── attention_denoising/
│   │   ├── gnn_denoising/
│   │   ├── hybrid_denoising/
│   │   ├── digress_denoising/
│   │   ├── spectral_denoising/
│   │   └── stages/          # Multi-stage experiments
│   ├── experiment_utils/    # Shared infrastructure
│   │   ├── data/            # Data loading and generation
│   │   ├── cloud/           # Cloud execution (Modal)
│   │   ├── base_lightningmodule.py
│   │   ├── run_experiment.py
│   │   ├── metrics.py
│   │   └── plotting.py
│   └── exp_configs/         # Hydra configuration files
│       ├── base_config_*.yaml
│       ├── models/
│       ├── data/
│       └── stages/
├── tests/                   # Test suite
└── docs/                    # Detailed documentation
```

## Documentation

For detailed documentation, see the [docs/](docs/) folder:

- [Architecture](docs/architecture.md) - System design and module organization
- [Configuration](docs/configuration.md) - Hydra config system and common overrides
- [Models](docs/models.md) - Model architectures and parameters
- [Data](docs/data.md) - Data pipeline, datasets, and noise types
- [Experiments](docs/experiments.md) - Running experiments and interpreting results
- [Cloud](docs/cloud.md) - Cloud execution with Modal
- [Extending](docs/extending.md) - Adding new models, datasets, and backends

## Model Architectures

**Spectral Denoisers**: The main focus of current experiments. Three architectures operating in the spectral domain:
- Linear PE: Â = V W V^T + bias
- Filter Bank: Polynomial spectral filters
- Self-Attention: Query-key attention on eigenvectors

**DiGress**: Diffusion-based transformer baseline for comparison.

**Attention Models**: Multi-layer transformer attention processing adjacency matrices directly.

**GNN Models**: Spectral graph neural networks using eigendecomposition embeddings. Variants include standard GNN, symmetric GNN (shared embeddings), and node-variant GNN.

**Hybrid Models**: Combine GNN embeddings with transformer-based denoising.

## Noise Types

The framework supports three noise models for training and evaluation:

- **Gaussian**: Additive Gaussian noise to adjacency matrices
- **Rotation**: Eigenspace rotation via skew-symmetric matrices
- **Digress**: Edge flipping with configurable probability

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tmgg --cov-report=html

# Run specific test file
uv run pytest tests/test_integration.py -v
```

## License

See LICENSE file for details.
