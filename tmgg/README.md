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
| `tmgg-experiment` | Unified stage runner (e.g., `+stage=stage1_poc`) |
| `tmgg-grid-search` | Hyperparameter grid search |
| `tmgg-wandb-export` | Export W&B metrics to CSV |
| `tmgg-tb-export` | Export TensorBoard metrics |
| `tmgg-modal-stage1` | Modal stage 1 runner |
| `tmgg-modal-stage2` | Modal stage 2 runner |
| `tmgg-eigenstructure` | Eigenstructure study (collect, analyze, noised, compare) |
| `tmgg-embedding-study` | Embedding dimension study (run, analyze) |

All commands support Hydra overrides:

```bash
# Override model parameters
uv run tmgg-attention model.num_layers=16 model.num_heads=8

# Override training steps and learning rate
uv run tmgg-gnn trainer.max_steps=50000 model.learning_rate=0.001

# Hyperparameter sweep
uv run tmgg-attention --multirun model.num_layers=4,8,16
```

## Experiment Analysis

Scripts for analyzing W&B experiment results and generating reports:

| Script | Description |
|--------|-------------|
| `scripts/fetch_wandb_runs.py` | Fetch runs from W&B to JSON |
| `scripts/analyze_experiments.py` | Download data, hyperparameter importance analysis |
| `scripts/analyze_wandb_runs.py` | Analyze exported JSON with grouping/filtering |
| `scripts/experiment_breakdown.py` | Generate breakdown tables by semantic groupings |
| `scripts/semantic_analysis.py` | Statistical significance tests across groupings |

```bash
# Full analysis pipeline
uv run scripts/analyze_experiments.py

# Use cached data (skip download)
uv run scripts/analyze_experiments.py --skip-download

# Generate all breakdown reports
uv run scripts/experiment_breakdown.py --mode full
```

### Key Findings (Eigenstructure Study)

Analysis of 2013 W&B runs comparing graph denoising approaches:

- **DiGress outperforms Spectral**: Mean MSE 0.087 vs 0.187
- **Stage2c optimal**: Achieves best results (MSE 0.075)
- **k=32 optimal**: Higher k (50) doesn't improve performance
- **Avoid asymmetric attention**: 2.7x worse than symmetric
- **Architecture choice inconsequential**: GNN variants (gnn_all, gnn_v, gnn_qk) equivalent to default
- **Filter bank wins on specific datasets**: pyg_enzymes, ring_of_cliques show filter_bank advantage

See `eigenstructure_results_full/analysis_summary.md` for full analysis and `eigenstructure_results_full/architecture_comparison.md` for per-dataset architecture comparison.

## Project Structure

```
tmgg/
├── src/tmgg/
│   ├── models/              # Neural network architectures
│   │   ├── attention/       # Transformer attention models
│   │   ├── gnn/             # Graph neural networks
│   │   ├── hybrid/          # GNN + Transformer combinations
│   │   ├── layers/          # Shared layers (GCN, MHA, Eigen)
│   │   ├── embeddings/      # Graph embedding dimension analysis
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
│   │   ├── eigenstructure_study/  # Eigenstructure analysis tools
│   │   ├── embedding_study/       # Embedding dimension study
│   │   ├── base_lightningmodule.py
│   │   ├── run_experiment.py
│   │   ├── metrics.py
│   │   └── plotting.py
│   └── exp_configs/         # Hydra configuration files
│       ├── base_config_*.yaml
│       ├── models/
│       ├── data/
│       └── stage/
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

## Code Quality

Pre-commit hooks enforce code quality. The hooks are defined in `.pre-commit-config.yaml` but require setup at the git root (parent directory in this monorepo). Run checks manually:

```bash
# Linting and formatting
uv run ruff check --fix src/
uv run ruff format src/

# Type checking
uv run basedpyright --project pyproject.toml

# Module boundary enforcement
uv run tach check
```

## License

See LICENSE file for details.
