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
# Spectral denoising (main experiment type)
uv run tmgg-spectral-arch

# GNN-based denoising with custom training steps
uv run tmgg-gnn trainer.max_steps=50000

# Spectral denoising with specific eigenvector count
uv run tmgg-spectral-arch model.k=50

# Run with Weights & Biases logging (enabled by default when WANDB_API_KEY is set)
WANDB_API_KEY="your-api-key" uv run tmgg-spectral-arch
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
| `WANDB_API_KEY` | No | — | Weights & Biases API key. Needed when W&B logging is enabled; most training configs enable it by default. |

```bash
export WANDB_API_KEY="your-api-key"
uv run tmgg-spectral-arch
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tmgg-gnn` | GNN-based denoising |
| `tmgg-gnn-transformer` | GNN + Transformer hybrid denoising |
| `tmgg-digress` | DiGress transformer model |
| `tmgg-spectral-arch` | Spectral positional encoding denoising |
| `tmgg-gaussian-gen` | Gaussian diffusion generative |
| `tmgg-discrete-gen` | Discrete diffusion generative (DiGress) |
| `tmgg-discrete-eval` | Discrete diffusion evaluation |
| `tmgg-baseline` | Linear/MLP baseline denoising |
| `tmgg-experiment` | Unified stage runner (e.g., `+stage=stage1_poc`) |
| `tmgg-grid-search` | Hyperparameter grid search |
| `tmgg-wandb-export` | Export W&B metrics to CSV |
| `tmgg-tb-export` | Export TensorBoard metrics |
| `tmgg-eigenstructure` | Eigenstructure study (collect, analyze, noised, compare) |
| `tmgg-embedding-study` | Embedding dimension study (run, analyze) |

All commands support Hydra overrides:

```bash
# Override model parameters
uv run tmgg-spectral-arch model.k=50 model.d_k=128

# Override training steps and learning rate
uv run tmgg-gnn trainer.max_steps=50000 model.learning_rate=0.001

# Hyperparameter sweep
uv run tmgg-spectral-arch --multirun model.k=8,16,32
```

### W&B Project Naming

Each CLI command logs to a specific W&B project when W&B logging is enabled. Most
training configs inherit the default W&B logger from `_base_infra.yaml`; the
project name is set in the corresponding base config and can be overridden with
`wandb_project=...`.

| CLI Commands | W&B Project | Base Config |
|-------------|-------------|-------------|
| `tmgg-gnn`, `tmgg-gnn-transformer`, `tmgg-spectral-arch`, `tmgg-digress`, `tmgg-baseline` | `architecture-study` | `base_config_{gnn,gnn_transformer,spectral_arch,digress,baseline}.yaml` |
| `tmgg-gaussian-gen` | `gaussian-diffusion` | `base_config_gaussian_diffusion.yaml` |
| `tmgg-discrete-gen`, `tmgg-discrete-eval` | `discrete-diffusion` | `base_config_discrete_diffusion_generative.yaml` |
| `tmgg-grid-search` | `tmgg-grid-search-4k` | `grid_search_base.yaml` |

The shared `_base_infra.yaml` composes the common trainer, logger, callbacks, and
path settings that experiment-specific base configs build on top of.

## Experiment Analysis

W&B experiment data is managed through standalone scripts in `wandb-tools/`:

| Script | Description |
|--------|-------------|
| `wandb-tools/export_runs.py` | Export W&B runs to parquet files |
| `wandb-tools/aggregate_runs.py` | Aggregate and postprocess exported data |
| `wandb-tools/analyze_runs.py` | CLI analysis of aggregated run data |
| `wandb-tools/list_entities.py` | List accessible W&B teams and projects |

```bash
# Export runs from a W&B project
uv run wandb-tools/export_runs.py --entity graph_denoise_team --project architecture-study
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
│   │   ├── gnn/             # Graph neural networks
│   │   ├── layers/          # Shared layers (GCN, MHA, Eigen)
│   │   ├── embeddings/      # Graph embedding dimension analysis
│   │   ├── spectral_denoisers/
│   │   └── factory.py       # Registry-based model factory
│   ├── experiments/         # Experiment runners
│   │   ├── spectral_arch_denoising/
│   │   ├── digress_denoising/
│   │   ├── discrete_diffusion_generative/
│   │   ├── gnn_denoising/
│   │   ├── gnn_transformer_denoising/
│   │   ├── gaussian_diffusion_generative/
│   │   ├── lin_mlp_baseline_denoising/
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

The framework supports multiple noise models for training and evaluation:

- **Gaussian**: Additive Gaussian noise to adjacency matrices
- **Rotation**: Eigenspace rotation via skew-symmetric matrices
- **Digress**: Categorical transition matrices (Vignac et al. 2023), interpolating between identity and uniform distribution
- **Edge Flip**: Simple Bernoulli edge flipping
- **Logit**: Gaussian noise in logit space, producing soft adjacency values

## Testing

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
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

## SBM Repro Report Script

`scripts/sbm_repro_report.py` — self-contained `uv run --script`
(inline deps: `wandb`, `pandas`, `pyarrow`, `matplotlib`, `Pillow`)
that builds a comparison report across the 5 SBM repro variants
running on W&B (`vignac`, `pearl`, `pearl-spec`, `pearl-gnnconv-norm`,
`pearl-gnnconv-raw`).

```bash
./scripts/sbm_repro_report.py            # cache-aware (idempotent)
./scripts/sbm_repro_report.py --refresh  # bypass cache, re-fetch from W&B
```

The W&B API key is read from `GRAPH_DENOISE_TEAM_SERVICE` in `.env`.

**Outputs** (under `wandb_export/sbm-repro-report-2026-05-05/`):

- `data/<variant>/history.parquet` — per-metric `scan_history()` dump.
  One `(global_step, metric)` pair at a time, because
  `scan_history(keys=[...])` only emits rows that contain *every*
  listed key, and train/val/gen-val log on different cadences, so a
  single multi-key fetch returns the empty intersection.
- `data/<variant>/summary.json` — last-known summary metrics, run
  state, and W&B URL.
- `media/<variant>/<kind>_step<N>.png` — evenly spaced graph and
  adjacency sample renderings (`--n-images-per-kind`, default 4).
- `figures/curves_<metric>.png` — overlay plots of train loss
  (step + epoch), val NLL, the four gen-val MMDs, and sbm_accuracy.
  Long traces (>500 points) get a faint raw line plus a rolling-mean
  overlay (window ~1% of length) so trends survive visual saturation.
- `figures/timeline_{graph,adj}.png` — variant × step grids of the
  generated samples, for visual quality progression.
- `report.typ` — Typst document embedding all figures plus
  like-to-like comparison tables at the steps every variant evaluated.
- `report.pdf` — auto-compiled when `typst` is on `PATH`.

**Idempotency:** parquet/json/png artefacts are reused when present;
re-runs without `--refresh` skip the W&B fetch entirely. Figure
regeneration and Typst compile always run (they are cheap).

To extend the script to a new variant, append a `(label, project,
display_name)` triple to `RUNS` at the top of the file. To track
new metrics, add the W&B key to `TRAIN_KEYS`, `VAL_KEYS`, or `GEN_KEYS`
and append a `CURVE_PLOTS` entry.

## License

See LICENSE file for details.
