# TMGG: Graph Denoising Experiments

## Installation

From the tmgg directory:

```bash
uv venv
source .venv/bin/activate
rye sync
uv pip install -e .
```

For development with testing:

```bash
pip install -e ".[test]"
```

## Quick Start

### Running Experiments

Each experiment can be run with its command-line interface:

```bash
# Attention-based denoising
tmgg-attention

# GNN-based denoising  
tmgg-gnn

# Hybrid GNN+Transformer denoising
tmgg-hybrid
# Digress GraphTensformer denoising
tmgg-digress
```

### Configuration Management

Experiments use Hydra for configuration. You can override any parameter:

```bash
# Change model architecture
tmgg-attention model.num_layers=4 model.num_heads=16

# Change data parameters
tmgg-gnn data.num_nodes=50 data.batch_size=64

# Run hyperparameter sweep
tmgg-hybrid --multirun model.gnn_feature_dim_out=5,10,15
```

### Model Variants

#### Attention Models

- `multi_layer_attention`: Standard transformer with configurable layers/heads
- `digress`: Digress graph transformer

#### GNN Models  

- `standard_gnn`: Basic GNN with eigenvalue embeddings
- `nodevar_gnn`: Node-variant GNN for heterogeneous graphs
- `symmetric_gnn`: Symmetric GNN using shared embeddings

#### Hybrid Models

- `hybrid_with_transformer`: GNN embeddings + transformer denoising
- `hybrid_gnn_only`: GNN embeddings without transformer (baseline)

## Architecture

### Shared Components

**Models** (`tmgg.models`):

- Base classes for denoising models
- Attention mechanisms (multi-head, multi-layer)
- GNN variants (standard, node-variant, symmetric)
- Hybrid architectures (for `SequentialDenoising`)
- Digress model

**Experiment Utils** (`tmgg.experiment_utils`):

- Data generation (SBM graphs, noise functions) + a data model
- Metrics (eigenvalue error, subspace distance, reconstruction metrics)
- Plotting (training curves, denoising results, noise level comparisons)
- Statistical analysis (confidence intervals, hypothesis testing)

### Experiment Structure

Each experiment follows a consistent structure:

```
experiment_utils/data/data_module.py # shared data module wrapping dataset
experiments/{experiment_name}/
├── config/
│   ├── model/          # Model configurations
│   ├── data/           # Data configurations  
│   ├── trainer/        # Training configurations
│   └── experiment/     # Main experiment configurations
├── src/
│   ├── lightning_module.py  # PyTorch Lightning wrapper
│   └── runner.py            # Hydra entry point
└── pyproject.toml           # Package configuration
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=tmgg --cov-report=html
```

## Configuration Examples

### Custom Model Configuration

Create `config/model/custom_attention.yaml`:

```yaml
_target_: tmgg.experiments.attention_denoising.lightning_module.AttentionDenoisingLightningModule

d_model: 32
num_heads: 4
num_layers: 6
learning_rate: 0.0005
loss_type: "BCE"
```

Then run:

```bash
tmgg-attention model=custom_attention
```

### Multi-run Experiments

Using <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/> makes it easy to try variations

```bash
# Test different noise levels
tmgg-hybrid --multirun evaluation.noise_levels="[0.1,0.2,0.3]","[0.2,0.4,0.6]"

# Compare model architectures  
tmgg-gnn --multirun model=standard_gnn,nodevar_gnn,symmetric_gnn

# Hyperparameter optimization
tmgg-attention --multirun model.learning_rate=0.001,0.005,0.01 model.num_layers=4,8,12
```

## Development

### Adding New Models

1. Implement in `tmgg/src/tmgg/models/`
2. Add to `__init__.py` exports
3. Create configuration files in experiment modules
4. Add tests in `tmgg/tests/models/`

### Adding New Experiments

1. Create experiment directory structure
2. Optionally tweak data_module  if you add new data
3. Implement Lightning module
4. Create Hydra configurations
5. Add entry point to main `pyproject.toml`
6. Create tests for experiment-specific logic
