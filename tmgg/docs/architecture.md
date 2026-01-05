# Architecture

This document describes the system design, module organization, and how components interact.

## Directory Structure

```
src/tmgg/
├── models/                    # Neural network architectures
│   ├── base.py                # BaseModel, DenoisingModel base classes
│   ├── attention/             # Transformer attention models
│   ├── gnn/                   # Graph neural networks
│   ├── hybrid/                # GNN + Transformer combinations
│   ├── layers/                # Shared layers (GCN, MHA, EigenEmbedding)
│   ├── embeddings/            # Graph embedding dimension analysis
│   │   ├── base.py            # GraphEmbedding base class
│   │   ├── lpca.py            # Logistic PCA embeddings
│   │   ├── dot_product.py     # Dot product embeddings
│   │   ├── dot_threshold.py   # Threshold-based dot product
│   │   ├── distance_threshold.py
│   │   ├── orthogonal.py      # Orthogonal representations
│   │   ├── dimension_search.py  # Binary search for min dimension
│   │   └── fitters/           # Gradient and spectral fitting
│   └── spectral_denoisers/    # Spectral positional encoding models
├── experiments/               # Experiment runners
│   ├── attention_denoising/   # Each has lightning_module.py + runner.py
│   ├── gnn_denoising/
│   ├── hybrid_denoising/
│   ├── digress_denoising/
│   ├── spectral_denoising/
│   └── stages/                # Stage runners
├── experiment_utils/          # Shared infrastructure
│   ├── data/                  # Data loading, generation, noise
│   ├── cloud/                 # Cloud execution backends
│   ├── eigenstructure_study/  # Eigenstructure analysis tools
│   ├── embedding_study/       # Embedding dimension study
│   ├── base_lightningmodule.py
│   ├── run_experiment.py
│   ├── metrics.py
│   └── plotting.py
└── exp_configs/               # Hydra configuration files
    ├── base_config_*.yaml     # Top-level experiment configs
    ├── models/                # Model configurations
    ├── data/                  # Data configurations
    ├── base/                  # Trainer, logger configs
    └── stage/                 # Stage definitions
```

## Core Abstractions

### DenoisingModel

The base class for all denoising models, defined in `src/tmgg/models/base.py`. It provides:

- Domain transformations (`standard` or `inv-sigmoid`) for numerical stability
- Parameter counting via `parameter_count()`
- Configuration export via `get_config()`

```python
from tmgg.models.base import DenoisingModel

class MyModel(DenoisingModel):
    def __init__(self, ..., domain: str = "standard"):
        super().__init__(domain=domain)
        # Model setup

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        A_transformed = self._apply_domain_transform(A)
        # Process
        return self._apply_output_transform(output)
```

### DenoisingLightningModule

The PyTorch Lightning base class for experiments, defined in `src/tmgg/experiment_utils/base_lightningmodule.py`. It handles:

- Optimizer and scheduler configuration
- Noise generation (Gaussian, Rotation, Digress)
- Training, validation, and test steps
- Visualization logging

Experiment-specific modules inherit from this and implement `_make_model()`:

```python
class AttentionDenoisingLightningModule(DenoisingLightningModule):
    def _make_model(self, d_model, num_heads, num_layers, ...):
        return MultiLayerAttention(d_model, num_heads, num_layers, ...)
```

### GraphDataModule

The data loading abstraction in `src/tmgg/experiment_utils/data/data_module.py`. Supports:

- Multiple dataset types (SBM, NetworkX, PyG, synthetic)
- Train/val/test splitting
- Batch loading with noise injection

### CloudRunner and CloudRunnerFactory

The cloud execution abstraction in `src/tmgg/experiment_utils/cloud/`. The factory pattern allows registering multiple backends:

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

runner = CloudRunnerFactory.create("local")  # or "modal"
result = runner.run_experiment(config)
```

## Execution Flow

When you run an experiment:

```
CLI Entry Point (e.g., tmgg-attention)
    │
    ▼
@hydra.main decorator loads configuration
    │
    ▼
run_experiment(config)
    ├── set_seed(config.seed)
    ├── hydra.utils.instantiate(config.data) → GraphDataModule
    ├── hydra.utils.instantiate(config.model) → LightningModule
    ├── create_callbacks(config) → [ModelCheckpoint, EarlyStopping, ...]
    ├── create_loggers(config) → [TensorBoard, WandB, ...]
    ├── hydra.utils.instantiate(config.trainer) → Trainer
    ├── maybe_run_sanity_check()
    ├── trainer.fit(model, datamodule)
    ├── trainer.test(model, datamodule)
    └── final_eval() → evaluation at multiple noise levels
```

Each step is driven by the Hydra configuration. The `_target_` fields in YAML configs specify which classes to instantiate.

## Design Patterns

### Factory Pattern

`CloudRunnerFactory` registers and creates execution backends:

```python
CloudRunnerFactory.register("modal", ModalRunner)
runner = CloudRunnerFactory.create("modal", **kwargs)
```

### Strategy Pattern

Noise generators, loggers, and optimizers are interchangeable via configuration. The same training loop works with different strategies.

### Template Method

`DenoisingLightningModule` defines the training algorithm structure. Subclasses override `_make_model()` to specify the model architecture.

### Composition

`SequentialDenoisingModel` composes a GNN embedding model with a transformer denoiser:

```python
model = SequentialDenoisingModel(
    embedding_model=GNN(...),
    denoising_model=MultiLayerAttention(...)
)
```

## Key Files by Purpose

**To understand training flow:**
- `experiment_utils/run_experiment.py` - Main orchestration
- `experiment_utils/base_lightningmodule.py` - Training loop

**To understand models:**
- `models/base.py` - Base classes
- `models/gnn/gnn.py` - GNN implementation
- `models/attention/attention.py` - Attention implementation

**To understand data:**
- `experiment_utils/data/data_module.py` - Data loading
- `experiment_utils/data/noise_generators.py` - Noise models

**To understand configuration:**
- `exp_configs/base_config_*.yaml` - Top-level configs
- Individual model/data YAML files
