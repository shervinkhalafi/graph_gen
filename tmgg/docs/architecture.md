# Architecture

This document describes the system design, module organization, and how components interact.

## Directory Structure

```
src/tmgg/
├── models/                    # Neural network architectures
│   ├── base.py                # BaseModel, GraphModel base classes
│   ├── digress/               # DiGress GraphTransformer
│   ├── gnn/                   # Graph neural networks
│   ├── layers/                # Shared layers (GCN, MHA, EigenEmbedding)
│   ├── spectral_denoisers/    # Spectral positional encoding models
│   └── factory.py             # Registry-based model factory
├── diffusion/                 # Diffusion framework
│   ├── noise_process.py       # NoiseProcess ABC + implementations
│   ├── sampler.py             # Sampler ABC + implementations
│   └── schedule.py            # NoiseSchedule
├── experiments/               # Experiment runners (each has runner.py)
│   ├── discrete_diffusion_generative/
│   ├── gaussian_diffusion_generative/
│   ├── gnn_denoising/
│   ├── gnn_transformer_denoising/
│   ├── lin_mlp_baseline_denoising/
│   └── _shared_utils/         # Shared infrastructure
│       ├── base_graph_module.py    # BaseGraphModule (shared LightningModule base)
│       ├── diffusion_module.py     # DiffusionModule (multi-step diffusion)
│       ├── denoising_module.py     # SingleStepDenoisingModule
│       ├── graph_evaluator.py      # GraphEvaluator
│       ├── run_experiment.py
│       └── metrics.py
└── exp_configs/               # Hydra configuration files
    ├── base_config_*.yaml     # Top-level experiment configs
    ├── models/                # Model configurations
    └── base/                  # Trainer, logger configs
```

## Core Abstractions

### GraphModel

The base class for all graph models, defined in `src/tmgg/models/base.py`. It provides:

- A unified forward signature: `forward(data: GraphData, t: Tensor | None) -> GraphData`
- Parameter counting via `parameter_count()`
- Configuration export via `get_config()`

```python
from tmgg.models.base import GraphModel

class MyModel(GraphModel):
    def __init__(self, ...):
        super().__init__()

    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData:
        # Process graph data, return predicted clean graph
        return output
```

### BaseGraphModule / DiffusionModule

`BaseGraphModule` (in `_shared_utils/base_graph_module.py`) provides shared
Lightning infrastructure: model creation via `ModelRegistry`, optimizer/scheduler
setup, and batch device transfer. It carries no training logic.

`DiffusionModule` (in `_shared_utils/diffusion_module.py`) implements the
multi-step diffusion training loop, composing four injected components:
`NoiseProcess`, `Sampler`, `NoiseSchedule`, and `GraphEvaluator`.

`SingleStepDenoisingModule` (in `_shared_utils/denoising_module.py`) subclasses
`DiffusionModule` for single-step denoising experiments (T=1, no sampler).

### GraphDataModule

The data loading abstraction in `src/tmgg/data/data_module.py`. Supports:

- Multiple dataset types (SBM, NetworkX, PyG, synthetic)
- Train/val/test splitting
- Batch loading with noise injection

### Cloud Execution (Modal)

The Modal integration in `src/tmgg/modal/` provides cloud GPU execution. CLI tools in `tmgg.modal.cli` spawn and manage experiments:

- `tmgg.modal.cli.spawn_single` — run one experiment configuration on a Modal GPU
- `tmgg.modal.cli.launch_sweep` — run a sweep of configurations in parallel
- `tmgg.modal._functions` — Modal function definitions deployed to the cloud

## Execution Flow

When you run an experiment:

```
CLI Entry Point (e.g., tmgg-spectral-arch)
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

`MODEL_REGISTRY` in `models/factory.py` maps string identifiers to model constructors:

```python
from tmgg.models.factory import create_model

model = create_model("self_attention", {"k": 8, "d_k": 64})
```

### Strategy Pattern

Noise generators, loggers, and optimizers are interchangeable via configuration. The same training loop works with different strategies.

### Template Method

`DiffusionModule` defines the training algorithm structure. Subclasses can override `training_step` and `_compute_loss` to customize behavior.

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
- `experiments/_shared_utils/run_experiment.py` - Main orchestration
- `experiments/_shared_utils/base_graph_module.py` - Shared LightningModule base
- `experiments/_shared_utils/diffusion_module.py` - Multi-step diffusion training loop
- `experiments/_shared_utils/denoising_module.py` - Single-step denoising (subclass of DiffusionModule)
- `experiments/_shared_utils/optimizer_config.py` - Optimizer and LR scheduler configuration
- `experiments/_shared_utils/graph_evaluator.py` - Graph generation quality metrics

**To understand models:**
- `models/base.py` - Base classes
- `models/gnn/gnn.py` - GNN implementation
- `models/spectral_denoisers/` - Spectral denoiser implementations
- `models/factory.py` - Registry-based model factory

**To understand data:**
- `data/data_module.py` - Data loading
- `data/noise.py` - Low-level noise functions (Gaussian, rotation, DiGress, edge flip, logit)
- `data/noise_generators.py` - OOP noise generator wrappers

**To understand metrics:**
- `experiments/_shared_utils/metrics.py` - Reconstruction quality metrics
- `experiments/_shared_utils/mmd_metrics.py` - Graph distribution distance (MMD)

**To understand configuration:**
- `exp_configs/base_config_*.yaml` - Top-level configs
- Individual model/data YAML files
