# Extending the Framework

This document covers how to add new models, datasets, noise types, execution backends, and entirely new experiment types.

## Adding a New Model

### Step 1: Create the Model Class

Create a new file in `src/tmgg/models/` inheriting from `GraphModel`:

```python
# src/tmgg/models/mymodels/my_model.py
from typing import Any

import torch
import torch.nn as nn
from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.base import GraphModel

class MyModel(GraphModel):
    """My custom graph model."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Define layers...

    def forward(
        self, data: GraphData, t: torch.Tensor | None = None
    ) -> GraphData:
        """Forward pass.

        Parameters
        ----------
        data
            Graph data with node features X, edge features E, and node_mask.
        t
            Normalised diffusion timestep (bs,), or None.

        Returns
        -------
        GraphData
            Predicted clean graph (logits, not probabilities).
        """
        # Your model logic here
        ...

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
```

### Step 2: Choose a Training Module

For denoising experiments, use ``SingleStepDenoisingModule``. For generative
diffusion experiments, use ``DiffusionModule`` with injected noise process,
sampler, and evaluator components. Both create models through ``ModelRegistry``.

```python
# For single-step denoising:
from tmgg.experiments._shared_utils.denoising_module import SingleStepDenoisingModule

# For multi-step diffusion:
from tmgg.experiments._shared_utils.diffusion_module import DiffusionModule
```

### Step 3: Create the Runner

```python
# src/tmgg/experiments/my_experiment/runner.py
import hydra
from omegaconf import DictConfig
from tmgg.experiments._shared_utils.run_experiment import run_experiment

CONFIG_PATH = "../../exp_configs"

@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="base_config_my")
def main(config: DictConfig):
    return run_experiment(config)

if __name__ == "__main__":
    main()
```

### Step 4: Create Configuration Files

Base config (`exp_configs/base_config_my.yaml`):

```yaml
defaults:
  - models/my/my_model@model
  - data: sbm_default
  - base/trainer/default@trainer
  - base/logger/tensorboard@logger
  - _self_

experiment_name: "my_experiment"
seed: 42
```

Model config (`exp_configs/models/my/my_model.yaml`):

```yaml
_target_: tmgg.experiments.my_experiment.lightning_module.MyLightningModule

hidden_dim: 64
num_layers: 4

learning_rate: 0.001
loss_type: "MSE"
noise_type: ${data.noise_type}
noise_levels: ${data.noise_levels}
seed: ${seed}
```

### Step 5: Add CLI Entry Point

In `pyproject.toml`:

```toml
[project.scripts]
tmgg-my = "tmgg.experiments.my_experiment.runner:main"
```

### Step 6: Export the Model

In `src/tmgg/models/__init__.py`:

```python
from tmgg.models.mymodels.my_model import MyModel
```

### Step 7: Register in Model Factory

The framework dispatches model construction through a central registry in
`src/tmgg/models/factory.py`. Without registration, `create_model()` (and by
extension every Lightning module's `_make_model()`) cannot instantiate your
architecture.

`MODEL_REGISTRY` is a plain `dict[str, Callable[[dict, Any], nn.Module]]`.
The `@register_model(*names)` decorator adds one or more string keys that
map to a factory function. Duplicate names raise `ValueError` immediately,
so silent overwrites are impossible.

Register your model by adding a decorated factory at the bottom of
`src/tmgg/models/factory.py`:

```python
@register_model("my_model")
def _make_my_model(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.mymodels.my_model import MyModel

    return MyModel(
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 4),
    )
```

If your model should be reachable under multiple names (e.g. a short alias),
pass them all to the decorator:

```python
@register_model("my_model", "mm")
def _make_my_model(config: dict[str, Any]) -> nn.Module:
    ...
```

After registration, any caller can construct the model via:

```python
from tmgg.models.factory import create_model

model = create_model("my_model", {"hidden_dim": 128, "num_layers": 6})
```

`create_model` raises `ValueError` with a list of all registered types if the
key is unknown, which makes typos easy to diagnose.

> **Note:** Use lazy imports inside the factory function (as shown above) to
> avoid circular imports and keep module load time fast. Every existing
> registration in `factory.py` follows this pattern.

## Adding a New Dataset

### Step 1: Create a Wrapper

If using external data, create a wrapper in `src/tmgg/data/`:

```python
# src/tmgg/data/my_dataset.py
import torch
from torch.utils.data import Dataset

class MyDatasetWrapper(Dataset):
    """Wrapper for my custom dataset."""

    def __init__(self, data_path: str, num_samples: int = 1000):
        self.data_path = data_path
        self.num_samples = num_samples
        self._load_data()

    def _load_data(self):
        # Load your data
        self.graphs = [...]  # List of adjacency matrices

    def __len__(self):
        return len(self.graphs) * self.num_samples

    def __getitem__(self, idx):
        graph_idx = idx // self.num_samples
        return torch.tensor(self.graphs[graph_idx], dtype=torch.float32)
```

### Step 2: Register in GraphDataModule

In `src/tmgg/data/data_module.py`, add handling:

```python
def _create_dataset(self):
    if self.dataset_name == "my_dataset":
        from .my_dataset import MyDatasetWrapper
        return MyDatasetWrapper(**self.dataset_config)
    # ... existing code
```

### Step 3: Create Config

Create `exp_configs/data/my_data.yaml`:

```yaml
dataset_name: my_dataset
dataset_config:
  data_path: "/path/to/data"
  num_samples: 1000

batch_size: 64
noise_type: "gaussian"
noise_levels: [0.1, 0.2, 0.3]
```

## Adding a New Synthetic Graph Generator

### Step 1: Implement the Generator

In `src/tmgg/data/synthetic_graphs.py`:

```python
def generate_my_graphs(
    n: int,
    num_graphs: int,
    my_param: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate my custom graph type.

    Parameters
    ----------
    n
        Number of nodes per graph.
    num_graphs
        Number of graphs to generate.
    my_param
        My custom parameter.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (num_graphs, n, n) containing adjacency matrices.
    """
    if seed is not None:
        np.random.seed(seed)

    graphs = []
    for _ in range(num_graphs):
        # Generate your graph
        A = np.zeros((n, n))
        # ... your generation logic
        graphs.append(A)

    return np.array(graphs)
```

### Step 2: Register in SyntheticGraphDataset

Update the `VALID_TYPES` and dispatch in `SyntheticGraphDataset`:

```python
VALID_TYPES = {..., "my_type", "mt"}  # Add type and alias
TYPE_ALIASES = {..., "mt": "my_type"}

def _generate(self):
    if self.graph_type == "my_type":
        return generate_my_graphs(
            self.n, self.num_graphs, my_param=self.kwargs.get("my_param", 0.5)
        )
```

## Adding a New Noise Type

### Step 1: Implement the Noise Function

In `src/tmgg/data/noise_generators.py`:

```python
def add_my_noise(
    A: torch.Tensor,
    eps: float,
    **kwargs,
) -> torch.Tensor:
    """Apply my custom noise.

    Parameters
    ----------
    A
        Clean adjacency matrix (batch, n, n).
    eps
        Noise level.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix.
    """
    # Your noise implementation
    noise = torch.randn_like(A) * eps
    return A + noise
```

### Step 2: Register as a NoiseProcess

Implement a ``NoiseProcess`` subclass in ``src/tmgg/diffusion/noise_process.py``
that wraps your noise function, or use it directly within a custom training module.

### Step 3: Export

In `src/tmgg/data/__init__.py`:

```python
from .noise_generators import add_my_noise
```

## Adding a New Cloud Backend

### Step 1: Implement CloudRunner

```python
# src/tmgg/runners/my_runner.py
from tmgg.runners.base import CloudRunner, ExperimentResult

class MyCloudRunner(CloudRunner):
    """My custom cloud runner."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key

    def run_experiment(
        self,
        config,
        gpu_type: str = "standard",
        timeout_seconds: int = 3600,
    ) -> ExperimentResult:
        # Submit to your cloud
        job_id = self._submit(config, gpu_type)

        # Wait for completion
        result = self._wait(job_id, timeout_seconds)

        return ExperimentResult(
            run_id=job_id,
            config=config,
            metrics=result["metrics"],
            checkpoint_path=result.get("checkpoint"),
            status="completed",
            error_message=None,
            duration_seconds=result["duration"],
        )

    def run_sweep(
        self,
        configs,
        gpu_type: str = "standard",
        parallelism: int = 4,
        timeout_seconds: int = 3600,
    ) -> list[ExperimentResult]:
        # Run configs in parallel
        results = []
        for config in configs:
            results.append(self.run_experiment(config, gpu_type, timeout_seconds))
        return results

    def get_status(self, run_id: str) -> str:
        # Query your cloud
        return "completed"

    def cancel(self, run_id: str) -> bool:
        # Cancel job
        return True
```

### Step 2: Register the Runner

Export your runner from `src/tmgg/runners/__init__.py` so it can be imported directly:

```python
# In src/tmgg/runners/__init__.py
from tmgg.runners.my_runner import MyCloudRunner

__all__ = [
    # ... existing exports ...
    "MyCloudRunner",
]
```

Then use it:

```python
from tmgg.runners import MyCloudRunner

runner = MyCloudRunner(api_key="...")
result = runner.run_experiment(config)
```

## Adding a New Experiment Type

The sections above cover adding models, datasets, and noise within an existing experiment family. This section covers creating an entirely new experiment family from scratch — for instance, a graph VAE alongside the existing denoising and discrete diffusion families.

The end-to-end process requires six artifacts. We walk through each using a hypothetical `vae_graph` experiment.

### Step 1: Lightning Module

Subclass ``DiffusionModule`` or ``SingleStepDenoisingModule``. Both create
models through ``ModelRegistry`` and handle optimizer/scheduler configuration
automatically. ``DiffusionModule`` accepts injected noise process, sampler,
and evaluator components for multi-step diffusion. ``SingleStepDenoisingModule``
handles single-step denoising with a fixed T=1 schedule.

```python
# For custom training logic, subclass DiffusionModule or SingleStepDenoisingModule
from tmgg.experiments._shared_utils.diffusion_module import DiffusionModule

# Register your model type, then reference by name in config
```

If your experiment needs custom training/validation logic (e.g., ELBO loss with KL term), subclass ``DiffusionModule`` or ``SingleStepDenoisingModule`` and override ``training_step`` and ``validation_step``.

### Step 2: Base Config YAML

Create `src/tmgg/exp_configs/base_config_vae_graph.yaml`. The `defaults:` list composes config groups; `_self_` goes last so that explicit keys in this file take priority over defaults.

```yaml
# Base configuration for graph VAE experiments
#
# Run with: uv run tmgg-vae-graph
# Override: uv run tmgg-vae-graph model.latent_dim=64

defaults:
  - base_config_training
  - models/vae/vae_default@model
  - _self_

experiment_name: "vae_graph"
wandb_project: vae-graph

model:
  noise_type: ${data.noise_type}
  noise_levels: ${data.noise_levels}
  seed: ${seed}
```

### Step 3: Model Config

Create YAML files under `src/tmgg/exp_configs/models/vae/`. The `_target_` points to the Lightning module class. Use `${}` interpolations for values that come from the base config or CLI overrides — the batch config builder strips these during generation.

```yaml
# src/tmgg/exp_configs/models/vae/vae_default.yaml

_target_: tmgg.experiments.vae_graph.lightning_module.GraphVAELightningModule

latent_dim: 32
encoder_layers: 3

learning_rate: ${learning_rate}
weight_decay: ${weight_decay}
optimizer_type: ${optimizer_type}

noise_type: ${noise_type}
noise_levels: ${noise_levels}
loss_type: ${loss_type}
```

### Step 4: Runner Script

Create the Hydra entry point. The `config_path` is relative to the runner file's location.

```python
# src/tmgg/experiments/vae_graph/runner.py
import hydra
from omegaconf import DictConfig
from tmgg.experiments._shared_utils.run_experiment import run_experiment

CONFIG_PATH = "../../exp_configs"

@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="base_config_vae_graph")
def main(config: DictConfig):
    return run_experiment(config)

if __name__ == "__main__":
    main()
```

### Step 5: CLI Entry Point

Register the runner in `pyproject.toml` under `[project.scripts]`:

```toml
[project.scripts]
# ... existing entries ...
# VAE experiment CLI
tmgg-vae-graph = "tmgg.experiments.vae_graph.runner:main"
```

After adding this, reinstall with `uv pip install -e .` so the console script is available.

### Step 6: Stage Definition (Batch Runs)

To run sweeps on Modal, create a stage definition at `src/tmgg/modal/stage_definitions/stage_vae_graph.yaml`. This defines the architecture/hyperparameter grid and seed list for batch config generation.

```yaml
# Graph VAE: initial validation sweep

name: stage_vae_graph
base_config: base_config_vae_graph

architectures:
  - models/vae/vae_default

hyperparameters:
  learning_rate: [1e-4, 1e-3]
  "model.latent_dim": [16, 32, 64]

seeds: [1, 2, 3]

run_id_template: "vae_{arch}_{latent_dim}_{lr}_s{seed}"
```

Verify the generated configs with a dry run:

```bash
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage_vae_graph --output-dir /tmp/test --dry-run
```

### Checklist Summary

Before marking a new experiment type complete, confirm:

1. Lightning module calls `save_hyperparameters()` before `super().__init__()`.
2. Base config YAML exists in `src/tmgg/exp_configs/` with `- _self_` last in defaults.
3. At least one model config exists under `src/tmgg/exp_configs/models/<family>/`.
4. Runner script uses `@hydra.main(version_base="1.3", ...)` with correct `config_path`.
5. `pyproject.toml` has the `tmgg-<name>` entry point.
6. Stage YAML generates valid configs via `--dry-run`.
7. `uv run tmgg-<name> --help` prints the Hydra help without errors.

## Testing Your Extensions

Create tests in `tests/`:

```python
# tests/test_my_model.py
import pytest
import torch
from tmgg.models.mymodels.my_model import MyModel

class TestMyModel:
    def test_forward_shape(self):
        model = MyModel(hidden_dim=20, num_layers=2)
        x = torch.randn(4, 20, 20)
        output = model(x)
        assert output.shape == (4, 20, 20)

    def test_predict_binary(self):
        model = MyModel(hidden_dim=20, num_layers=2)
        model.eval()
        x = torch.randn(4, 20, 20)
        logits = model(x)
        # predict() converts raw logits to binary {0, 1} predictions
        preds = model.predict(logits)
        assert torch.all((preds == 0) | (preds == 1))
```

Run tests:

```bash
uv run pytest tests/test_my_model.py -v
```
