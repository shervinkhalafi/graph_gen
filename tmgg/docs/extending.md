# Extending the Framework

This document covers how to add new models, datasets, noise types, and execution backends.

## Adding a New Model

### Step 1: Create the Model Class

Create a new file in `src/tmgg/models/` inheriting from `DenoisingModel`:

```python
# src/tmgg/models/mymodels/my_model.py
import torch
import torch.nn as nn
from tmgg.models.base import DenoisingModel

class MyModel(DenoisingModel):
    """My custom denoising model."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        domain: str = "standard",
    ):
        super().__init__(domain=domain)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            A: Adjacency matrix (batch, n, n)

        Returns:
            Reconstructed adjacency (batch, n, n)
        """
        # Apply input domain transformation
        x = self._apply_domain_transform(A)

        # Process
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output(x)

        # Apply output domain transformation
        return self._apply_output_transform(x)

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "domain": self.domain,
        }
```

### Step 2: Create the Lightning Module

Create a Lightning module in `src/tmgg/experiments/`:

```python
# src/tmgg/experiments/my_experiment/lightning_module.py
from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.mymodels.my_model import MyModel

class MyLightningModule(DenoisingLightningModule):
    """Lightning module for MyModel."""

    def _make_model(
        self,
        hidden_dim: int,
        num_layers: int,
        domain: str = "standard",
        **kwargs,
    ) -> MyModel:
        return MyModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            domain=domain,
        )
```

### Step 3: Create the Runner

```python
# src/tmgg/experiments/my_experiment/runner.py
import hydra
from omegaconf import DictConfig
from tmgg.experiment_utils.run_experiment import run_experiment

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
domain: "standard"

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

## Adding a New Dataset

### Step 1: Create a Wrapper

If using external data, create a wrapper in `src/tmgg/experiment_utils/data/`:

```python
# src/tmgg/experiment_utils/data/my_dataset.py
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

In `src/tmgg/experiment_utils/data/data_module.py`, add handling:

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

In `src/tmgg/experiment_utils/data/synthetic_graphs.py`:

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

In `src/tmgg/experiment_utils/data/noise_generators.py`:

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

### Step 2: Register in Lightning Module

In `src/tmgg/experiment_utils/base_lightningmodule.py`, add to noise dispatch:

```python
def _apply_noise(self, batch, noise_level):
    if self.noise_type == "my_noise":
        return add_my_noise(batch, noise_level, **self.noise_kwargs)
    # ... existing code
```

### Step 3: Export

In `src/tmgg/experiment_utils/data/__init__.py`:

```python
from .noise_generators import add_my_noise
```

## Adding a New Cloud Backend

### Step 1: Implement CloudRunner

```python
# src/tmgg/experiment_utils/cloud/my_runner.py
from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult

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

In `src/tmgg/experiment_utils/cloud/factory.py`:

```python
def _try_register_my_cloud() -> None:
    try:
        from .my_runner import MyCloudRunner
        CloudRunnerFactory.register("mycloud", MyCloudRunner)
    except ImportError:
        pass

# Call during module initialization
_try_register_my_cloud()
```

Or register at runtime:

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory
from my_package import MyCloudRunner

CloudRunnerFactory.register("mycloud", MyCloudRunner)
```

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

    def test_output_range(self):
        model = MyModel(hidden_dim=20, num_layers=2)
        model.eval()
        x = torch.randn(4, 20, 20)
        output = model(x)
        # In eval mode with standard domain, output is sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)
```

Run tests:

```bash
uv run pytest tests/test_my_model.py -v
```
