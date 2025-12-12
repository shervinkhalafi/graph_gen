# Cloud Execution

This document covers cloud execution using the CloudRunner abstraction and Modal integration.

## Architecture

```
ExperimentCoordinator
    │
    ├── CloudRunnerFactory (creates runners)
    │   ├── LocalRunner (built-in, subprocess)
    │   └── ModalRunner (optional, cloud GPUs)
    │
    └── CloudStorage (result persistence)
        ├── LocalStorage
        └── S3Storage
```

## CloudRunnerFactory

The factory pattern allows registering multiple execution backends.

### Basic Usage

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

# List available backends
print(CloudRunnerFactory.available_backends())  # ['local', 'modal']

# Create a runner
runner = CloudRunnerFactory.create("local")

# Run an experiment
result = runner.run_experiment(config, gpu_type="standard", timeout_seconds=3600)
```

### Available Backends

| Backend | Description | Installation |
|---------|-------------|--------------|
| `local` | Subprocess execution | Built-in |
| `modal` | Cloud GPUs via Modal | `pip install tmgg-modal` |

## LocalRunner

The default runner executes experiments in a subprocess. No additional setup required.

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

runner = CloudRunnerFactory.create("local", output_dir="./results")
result = runner.run_experiment(config)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | Path | `./outputs` | Output directory for results |

## ModalRunner

Executes experiments on Modal cloud infrastructure with GPU support.

### Setup

1. Install the Modal package:
   ```bash
   pip install tmgg-modal
   ```

2. Authenticate with Modal:
   ```bash
   modal token new
   ```

3. The runner auto-registers when `tmgg_modal` is available.

### Usage

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

runner = CloudRunnerFactory.create("modal")
result = runner.run_experiment(
    config,
    gpu_type="A10G",
    timeout_seconds=3600
)
```

### GPU Types

| GPU Type | Description |
|----------|-------------|
| `standard` | Default GPU (T4 or similar) |
| `A10G` | NVIDIA A10G |
| `A100` | NVIDIA A100 |

## ExperimentCoordinator

Manages multi-experiment runs with configuration generation and result aggregation.

### Basic Usage

```python
from tmgg.experiment_utils.cloud import ExperimentCoordinator

coordinator = ExperimentCoordinator(
    backend="local",  # or "modal"
    base_config_path=Path("exp_configs/"),
)

# Run a stage
result = coordinator.run_stage(
    stage_config,
    base_config,
    parallelism=4,
    resume=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | "local" | Execution backend |
| `storage` | CloudStorage | LocalStorage | Result storage |
| `base_config_path` | Path | None | Path to config directory |
| `cache_dir` | Path | `./cache` | Local cache directory |

### Running Sweeps

The coordinator generates all combinations of architectures, datasets, hyperparameters, and seeds:

```python
from tmgg.experiment_utils.cloud import ExperimentCoordinator, StageConfig

stage = StageConfig(
    name="my_sweep",
    architectures=["models/gnn/standard", "models/attention/multi_layer"],
    datasets=["data/sbm_default"],
    hyperparameter_space={
        "learning_rate": [1e-4, 1e-3],
        "model.num_layers": [4, 8],
    },
    seeds=[1, 2, 3],
)

coordinator = ExperimentCoordinator(backend="modal")
result = coordinator.run_stage(stage, base_config, parallelism=8)
```

## StageConfig

Configuration for experimental stages.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Stage identifier |
| `architectures` | list[str] | Model config paths |
| `datasets` | list[str] | Data config paths |
| `hyperparameter_space` | dict | Parameters to sweep |
| `num_trials` | int | Bayesian optimization trials |
| `seeds` | list[int] | Random seeds |
| `gpu_type` | str | GPU tier |
| `timeout_seconds` | int | Max runtime per experiment |

### Loading from YAML

```python
from tmgg.experiment_utils.cloud import StageConfig
from pathlib import Path

stage = StageConfig.from_yaml(Path("exp_configs/stages/stage1_poc.yaml"))
```

## StageResult

Aggregated results from a stage run.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `stage_name` | str | Stage identifier |
| `experiments` | list[ExperimentResult] | Individual results |
| `best_config` | dict | Configuration with best val loss |
| `best_metrics` | dict | Metrics from best run |
| `summary` | dict | Aggregated statistics |
| `started_at` | str | Start timestamp |
| `completed_at` | str | End timestamp |

### Summary Statistics

```python
result.summary = {
    "total_experiments": 24,
    "completed": 22,
    "failed": 2,
    "success_rate": 0.917,
    "mean_duration_seconds": 342.5,
    "total_duration_seconds": 7535.0,
}
```

## ExperimentResult

Result from a single experiment run.

```python
@dataclass
class ExperimentResult:
    run_id: str
    config: dict
    metrics: dict          # {"val_loss": 0.123, "test_loss": 0.145}
    checkpoint_path: str | None
    status: str            # "completed", "failed", "timeout"
    error_message: str | None
    duration_seconds: float
```

## Resuming Experiments

The coordinator supports resuming interrupted sweeps:

```python
result = coordinator.run_stage(
    stage_config,
    base_config,
    resume=True,  # Skip completed experiments
)
```

Completed experiments are tracked via storage and skipped on resume.

## Custom Backends

To add a custom execution backend:

```python
from tmgg.experiment_utils.cloud import CloudRunner, CloudRunnerFactory

class MyCloudRunner(CloudRunner):
    def run_experiment(self, config, gpu_type="standard", timeout_seconds=3600):
        # Implementation
        return ExperimentResult(...)

    def run_sweep(self, configs, gpu_type, parallelism, timeout_seconds):
        # Implementation
        return [ExperimentResult(...), ...]

    def get_status(self, run_id):
        return "running"  # or "completed", "failed"

    def cancel(self, run_id):
        return True

# Register
CloudRunnerFactory.register("mycloud", MyCloudRunner)
```
