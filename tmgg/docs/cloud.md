# Cloud Execution

TMGG supports cloud execution via Modal, enabling GPU-accelerated experiments without local hardware. This document explains the architecture, setup, and usage.

## Architecture

```
Experiment Runner
    │
    ├── CloudRunnerFactory
    │   ├── LocalRunner (subprocess, built-in)
    │   └── ModalRunner (cloud GPUs)
    │
    └── Storage
        ├── LocalStorage
        └── TigrisStorage (S3-compatible)
```

The `ModalRunner` deploys experiments to Modal's serverless infrastructure, which provisions GPUs on demand and scales automatically.

## Prerequisites

Before using Modal execution, you need:

1. **Modal account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI authentication**:
   ```bash
   modal token new
   ```
3. **Secrets**: Tigris storage credentials and W&B API key (see [Secrets Configuration](#secrets-configuration))

## Secrets Configuration

Modal requires two secret groups for TMGG experiments:

### Required Secrets

| Secret Group | Keys | Purpose |
|--------------|------|---------|
| `tigris-credentials` | `TMGG_TIGRIS_BUCKET`, `TMGG_TIGRIS_ACCESS_KEY`, `TMGG_TIGRIS_SECRET_KEY` | Checkpoint and metrics storage |
| `wandb-credentials` | `WANDB_API_KEY` | Experiment tracking |

### Setting Up Secrets

**Option 1: Via Doppler (recommended)**

If you use Doppler for secrets management, mise creates Modal secrets automatically:

```bash
mise run modal-secrets
```

This maps Doppler environment variables to Modal secrets:
- `TMGG_S3_BUCKET` → `TMGG_TIGRIS_BUCKET`
- `AWS_ACCESS_KEY_ID` → `TMGG_TIGRIS_ACCESS_KEY`
- `AWS_SECRET_ACCESS_KEY` → `TMGG_TIGRIS_SECRET_KEY`
- `WANDB_API_KEY` → `WANDB_API_KEY`

**Option 2: Manual creation**

```bash
modal secret create tigris-credentials \
    TMGG_TIGRIS_BUCKET="your-bucket" \
    TMGG_TIGRIS_ACCESS_KEY="your-key" \
    TMGG_TIGRIS_SECRET_KEY="your-secret"

modal secret create wandb-credentials \
    WANDB_API_KEY="your-wandb-key"
```

## Deployment

Deploy the TMGG Modal app before running experiments:

```bash
mise run modal-deploy
```

This command:
1. Creates/updates Modal secrets from Doppler
2. Deploys the `tmgg-spectral` app with two functions:
   - `modal_execute_task` (standard GPU)
   - `modal_execute_task_fast` (A100 GPU)

For development with hot reload:

```bash
mise run modal-serve
```

## Running Experiments

### Single Experiment

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

runner = CloudRunnerFactory.create("modal")
result = runner.run_experiment(config, gpu_type="standard")
```

### Via Hydra CLI

```bash
uv run tmgg-stage2 run_on_modal=true
```

### Sweeps

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

runner = CloudRunnerFactory.create("modal")
results = runner.run_sweep(
    configs,
    gpu_type="standard",
    parallelism=4,
    timeout_seconds=1800,
)
```

## GPU Tiers

| Tier | GPU | Default Timeout | Typical Use |
|------|-----|-----------------|-------------|
| `debug` | T4 | 10 minutes | Quick tests |
| `standard` | A10G | 30 minutes | Most experiments |
| `fast` | A100-40GB | 1 hour | Large models, fast iteration |
| `multi` | A100-40GB × 2 | 2 hours | DiGress, multi-GPU training |
| `h100` | H100 | 1 hour | Maximum throughput |

Select a tier via the `gpu_type` parameter:

```python
result = runner.run_experiment(config, gpu_type="fast")
```

## LocalRunner

The default runner executes experiments in a subprocess on your local machine:

```python
from tmgg.experiment_utils.cloud import CloudRunnerFactory

runner = CloudRunnerFactory.create("local", output_dir="./results")
result = runner.run_experiment(config)
```

No additional setup required.

## ExperimentCoordinator

The coordinator manages multi-experiment runs with configuration generation and result aggregation:

```python
from tmgg.experiment_utils.cloud import ExperimentCoordinator

coordinator = ExperimentCoordinator(
    backend="modal",
    base_config_path=Path("exp_configs/"),
)

result = coordinator.run_stage(
    stage_config,
    base_config,
    parallelism=4,
    resume=True,
)
```

### Running Sweeps

The coordinator generates combinations of architectures, datasets, hyperparameters, and seeds:

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

## Result Types

### ExperimentResult

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

### StageResult

```python
@dataclass
class StageResult:
    stage_name: str
    experiments: list[ExperimentResult]
    best_config: dict
    best_metrics: dict
    summary: dict
    started_at: str
    completed_at: str
```

The summary contains aggregate statistics:

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

## Resuming Interrupted Sweeps

The coordinator tracks completed experiments and skips them on resume:

```python
result = coordinator.run_stage(
    stage_config,
    base_config,
    resume=True,
)
```

## Troubleshooting

### "Modal app not deployed"

Run `mise run modal-deploy` to deploy the app.

### "Function not hydrated"

The deployment check uses `modal.Function.from_name()` to verify deployment. Ensure the app name matches (`tmgg-spectral`).

### SyntaxError in Modal container

The Modal image uses Python 3.12. Ensure your code doesn't use features beyond 3.12.

### Image build timeout

Large dependencies (PyTorch, etc.) are installed via `uv_pip_install` for faster builds. If builds still timeout, check Modal's status page.

### Secrets not found

Verify secrets exist:
```bash
modal secret list
```

Recreate if needed:
```bash
mise run modal-secrets
```
