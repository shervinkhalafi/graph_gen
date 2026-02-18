# Cloud Execution

TMGG supports cloud execution via Modal, enabling GPU-accelerated experiments without local hardware. This document explains the architecture, setup, and usage.

## Architecture

```
Experiment Runner
    │
    ├── CloudRunner (abstract base)
    │   ├── LocalRunner (subprocess, built-in)
    │   └── ModalRunner (cloud GPUs)
    │
    ├── ExperimentCoordinator (sweep orchestration)
    │
    └── Storage
        ├── LocalStorage
        └── S3Storage (Tigris-compatible)
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
2. Deploys the `tmgg-spectral-arch` app with two functions:
   - `modal_execute_task` (standard GPU)
   - `modal_execute_task_fast` (A100 GPU)

For development with hot reload:

```bash
mise run modal-serve
```

## Running Experiments

### Single Experiment

```python
from tmgg.modal.runner import ModalRunner

runner = ModalRunner()
result = runner.run_experiment(config, gpu_type="standard")
```

### Via Hydra CLI

```bash
uv run tmgg-experiment +stage=stage2_validation run_on_modal=true
```

### Sweeps

```python
from tmgg.modal.runner import ModalRunner

runner = ModalRunner()
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
from tmgg.experiment_utils.cloud import LocalRunner

runner = LocalRunner(output_dir="./results")
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

## Config serialization for remote execution

A generated config must survive serialization, network transit, and reconstruction inside a Modal container before it can drive an experiment. Understanding this pipeline is essential for debugging Modal execution failures, since errors can originate at any stage.

### Data flow

```
config_builder.build()
  │  DictConfig with ${...} interpolations
  ▼
prepare_config_for_remote()          ← strips paths, logger, resolves interpolations
  │  plain dict (JSON-safe)
  ▼
JSON file on disk  ─── or ───  ModalRunner wraps directly
  │                                │
  ▼                                ▼
spawn_single / launch_sweep     runner._prepare_task()
  │  reads JSON, builds dict       │  calls prepare_config_for_remote(),
  │                                │  wraps into TaskInput
  ▼                                ▼
task_dict sent to Modal via .spawn() / .remote()
  │
  ▼
modal_execute_task(task_dict)        ← Modal container entry point
  │  reconstructs TaskInput(**task_dict)
  ▼
execute_task(task, get_storage)      ← resolves paths, rebuilds OmegaConf, runs experiment
  │
  ▼
run_experiment(config) → TaskOutput
```

Two invocation paths converge at the Modal function boundary. The **programmatic path** (`ModalRunner`) calls `prepare_config_for_remote()` directly, then wraps the result into a `TaskInput`. The **CLI path** (`spawn_single`, `launch_sweep`) reads a pre-serialized JSON file and builds the `task_dict` manually. Both produce the same `dict[str, Any]` that Modal transmits to the container.

### `prepare_config_for_remote()`

Defined in `src/tmgg/experiment_utils/task.py`. Transforms a live `DictConfig` into a JSON-safe dict by handling three categories of values that cannot survive serialization:

1. **Paths** (`paths.output_dir`, `paths.results_dir`) are set to `None`. These reference `${hydra:runtime.output_dir}`, which only resolves during a Hydra run. The worker sets them from `TMGG_OUTPUT_BASE` and the run ID.

2. **Logger config** is removed entirely. It contains `${oc.env:WANDB_API_KEY}` and similar interpolations that require environment variables unavailable at serialization time. Before removal, W&B settings (entity, project, tags, log_model) are extracted into a `_wandb_config` dict that travels with the config. The worker's `create_loggers()` reconstructs loggers from this dict plus the container's environment.

3. **All remaining interpolations** are resolved via `OmegaConf.to_container(resolve=True)`. If any interpolation is unresolvable, this raises `ValueError` immediately rather than letting a broken config reach the worker.

### `TaskInput` and `TaskOutput`

These dataclasses (in `task.py`) define the serialization contract between caller and worker.

**`TaskInput`** contains everything the worker needs:
- `config: dict` — the serialized config (output of `prepare_config_for_remote()`)
- `run_id: str` — unique identifier for this experiment
- `gpu_tier: str` — which GPU tier was requested (informational inside the container)
- `timeout_seconds: int` — maximum wall-clock time
- `additional_tags: list[str]` — extra W&B tags merged at execution time

**`TaskOutput`** is returned to the caller:
- `run_id`, `status` (`completed` / `failed` / `timeout`), `metrics: dict`
- `checkpoint_uri` — remote URI on Tigris, if a checkpoint was uploaded
- `error_message` — populated only on failure
- `started_at`, `completed_at` — ISO timestamps; `duration_seconds` — wall-clock time

Both are transmitted as plain dicts (via `dataclasses.asdict`), not as pickled objects.

### Worker-side reconstruction (`execute_task`)

`execute_task()` runs inside the Modal container. It reverses the serialization steps:

1. **Path resolution.** Reads `TMGG_OUTPUT_BASE` from the environment (defaults to `/data/outputs` on Modal volumes, `./outputs` locally) and constructs `{base}/{run_id}/` for output and `{base}/{run_id}/results/` for results.

2. **Config reconstruction.** Wraps the plain dict back into an `OmegaConf` `DictConfig`, injects the resolved paths, and merges `additional_tags` into `_wandb_config.tags`.

3. **Confirmation tracking.** Writes `confirmation.json` to the output directory at start, completion, and failure. This provides a storage-independent audit trail on the Modal volume, useful when W&B or Tigris uploads fail.

4. **Experiment execution.** Calls `run_experiment(config)`, which handles the full training/testing/evaluation loop. On success, uploads the best checkpoint and final metrics to Tigris via the storage backend.

5. **Error propagation.** On failure, records the error in both Tigris and the volume confirmation file, then re-raises the exception. There is no graceful fallback.

## Troubleshooting

### "Modal app not deployed"

Run `mise run modal-deploy` to deploy the app.

### "Function not hydrated"

The deployment check uses `modal.Function.from_name()` to verify deployment. Ensure the app name matches (`tmgg-spectral-arch`).

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
