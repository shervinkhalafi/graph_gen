"""Unified task execution for cloud backends.

This module provides backend-agnostic task execution logic that both Modal
and Ray runners use. It separates serializable task inputs from execution-time
concerns (paths, storage, env vars) that are resolved inside the worker.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from omegaconf import DictConfig, OmegaConf, open_dict

if TYPE_CHECKING:
    from typing import Protocol

    class StorageProtocol(Protocol):
        """Protocol for storage backends (Tigris, local, etc.)."""

        def upload_metrics(self, metrics: dict[str, object], run_id: str) -> str: ...
        def upload_checkpoint(self, local_path: Path, run_id: str) -> str: ...


@dataclass
class TaskInput:
    """Input to a single experiment task.

    This structure is serializable and contains everything needed to run an
    experiment, except for execution-time values (paths, credentials) that
    are resolved inside the worker from environment variables.

    Attributes
    ----------
    config
        Serialized OmegaConf configuration. Must NOT contain env interpolations
        or Hydra runtime references. Paths should be None (set by worker).
    run_id
        Unique identifier for this run.
    gpu_tier
        GPU tier for this experiment ('debug', 'standard', 'fast', 'multi', 'h100').
    timeout_seconds
        Maximum runtime before the task is considered timed out.
    additional_tags
        Extra W&B tags to merge with config-defined tags at execution time.
    """

    config: dict[str, Any]
    run_id: str
    gpu_tier: str = "standard"
    timeout_seconds: int = 3600
    additional_tags: list[str] = field(default_factory=list)


@dataclass
class TaskOutput:
    """Output from a single experiment task.

    This is what gets uploaded to storage and returned to callers.
    It contains the final metrics and status of the experiment.

    Attributes
    ----------
    run_id
        Unique identifier matching the input.
    status
        Completion status: 'completed', 'failed', or 'timeout'.
    metrics
        Final metrics from training (e.g., best_val_loss).
    checkpoint_uri
        Remote URI of the best model checkpoint, if uploaded.
    error_message
        Error details if status is 'failed'.
    started_at
        ISO timestamp when execution started.
    completed_at
        ISO timestamp when execution finished.
    duration_seconds
        Wall-clock time for the experiment.
    """

    run_id: str
    status: Literal["completed", "failed", "timeout"]
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_uri: str | None = None
    error_message: str | None = None
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0


def _extract_wandb_config(config: DictConfig) -> dict[str, Any] | None:
    """Extract W&B config values that can be resolved without env vars.

    This preserves entity, project pattern, tags, etc. so Modal workers
    use the same W&B settings as local execution.

    Parameters
    ----------
    config
        The Hydra configuration (before logger is stripped).

    Returns
    -------
    dict or None
        Extracted W&B config, or None if no W&B logger configured.
    """
    if "logger" not in config:
        return None

    # logger can be a list of logger configs
    logger_configs = config.logger
    if not isinstance(logger_configs, list):
        logger_configs = [logger_configs]

    for logger_cfg in logger_configs:
        if isinstance(logger_cfg, dict) and "wandb" in logger_cfg:
            wandb = logger_cfg["wandb"]
            # Extract values that don't use ${oc.env:...}
            # Project pattern is "tmgg-${oc.select:stage,default}" - resolve stage
            stage = config.get("stage", "default") or "default"
            return {
                "entity": wandb.get("entity"),
                "project": f"tmgg-{stage}",
                "log_model": wandb.get("log_model", False),
                "tags": list(wandb.get("tags", [])),
            }

    return None


def prepare_config_for_remote(
    config: DictConfig, run_id: str | None = None
) -> dict[str, Any]:
    """Prepare a config for remote execution.

    This function strips out execution-time dependencies (paths, logger config
    with env interpolations) and returns a serializable dict. The worker will
    reconstruct these values from its environment.

    Parameters
    ----------
    config
        The original Hydra configuration.
    run_id
        Optional run ID to embed in the config. If None, a UUID is generated.

    Returns
    -------
    dict
        Serializable configuration dict ready for remote execution.

    Notes
    -----
    The returned config has:
    - `paths.output_dir` and `paths.results_dir` set to None (resolved by worker)
    - `logger` removed (reconstructed by worker based on env vars)
    - `run_id` set to the provided or generated value
    - All other interpolations resolved
    """
    run_id = run_id or str(uuid.uuid4())[:8]

    # Create a mutable copy
    config_copy = OmegaConf.create(OmegaConf.to_container(config, resolve=False))

    # Set run_id
    config_copy.run_id = run_id

    # Clear paths - they will be set by the worker based on run_id and env
    # We use explicit None rather than removing to ensure the structure exists
    if "paths" not in config_copy:
        config_copy.paths = OmegaConf.create({})
    config_copy.paths.output_dir = None
    config_copy.paths.results_dir = None

    # Extract W&B config before removing logger - preserve entity, project, etc.
    # so Modal workers use the same W&B settings as local execution
    # config_copy is always DictConfig (created from dict), but OmegaConf.create returns union
    if not isinstance(config_copy, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(config_copy)}")
    wandb_config = _extract_wandb_config(config_copy)

    # Remove logger config - it uses ${oc.env:...} interpolations that cannot
    # be resolved at serialization time. The worker's create_loggers() will
    # reconstruct loggers using _wandb_config and available env vars.
    if "logger" in config_copy:
        with open_dict(config_copy):
            # open_dict makes DictConfig mutable and enables dict-like deletion
            # Type checker sees DictConfig|ListConfig union and assumes ListConfig.pop(int)
            config_copy.pop("logger")  # pyright: ignore[reportArgumentType]

    # Store extracted W&B config for worker to use
    if wandb_config:
        config_copy._wandb_config = OmegaConf.create(wandb_config)

    # Now resolve everything else (model, data, trainer, etc.)
    # This will fail if there are unresolvable interpolations, which is correct
    # behavior - we want to catch config errors early.
    try:
        resolved = OmegaConf.to_container(config_copy, resolve=True)
    except Exception as e:
        raise ValueError(
            f"Config contains unresolvable interpolations. Ensure all non-path/logger interpolations can be resolved. Error: {e}"
        ) from e

    if not isinstance(resolved, dict):
        raise TypeError(
            f"Expected dict from OmegaConf.to_container, got {type(resolved)}"
        )

    return cast(dict[str, Any], resolved)


StorageGetter = "Callable[[], StorageProtocol | None]"


def _resolve_execution_paths(run_id: str) -> tuple[str, str]:
    """Resolve output paths for this execution environment.

    Reads TMGG_OUTPUT_BASE from environment, defaulting to /data/outputs
    for Modal containers or ./outputs for local execution.

    Parameters
    ----------
    run_id
        Unique run identifier.

    Returns
    -------
    tuple[str, str]
        (output_dir, results_dir) paths for this run.
    """
    # Modal volumes mount at /data/outputs, local uses ./outputs
    default_base = "/data/outputs" if os.path.exists("/data") else "./outputs"
    output_base = os.environ.get("TMGG_OUTPUT_BASE", default_base)
    output_dir = f"{output_base}/{run_id}"
    results_dir = f"{output_dir}/results"
    return output_dir, results_dir


def _import_run_experiment():
    """Lazy import to avoid circular dependencies at module load time."""
    from tmgg.experiment_utils.run_experiment import run_experiment

    return run_experiment


# For testing: this can be replaced with a mock
run_experiment = _import_run_experiment


def execute_task(
    task: TaskInput,
    get_storage: Callable[[], StorageProtocol | None] | None = None,
) -> TaskOutput:
    """Execute a single experiment task.

    This function runs INSIDE the worker (Modal container or Ray worker).
    It has access to environment variables and the local/mounted filesystem.

    The function:
    1. Resolves execution-time paths from run_id and environment
    2. Reconstructs the full OmegaConf config with proper paths
    3. Calls run_experiment() to perform training
    4. Uploads results to storage (if get_storage provided)
    5. Returns TaskOutput with final metrics

    Parameters
    ----------
    task
        The task input containing config and metadata.
    get_storage
        Optional callable that returns a storage backend. When running on
        Modal, pass `tmgg.modal.storage.get_storage_from_env`. When running
        locally without storage, pass None.

    Returns
    -------
    TaskOutput
        The result of the experiment execution.

    Raises
    ------
    Exception
        Re-raises any exception from run_experiment() after recording
        the failure in storage. We fail loudly, no graceful fallback.
    """
    # Get the actual run_experiment function (lazy loaded)
    _run_experiment = run_experiment() if callable(run_experiment) else run_experiment

    start_time = time.time()
    started_at = datetime.now().isoformat()
    run_id = task.run_id

    # Resolve execution-time paths
    output_dir, results_dir = _resolve_execution_paths(run_id)

    # Reconstruct OmegaConf with proper paths
    config = OmegaConf.create(task.config)
    config.paths = OmegaConf.create(
        {
            "output_dir": output_dir,
            "results_dir": results_dir,
        }
    )

    # Ensure run_id is set (should already be, but be explicit)
    config.run_id = run_id

    # Merge additional_tags into W&B config
    if task.additional_tags:
        if "_wandb_config" not in config:
            config._wandb_config = OmegaConf.create({"tags": []})
        existing_tags = list(config._wandb_config.get("tags", []))
        config._wandb_config.tags = existing_tags + task.additional_tags

    # Get storage for result uploads (if storage getter provided)
    storage = get_storage() if get_storage else None

    try:
        # Run the actual experiment
        result = _run_experiment(config)
        duration = time.time() - start_time

        # Extract metrics
        metrics = {
            "best_val_loss": result.get("best_val_loss", float("inf")),
        }

        # Upload checkpoint if available
        checkpoint_uri = None
        best_model_path = result.get("best_model_path")
        if storage and best_model_path and isinstance(best_model_path, str):
            checkpoint_path = Path(best_model_path)
            if checkpoint_path.exists():
                checkpoint_uri = storage.upload_checkpoint(checkpoint_path, run_id)

        output = TaskOutput(
            run_id=run_id,
            status="completed",
            metrics=metrics,
            checkpoint_uri=checkpoint_uri,
            error_message=None,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            duration_seconds=duration,
        )

        # Upload metrics to storage (source of truth for completion)
        if storage:
            _ = storage.upload_metrics(asdict(output), run_id)

        return output

    except Exception as e:
        duration = time.time() - start_time
        output = TaskOutput(
            run_id=run_id,
            status="failed",
            metrics={},
            checkpoint_uri=None,
            error_message=str(e),
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            duration_seconds=duration,
        )

        # Upload failure record to storage
        if storage:
            _ = storage.upload_metrics(asdict(output), run_id)

        # Re-raise to propagate to caller - no graceful fallback
        raise
