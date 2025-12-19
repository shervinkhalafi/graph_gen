"""Modal-specific CloudRunner implementation.

Provides ModalRunner that executes TMGG experiments on Modal GPUs
with Tigris storage for checkpoints and metrics.

Architecture:
    The ModalRunner uses the unified TaskInput/TaskOutput abstraction from
    tmgg.experiment_utils.task. Modal functions are thin wrappers around
    execute_task(), which handles config reconstruction, experiment execution,
    and storage uploads.

    Execution modes:
    - Blocking: run_experiment(), run_sweep() - wait for results
    - Detached: spawn_experiment(), spawn_sweep() - fire-and-forget
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import modal
from omegaconf import DictConfig

from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult
from tmgg.experiment_utils.task import (
    TaskInput,
    execute_task,
    prepare_config_for_remote,
)
from tmgg.modal.app import DEFAULT_TIMEOUTS, GPU_CONFIGS, app
from tmgg.modal.image import create_tmgg_image
from tmgg.modal.storage import TigrisStorage, get_storage_from_env
from tmgg.modal.volumes import get_volume_mounts

if TYPE_CHECKING:
    from modal.functions import FunctionCall

logger = logging.getLogger(__name__)

# App name must match the one defined in app.py
MODAL_APP_NAME = "tmgg-spectral"


class ModalNotDeployedError(Exception):
    """Raised when the Modal app is not deployed."""

    pass


def check_modal_deployment() -> None:
    """Verify that the tmgg-spectral Modal app is deployed.

    This function checks if the Modal functions are accessible, which requires
    the app to have been deployed via `modal deploy`. If the app is not deployed,
    it raises ModalNotDeployedError with instructions.

    Raises
    ------
    ModalNotDeployedError
        If the Modal app is not deployed or functions are not accessible.
    """
    try:
        # Try to get a reference to the deployed function
        # from_name() returns a handle to the deployed function
        fn = modal.Function.from_name(MODAL_APP_NAME, "modal_execute_task")
        # hydrate() forces resolution - this will fail if not deployed
        fn.hydrate()
    except modal.exception.NotFoundError as e:
        raise ModalNotDeployedError(
            f"Modal app '{MODAL_APP_NAME}' is not deployed. "
            + "Run 'mise run modal-deploy' or 'uv run modal deploy src/tmgg/modal/runner.py' first.\n"
            + f"Original error: {e}"
        ) from e
    except Exception as e:
        # Re-raise unexpected errors with context
        raise ModalNotDeployedError(
            f"Failed to verify Modal deployment for app '{MODAL_APP_NAME}': {e}"
        ) from e


# Create experiment image, with fallback for testing
try:
    from tmgg.modal.paths import discover_tmgg_path

    _tmgg_path = discover_tmgg_path()
    experiment_image = create_tmgg_image(_tmgg_path)
except Exception:
    # During testing with mocked modal, image creation may fail
    # This is fine as the decorated functions won't actually run
    experiment_image = None


# Modal secrets for Tigris storage and W&B
tigris_secret = modal.Secret.from_name(
    "tigris-credentials",
    required_keys=[
        "TMGG_TIGRIS_BUCKET",
        "TMGG_TIGRIS_ACCESS_KEY",
        "TMGG_TIGRIS_SECRET_KEY",
    ],
)

wandb_secret = modal.Secret.from_name(
    "wandb-credentials",
    required_keys=[
        "WANDB_API_KEY",
    ],
)


def _get_timeout_for_gpu(gpu_type: str) -> int:
    """Get the timeout in seconds for a given GPU tier."""
    return DEFAULT_TIMEOUTS.get(gpu_type, DEFAULT_TIMEOUTS["standard"])


@app.function(
    name="modal_execute_task",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=DEFAULT_TIMEOUTS["standard"],
    scaledown_window=2,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_execute_task(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute a single task on Modal using the unified task abstraction.

    This is the primary Modal function for experiment execution. It wraps
    the backend-agnostic execute_task() function, providing Modal-specific
    storage integration via get_storage_from_env().

    Parameters
    ----------
    task_dict
        Serialized TaskInput as a dictionary (via asdict()).

    Returns
    -------
    dict
        TaskOutput as a dictionary (via asdict()).
    """
    # Reconstruct TaskInput from dict
    task = TaskInput(**task_dict)

    # Execute using the unified task abstraction
    # Storage is obtained from Modal secrets via get_storage_from_env
    output = execute_task(task, get_storage=get_storage_from_env)

    return asdict(output)


@app.function(
    name="modal_execute_task_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    timeout=DEFAULT_TIMEOUTS["fast"],
    scaledown_window=2,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_execute_task_fast(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute task on fast (A100) GPU - delegates to modal_execute_task."""
    return modal_execute_task.local(task_dict)


@app.function(
    name="modal_execute_task_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    timeout=DEFAULT_TIMEOUTS["debug"],
    scaledown_window=2,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_execute_task_debug(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute task on debug (T4) GPU - delegates to modal_execute_task."""
    return modal_execute_task.local(task_dict)


# Legacy function names for backward compatibility during transition
# These will be removed once all callers migrate to modal_execute_task
run_single_experiment = modal_execute_task
run_single_experiment_fast = modal_execute_task_fast


@dataclass
class SpawnedTask:
    """Handle for a spawned Modal task.

    Provides the run_id for tracking and methods to wait for completion
    or check status via storage.

    Attributes
    ----------
    run_id
        Unique identifier for the spawned experiment.
    gpu_tier
        GPU tier used for execution.
    function_call
        Modal FunctionCall handle for polling (optional, may not be available
        after process restart).
    """

    run_id: str
    gpu_tier: str
    function_call: FunctionCall[dict[str, Any]] | None = None


class ModalRunner(CloudRunner):
    """Modal-specific experiment runner.

    Executes experiments on Modal GPUs with automatic scaling
    and parallel execution support.

    Requires the Modal app to be pre-deployed. If not deployed,
    instantiation will fail with ModalNotDeployedError.

    Execution Modes
    ---------------
    - **Blocking**: `run_experiment()` / `run_sweep()` wait for results
    - **Detached**: `spawn_experiment()` / `spawn_sweep()` return immediately

    The detached mode is preferred for long-running sweeps where you want
    to fire-and-forget. Results are stored in Tigris and can be retrieved
    later via run_id.
    """

    gpu_type: str
    storage: TigrisStorage | None

    def __init__(
        self,
        gpu_type: str = "debug",
        storage: TigrisStorage | None = None,
        skip_deployment_check: bool = False,
    ):
        """Initialize Modal runner.

        Parameters
        ----------
        gpu_type
            Default GPU tier for experiments.
        storage
            Tigris storage for results. If None, creates from env.
        skip_deployment_check
            If True, skip the deployment verification. Only use for testing
            or when you're certain the app is deployed.

        Raises
        ------
        ModalNotDeployedError
            If the Modal app is not deployed and skip_deployment_check is False.
        """
        if not skip_deployment_check:
            check_modal_deployment()

        self.gpu_type = gpu_type
        self.storage = storage or get_storage_from_env()
        self._active_runs: dict[str, SpawnedTask] = {}

    def _create_task_input(
        self,
        config: DictConfig,
        gpu_tier: str,
        timeout_seconds: int | None = None,
    ) -> TaskInput:
        """Create a TaskInput from a Hydra config.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_tier
            GPU tier for execution.
        timeout_seconds
            Optional timeout override.

        Returns
        -------
        TaskInput
            Serializable task input ready for remote execution.
        """
        # Use the unified prepare_config_for_remote to strip env vars
        config_dict = prepare_config_for_remote(config)
        run_id = config_dict.get("run_id", config_dict["run_id"])
        timeout = timeout_seconds or _get_timeout_for_gpu(gpu_tier)

        return TaskInput(
            config=config_dict,
            run_id=run_id,
            gpu_tier=gpu_tier,
            timeout_seconds=timeout,
        )

    def _select_modal_function(self, gpu_tier: str) -> modal.Function:
        """Select the appropriate Modal function for the GPU tier.

        Uses modal.Function.from_name() to get references to deployed functions,
        which is required when calling from outside Modal (i.e., from local machine).

        Tier mapping:
        - debug → T4 (modal_execute_task_debug)
        - standard → A10G (modal_execute_task)
        - fast, multi, h100 → A100 (modal_execute_task_fast)
        """
        if gpu_tier == "debug":
            func_name = "modal_execute_task_debug"
        elif gpu_tier in ("fast", "multi", "h100"):
            func_name = "modal_execute_task_fast"
        else:
            func_name = "modal_execute_task"
        return modal.Function.from_name(MODAL_APP_NAME, func_name)

    # =========================================================================
    # Spawn Methods (Detached Execution)
    # =========================================================================

    def spawn_experiment(
        self,
        config: DictConfig,
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
    ) -> SpawnedTask:
        """Spawn a single experiment without waiting for results.

        Uses Modal's `.spawn()` for fire-and-forget execution. Results are
        uploaded to Tigris storage and can be retrieved via run_id.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            GPU tier override.
        timeout_seconds
            Timeout override.

        Returns
        -------
        SpawnedTask
            Handle with run_id for tracking.
        """
        gpu = gpu_type or self.gpu_type
        task_input = self._create_task_input(config, gpu, timeout_seconds)
        task_dict = asdict(task_input)

        # Use spawn() for detached execution
        modal_fn = self._select_modal_function(gpu)
        function_call = modal_fn.spawn(task_dict)

        spawned = SpawnedTask(
            run_id=task_input.run_id,
            gpu_tier=gpu,
            function_call=function_call,
        )
        self._active_runs[task_input.run_id] = spawned

        logger.info(f"Spawned experiment {task_input.run_id} on {gpu} GPU (detached)")
        return spawned

    def spawn_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
    ) -> list[SpawnedTask]:
        """Spawn multiple experiments without waiting for results.

        Each experiment is spawned independently using `.spawn()`.
        Results are uploaded to Tigris and can be retrieved via run_ids.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            GPU tier for all experiments.
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[SpawnedTask]
            Handles with run_ids for tracking.
        """
        gpu = gpu_type or self.gpu_type
        spawned_tasks: list[SpawnedTask] = []

        for config in configs:
            task = self.spawn_experiment(config, gpu, timeout_seconds)
            spawned_tasks.append(task)

        logger.info(f"Spawned {len(spawned_tasks)} experiments on {gpu} GPU (detached)")
        return spawned_tasks

    # =========================================================================
    # Blocking Methods (Wait for Results)
    # =========================================================================

    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
    ) -> ExperimentResult:
        """Run a single experiment and wait for results.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            GPU tier override.
        timeout_seconds
            Timeout override.

        Returns
        -------
        ExperimentResult
            Result of the experiment.
        """
        gpu = gpu_type or self.gpu_type
        task_input = self._create_task_input(config, gpu, timeout_seconds)
        task_dict = asdict(task_input)

        # Use remote() for blocking execution
        modal_fn = self._select_modal_function(gpu)
        result_dict: dict[str, Any] = modal_fn.remote(task_dict)

        # Convert TaskOutput dict to ExperimentResult
        return self._task_output_to_result(result_dict, task_input.config)

    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str | None = None,
        parallelism: int = 4,
        timeout_seconds: int | None = None,
    ) -> list[ExperimentResult]:
        """Run multiple experiments in parallel and wait for all results.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            GPU tier for all experiments.
        parallelism
            Maximum concurrent experiments (handled by Modal).
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[ExperimentResult]
            Results from all experiments.
        """
        gpu = gpu_type or self.gpu_type
        task_inputs = [
            self._create_task_input(c, gpu, timeout_seconds) for c in configs
        ]
        task_dicts = [asdict(t) for t in task_inputs]

        # Use Modal's map() for parallel blocking execution
        modal_fn = self._select_modal_function(gpu)
        result_dicts = list(modal_fn.map(task_dicts))

        return [
            self._task_output_to_result(rd, ti.config)
            for rd, ti in zip(result_dicts, task_inputs, strict=True)
        ]

    def _task_output_to_result(
        self, output_dict: dict[str, Any], config: dict[str, Any]
    ) -> ExperimentResult:
        """Convert TaskOutput dict to ExperimentResult."""
        return ExperimentResult(
            run_id=output_dict["run_id"],
            config=config,
            metrics=output_dict.get("metrics", {}),
            checkpoint_path=output_dict.get("checkpoint_uri"),
            status=output_dict.get("status", "unknown"),
            error_message=output_dict.get("error_message"),
            duration_seconds=output_dict.get("duration_seconds", 0.0),
        )

    # =========================================================================
    # Status and Control
    # =========================================================================

    def get_status(self, run_id: str) -> str:
        """Get status of an experiment by checking Tigris storage.

        This is the source of truth for experiment completion, as it works
        across process restarts (unlike Modal FunctionCall handles).

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        str
            One of 'pending', 'running', 'completed', 'failed', 'unknown'.
        """
        if not self.storage:
            return (
                self._active_runs.get(run_id, SpawnedTask(run_id, "")).gpu_tier
                and "running"
                or "unknown"
            )

        # Check storage for completion
        try:
            # Storage.exists() checks for metrics file
            if self.storage.exists(f"metrics/{run_id}.json"):
                metrics = self.storage.download_metrics(run_id)
                return metrics.get("status", "completed")
        except Exception:
            pass

        # Check if we have an active spawn handle
        if run_id in self._active_runs:
            return "running"

        return "unknown"

    def cancel(self, run_id: str) -> bool:
        """Cancel a running experiment.

        Modal doesn't support direct cancellation of spawned functions.
        This removes the local tracking but the remote task continues.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        bool
            False (cancellation not supported).
        """
        if run_id in self._active_runs:
            del self._active_runs[run_id]
        return False

    @property
    def supports_spawn(self) -> bool:
        """ModalRunner supports detached (spawn) execution."""
        return True


def create_runner(gpu_type: str = "debug") -> ModalRunner:
    """Factory function to create a ModalRunner.

    Parameters
    ----------
    gpu_type
        Default GPU tier.

    Returns
    -------
    ModalRunner
        Configured runner instance.
    """
    storage = get_storage_from_env()
    return ModalRunner(gpu_type=gpu_type, storage=storage)
