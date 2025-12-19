"""Ray-based CloudRunner implementation for local distributed execution.

Provides RayRunner that executes TMGG experiments on a local Ray cluster
with concurrency control and optional GPU scheduling. Uses the same unified
TaskInput/TaskOutput abstraction as ModalRunner.

Ray auto-initializes a local cluster if not already connected, making it
suitable for development and single-machine parallelism without external
infrastructure.

Example
-------
    >>> from tmgg.experiment_utils.cloud.ray_runner import RayRunner
    >>> runner = RayRunner(max_concurrent=4)
    >>> result = runner.run_experiment(config)

Note
----
Requires the `ray` optional dependency: `pip install tmgg[ray]`
"""

# pyright: reportMissingImports=false
# Ray is an optional dependency - these imports are checked at runtime

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, override

from omegaconf import DictConfig

from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult
from tmgg.experiment_utils.task import TaskInput, execute_task

if TYPE_CHECKING:
    import ray

logger = logging.getLogger(__name__)


def _ensure_ray_available() -> None:
    """Check that ray is installed and importable.

    Raises
    ------
    ImportError
        If ray is not installed.
    """
    try:
        import ray  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as e:
        raise ImportError(
            "Ray is not installed. Install it with: pip install tmgg[ray]"
        ) from e


def _get_ray():
    """Lazy import ray to avoid import errors when ray is not installed."""
    import ray

    return ray


# Remote task function - defined at module level but only usable after ray.init()
# We use a factory to avoid issues with ray not being initialized at import time.
_ray_execute_task_fn = None


def _get_ray_execute_task():
    """Get or create the ray remote task function.

    This lazy initialization ensures ray.remote decorator is applied
    only after ray is imported and initialized.
    """
    global _ray_execute_task_fn
    if _ray_execute_task_fn is None:
        ray = _get_ray()

        @ray.remote
        def ray_execute_task(task_dict: dict[str, Any]) -> dict[str, Any]:
            """Execute a single task on Ray using the unified task abstraction.

            This function runs inside a Ray worker. It wraps the backend-agnostic
            execute_task() function. Storage is not used for local Ray execution
            (results are returned directly via Ray's object store).

            Parameters
            ----------
            task_dict
                Serialized TaskInput as a dictionary.

            Returns
            -------
            dict
                TaskOutput as a dictionary.
            """
            task = TaskInput(**task_dict)
            # No cloud storage for local Ray - results returned directly
            output = execute_task(task, get_storage=None)
            return asdict(output)

        _ray_execute_task_fn = ray_execute_task

    return _ray_execute_task_fn


@dataclass
class RaySpawnedTask:
    """Handle for a spawned Ray task.

    Provides the run_id for tracking and the Ray ObjectRef for waiting
    on completion or checking status.

    Attributes
    ----------
    run_id
        Unique identifier for the spawned experiment.
    object_ref
        Ray ObjectRef for polling or getting results.
    """

    run_id: str
    object_ref: ray.ObjectRef  # type: ignore[name-defined]


class RayRunner(CloudRunner):
    """Ray-based experiment runner for local distributed execution.

    Executes experiments on a local Ray cluster with configurable
    concurrency and optional GPU scheduling. Automatically initializes
    Ray if not already connected.

    Unlike ModalRunner, RayRunner does not use cloud storage - results
    are returned directly through Ray's object store. This makes it
    suitable for development and single-machine parallelism.

    Attributes
    ----------
    max_concurrent
        Maximum number of concurrent tasks.
    num_gpus_per_task
        GPU fraction per task (0.0 for CPU-only, 1.0 for one GPU each).
    num_cpus_per_task
        CPU cores per task.
    """

    max_concurrent: int
    num_gpus_per_task: float
    num_cpus_per_task: int

    def __init__(
        self,
        max_concurrent: int = 4,
        num_gpus_per_task: float = 0.0,
        num_cpus_per_task: int = 1,
    ):
        """Initialize Ray runner.

        Parameters
        ----------
        max_concurrent
            Maximum concurrent experiments. Controls parallelism via
            Ray's resource scheduling (each task requests 1/max_concurrent
            of the 'experiment' resource).
        num_gpus_per_task
            GPU fraction per task. Set to 0.0 for CPU-only execution,
            1.0 to allocate one GPU per task.
        num_cpus_per_task
            CPU cores per task.

        Raises
        ------
        ImportError
            If ray is not installed.
        """
        _ensure_ray_available()
        ray = _get_ray()

        # Auto-init if not connected
        if not ray.is_initialized():  # pyright: ignore[reportUnknownMemberType]
            logger.info("Initializing local Ray cluster")
            ray.init(  # pyright: ignore[reportUnknownMemberType]
                ignore_reinit_error=True,
                # Define custom resource for concurrency control
                resources={"experiment_slot": max_concurrent},
            )

        self.max_concurrent = max_concurrent
        self.num_gpus_per_task = num_gpus_per_task
        self.num_cpus_per_task = num_cpus_per_task
        self._active_runs: dict[str, RaySpawnedTask] = {}

    def _create_task_input(
        self,
        config: DictConfig,
        timeout_seconds: int,
    ) -> TaskInput:
        """Create a TaskInput from a Hydra config.

        Parameters
        ----------
        config
            Hydra configuration.
        timeout_seconds
            Timeout for the task.

        Returns
        -------
        TaskInput
            Serializable task input ready for remote execution.
        """
        from tmgg.experiment_utils.task import prepare_config_for_remote

        config_dict = prepare_config_for_remote(config)
        run_id = config_dict.get("run_id", config_dict["run_id"])

        return TaskInput(
            config=config_dict,
            run_id=run_id,
            gpu_tier="local",  # Ray doesn't use GPU tiers
            timeout_seconds=timeout_seconds,
        )

    def _get_remote_options(self) -> dict[str, Any]:
        """Get Ray remote options for task execution."""
        options: dict[str, Any] = {
            "num_cpus": self.num_cpus_per_task,
            # Use experiment_slot resource for concurrency control
            "resources": {"experiment_slot": 1},
        }
        if self.num_gpus_per_task > 0:
            options["num_gpus"] = self.num_gpus_per_task
        return options

    # =========================================================================
    # Spawn Methods (Detached Execution)
    # =========================================================================

    def spawn_experiment(
        self,
        config: DictConfig,
        timeout_seconds: int = 3600,
    ) -> RaySpawnedTask:
        """Spawn a single experiment without waiting for results.

        Uses Ray's `.remote()` for non-blocking execution. Results can be
        retrieved later via `ray.get()` on the returned object ref.

        Parameters
        ----------
        config
            Hydra configuration.
        timeout_seconds
            Timeout for the task.

        Returns
        -------
        RaySpawnedTask
            Handle with run_id and ObjectRef for tracking.
        """
        task_input = self._create_task_input(config, timeout_seconds)
        task_dict = asdict(task_input)

        # Get the remote function with options
        ray_task = _get_ray_execute_task()
        options = self._get_remote_options()
        object_ref = ray_task.options(**options).remote(task_dict)  # pyright: ignore[reportFunctionMemberAccess,reportOptionalMemberAccess]

        spawned = RaySpawnedTask(
            run_id=task_input.run_id,
            object_ref=object_ref,
        )
        self._active_runs[task_input.run_id] = spawned

        logger.info(f"Spawned experiment {task_input.run_id} on Ray (detached)")
        return spawned

    def spawn_sweep(
        self,
        configs: list[DictConfig],
        timeout_seconds: int = 3600,
    ) -> list[RaySpawnedTask]:
        """Spawn multiple experiments without waiting for results.

        Each experiment is spawned independently. Ray's scheduler handles
        concurrency based on the experiment_slot resource.

        Parameters
        ----------
        configs
            List of configurations.
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[RaySpawnedTask]
            Handles with run_ids for tracking.
        """
        spawned_tasks: list[RaySpawnedTask] = []

        for config in configs:
            task = self.spawn_experiment(config, timeout_seconds)
            spawned_tasks.append(task)

        logger.info(f"Spawned {len(spawned_tasks)} experiments on Ray (detached)")
        return spawned_tasks

    # =========================================================================
    # Blocking Methods (Wait for Results)
    # =========================================================================

    @override
    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str = "debug",
        timeout_seconds: int = 3600,
    ) -> ExperimentResult:
        """Run a single experiment and wait for results.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            Ignored for Ray (uses num_gpus_per_task from init).
        timeout_seconds
            Timeout for the task.

        Returns
        -------
        ExperimentResult
            Result of the experiment.
        """
        ray = _get_ray()
        spawned = self.spawn_experiment(config, timeout_seconds)

        # Block until complete
        result_dict: dict[str, Any] = ray.get(spawned.object_ref)  # pyright: ignore[reportUnknownMemberType]

        # Clean up tracking
        if spawned.run_id in self._active_runs:
            del self._active_runs[spawned.run_id]

        return self._task_output_to_result(result_dict, spawned.run_id, config)

    @override
    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str = "debug",
        parallelism: int = 4,
        timeout_seconds: int = 3600,
    ) -> list[ExperimentResult]:
        """Run multiple experiments in parallel and wait for all results.

        Ray's scheduler handles parallelism via the experiment_slot resource
        defined at initialization (max_concurrent). The parallelism parameter
        is ignored.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            Ignored for Ray.
        parallelism
            Ignored (uses max_concurrent from init).
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[ExperimentResult]
            Results from all experiments.
        """
        ray = _get_ray()
        spawned_tasks = self.spawn_sweep(configs, timeout_seconds)

        # Collect all results
        object_refs = [t.object_ref for t in spawned_tasks]
        result_dicts: list[dict[str, Any]] = ray.get(object_refs)  # pyright: ignore[reportUnknownMemberType]

        # Clean up tracking
        for task in spawned_tasks:
            if task.run_id in self._active_runs:
                del self._active_runs[task.run_id]

        return [
            self._task_output_to_result(rd, task.run_id, config)
            for rd, task, config in zip(
                result_dicts, spawned_tasks, configs, strict=True
            )
        ]

    def _task_output_to_result(
        self, output_dict: dict[str, Any], run_id: str, config: DictConfig
    ) -> ExperimentResult:
        """Convert TaskOutput dict to ExperimentResult."""
        from typing import cast

        from omegaconf import OmegaConf

        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"Expected dict from OmegaConf.to_container, got {type(config_dict)}"
            )

        return ExperimentResult(
            run_id=run_id,
            config=cast(dict[str, Any], config_dict),
            metrics=output_dict.get("metrics", {}),
            checkpoint_path=output_dict.get("checkpoint_uri"),
            status=output_dict.get("status", "unknown"),
            error_message=output_dict.get("error_message"),
            duration_seconds=output_dict.get("duration_seconds", 0.0),
        )

    # =========================================================================
    # Status and Control
    # =========================================================================

    @override
    def get_status(self, run_id: str) -> str:
        """Get status of an experiment.

        Uses Ray's object store to check if the result is ready.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        str
            One of 'pending', 'running', 'completed', 'failed', 'unknown'.
        """
        ray = _get_ray()

        if run_id not in self._active_runs:
            return "unknown"

        spawned = self._active_runs[run_id]

        # Check if result is ready without blocking
        ready, _ = ray.wait([spawned.object_ref], timeout=0)  # pyright: ignore[reportUnknownMemberType]

        if not ready:
            return "running"

        # Result is ready - check if it succeeded or failed
        try:
            result_dict: dict[str, Any] = ray.get(spawned.object_ref)  # pyright: ignore[reportUnknownMemberType]
            return str(result_dict.get("status", "completed"))
        except Exception:
            return "failed"

    @override
    def cancel(self, run_id: str) -> bool:
        """Cancel a running experiment.

        Uses Ray's task cancellation to attempt to stop the task.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        bool
            True if cancellation was requested (may not succeed immediately).
        """
        ray = _get_ray()

        if run_id not in self._active_runs:
            return False

        spawned = self._active_runs[run_id]
        try:
            ray.cancel(spawned.object_ref)  # pyright: ignore[reportUnknownMemberType]
            del self._active_runs[run_id]
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        """Shutdown the Ray cluster.

        Call this when done with the runner to clean up Ray resources.
        """
        ray = _get_ray()
        if ray.is_initialized():  # pyright: ignore[reportUnknownMemberType]
            ray.shutdown()  # pyright: ignore[reportUnknownMemberType]

    @property
    def supports_spawn(self) -> bool:
        """RayRunner supports detached (spawn) execution."""
        return True
