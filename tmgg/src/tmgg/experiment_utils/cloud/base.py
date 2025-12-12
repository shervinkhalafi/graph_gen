"""Abstract base classes for cloud experiment execution.

Defines the CloudRunner interface that backends (Modal, Ray, local) implement,
along with common data structures for experiment results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


@dataclass
class ExperimentResult:
    """Result from a single experiment run.

    Attributes
    ----------
    run_id
        Unique identifier for this run.
    config
        The configuration used for this run.
    metrics
        Final metrics from training (val_loss, test_loss, etc.).
    checkpoint_path
        Path to the best model checkpoint (local or remote).
    status
        Completion status: 'completed', 'failed', or 'timeout'.
    error_message
        Error details if status is 'failed'.
    duration_seconds
        Wall-clock time for the experiment.
    """

    run_id: str
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    status: str = "completed"
    error_message: str | None = None
    duration_seconds: float = 0.0


class CloudRunner(ABC):
    """Abstract base for cloud execution backends.

    Implementations handle the specifics of launching experiments on
    different platforms (Modal, Ray, local subprocess) while providing
    a unified interface for the experiment coordinator.
    """

    @abstractmethod
    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str = "standard",
        timeout_seconds: int = 3600,
    ) -> ExperimentResult:
        """Run a single experiment with the given configuration.

        Parameters
        ----------
        config
            Hydra configuration for the experiment.
        gpu_type
            GPU tier to request ('debug', 'standard', 'fast', 'multi').
        timeout_seconds
            Maximum runtime before the job is killed.

        Returns
        -------
        ExperimentResult
            The result of the experiment run.
        """
        ...

    @abstractmethod
    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str = "standard",
        parallelism: int = 4,
        timeout_seconds: int = 3600,
    ) -> list[ExperimentResult]:
        """Run multiple experiments in parallel.

        Parameters
        ----------
        configs
            List of Hydra configurations for each experiment.
        gpu_type
            GPU tier to request for all experiments.
        parallelism
            Maximum number of concurrent experiments.
        timeout_seconds
            Maximum runtime per experiment.

        Returns
        -------
        list[ExperimentResult]
            Results from all experiments, in the same order as configs.
        """
        ...

    @abstractmethod
    def get_status(self, run_id: str) -> str:
        """Get the current status of a running experiment.

        Parameters
        ----------
        run_id
            The run identifier returned from run_experiment.

        Returns
        -------
        str
            One of 'pending', 'running', 'completed', 'failed', 'timeout'.
        """
        ...

    @abstractmethod
    def cancel(self, run_id: str) -> bool:
        """Cancel a running experiment.

        Parameters
        ----------
        run_id
            The run identifier to cancel.

        Returns
        -------
        bool
            True if cancellation was successful.
        """
        ...


class LocalRunner(CloudRunner):
    """Local execution backend for development and testing.

    Runs experiments in the current process without any cloud infrastructure.
    Useful for debugging configurations before deploying to cloud.
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize local runner.

        Parameters
        ----------
        output_dir
            Directory for experiment outputs. Defaults to ./outputs.
        """
        self.output_dir = output_dir or Path("./outputs")
        self._active_runs: dict[str, str] = {}

    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str = "standard",
        timeout_seconds: int = 3600,
    ) -> ExperimentResult:
        """Run experiment locally in current process."""
        import time
        import uuid

        from omegaconf import OmegaConf

        from tmgg.experiment_utils.run_experiment import run_experiment

        run_id = str(uuid.uuid4())[:8]
        self._active_runs[run_id] = "running"
        start_time = time.time()

        try:
            result = run_experiment(config)
            duration = time.time() - start_time
            self._active_runs[run_id] = "completed"

            return ExperimentResult(
                run_id=run_id,
                config=OmegaConf.to_container(config, resolve=True),
                metrics={
                    "best_val_loss": result.get("best_val_loss", float("inf")),
                },
                checkpoint_path=result.get("best_model_path"),
                status="completed",
                duration_seconds=duration,
            )
        except Exception:
            self._active_runs[run_id] = "failed"
            raise  # Crash the sweep with original traceback

    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str = "standard",
        parallelism: int = 4,
        timeout_seconds: int = 3600,
    ) -> list[ExperimentResult]:
        """Run experiments sequentially (no parallelism in local mode)."""
        return [
            self.run_experiment(config, gpu_type, timeout_seconds)
            for config in configs
        ]

    def get_status(self, run_id: str) -> str:
        """Get status of a local run."""
        return self._active_runs.get(run_id, "unknown")

    def cancel(self, run_id: str) -> bool:
        """Cancel not supported for local runs."""
        return False
