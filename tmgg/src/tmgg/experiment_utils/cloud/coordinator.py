"""Experiment coordinator for managing multi-experiment runs.

Handles stage execution, result aggregation, and experiment tracking
across distributed runs.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.cloud.base import (
    CloudRunner,
    ExperimentResult,
    LocalRunner,
    SpawnedTask,
)
from tmgg.experiment_utils.cloud.storage import CloudStorage, LocalStorage


@dataclass
class StageConfig:
    """Configuration for an experimental stage.

    Attributes
    ----------
    name
        Stage identifier (e.g., 'stage1_poc').
    architectures
        List of model configurations to run.
    datasets
        List of dataset configurations to use.
    hyperparameter_space
        Hyperparameters to sweep (learning_rate, weight_decay, etc.).
    num_trials
        Number of Bayesian optimization trials per architecture.
    seeds
        Random seeds for reproducibility.
    gpu_type
        GPU tier for this stage.
    timeout_seconds
        Maximum runtime per experiment.
    """

    name: str
    architectures: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    hyperparameter_space: dict[str, list[Any]] = field(default_factory=dict)
    num_trials: int = 4
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    gpu_type: str = "debug"
    timeout_seconds: int = 3600

    @classmethod
    def from_yaml(cls, path: Path) -> "StageConfig":
        """Load stage configuration from YAML file."""
        config_raw = OmegaConf.load(path)
        if not isinstance(config_raw, DictConfig):
            raise TypeError(f"Expected DictConfig, got {type(config_raw)}")
        config = config_raw

        # Sweep metadata is nested under _sweep_config
        sweep_config_raw = config.get("_sweep_config", {})
        sweep_config = (
            sweep_config_raw if isinstance(sweep_config_raw, dict | DictConfig) else {}
        )

        # Handle both OmegaConf and plain dict
        hp_space_raw = sweep_config.get("hyperparameter_space", {})
        hp_space: dict[str, list[Any]]
        if OmegaConf.is_config(hp_space_raw):
            hp_space = cast(
                dict[str, list[Any]], OmegaConf.to_container(hp_space_raw, resolve=True)
            )
        else:
            hp_space = cast(dict[str, list[Any]], hp_space_raw)

        stage_name_raw = config.get("stage", config.get("name", path.stem))
        stage_name = str(stage_name_raw) if stage_name_raw is not None else path.stem

        return cls(
            name=stage_name,
            architectures=list(sweep_config.get("architectures", [])),
            datasets=list(sweep_config.get("datasets", [])),
            hyperparameter_space=hp_space,
            num_trials=int(sweep_config.get("num_trials", 4)),
            seeds=list(sweep_config.get("seeds", [1, 2, 3])),
            gpu_type=str(sweep_config.get("gpu_type", "debug")),
            timeout_seconds=int(sweep_config.get("timeout_seconds", 3600)),
        )


@dataclass
class StageResult:
    """Aggregated results from a stage run.

    Attributes
    ----------
    stage_name
        Name of the stage.
    experiments
        Individual experiment results.
    best_config
        Configuration that achieved best validation loss.
    best_metrics
        Metrics from the best run.
    summary
        Aggregated statistics across all runs.
    started_at
        Timestamp when stage started.
    completed_at
        Timestamp when stage completed.
    """

    stage_name: str
    experiments: list[ExperimentResult] = field(default_factory=list)
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "experiments": [
                {
                    "run_id": e.run_id,
                    "status": e.status,
                    "metrics": e.metrics,
                    "duration_seconds": e.duration_seconds,
                }
                for e in self.experiments
            ],
            "best_config": self.best_config,
            "best_metrics": self.best_metrics,
            "summary": self.summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ExperimentCoordinator:
    """Coordinates multi-experiment runs with checkpointing.

    Manages the execution of experimental stages, handling configuration
    generation, result aggregation, and persistence to cloud storage.
    """

    def __init__(
        self,
        runner: CloudRunner | None = None,
        storage: CloudStorage | None = None,
        base_config_path: Path | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize the coordinator.

        Parameters
        ----------
        runner
            Cloud runner for experiment execution. Defaults to LocalRunner.
        storage
            Cloud storage backend for checkpoints and metrics.
            Defaults to LocalStorage.
        base_config_path
            Path to base Hydra configuration directory.
        cache_dir
            Local cache directory for temporary files.
        """
        self.runner = runner or LocalRunner()
        self.storage = storage or LocalStorage()
        self.base_config_path = base_config_path
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_configs(
        self,
        stage: StageConfig,
        base_config: DictConfig,
    ) -> Iterator[DictConfig]:
        """Generate experiment configurations for a stage.

        Creates the cartesian product of architectures, datasets,
        hyperparameters, and seeds.

        Parameters
        ----------
        stage
            Stage configuration defining the sweep space.
        base_config
            Base Hydra configuration to extend.

        Yields
        ------
        DictConfig
            Individual experiment configuration.
        """
        import itertools

        # Build hyperparameter combinations
        hp_keys = list(stage.hyperparameter_space.keys())
        hp_values = [stage.hyperparameter_space[k] for k in hp_keys]
        hp_combos = list(itertools.product(*hp_values)) if hp_values else [()]

        # Loop seeds outermost: run all configs with seed 1, then seed 2, etc.
        # This ensures we get one result per config before repeating with different seeds.
        for seed in stage.seeds:
            for arch in stage.architectures or [None]:
                for dataset in stage.datasets or [None]:
                    for hp_combo in hp_combos:
                        config_raw = OmegaConf.create(
                            OmegaConf.to_container(base_config, resolve=True)
                        )
                        config = cast(DictConfig, config_raw)

                        # Apply architecture override by loading and merging config
                        if arch:
                            if self.base_config_path is None:
                                raise RuntimeError(
                                    "base_config_path must be set to apply architecture overrides"
                                )
                            arch_path = self.base_config_path / f"{arch}.yaml"
                            if arch_path.exists():
                                arch_config_raw = OmegaConf.load(arch_path)
                                arch_config = cast(DictConfig, arch_config_raw)
                                # If _target_ changes (different model class), replace entirely
                                # to avoid leftover params from base config (e.g. model_type)
                                if arch_config.get("_target_") != config.model.get(
                                    "_target_"
                                ):
                                    config.model = arch_config
                                else:
                                    config.model = OmegaConf.merge(
                                        config.model, arch_config
                                    )
                            else:
                                raise FileNotFoundError(
                                    f"Architecture config not found: {arch_path}"
                                )

                        # Apply dataset override by loading and merging config
                        if dataset:
                            if self.base_config_path is None:
                                raise RuntimeError(
                                    "base_config_path must be set to apply dataset overrides"
                                )
                            dataset_path = self.base_config_path / f"{dataset}.yaml"
                            if dataset_path.exists():
                                dataset_config = OmegaConf.load(dataset_path)
                                config.data = OmegaConf.merge(
                                    config.data, dataset_config
                                )
                            else:
                                raise FileNotFoundError(
                                    f"Dataset config not found: {dataset_path}"
                                )

                        # Apply hyperparameters
                        for key, value in zip(hp_keys, hp_combo, strict=False):
                            OmegaConf.update(config, key, value)
                            # Keep data.noise_levels in sync with noise_levels
                            # (interpolation ${noise_levels} was resolved at config creation)
                            if key == "noise_levels":
                                OmegaConf.update(config, "data.noise_levels", value)

                        # Apply seed
                        config.seed = seed

                        # Training settings come from Hydra config inheritance
                        # (base_config_spectral.yaml -> base/trainer/default.yaml)

                        # Generate deterministic run ID from config signature
                        run_id_parts = [stage.name]
                        if arch:
                            run_id_parts.append(
                                arch.split("/")[-1].replace(".yaml", "")
                            )
                        if dataset:
                            run_id_parts.append(
                                dataset.split("/")[-1].replace(".yaml", "")
                            )
                        # Include hyperparameters in run_id for uniqueness
                        for key, value in zip(hp_keys, hp_combo, strict=False):
                            short_key = key.split(".")[-1]
                            run_id_parts.append(f"{short_key}_{value}")
                        run_id_parts.append(f"s{seed}")
                        config.run_id = "_".join(run_id_parts)

                        # Isolate output directory per sweep run
                        sweep_output_dir = (
                            Path("outputs/sweeps") / stage.name / config.run_id
                        )
                        config.paths.output_dir = str(sweep_output_dir)
                        config.paths.results_dir = str(sweep_output_dir / "results")

                        # Make W&B run name unique
                        if "wandb" in config and config.wandb is not None:
                            base_name = config.wandb.get("name", "run")
                            config.wandb.name = f"{base_name}_{config.run_id}"

                        yield config

    def run_stage(
        self,
        stage: StageConfig,
        base_config: DictConfig,
        resume: bool = True,
    ) -> StageResult:
        """Execute all experiments in a stage.

        Parameters
        ----------
        stage
            Stage configuration.
        base_config
            Base Hydra configuration.
        resume
            If True, skip experiments with existing results.

        Returns
        -------
        StageResult
            Aggregated results from the stage.
        """
        started_at = datetime.now().isoformat()
        configs = list(self.generate_configs(stage, base_config))

        # Filter out completed runs if resuming
        if resume:
            configs = [
                c for c in configs if not self._check_completed(c.get("run_id", ""))
            ]

        if not configs:
            return self._load_stage_result(stage.name)

        # Run experiments
        results = self.runner.run_sweep(
            configs,
            gpu_type=stage.gpu_type,
            timeout_seconds=stage.timeout_seconds,
        )

        # Persist individual results
        for result in results:
            self._persist_result(result)

        # Aggregate results
        stage_result = self._aggregate_results(stage.name, results, started_at)

        # Persist stage result
        self._persist_stage_result(stage_result)

        return stage_result

    # =========================================================================
    # Async/Spawn Methods (for detached execution)
    # =========================================================================

    def spawn_stage(
        self,
        stage: StageConfig,
        base_config: DictConfig,
        resume: bool = True,
    ) -> list[SpawnedTask]:
        """Spawn all experiments in a stage without waiting.

        Fire-and-forget execution. Use get_stage_status() to poll and
        wait_for_stage() to block until completion.

        Parameters
        ----------
        stage
            Stage configuration.
        base_config
            Base Hydra configuration.
        resume
            If True, skip experiments with existing results.

        Returns
        -------
        list[SpawnedTask]
            Handles for tracking spawned experiments.

        Raises
        ------
        NotImplementedError
            If the runner does not support spawn execution.
        """
        if not self.runner.supports_spawn:
            raise NotImplementedError(
                f"{self.runner.__class__.__name__} does not support spawn execution. "
                + "Use run_stage() for blocking execution, or use ModalRunner/RayRunner."
            )

        configs = list(self.generate_configs(stage, base_config))

        # Filter out completed runs if resuming
        if resume:
            configs = [
                c for c in configs if not self._check_completed(c.get("run_id", ""))
            ]

        if not configs:
            return []

        # Spawn experiments
        return self.runner.spawn_sweep(
            configs,
            gpu_type=stage.gpu_type,
            timeout_seconds=stage.timeout_seconds,
        )

    def get_stage_status(
        self,
        spawned_tasks: list[SpawnedTask],
    ) -> dict[str, str]:
        """Get status of all experiments in a spawned stage.

        Polls storage and runner to determine experiment status.

        Parameters
        ----------
        spawned_tasks
            List of spawned task handles from spawn_stage().

        Returns
        -------
        dict[str, str]
            Mapping of run_id to status ('pending', 'running', 'completed', 'failed').
        """
        status_map: dict[str, str] = {}

        for task in spawned_tasks:
            run_id = task.run_id

            # First check storage (source of truth for completion)
            if self._check_completed(run_id):
                # Load metrics to get actual status
                try:
                    metrics = self.storage.download_metrics(f"results/{run_id}")
                    status_map[run_id] = metrics.get("status", "completed")
                except Exception:
                    status_map[run_id] = "completed"
            else:
                # Not in storage - check runner status
                status_map[run_id] = self.runner.get_status(run_id)

        return status_map

    def wait_for_stage(
        self,
        spawned_tasks: list[SpawnedTask],
        stage_name: str,
        poll_interval_seconds: float = 30.0,
        timeout_seconds: float | None = None,
    ) -> StageResult:
        """Wait for all spawned experiments to complete.

        Polls storage periodically until all experiments finish.

        Parameters
        ----------
        spawned_tasks
            List of spawned task handles from spawn_stage().
        stage_name
            Name of the stage for result aggregation.
        poll_interval_seconds
            Time between status polls.
        timeout_seconds
            Maximum wait time. None means wait indefinitely.

        Returns
        -------
        StageResult
            Aggregated results from the stage.

        Raises
        ------
        TimeoutError
            If timeout_seconds is exceeded before all experiments complete.
        """
        import time

        started_at = datetime.now().isoformat()
        start_time = time.time()
        run_ids = [t.run_id for t in spawned_tasks]

        while True:
            # Check status of all runs
            status_map = self.get_stage_status(spawned_tasks)

            # Count completed (including failed)
            terminal_statuses = {"completed", "failed", "timeout"}
            completed_count = sum(
                1 for status in status_map.values() if status in terminal_statuses
            )

            if completed_count == len(run_ids):
                break

            # Check timeout
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        f"Stage {stage_name} did not complete within {timeout_seconds}s. "
                        + f"Completed: {completed_count}/{len(run_ids)}"
                    )

            # Wait before next poll
            time.sleep(poll_interval_seconds)

        # Load results from storage
        results: list[ExperimentResult] = []
        for run_id in run_ids:
            try:
                metrics = self.storage.download_metrics(f"results/{run_id}")
                results.append(
                    ExperimentResult(
                        run_id=run_id,
                        config=metrics.get("config", {}),
                        metrics=metrics.get("metrics", {}),
                        checkpoint_path=metrics.get("checkpoint_path"),
                        status=metrics.get("status", "unknown"),
                        error_message=metrics.get("error_message"),
                        duration_seconds=metrics.get("duration_seconds", 0.0),
                    )
                )
            except Exception as e:
                # Failed to load result - mark as unknown
                results.append(
                    ExperimentResult(
                        run_id=run_id,
                        config={},
                        status="unknown",
                        error_message=f"Failed to load result: {e}",
                    )
                )

        # Aggregate and persist
        stage_result = self._aggregate_results(stage_name, results, started_at)
        self._persist_stage_result(stage_result)

        return stage_result

    def _check_completed(self, run_id: str) -> bool:
        """Check if a run has already completed."""
        return self.storage.exists(f"results/{run_id}.json")

    def _persist_result(self, result: ExperimentResult) -> None:
        """Persist individual experiment result."""
        self.storage.upload_metrics(
            {
                "run_id": result.run_id,
                "status": result.status,
                "metrics": result.metrics,
                "checkpoint_path": result.checkpoint_path,
                "error_message": result.error_message,
                "duration_seconds": result.duration_seconds,
                "config": result.config,
            },
            f"results/{result.run_id}",
        )

    def _persist_stage_result(self, result: StageResult) -> None:
        """Persist aggregated stage result."""
        self.storage.upload_metrics(result.to_dict(), f"stages/{result.stage_name}")

    def _load_stage_result(self, stage_name: str) -> StageResult:
        """Load existing stage result from storage."""
        try:
            data = self.storage.download_metrics(f"stages/{stage_name}")
            return StageResult(
                stage_name=data["stage_name"],
                best_config=data.get("best_config"),
                best_metrics=data.get("best_metrics"),
                summary=data.get("summary", {}),
                started_at=data.get("started_at", ""),
                completed_at=data.get("completed_at", ""),
            )
        except Exception:
            return StageResult(stage_name=stage_name)

    def _aggregate_results(
        self,
        stage_name: str,
        results: list[ExperimentResult],
        started_at: str,
    ) -> StageResult:
        """Aggregate individual results into stage result."""
        completed_at = datetime.now().isoformat()

        # Find best result by validation loss
        completed_results = [r for r in results if r.status == "completed"]
        best_result = None
        if completed_results:
            best_result = min(
                completed_results,
                key=lambda r: r.metrics.get("best_val_loss", float("inf")),
            )

        # Compute summary statistics
        total = len(results)
        completed = len(completed_results)
        failed = len([r for r in results if r.status == "failed"])
        durations = [r.duration_seconds for r in results if r.duration_seconds > 0]

        summary = {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
            "mean_duration_seconds": sum(durations) / len(durations)
            if durations
            else 0,
            "total_duration_seconds": sum(durations),
        }

        return StageResult(
            stage_name=stage_name,
            experiments=results,
            best_config=best_result.config if best_result else None,
            best_metrics=best_result.metrics if best_result else None,
            summary=summary,
            started_at=started_at,
            completed_at=completed_at,
        )

    def get_best_config(self, stage_name: str) -> dict[str, Any] | None:
        """Retrieve the best configuration from a completed stage."""
        result = self._load_stage_result(stage_name)
        return result.best_config

    def export_results(self, stage_name: str, output_path: Path) -> None:
        """Export stage results to a local JSON file."""
        result = self._load_stage_result(stage_name)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
