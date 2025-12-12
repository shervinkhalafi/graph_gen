"""Experiment coordinator for managing multi-experiment runs.

Handles stage execution, result aggregation, and experiment tracking
across distributed runs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult
from tmgg.experiment_utils.cloud.factory import CloudRunnerFactory
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
    gpu_type: str = "standard"
    timeout_seconds: int = 3600

    @classmethod
    def from_yaml(cls, path: Path) -> "StageConfig":
        """Load stage configuration from YAML file."""
        config = OmegaConf.load(path)

        # Sweep metadata is nested under _sweep_config
        sweep_config = config.get("_sweep_config", {})

        # Handle both OmegaConf and plain dict
        hp_space = sweep_config.get("hyperparameter_space", {})
        if OmegaConf.is_config(hp_space):
            hp_space = OmegaConf.to_container(hp_space, resolve=True)

        return cls(
            name=config.get("stage", config.get("name", path.stem)),
            architectures=list(sweep_config.get("architectures", [])),
            datasets=list(sweep_config.get("datasets", [])),
            hyperparameter_space=hp_space,
            num_trials=sweep_config.get("num_trials", 4),
            seeds=list(sweep_config.get("seeds", [1, 2, 3])),
            gpu_type=sweep_config.get("gpu_type", "standard"),
            timeout_seconds=sweep_config.get("timeout_seconds", 3600),
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
        storage: CloudStorage | None = None,
        runner: CloudRunner | None = None,
        backend: str = "local",
        base_config_path: Path | None = None,
        cache_dir: Path | None = None,
        **runner_kwargs,
    ):
        """Initialize the coordinator.

        Parameters
        ----------
        storage
            Cloud storage backend for checkpoints and metrics.
            Defaults to LocalStorage.
        runner
            Cloud runner for experiment execution. If provided, takes
            precedence over backend parameter.
        backend
            Backend name to use via CloudRunnerFactory (e.g., "local", "modal").
            Ignored if runner is provided directly.
        base_config_path
            Path to base Hydra configuration directory.
        cache_dir
            Local cache directory for temporary files.
        **runner_kwargs
            Additional arguments passed to CloudRunnerFactory.create()
            when using backend parameter.
        """
        self.storage = storage or LocalStorage()
        self.runner = runner or CloudRunnerFactory.create(backend, **runner_kwargs)
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
                        config = OmegaConf.create(
                            OmegaConf.to_container(base_config, resolve=True)
                        )

                        # Apply architecture override by loading and merging config
                        if arch:
                            arch_path = self.base_config_path / f"{arch}.yaml"
                            if arch_path.exists():
                                arch_config = OmegaConf.load(arch_path)
                                # If _target_ changes (different model class), replace entirely
                                # to avoid leftover params from base config (e.g. model_type)
                                if arch_config.get("_target_") != config.model.get("_target_"):
                                    config.model = arch_config
                                else:
                                    config.model = OmegaConf.merge(config.model, arch_config)
                            else:
                                raise FileNotFoundError(
                                    f"Architecture config not found: {arch_path}"
                                )

                        # Apply dataset override by loading and merging config
                        if dataset:
                            dataset_path = self.base_config_path / f"{dataset}.yaml"
                            if dataset_path.exists():
                                dataset_config = OmegaConf.load(dataset_path)
                                config.data = OmegaConf.merge(config.data, dataset_config)
                            else:
                                raise FileNotFoundError(
                                    f"Dataset config not found: {dataset_path}"
                                )

                        # Apply hyperparameters
                        for key, value in zip(hp_keys, hp_combo):
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
                            run_id_parts.append(arch.split("/")[-1].replace(".yaml", ""))
                        if dataset:
                            run_id_parts.append(dataset.split("/")[-1].replace(".yaml", ""))
                        # Include hyperparameters in run_id for uniqueness
                        for key, value in zip(hp_keys, hp_combo):
                            short_key = key.split(".")[-1]
                            run_id_parts.append(f"{short_key}_{value}")
                        run_id_parts.append(f"s{seed}")
                        config.run_id = "_".join(run_id_parts)

                        # Isolate output directory per sweep run
                        sweep_output_dir = Path("outputs/sweeps") / stage.name / config.run_id
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
        parallelism: int = 4,
        resume: bool = True,
    ) -> StageResult:
        """Execute all experiments in a stage.

        Parameters
        ----------
        stage
            Stage configuration.
        base_config
            Base Hydra configuration.
        parallelism
            Maximum concurrent experiments.
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
                c for c in configs
                if not self._check_completed(c.get("run_id", ""))
            ]

        if not configs:
            return self._load_stage_result(stage.name)

        # Run experiments
        results = self.runner.run_sweep(
            configs,
            gpu_type=stage.gpu_type,
            parallelism=parallelism,
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
            "mean_duration_seconds": sum(durations) / len(durations) if durations else 0,
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
