"""Custom Hydra launcher that dispatches to CloudRunner backends.

Replaces the hacky sys.argv injection pattern with proper Hydra integration.
Supports local, Modal, and Ray execution backends.
"""

from collections.abc import Sequence
from typing import Any, cast, override

from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult, LocalRunner


class TmggLauncher(Launcher):
    """Custom Hydra launcher for TMGG experiments.

    Dispatches jobs to CloudRunner backends (LocalRunner, ModalRunner, RayRunner)
    based on configuration. Converts between Hydra's job paradigm and CloudRunner's
    experiment interface.
    """

    def __init__(self) -> None:
        """Initialize launcher state."""
        self._runner: CloudRunner | None = None
        self._config: DictConfig | None = None
        self._hydra_context: HydraContext | None = None

    @override
    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Configure launcher with Hydra context and select runner.

        Parameters
        ----------
        hydra_context
            Hydra's context containing config loader and other state.
        task_function
            The decorated main function (unused, we call CloudRunner directly).
        config
            The resolved Hydra configuration.
        """
        self._hydra_context = hydra_context
        self._config = config
        self._runner = self._select_runner(config)

    @override
    def launch(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
    ) -> Sequence[JobReturn]:
        """Execute jobs via CloudRunner.

        Parameters
        ----------
        job_overrides
            List of override sequences, one per job. Each override is a list
            of strings like ["model=test", "lr=0.01"].
        initial_job_idx
            Starting index for job numbering.

        Returns
        -------
        Sequence[JobReturn]
            Hydra JobReturn objects for each executed job.
        """
        if self._runner is None or self._config is None:
            raise RuntimeError("Launcher.setup() must be called before launch()")

        results: list[JobReturn] = []
        for idx, overrides in enumerate(job_overrides):
            job_idx = initial_job_idx + idx
            cfg = self._apply_overrides(self._config, list(overrides))

            # Get launcher-specific settings
            launcher_cfg = self._get_launcher_config(cfg)
            gpu_type = launcher_cfg.get("gpu_type", "debug")
            timeout = launcher_cfg.get("timeout_seconds", 3600)

            # Execute via CloudRunner
            experiment_result = self._runner.run_experiment(
                cfg,
                gpu_type=gpu_type,
                timeout_seconds=timeout,
            )

            job_return = self._result_to_job_return(
                experiment_result,
                overrides=list(overrides),
                idx=job_idx,
            )
            results.append(job_return)

        return results

    def _select_runner(self, config: DictConfig) -> CloudRunner:
        """Select appropriate CloudRunner based on configuration.

        Parameters
        ----------
        config
            Hydra configuration containing launcher settings.

        Returns
        -------
        CloudRunner
            Configured runner instance.
        """
        launcher_cfg = self._get_launcher_config(config)

        if launcher_cfg.get("use_modal", False):
            gpu_type = launcher_cfg.get("gpu_type", "debug")
            return self._create_modal_runner(gpu_type=gpu_type)
        elif launcher_cfg.get("use_ray", False):
            return self._create_ray_runner()
        elif launcher_cfg.get("use_slurm", False):
            return self._create_slurm_runner(launcher_cfg)
        else:
            return self._create_local_runner()

    def _get_launcher_config(self, config: DictConfig) -> dict[str, Any]:
        """Extract launcher configuration from Hydra config.

        Parameters
        ----------
        config
            Full Hydra configuration.

        Returns
        -------
        dict
            Launcher-specific settings (use_modal, use_ray, gpu_type, etc.).
        """
        hydra_cfg = OmegaConf.to_container(config.get("hydra", {}), resolve=True)
        if not isinstance(hydra_cfg, dict):
            return {}
        return hydra_cfg.get("launcher", {})  # type: ignore[return-value]

    def _apply_overrides(
        self, base_config: DictConfig, overrides: list[str]
    ) -> DictConfig:
        """Apply job-specific overrides to base configuration.

        Parameters
        ----------
        base_config
            Base configuration from setup().
        overrides
            Job-specific override strings.

        Returns
        -------
        DictConfig
            New configuration with overrides applied.
        """
        container = OmegaConf.to_container(base_config, resolve=True)
        cfg = cast(DictConfig, OmegaConf.create(container))

        for ovr in overrides:
            if "=" not in ovr:
                continue
            key, value = ovr.split("=", 1)
            key = key.lstrip("+")  # Handle Hydra append syntax
            OmegaConf.update(cfg, key, value)

        return cfg

    @staticmethod
    def _result_to_job_return(
        result: ExperimentResult,
        overrides: list[str],
        idx: int,
    ) -> JobReturn:
        """Convert ExperimentResult to Hydra JobReturn.

        Parameters
        ----------
        result
            CloudRunner experiment result.
        overrides
            The overrides used for this job.
        idx
            Job index (stored in return value for tracking).

        Returns
        -------
        JobReturn
            Hydra-compatible job result.
        """
        if result.status == "completed":
            status = JobStatus.COMPLETED
        else:
            status = JobStatus.FAILED

        # Include idx in return value for tracking
        return_value = {**result.metrics, "_job_idx": idx}

        return JobReturn(
            _return_value=return_value,
            overrides=overrides,
            cfg=OmegaConf.create(result.config),
            hydra_cfg=None,
            working_dir=None,
            task_name=None,
            status=status,
        )

    def _create_local_runner(self) -> CloudRunner:
        """Create LocalRunner instance."""
        return LocalRunner()

    def _create_modal_runner(self, gpu_type: str = "debug") -> CloudRunner:
        """Create ModalRunner instance.

        Parameters
        ----------
        gpu_type
            GPU tier to request on Modal.

        Returns
        -------
        CloudRunner
            Configured ModalRunner.
        """
        from tmgg.modal.runner import ModalRunner

        return ModalRunner(gpu_type=gpu_type)

    def _create_ray_runner(self) -> CloudRunner:
        """Create RayRunner instance.

        Returns
        -------
        CloudRunner
            Configured RayRunner.
        """
        from tmgg.experiment_utils.cloud.ray_runner import RayRunner

        return RayRunner()

    def _create_slurm_runner(self, launcher_cfg: dict[str, Any]) -> CloudRunner:
        """Create SlurmRunner instance.

        Parameters
        ----------
        launcher_cfg
            Launcher configuration with SLURM-specific settings.

        Returns
        -------
        CloudRunner
            Configured SlurmRunner.
        """
        from tmgg.experiment_utils.cloud.slurm_runner import SlurmConfig, SlurmRunner

        slurm_config = SlurmConfig(
            partition=launcher_cfg.get("slurm_partition", "gpu"),
            nodes=launcher_cfg.get("slurm_nodes", 1),
            cpus_per_task=launcher_cfg.get("slurm_cpus_per_task", 4),
            gpus_per_task=launcher_cfg.get("slurm_gpus_per_task", 1),
            time_limit=launcher_cfg.get("slurm_time_limit", "04:00:00"),
            mem_per_cpu=launcher_cfg.get("slurm_mem_per_cpu", "4GB"),
            setup_commands=launcher_cfg.get("slurm_setup_commands", []),
        )
        return SlurmRunner(slurm_config=slurm_config)
