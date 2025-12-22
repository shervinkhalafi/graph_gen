"""Tests for the custom Hydra launcher plugin.

Test rationale:
    The TmggLauncher is a custom Hydra launcher plugin that dispatches jobs to
    CloudRunner backends (LocalRunner, ModalRunner, RayRunner). These tests verify
    that the launcher correctly:
    - Initializes with the appropriate runner based on configuration
    - Executes single and multi-run jobs via CloudRunner
    - Converts ExperimentResult to Hydra's JobReturn format
    - Handles job failures appropriately

Assumptions:
    - Tests use mocked CloudRunner implementations to avoid actual experiment execution
    - The launcher receives Hydra-style config and job_overrides
    - All tests run without actual GPU or cloud resources

Invariants:
    - setup() must be called before launch()
    - LocalRunner is used when use_modal=False, use_ray=False, use_slurm=False
    - ModalRunner is used when use_modal=True
    - RayRunner is used when use_ray=True
    - SlurmRunner is used when use_slurm=True
    - Each job override sequence produces one ExperimentResult and one JobReturn
"""

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hydra.core.utils import JobReturn, JobStatus
from omegaconf import OmegaConf

from tmgg.experiment_utils.cloud.base import ExperimentResult


class TestTmggLauncherRunnerSelection:
    """Tests for runner selection based on configuration."""

    def test_selects_local_runner_by_default(self) -> None:
        """LocalRunner is selected when no cloud backend is specified."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {"use_modal": False, "use_ray": False}},
            }
        )

        with patch.object(launcher, "_create_local_runner") as mock_local:
            mock_local.return_value = MagicMock()
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        mock_local.assert_called_once()

    def test_selects_modal_runner_when_configured(self) -> None:
        """ModalRunner is selected when use_modal=True."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {
                    "launcher": {
                        "use_modal": True,
                        "use_ray": False,
                        "gpu_type": "a10g",
                    }
                },
            }
        )

        with patch.object(launcher, "_create_modal_runner") as mock_modal:
            mock_modal.return_value = MagicMock()
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        mock_modal.assert_called_once_with(gpu_type="a10g")

    def test_selects_ray_runner_when_configured(self) -> None:
        """RayRunner is selected when use_ray=True."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {
                    "launcher": {
                        "use_modal": False,
                        "use_ray": True,
                    }
                },
            }
        )

        with patch.object(launcher, "_create_ray_runner") as mock_ray:
            mock_ray.return_value = MagicMock()
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        mock_ray.assert_called_once()

    def test_selects_slurm_runner_when_configured(self) -> None:
        """SlurmRunner is selected when use_slurm=True."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {
                    "launcher": {
                        "use_modal": False,
                        "use_ray": False,
                        "use_slurm": True,
                        "slurm_partition": "gpu",
                        "slurm_nodes": 4,
                        "slurm_cpus_per_task": 8,
                        "slurm_gpus_per_task": 1,
                        "slurm_time_limit": "08:00:00",
                        "slurm_mem_per_cpu": "4GB",
                        "slurm_setup_commands": [],
                    }
                },
            }
        )

        with patch.object(launcher, "_create_slurm_runner") as mock_slurm:
            mock_slurm.return_value = MagicMock()
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        # Verify _create_slurm_runner was called with the launcher config dict
        mock_slurm.assert_called_once()
        call_args = mock_slurm.call_args[0][0]
        assert call_args["slurm_partition"] == "gpu"
        assert call_args["slurm_nodes"] == 4
        assert call_args["slurm_cpus_per_task"] == 8
        assert call_args["slurm_gpus_per_task"] == 1
        assert call_args["slurm_time_limit"] == "08:00:00"


class TestTmggLauncherJobExecution:
    """Tests for job execution via the launcher."""

    @pytest.fixture
    def mock_runner(self) -> MagicMock:
        """Create a mock CloudRunner that returns successful results."""
        runner = MagicMock()
        runner.run_experiment.return_value = ExperimentResult(
            run_id="test-123",
            config={"model": "test"},
            metrics={"val_loss": 0.5},
            status="completed",
            duration_seconds=10.0,
        )
        return runner

    @pytest.fixture
    def configured_launcher(self, mock_runner: MagicMock) -> Any:
        """Create a TmggLauncher with mocked runner."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {"use_modal": False, "use_ray": False}},
                "model": "baseline",
            }
        )

        with patch.object(launcher, "_create_local_runner", return_value=mock_runner):
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        return launcher

    def test_launch_single_job(
        self, configured_launcher: Any, mock_runner: MagicMock
    ) -> None:
        """Single job execution produces one JobReturn."""
        job_overrides: Sequence[Sequence[str]] = [["model=test_model"]]

        results = configured_launcher.launch(job_overrides, initial_job_idx=0)

        assert len(results) == 1
        assert isinstance(results[0], JobReturn)
        mock_runner.run_experiment.assert_called_once()

    def test_launch_multirun_jobs(
        self, configured_launcher: Any, mock_runner: MagicMock
    ) -> None:
        """Multiple job overrides produce multiple JobReturns."""
        job_overrides: Sequence[Sequence[str]] = [
            ["model=model_a", "lr=0.01"],
            ["model=model_b", "lr=0.01"],
            ["model=model_a", "lr=0.001"],
        ]

        results = configured_launcher.launch(job_overrides, initial_job_idx=0)

        assert len(results) == 3
        assert all(isinstance(r, JobReturn) for r in results)
        assert mock_runner.run_experiment.call_count == 3

    def test_job_idx_preserved(
        self, configured_launcher: Any, mock_runner: MagicMock
    ) -> None:
        """Job index is correctly assigned starting from initial_job_idx."""
        job_overrides: Sequence[Sequence[str]] = [["model=a"], ["model=b"]]

        results = configured_launcher.launch(job_overrides, initial_job_idx=5)

        # Job indices are stored in return_value as _job_idx
        assert results[0].return_value["_job_idx"] == 5
        assert results[1].return_value["_job_idx"] == 6


class TestJobReturnConversion:
    """Tests for ExperimentResult to JobReturn conversion."""

    def test_successful_result_converts_to_completed_status(self) -> None:
        """Completed ExperimentResult maps to JobStatus.COMPLETED."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        experiment_result = ExperimentResult(
            run_id="abc123",
            config={"model": "test"},
            metrics={"val_loss": 0.25},
            status="completed",
            duration_seconds=60.0,
        )

        job_return = TmggLauncher._result_to_job_return(
            experiment_result,
            overrides=["model=test"],
            idx=0,
        )

        assert job_return.status == JobStatus.COMPLETED
        assert job_return.return_value["val_loss"] == 0.25
        assert job_return.return_value["_job_idx"] == 0

    def test_failed_result_converts_to_failed_status(self) -> None:
        """Failed ExperimentResult maps to JobStatus.FAILED."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        experiment_result = ExperimentResult(
            run_id="abc123",
            config={"model": "test"},
            metrics={},
            status="failed",
            error_message="CUDA out of memory",
            duration_seconds=5.0,
        )

        job_return = TmggLauncher._result_to_job_return(
            experiment_result,
            overrides=["model=test"],
            idx=0,
        )

        assert job_return.status == JobStatus.FAILED
        # For failed jobs, access _return_value directly (return_value raises)
        assert job_return._return_value["_job_idx"] == 0


class TestLauncherErrorHandling:
    """Tests for error handling during job execution."""

    def test_runner_exception_raises(self) -> None:
        """Runner exception crashes the launcher (fail-fast policy)."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {"use_modal": False, "use_ray": False}},
                "model": "baseline",
            }
        )

        failing_runner = MagicMock()
        failing_runner.run_experiment.side_effect = RuntimeError("Experiment failed")

        with patch.object(
            launcher, "_create_local_runner", return_value=failing_runner
        ):
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        # Per project conventions: no graceful fallback, crash loudly
        with pytest.raises(RuntimeError, match="Experiment failed"):
            launcher.launch([["model=test"]], initial_job_idx=0)


class TestLauncherConfiguration:
    """Tests for launcher configuration via Hydra ConfigStore."""

    def test_launcher_config_dataclass_exists(self) -> None:
        """TmggLauncherConf dataclass is properly defined."""
        from tmgg.hydra_plugins.tmgg_launcher.config import TmggLauncherConf

        # Verify required attributes exist with correct defaults
        config = TmggLauncherConf()
        assert config.use_modal is False
        assert config.use_ray is False
        assert config.use_slurm is False
        assert config.gpu_type == "debug"
        assert config.parallelism == 4
        assert config.timeout_seconds == 3600

        # SLURM-specific defaults
        assert config.slurm_partition == "gpu"
        assert config.slurm_nodes == 1
        assert config.slurm_cpus_per_task == 4
        assert config.slurm_gpus_per_task == 1
        assert config.slurm_time_limit == "04:00:00"
        assert config.slurm_mem_per_cpu == "4GB"
        assert config.slurm_setup_commands == []

    def test_launcher_registered_in_config_store(self) -> None:
        """Launcher config is registered in Hydra's ConfigStore."""
        from hydra.core.config_store import ConfigStore

        # Import the module to trigger registration
        import tmgg.hydra_plugins.tmgg_launcher  # noqa: F401

        cs = ConfigStore.instance()

        # The config should be registered under hydra/launcher group
        configs = cs.list("hydra/launcher")
        assert "tmgg.yaml" in configs
