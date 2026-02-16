"""Tests for the SlurmRunner cloud backend.

Test rationale:
    SlurmRunner generates sbatch scripts and submits jobs to SLURM clusters.
    Since SLURM is not available in the test environment, we mock all subprocess
    calls (sbatch, squeue, sacct, scancel) to verify correct behavior.

Assumptions:
    - All SLURM commands are mocked via unittest.mock.patch
    - Tests verify script generation, job submission, status checking, and result parsing
    - No actual SLURM cluster or jobs are created

Invariants:
    - Generated sbatch scripts contain correct #SBATCH directives
    - ray symmetric-run is used for cluster initialization
    - TMGG_RESULT: JSON line is used for result parsing
    - SLURM job states map correctly to CloudRunner status strings
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from tmgg.experiment_utils.cloud.slurm_runner import (
    SlurmConfig,
    SlurmRunner,
    SlurmSpawnedTask,
)


class TestSlurmScriptGeneration:
    """Tests for sbatch script generation."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> SlurmRunner:
        """Create a SlurmRunner with test configuration."""
        config = SlurmConfig(
            partition="test-gpu",
            nodes=2,
            cpus_per_task=4,
            gpus_per_task=1,
            time_limit="02:00:00",
            mem_per_cpu="4GB",
        )
        return SlurmRunner(slurm_config=config, output_dir=tmp_path)

    @pytest.fixture
    def sample_config(self) -> Any:
        """Create a sample experiment config."""
        return OmegaConf.create(
            {
                "run_id": "test-run-123",
                "model": {"hidden_dim": 32},
                "trainer": {"max_steps": 100},
            }
        )

    def test_generates_valid_sbatch_directives(
        self, runner: SlurmRunner, sample_config: Any
    ) -> None:
        """Script contains correct #SBATCH lines matching config."""
        from tmgg.experiment_utils.task import TaskInput

        task_input = TaskInput(
            config={"run_id": "test-123", "model": {}},
            run_id="test-123",
            gpu_tier="slurm",
            timeout_seconds=3600,
        )

        script = runner._generate_sbatch_script(task_input)

        # Verify SBATCH directives
        assert "#SBATCH --partition=test-gpu" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --gpus-per-task=1" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --mem-per-cpu=4GB" in script
        assert "#SBATCH --exclusive" in script
        assert "#SBATCH --tasks-per-node=1" in script

    def test_includes_ray_symmetric_run_command(
        self, runner: SlurmRunner, sample_config: Any
    ) -> None:
        """Script uses ray symmetric-run with correct arguments."""
        from tmgg.experiment_utils.task import TaskInput

        task_input = TaskInput(
            config={"run_id": "test-123", "model": {}},
            run_id="test-123",
            gpu_tier="slurm",
            timeout_seconds=3600,
        )

        script = runner._generate_sbatch_script(task_input)

        assert "ray symmetric-run" in script
        assert '--address "$ip_head"' in script
        assert '--min-nodes "$SLURM_JOB_NUM_NODES"' in script
        assert '--num-cpus="${SLURM_CPUS_PER_TASK}"' in script
        assert '--num-gpus="${SLURM_GPUS_PER_TASK}"' in script

    def test_includes_setup_commands(self, tmp_path: Path) -> None:
        """Environment setup commands are included in script."""
        config = SlurmConfig(
            partition="gpu",
            setup_commands=[
                "module load cuda/12.1",
                "source ~/.bashrc && conda activate tmgg",
            ],
        )
        runner = SlurmRunner(slurm_config=config, output_dir=tmp_path)

        from tmgg.experiment_utils.task import TaskInput

        task_input = TaskInput(
            config={"run_id": "test-123"},
            run_id="test-123",
            gpu_tier="slurm",
            timeout_seconds=3600,
        )

        script = runner._generate_sbatch_script(task_input)

        assert "module load cuda/12.1" in script
        assert "source ~/.bashrc && conda activate tmgg" in script

    def test_includes_execute_task_python_code(
        self, runner: SlurmRunner, sample_config: Any
    ) -> None:
        """Script contains Python code to execute task and print result."""
        from tmgg.experiment_utils.task import TaskInput

        task_input = TaskInput(
            config={"run_id": "test-123"},
            run_id="test-123",
            gpu_tier="slurm",
            timeout_seconds=3600,
        )

        script = runner._generate_sbatch_script(task_input)

        assert (
            "from tmgg.experiment_utils.task import TaskInput, execute_task" in script
        )
        assert "TMGG_RESULT:" in script
        assert "json.dumps" in script

    def test_script_output_paths_use_run_id(
        self, runner: SlurmRunner, tmp_path: Path
    ) -> None:
        """Script output/error paths include run_id."""
        from tmgg.experiment_utils.task import TaskInput

        task_input = TaskInput(
            config={"run_id": "unique-run-456"},
            run_id="unique-run-456",
            gpu_tier="slurm",
            timeout_seconds=3600,
        )

        script = runner._generate_sbatch_script(task_input)

        assert f"{tmp_path}/unique-run-456.out" in script
        assert f"{tmp_path}/unique-run-456.err" in script


class TestSlurmSubmission:
    """Tests for job submission with mocked subprocess."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> SlurmRunner:
        """Create a SlurmRunner with test configuration."""
        return SlurmRunner(output_dir=tmp_path)

    @pytest.fixture
    def sample_config(self) -> Any:
        """Create a sample experiment config."""
        return OmegaConf.create({"run_id": "test-run", "model": {}})

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_spawn_calls_sbatch(
        self, mock_run: MagicMock, runner: SlurmRunner, sample_config: Any
    ) -> None:
        """spawn_experiment() calls sbatch with script path."""
        mock_run.return_value = MagicMock(stdout="Submitted batch job 12345")

        spawned = runner.spawn_experiment(sample_config)

        # Verify sbatch was called
        calls = [c for c in mock_run.call_args_list if "sbatch" in str(c)]
        assert len(calls) == 1
        assert spawned.job_id == "12345"

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_parses_job_id_from_sbatch_output(
        self, mock_run: MagicMock, runner: SlurmRunner, sample_config: Any
    ) -> None:
        """Job ID is correctly extracted from sbatch output."""
        mock_run.return_value = MagicMock(stdout="Submitted batch job 98765\n")

        spawned = runner.spawn_experiment(sample_config)

        assert spawned.job_id == "98765"
        assert isinstance(spawned, SlurmSpawnedTask)

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_creates_sbatch_script_file(
        self,
        mock_run: MagicMock,
        runner: SlurmRunner,
        sample_config: Any,
        tmp_path: Path,
    ) -> None:
        """spawn_experiment() writes sbatch script to output directory."""
        mock_run.return_value = MagicMock(stdout="Submitted batch job 11111")

        spawned = runner.spawn_experiment(sample_config)

        script_path = tmp_path / f"{spawned.run_id}.sbatch"
        assert script_path.exists()
        content = script_path.read_text()
        assert "#!/bin/bash" in content
        assert "#SBATCH" in content


class TestSlurmStatus:
    """Tests for status checking with mocked squeue/sacct."""

    @pytest.fixture
    def runner_with_job(self, tmp_path: Path) -> tuple[SlurmRunner, str]:
        """Create a runner with a tracked job."""
        runner = SlurmRunner(output_dir=tmp_path)
        # Manually add a tracked job
        run_id = "tracked-run-123"
        runner._active_jobs[run_id] = SlurmSpawnedTask(
            run_id=run_id,
            job_id="54321",
        )
        return runner, run_id

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_pending_status(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """PENDING state maps to 'pending'."""
        runner, run_id = runner_with_job
        mock_run.return_value = MagicMock(stdout="PENDING", returncode=0)

        status = runner.get_status(run_id)

        assert status == "pending"

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_running_status(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """RUNNING state maps to 'running'."""
        runner, run_id = runner_with_job
        mock_run.return_value = MagicMock(stdout="RUNNING", returncode=0)

        status = runner.get_status(run_id)

        assert status == "running"

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_completed_via_squeue(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """COMPLETED in squeue maps to 'completed'."""
        runner, run_id = runner_with_job
        mock_run.return_value = MagicMock(stdout="COMPLETED", returncode=0)

        status = runner.get_status(run_id)

        assert status == "completed"

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_completed_via_sacct(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """Job not in squeue but COMPLETED in sacct maps to 'completed'."""
        runner, run_id = runner_with_job

        # First call (squeue) returns empty, second call (sacct) returns COMPLETED
        mock_run.side_effect = [
            MagicMock(stdout="", returncode=0),  # squeue
            MagicMock(stdout="COMPLETED", returncode=0),  # sacct
        ]

        status = runner.get_status(run_id)

        assert status == "completed"

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_failed_status(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """FAILED state maps to 'failed'."""
        runner, run_id = runner_with_job
        mock_run.return_value = MagicMock(stdout="FAILED", returncode=0)

        status = runner.get_status(run_id)

        assert status == "failed"

    def test_unknown_for_untracked_job(self, tmp_path: Path) -> None:
        """Returns 'unknown' for jobs not in _active_jobs."""
        runner = SlurmRunner(output_dir=tmp_path)

        status = runner.get_status("nonexistent-run")

        assert status == "unknown"


class TestSlurmResultParsing:
    """Tests for output file parsing."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> SlurmRunner:
        """Create a SlurmRunner with test configuration."""
        return SlurmRunner(output_dir=tmp_path)

    def test_extracts_tmgg_result_from_output(
        self, runner: SlurmRunner, tmp_path: Path
    ) -> None:
        """TMGG_RESULT: JSON line is parsed correctly."""
        run_id = "parse-test-123"
        spawned = SlurmSpawnedTask(run_id=run_id, job_id="999")

        # Write output file with result
        output_file = tmp_path / f"{run_id}.out"
        result_json = '{"metrics": {"val_loss": 0.25}, "status": "completed", "duration_seconds": 120.5}'
        output_file.write_text(f"Some logs\nTMGG_RESULT:{result_json}\nMore logs\n")

        config = OmegaConf.create({"run_id": run_id, "model": {}})
        result = runner._parse_result(spawned, config, duration=100.0)

        assert result.status == "completed"
        assert result.metrics == {"val_loss": 0.25}
        assert result.duration_seconds == 120.5

    def test_returns_failed_for_missing_output(
        self, runner: SlurmRunner, tmp_path: Path
    ) -> None:
        """Returns failed result when output file doesn't exist."""
        run_id = "missing-output"
        spawned = SlurmSpawnedTask(run_id=run_id, job_id="888")

        config = OmegaConf.create({"run_id": run_id})
        result = runner._parse_result(spawned, config, duration=50.0)

        assert result.status == "failed"
        assert result.duration_seconds == 50.0

    def test_returns_failed_for_missing_result_line(
        self, runner: SlurmRunner, tmp_path: Path
    ) -> None:
        """Returns failed when output exists but lacks TMGG_RESULT line."""
        run_id = "no-result-line"
        spawned = SlurmSpawnedTask(run_id=run_id, job_id="777")

        output_file = tmp_path / f"{run_id}.out"
        output_file.write_text("Just some logs\nNo result here\n")

        config = OmegaConf.create({"run_id": run_id})
        result = runner._parse_result(spawned, config, duration=30.0)

        assert result.status == "failed"

    def test_includes_stderr_in_error_message(
        self, runner: SlurmRunner, tmp_path: Path
    ) -> None:
        """Error file content is included in error message for failed jobs."""
        run_id = "with-stderr"
        spawned = SlurmSpawnedTask(run_id=run_id, job_id="666")

        # Write empty output (no result) and error file
        (tmp_path / f"{run_id}.out").write_text("")
        (tmp_path / f"{run_id}.err").write_text("CUDA error: out of memory\n")

        config = OmegaConf.create({"run_id": run_id})
        result = runner._parse_result(spawned, config, duration=10.0)

        assert result.status == "failed"
        assert "CUDA error: out of memory" in (result.error_message or "")

    def test_handles_json_parse_error(
        self, runner: SlurmRunner, tmp_path: Path
    ) -> None:
        """Gracefully handles malformed JSON in result line."""
        run_id = "bad-json"
        spawned = SlurmSpawnedTask(run_id=run_id, job_id="555")

        output_file = tmp_path / f"{run_id}.out"
        output_file.write_text("TMGG_RESULT:{not valid json}\n")

        config = OmegaConf.create({"run_id": run_id})
        result = runner._parse_result(spawned, config, duration=5.0)

        assert result.status == "failed"
        assert "Failed to parse result JSON" in (result.error_message or "")


class TestSlurmCancel:
    """Tests for job cancellation."""

    @pytest.fixture
    def runner_with_job(self, tmp_path: Path) -> tuple[SlurmRunner, str]:
        """Create a runner with a tracked job."""
        runner = SlurmRunner(output_dir=tmp_path)
        run_id = "cancel-test-run"
        runner._active_jobs[run_id] = SlurmSpawnedTask(
            run_id=run_id,
            job_id="12345",
        )
        return runner, run_id

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_cancel_calls_scancel(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """cancel() calls scancel with job ID."""
        runner, run_id = runner_with_job

        result = runner.cancel(run_id)

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["scancel", "12345"]

    @patch("tmgg.experiment_utils.cloud.slurm_runner.subprocess.run")
    def test_cancel_removes_from_active_jobs(
        self, mock_run: MagicMock, runner_with_job: tuple[SlurmRunner, str]
    ) -> None:
        """Cancelled job is removed from _active_jobs."""
        runner, run_id = runner_with_job

        runner.cancel(run_id)

        assert run_id not in runner._active_jobs

    def test_cancel_returns_false_for_unknown_job(self, tmp_path: Path) -> None:
        """cancel() returns False for jobs not in _active_jobs."""
        runner = SlurmRunner(output_dir=tmp_path)

        result = runner.cancel("unknown-run")

        assert result is False


class TestSlurmRunnerProperties:
    """Tests for SlurmRunner properties and configuration."""

    def test_supports_spawn(self, tmp_path: Path) -> None:
        """SlurmRunner reports spawn support."""
        runner = SlurmRunner(output_dir=tmp_path)

        assert runner.supports_spawn is True

    def test_default_config(self, tmp_path: Path) -> None:
        """Default SlurmConfig values are sensible."""
        runner = SlurmRunner(output_dir=tmp_path)

        assert runner.slurm_config.partition == "gpu"
        assert runner.slurm_config.nodes == 1
        assert runner.slurm_config.cpus_per_task == 4
        assert runner.slurm_config.gpus_per_task == 1
        assert runner.slurm_config.time_limit == "04:00:00"

    def test_custom_config(self, tmp_path: Path) -> None:
        """Custom SlurmConfig is applied correctly."""
        config = SlurmConfig(
            partition="research",
            nodes=8,
            cpus_per_task=16,
            gpus_per_task=4,
            time_limit="24:00:00",
        )
        runner = SlurmRunner(slurm_config=config, output_dir=tmp_path)

        assert runner.slurm_config.partition == "research"
        assert runner.slurm_config.nodes == 8
        assert runner.slurm_config.gpus_per_task == 4
