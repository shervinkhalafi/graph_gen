"""Tests for Modal runner and result status tracking.

Test rationale:
    The Modal runner orchestrates experiment execution on cloud GPUs.
    Result status tracking enables intelligent resume behavior for long sweeps.

    These components are critical because:
    1. GPU time is expensive (Stage 2 = 166.5 hours)
    2. Incorrect resume logic can skip incomplete experiments or re-run completed ones
    3. Timeout handling prevents silent failures

Invariants:
    - ModalRunner.run_experiment returns ExperimentResult
    - ResultStatus correctly categorizes experiment outcomes
    - filter_configs_by_status respects skip_statuses parameter
"""

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock modal before importing tmgg_modal modules to avoid image creation at import time
# Create a comprehensive mock with exception submodule
mock_modal = MagicMock()
mock_modal.exception = MagicMock()
mock_modal.exception.NotFoundError = type("NotFoundError", (Exception,), {})
sys.modules["modal"] = mock_modal
sys.modules["modal.exception"] = mock_modal.exception

# Use importlib to load modal modules directly without going through tmgg.__init__
# This avoids triggering unrelated import errors in other parts of the package
_modal_base_path = Path(__file__).parent.parent.parent / "src" / "tmgg" / "modal"
_experiment_utils_path = (
    Path(__file__).parent.parent.parent / "src" / "tmgg" / "experiment_utils"
)


def _load_module(name: str, path: Path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the modules we need - order matters due to dependencies

# Load experiment_utils.task first (runner depends on it)
_task_module = _load_module(
    "tmgg.experiment_utils.task", _experiment_utils_path / "task.py"
)

# Load modal modules
_app_module = _load_module("tmgg.modal.app", _modal_base_path / "app.py")
_storage_module = _load_module("tmgg.modal.storage", _modal_base_path / "storage.py")
_volumes_module = _load_module("tmgg.modal.volumes", _modal_base_path / "volumes.py")
_image_module = _load_module("tmgg.modal.image", _modal_base_path / "image.py")
_paths_module = _load_module("tmgg.modal.paths", _modal_base_path / "paths.py")

# Load cloud base for ExperimentResult
_cloud_base_path = _experiment_utils_path / "cloud" / "base.py"
_cloud_base_module = _load_module("tmgg.experiment_utils.cloud.base", _cloud_base_path)

# Now load runner after its dependencies are available
_runner_module = _load_module("tmgg.modal.runner", _modal_base_path / "runner.py")

# Load result_status module
_result_status_module = _load_module(
    "tmgg.modal.result_status", _modal_base_path / "result_status.py"
)

# Extract what we need from loaded modules
GPU_CONFIGS = _app_module.GPU_CONFIGS
DEFAULT_TIMEOUTS = _app_module.DEFAULT_TIMEOUTS
ModalRunner = _runner_module.ModalRunner
ModalNotDeployedError = _runner_module.ModalNotDeployedError
SpawnedTask = _runner_module.SpawnedTask
check_modal_deployment = _runner_module.check_modal_deployment
create_runner = _runner_module.create_runner
modal_execute_task = _runner_module.modal_execute_task
ExperimentResult = _cloud_base_module.ExperimentResult
ResultStatus = _result_status_module.ResultStatus
check_result_status = _result_status_module.check_result_status
filter_configs_by_status = _result_status_module.filter_configs_by_status
summarize_status_map = _result_status_module.summarize_status_map
TaskInput = _task_module.TaskInput
TaskOutput = _task_module.TaskOutput


class TestResultStatus:
    """Tests for ResultStatus enum and check_result_status function."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return MagicMock()

    def test_missing_when_result_not_found(self, mock_storage):
        """Should return MISSING when storage.exists returns False."""
        mock_storage.exists.return_value = False

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.MISSING

    def test_complete_with_valid_metrics(self, mock_storage):
        """Should return COMPLETE when all required metrics present."""
        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.123},
            "completed_at": datetime.now().isoformat(),
        }

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.COMPLETE

    def test_partial_when_metric_missing(self, mock_storage):
        """Should return PARTIAL when required metric is None."""
        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"other_metric": 0.5},  # Missing best_val_loss
        }

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.PARTIAL

    def test_partial_when_metric_is_inf(self, mock_storage):
        """Should return PARTIAL when metric is infinity (failed run)."""
        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": float("inf")},
        }

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.PARTIAL

    def test_stale_when_older_than_threshold(self, mock_storage):
        """Should return STALE when result is older than max_age_hours."""
        mock_storage.exists.return_value = True
        old_time = datetime.now() - timedelta(hours=48)
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.123},
            "completed_at": old_time.isoformat(),
        }

        status = check_result_status(
            mock_storage,
            "run-123",
            max_age_hours=24,  # 24 hour threshold
        )

        assert status == ResultStatus.STALE

    def test_complete_when_within_age_threshold(self, mock_storage):
        """Should return COMPLETE when result is within max_age_hours."""
        mock_storage.exists.return_value = True
        recent_time = datetime.now() - timedelta(hours=12)
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.123},
            "completed_at": recent_time.isoformat(),
        }

        status = check_result_status(
            mock_storage,
            "run-123",
            max_age_hours=24,  # 24 hour threshold
        )

        assert status == ResultStatus.COMPLETE

    def test_custom_required_metrics(self, mock_storage):
        """Should check custom required_metrics list."""
        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.1, "accuracy": 0.95},
        }

        # Missing "f1_score" from custom requirements
        status = check_result_status(
            mock_storage,
            "run-123",
            required_metrics=["best_val_loss", "accuracy", "f1_score"],
        )

        assert status == ResultStatus.PARTIAL


class TestFilterConfigsByStatus:
    """Tests for filter_configs_by_status function."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return MagicMock()

    @pytest.fixture
    def sample_configs(self):
        """Sample experiment configs for testing."""
        return [
            {"run_id": "run-1", "lr": 1e-3},
            {"run_id": "run-2", "lr": 1e-4},
            {"run_id": "run-3", "lr": 1e-5},
        ]

    def test_returns_all_configs_when_storage_is_none(self, sample_configs):
        """When storage is None, all configs should be returned as MISSING."""
        filtered, status_map = filter_configs_by_status(None, sample_configs)

        assert len(filtered) == 3
        assert all(s == ResultStatus.MISSING for s in status_map.values())

    def test_filters_complete_results_by_default(self, mock_storage, sample_configs):
        """By default, should skip COMPLETE results."""

        # run-1 is complete, run-2 and run-3 are missing
        def mock_exists(key):
            return "run-1" in key

        def mock_download(run_id):
            if "run-1" in run_id:
                return {"metrics": {"best_val_loss": 0.1}}
            raise FileNotFoundError()

        mock_storage.exists.side_effect = mock_exists
        mock_storage.download_metrics.side_effect = mock_download

        filtered, status_map = filter_configs_by_status(mock_storage, sample_configs)

        # run-1 should be skipped
        assert len(filtered) == 2
        assert all(cfg["run_id"] != "run-1" for cfg in filtered)
        assert status_map["run-1"] == ResultStatus.COMPLETE
        assert status_map["run-2"] == ResultStatus.MISSING

    def test_custom_skip_statuses(self, mock_storage, sample_configs):
        """Should respect custom skip_statuses parameter."""
        # All are complete
        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {"metrics": {"best_val_loss": 0.1}}

        # Don't skip anything
        filtered, _ = filter_configs_by_status(
            mock_storage, sample_configs, skip_statuses=set()
        )

        assert len(filtered) == 3

    def test_status_map_contains_all_run_ids(self, mock_storage, sample_configs):
        """Status map should contain entries for all configs."""
        mock_storage.exists.return_value = False

        _, status_map = filter_configs_by_status(mock_storage, sample_configs)

        assert set(status_map.keys()) == {"run-1", "run-2", "run-3"}


class TestSummarizeStatusMap:
    """Tests for summarize_status_map function."""

    def test_summarizes_all_statuses(self):
        """Should include counts for all present statuses."""
        status_map = {
            "run-1": ResultStatus.COMPLETE,
            "run-2": ResultStatus.COMPLETE,
            "run-3": ResultStatus.PARTIAL,
            "run-4": ResultStatus.MISSING,
        }

        summary = summarize_status_map(status_map)

        assert "2 complete" in summary
        assert "1 partial" in summary
        assert "1 missing" in summary
        assert "stale" not in summary  # No stale results

    def test_empty_map_returns_no_results(self):
        """Empty status map should return 'no results'."""
        summary = summarize_status_map({})

        assert summary == "no results"


class TestDeploymentCheck:
    """Tests for check_modal_deployment() and ModalNotDeployedError.

    Test rationale:
        Pre-deployment verification prevents cryptic errors when users
        try to run experiments before deploying the Modal app. The check
        should fail fast with clear instructions.
    """

    def test_raises_when_function_not_found(self):
        """Should raise ModalNotDeployedError when function lookup fails."""
        # Configure the mock to raise NotFoundError on hydrate
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = mock_modal.exception.NotFoundError(
            "Function not found"
        )
        mock_modal.Function.from_name.return_value = mock_fn

        with pytest.raises(ModalNotDeployedError) as exc_info:
            check_modal_deployment()

        assert "not deployed" in str(exc_info.value)
        assert "mise run modal-deploy" in str(exc_info.value)

    def test_succeeds_when_function_deployed(self):
        """Should not raise when function is accessible."""
        # Configure the mock to succeed
        mock_fn = MagicMock()
        mock_fn.hydrate.return_value = None  # Success
        mock_modal.Function.from_name.return_value = mock_fn

        # Should not raise
        check_modal_deployment()

        # Verify it tried to look up and hydrate
        mock_modal.Function.from_name.assert_called_with(
            "tmgg-spectral", "run_single_experiment"
        )
        mock_fn.hydrate.assert_called_once()

    def test_error_message_includes_instructions(self):
        """Error message should include deployment command."""
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = mock_modal.exception.NotFoundError("Not found")
        mock_modal.Function.from_name.return_value = mock_fn

        with pytest.raises(ModalNotDeployedError) as exc_info:
            check_modal_deployment()

        error_msg = str(exc_info.value)
        assert "uv run modal deploy" in error_msg
        assert "tmgg-spectral" in error_msg


class TestModalRunner:
    """Tests for ModalRunner experiment execution."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return MagicMock()

    def test_runner_initialization(self, mock_storage):
        """ModalRunner should initialize with gpu_type and storage."""
        runner = ModalRunner(
            gpu_type="fast", storage=mock_storage, skip_deployment_check=True
        )

        assert runner.gpu_type == "fast"
        assert runner.storage is mock_storage

    def test_runner_checks_deployment_by_default(self, mock_storage):
        """ModalRunner should call check_modal_deployment by default."""
        # Configure mock to fail deployment check
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = mock_modal.exception.NotFoundError("Not deployed")
        mock_modal.Function.from_name.return_value = mock_fn

        with pytest.raises(ModalNotDeployedError):
            ModalRunner(gpu_type="standard", storage=mock_storage)

    def test_runner_skips_deployment_check_when_requested(self, mock_storage):
        """ModalRunner should skip deployment check when skip_deployment_check=True."""
        # Configure mock to fail deployment check
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = mock_modal.exception.NotFoundError("Not deployed")
        mock_modal.Function.from_name.return_value = mock_fn

        # Should not raise when skip_deployment_check=True
        runner = ModalRunner(
            gpu_type="standard", storage=mock_storage, skip_deployment_check=True
        )

        assert runner.gpu_type == "standard"

    def test_create_runner_factory(self):
        """create_runner should create ModalRunner with default settings."""
        # Configure mock for successful deployment check
        mock_fn = MagicMock()
        mock_fn.hydrate.return_value = None
        mock_modal.Function.from_name.return_value = mock_fn

        with patch.dict("os.environ", {}, clear=True):
            runner = create_runner(gpu_type="standard")

            assert runner.gpu_type == "standard"
            # Storage should be None when env not configured
            assert runner.storage is None


class TestExperimentResult:
    """Tests for ExperimentResult dataclass structure."""

    def test_result_from_dict(self):
        """ExperimentResult should be constructable from result dict."""
        result_dict = {
            "run_id": "test-123",
            "config": {"lr": 1e-4},
            "metrics": {"best_val_loss": 0.5},
            "checkpoint_path": "s3://bucket/checkpoints/test-123/model.ckpt",
            "status": "completed",
            "duration_seconds": 3600.0,
        }

        result = ExperimentResult(**result_dict)

        assert result.run_id == "test-123"
        assert result.metrics["best_val_loss"] == 0.5
        assert result.status == "completed"

    def test_result_with_error(self):
        """Failed experiments should capture error message."""
        result = ExperimentResult(
            run_id="failed-run",
            config={},
            metrics={},
            status="failed",
            error_message="CUDA out of memory",
            duration_seconds=120.0,
        )

        assert result.status == "failed"
        assert result.error_message is not None and "CUDA" in result.error_message


class TestGPUConfigs:
    """Tests for GPU configuration constants."""

    def test_gpu_configs_defined(self):
        """GPU_CONFIGS should define expected tiers."""
        assert "standard" in GPU_CONFIGS
        assert "fast" in GPU_CONFIGS

    def test_default_timeouts_match_gpu_tiers(self):
        """DEFAULT_TIMEOUTS should have entries for GPU tiers."""
        # All GPU configs should have corresponding timeouts
        for tier in GPU_CONFIGS:
            assert tier in DEFAULT_TIMEOUTS, f"Missing timeout for {tier}"

    def test_timeouts_are_reasonable(self):
        """Timeouts should be in reasonable range (10 min to 24 hours)."""
        for tier, timeout in DEFAULT_TIMEOUTS.items():
            assert timeout >= 600, f"Timeout for {tier} too short"
            assert timeout <= 86400, f"Timeout for {tier} too long"


class TestSpawnedTask:
    """Tests for SpawnedTask dataclass.

    Test rationale:
        SpawnedTask is the handle returned by spawn_experiment() that allows
        tracking experiment status. It must be serializable (for persistence)
        and contain all necessary tracking information.
    """

    def test_spawned_task_creation(self):
        """SpawnedTask should be creatable with required fields."""
        task = SpawnedTask(run_id="test-123", gpu_tier="standard")

        assert task.run_id == "test-123"
        assert task.gpu_tier == "standard"
        assert task.function_call is None  # Optional

    def test_spawned_task_with_function_call(self):
        """SpawnedTask should accept optional function_call handle."""
        mock_call = MagicMock()
        task = SpawnedTask(
            run_id="test-456",
            gpu_tier="fast",
            function_call=mock_call,
        )

        assert task.function_call is mock_call


class TestModalExecuteTask:
    """Tests for the modal_execute_task Modal function.

    Test rationale:
        modal_execute_task wraps the unified execute_task() logic. It must
        correctly convert TaskInput dict -> TaskInput -> execute_task -> TaskOutput dict.
    """

    def test_modal_execute_task_uses_task_abstraction(self):
        """modal_execute_task should use the unified TaskInput/TaskOutput."""
        # The function signature accepts dict[str, Any] and returns dict[str, Any]
        # Verify it's properly decorated as a Modal function
        assert hasattr(modal_execute_task, "spawn")
        assert hasattr(modal_execute_task, "remote")
        assert hasattr(modal_execute_task, "map")


class TestSpawnMethods:
    """Tests for ModalRunner spawn methods.

    Test rationale:
        spawn_experiment() and spawn_sweep() enable fire-and-forget execution.
        They must return SpawnedTask handles with valid run_ids and call
        Modal's spawn() method for detached execution.
    """

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return MagicMock()

    @pytest.fixture
    def runner(self, mock_storage):
        """Create a ModalRunner with deployment check skipped."""
        return ModalRunner(
            gpu_type="standard",
            storage=mock_storage,
            skip_deployment_check=True,
        )

    def test_spawn_experiment_returns_spawned_task(self, runner):
        """spawn_experiment should return SpawnedTask with run_id."""
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64},
                "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
            }
        )

        # Mock the Modal function's spawn method
        mock_fn_call = MagicMock()
        mock_modal.Function.from_name.return_value.spawn.return_value = mock_fn_call

        task = runner.spawn_experiment(config)

        assert isinstance(task, SpawnedTask)
        assert task.run_id is not None
        assert len(task.run_id) == 8  # UUID[:8]
        assert task.gpu_tier == "standard"

    def test_spawn_sweep_returns_list_of_spawned_tasks(self, runner):
        """spawn_sweep should return list of SpawnedTask handles."""
        from omegaconf import OmegaConf

        configs = [
            OmegaConf.create(
                {
                    "model": {"hidden_dim": dim},
                    "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
                }
            )
            for dim in [32, 64, 128]
        ]

        # Mock the Modal function's spawn method
        mock_fn_call = MagicMock()
        mock_modal.Function.from_name.return_value.spawn.return_value = mock_fn_call

        tasks = runner.spawn_sweep(configs)

        assert len(tasks) == 3
        assert all(isinstance(t, SpawnedTask) for t in tasks)
        # Each task should have a unique run_id
        run_ids = [t.run_id for t in tasks]
        assert len(set(run_ids)) == 3

    def test_spawn_experiment_tracks_active_runs(self, runner):
        """spawn_experiment should track spawned tasks in _active_runs."""
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64},
                "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
            }
        )

        mock_fn_call = MagicMock()
        mock_modal.Function.from_name.return_value.spawn.return_value = mock_fn_call

        task = runner.spawn_experiment(config)

        assert task.run_id in runner._active_runs
        assert runner._active_runs[task.run_id] is task
