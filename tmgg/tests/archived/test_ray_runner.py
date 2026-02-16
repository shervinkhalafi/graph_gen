"""Tests for RayRunner distributed execution.

Test rationale:
    RayRunner provides local distributed execution using Ray for parallel
    experiment runs. Since ray is an optional dependency, these tests mock
    the ray module to verify the runner's orchestration logic without
    requiring ray to be installed.

    Invariants:
    - RayRunner raises ImportError if ray is not installed
    - spawn_experiment returns RaySpawnedTask with valid run_id
    - run_experiment blocks and returns ExperimentResult
    - get_status checks ray object readiness
    - cancel attempts to stop ray tasks
"""

# pyright: reportAttributeAccessIssue=false
# The ray_runner module is loaded dynamically via importlib and we need to
# reset internal state (_ray_execute_task_fn) between tests

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


# Create comprehensive ray mock before importing the module
def create_ray_mock():
    """Create a mock ray module with all required functionality."""
    mock_ray = MagicMock()

    # Mock is_initialized
    mock_ray.is_initialized.return_value = False

    # Mock init
    mock_ray.init.return_value = None

    # Mock get - returns the result
    mock_ray.get.return_value = {
        "run_id": "test123",
        "status": "completed",
        "metrics": {"best_val_loss": 0.1},
        "checkpoint_uri": None,
        "error_message": None,
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T01:00:00",
        "duration_seconds": 3600.0,
    }

    # Mock wait - returns ([ref], [])
    mock_ray.wait.return_value = ([MagicMock()], [])

    # Mock cancel
    mock_ray.cancel.return_value = None

    # Mock shutdown
    mock_ray.shutdown.return_value = None

    # Mock ObjectRef
    mock_ray.ObjectRef = MagicMock

    # Mock remote decorator
    def mock_remote(func=None, **kwargs):
        if func is None:
            return lambda f: mock_remote(f)
        wrapped = MagicMock(wraps=func)
        wrapped.options = MagicMock(return_value=wrapped)
        wrapped.remote = MagicMock(return_value=MagicMock())
        return wrapped

    mock_ray.remote = mock_remote

    return mock_ray


# Install mock before importing
mock_ray = create_ray_mock()
sys.modules["ray"] = mock_ray

# Use importlib to load modules directly
_base_path = Path(__file__).parent.parent.parent.parent / "src" / "tmgg"
_cloud_path = _base_path / "experiment_utils" / "cloud"
_experiment_utils_path = _base_path / "experiment_utils"


def _load_module(name: str, path: Path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load dependencies first
_base_module = _load_module("tmgg.experiment_utils.cloud.base", _cloud_path / "base.py")
_task_module = _load_module(
    "tmgg.experiment_utils.task", _experiment_utils_path / "task.py"
)

# Load ray_runner
_ray_runner_module = _load_module(
    "tmgg.experiment_utils.cloud.ray_runner", _cloud_path / "ray_runner.py"
)

# Extract classes
RayRunner = _ray_runner_module.RayRunner
RaySpawnedTask = _ray_runner_module.RaySpawnedTask
ExperimentResult = _base_module.ExperimentResult
TaskInput = _task_module.TaskInput
TaskOutput = _task_module.TaskOutput


class TestRayRunnerInit:
    """Tests for RayRunner initialization."""

    def test_init_auto_initializes_ray(self):
        """RayRunner initializes ray cluster if not already connected."""
        mock_ray.is_initialized.return_value = False

        runner = RayRunner(max_concurrent=2)

        assert runner.max_concurrent == 2
        assert runner.num_gpus_per_task == 0.0
        assert runner.num_cpus_per_task == 1

    def test_init_skips_if_ray_connected(self):
        """RayRunner skips init if ray already connected."""
        mock_ray.is_initialized.return_value = True
        mock_ray.init.reset_mock()

        _ = RayRunner(max_concurrent=4)

        # init should not be called
        mock_ray.init.assert_not_called()

    def test_init_with_gpu_config(self):
        """RayRunner accepts GPU configuration."""
        mock_ray.is_initialized.return_value = True

        runner = RayRunner(
            max_concurrent=2,
            num_gpus_per_task=1.0,
            num_cpus_per_task=4,
        )

        assert runner.num_gpus_per_task == 1.0
        assert runner.num_cpus_per_task == 4


class TestRaySpawnedTask:
    """Tests for RaySpawnedTask dataclass."""

    def test_spawned_task_creation(self):
        """RaySpawnedTask stores run_id and object_ref."""
        mock_ref = MagicMock()
        task = RaySpawnedTask(run_id="test123", object_ref=mock_ref)

        assert task.run_id == "test123"
        assert task.object_ref == mock_ref


class TestRayRunnerSpawn:
    """Tests for spawn methods."""

    @pytest.fixture
    def runner(self):
        """Create RayRunner for testing."""
        mock_ray.is_initialized.return_value = True
        return RayRunner(max_concurrent=4)

    @pytest.fixture
    def sample_config(self):
        """Create a sample OmegaConf configuration."""
        return OmegaConf.create(
            {
                "run_id": "spawn_test",
                "model": {"hidden_dim": 64},
                "data": {"batch_size": 32},
                "trainer": {"max_epochs": 10},
                "paths": {"output_dir": None, "results_dir": None},
            }
        )

    def test_spawn_experiment_returns_spawned_task(self, runner, sample_config):
        """spawn_experiment returns RaySpawnedTask with valid run_id."""
        # Reset the ray_execute_task function (use setattr for dynamic module access)
        _ray_runner_module._ray_execute_task_fn = None

        spawned = runner.spawn_experiment(sample_config)

        assert isinstance(spawned, RaySpawnedTask)
        # prepare_config_for_remote generates a new UUID-based run_id
        assert len(spawned.run_id) == 8  # UUID[:8] format
        assert spawned.object_ref is not None

    def test_spawn_experiment_tracks_active_runs(self, runner, sample_config):
        """spawn_experiment adds to active runs tracking."""
        _ray_runner_module._ray_execute_task_fn = None

        spawned = runner.spawn_experiment(sample_config)

        assert spawned.run_id in runner._active_runs
        assert runner._active_runs[spawned.run_id] == spawned

    def test_spawn_sweep_spawns_all_configs(self, runner):
        """spawn_sweep spawns all provided configurations."""
        _ray_runner_module._ray_execute_task_fn = None

        configs = [
            OmegaConf.create(
                {
                    "run_id": f"sweep_{i}",
                    "model": {"hidden_dim": 64},
                    "paths": {"output_dir": None, "results_dir": None},
                }
            )
            for i in range(3)
        ]

        spawned_tasks = runner.spawn_sweep(configs)

        assert len(spawned_tasks) == 3
        assert all(isinstance(t, RaySpawnedTask) for t in spawned_tasks)
        # Each task gets a unique run_id (UUID[:8] format)
        run_ids = [t.run_id for t in spawned_tasks]
        assert len(set(run_ids)) == 3  # All unique
        assert all(len(rid) == 8 for rid in run_ids)


class TestRayRunnerBlocking:
    """Tests for blocking execution methods."""

    @pytest.fixture
    def runner(self):
        """Create RayRunner for testing."""
        mock_ray.is_initialized.return_value = True
        return RayRunner(max_concurrent=4)

    @pytest.fixture
    def sample_config(self):
        """Create a sample OmegaConf configuration."""
        return OmegaConf.create(
            {
                "run_id": "blocking_test",
                "model": {"hidden_dim": 64},
                "paths": {"output_dir": None, "results_dir": None},
            }
        )

    def test_run_experiment_returns_result(self, runner, sample_config):
        """run_experiment returns ExperimentResult after waiting."""
        _ray_runner_module._ray_execute_task_fn = None

        result = runner.run_experiment(sample_config)

        assert isinstance(result, ExperimentResult)
        # run_id is generated by prepare_config_for_remote (UUID[:8] format)
        assert len(result.run_id) == 8
        assert result.status == "completed"

    def test_run_experiment_cleans_up_tracking(self, runner, sample_config):
        """run_experiment removes from active runs after completion."""
        _ray_runner_module._ray_execute_task_fn = None

        result = runner.run_experiment(sample_config)

        # Should not be in active runs anymore
        assert result.run_id not in runner._active_runs

    def test_run_sweep_returns_all_results(self, runner):
        """run_sweep returns results for all configs."""
        _ray_runner_module._ray_execute_task_fn = None

        # Mock ray.get to return a list of results
        mock_ray.get.return_value = [
            {
                "run_id": f"sweep_{i}",
                "status": "completed",
                "metrics": {"best_val_loss": 0.1 * i},
                "checkpoint_uri": None,
                "error_message": None,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T01:00:00",
                "duration_seconds": 3600.0,
            }
            for i in range(3)
        ]

        configs = [
            OmegaConf.create(
                {
                    "run_id": f"sweep_{i}",
                    "model": {"hidden_dim": 64},
                    "paths": {"output_dir": None, "results_dir": None},
                }
            )
            for i in range(3)
        ]

        results = runner.run_sweep(configs)

        assert len(results) == 3
        assert all(isinstance(r, ExperimentResult) for r in results)


class TestRayRunnerStatus:
    """Tests for status and control methods."""

    @pytest.fixture
    def runner(self):
        """Create RayRunner for testing."""
        mock_ray.is_initialized.return_value = True
        return RayRunner(max_concurrent=4)

    def test_get_status_unknown_for_missing_run(self, runner):
        """get_status returns 'unknown' for untracked runs."""
        status = runner.get_status("nonexistent")

        assert status == "unknown"

    def test_get_status_running_if_not_ready(self, runner):
        """get_status returns 'running' if result not ready."""
        mock_ref = MagicMock()
        runner._active_runs["test123"] = RaySpawnedTask(
            run_id="test123", object_ref=mock_ref
        )
        mock_ray.wait.return_value = ([], [mock_ref])  # Not ready

        status = runner.get_status("test123")

        assert status == "running"

    def test_get_status_completed_if_ready(self, runner):
        """get_status returns 'completed' if result is ready."""
        mock_ref = MagicMock()
        runner._active_runs["test123"] = RaySpawnedTask(
            run_id="test123", object_ref=mock_ref
        )
        mock_ray.wait.return_value = ([mock_ref], [])  # Ready
        mock_ray.get.return_value = {"status": "completed"}

        status = runner.get_status("test123")

        assert status == "completed"

    def test_cancel_returns_false_for_missing_run(self, runner):
        """cancel returns False for untracked runs."""
        result = runner.cancel("nonexistent")

        assert result is False

    def test_cancel_removes_from_tracking(self, runner):
        """cancel removes run from active tracking."""
        mock_ref = MagicMock()
        runner._active_runs["test123"] = RaySpawnedTask(
            run_id="test123", object_ref=mock_ref
        )

        result = runner.cancel("test123")

        assert result is True
        assert "test123" not in runner._active_runs

    def test_shutdown_calls_ray_shutdown(self, runner):
        """shutdown calls ray.shutdown if initialized."""
        mock_ray.is_initialized.return_value = True
        mock_ray.shutdown.reset_mock()

        runner.shutdown()

        mock_ray.shutdown.assert_called_once()


class TestRayNotInstalled:
    """Tests for behavior when ray is not installed."""

    def test_ensure_ray_available_raises_if_not_installed(self):
        """_ensure_ray_available raises ImportError if ray not installed."""
        # Temporarily remove ray from sys.modules
        original = sys.modules.get("ray")

        with patch.dict(sys.modules, {"ray": None}):
            # Reload the function to get fresh import check
            del sys.modules["ray"]

            with pytest.raises(ImportError, match="Ray is not installed"):
                _ray_runner_module._ensure_ray_available()

        # Restore
        if original:
            sys.modules["ray"] = original
