"""Tests for unified task execution module.

Test Rationale
--------------
These tests verify the task abstraction layer that enables unified execution
across Modal and Ray backends. The key invariants are:

1. TaskInput/TaskOutput must be fully serializable (JSON-compatible via asdict)
2. prepare_config_for_remote() must strip env-dependent interpolations
3. Path resolution must work in both Modal (mounted /data) and local environments
4. Failed experiments must still upload metrics before re-raising
"""

from __future__ import annotations

# Use importlib to load the module directly without going through tmgg.__init__
# This avoids triggering unrelated import errors in other parts of the package
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

_task_path = (
    Path(__file__).parent.parent.parent
    / "src"
    / "tmgg"
    / "experiment_utils"
    / "task.py"
)
_spec = importlib.util.spec_from_file_location("tmgg.experiment_utils.task", _task_path)
assert _spec is not None and _spec.loader is not None
_task_module = importlib.util.module_from_spec(_spec)
sys.modules["tmgg.experiment_utils.task"] = _task_module
_spec.loader.exec_module(_task_module)

TaskInput = _task_module.TaskInput
TaskOutput = _task_module.TaskOutput
_resolve_execution_paths = _task_module._resolve_execution_paths
execute_task = _task_module.execute_task
prepare_config_for_remote = _task_module.prepare_config_for_remote


class TestTaskInputOutput:
    """Tests for TaskInput and TaskOutput dataclasses."""

    def test_task_input_serializable(self) -> None:
        """TaskInput should be JSON-serializable via asdict."""
        task = TaskInput(
            config={"model": {"hidden_dim": 64}, "seed": 42},
            run_id="abc123",
            gpu_tier="standard",
            timeout_seconds=3600,
        )
        serialized = asdict(task)
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)

        assert deserialized["run_id"] == "abc123"
        assert deserialized["config"]["model"]["hidden_dim"] == 64

    def test_task_output_serializable(self) -> None:
        """TaskOutput should be JSON-serializable via asdict."""
        output = TaskOutput(
            run_id="abc123",
            status="completed",
            metrics={"best_val_loss": 0.5},
            checkpoint_uri="s3://bucket/checkpoints/abc123/best.ckpt",
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T01:00:00",
            duration_seconds=3600.0,
        )
        serialized = asdict(output)
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)

        assert deserialized["status"] == "completed"
        assert deserialized["metrics"]["best_val_loss"] == 0.5

    def test_task_input_defaults(self) -> None:
        """TaskInput should have sensible defaults."""
        task = TaskInput(config={}, run_id="test")
        assert task.gpu_tier == "standard"
        assert task.timeout_seconds == 3600

    def test_task_output_defaults(self) -> None:
        """TaskOutput should have sensible defaults for optional fields."""
        output = TaskOutput(run_id="test", status="completed")
        assert output.metrics == {}
        assert output.checkpoint_uri is None
        assert output.error_message is None


class TestPrepareConfigForRemote:
    """Tests for prepare_config_for_remote() function."""

    def test_strips_paths(self) -> None:
        """Config paths should be set to None for worker resolution."""
        config = OmegaConf.create(
            {
                "paths": {
                    "output_dir": "/local/path/to/outputs",
                    "results_dir": "/local/path/to/results",
                },
                "model": {"hidden_dim": 64},
                "seed": 42,
            }
        )
        prepared = prepare_config_for_remote(config, run_id="test123")

        assert prepared["paths"]["output_dir"] is None
        assert prepared["paths"]["results_dir"] is None
        assert prepared["model"]["hidden_dim"] == 64

    def test_removes_logger_config(self) -> None:
        """Logger config should be removed (uses env interpolations)."""
        config = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp/out", "results_dir": "/tmp/res"},
                "logger": [
                    {"tensorboard": {"save_dir": "s3://bucket/tb"}},
                    {"wandb": {"project": "test"}},
                ],
                "model": {"hidden_dim": 64},
            }
        )
        prepared = prepare_config_for_remote(config, run_id="test123")

        assert "logger" not in prepared
        assert prepared["model"]["hidden_dim"] == 64

    def test_sets_run_id(self) -> None:
        """Run ID should be set in the prepared config."""
        config = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
                "model": {"hidden_dim": 64},
            }
        )
        prepared = prepare_config_for_remote(config, run_id="explicit_id")
        assert prepared["run_id"] == "explicit_id"

    def test_generates_run_id_if_not_provided(self) -> None:
        """Run ID should be auto-generated if not provided."""
        config = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
            }
        )
        prepared = prepare_config_for_remote(config)
        assert "run_id" in prepared
        assert len(prepared["run_id"]) == 8  # UUID[:8]

    def test_resolves_simple_interpolations(self) -> None:
        """Non-env interpolations should be resolved."""
        config = OmegaConf.create(
            {
                "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
                "model": {
                    "hidden_dim": 64,
                    "output_dim": "${model.hidden_dim}",
                },
            }
        )
        prepared = prepare_config_for_remote(config, run_id="test")
        assert prepared["model"]["output_dim"] == 64

    def test_creates_paths_if_missing(self) -> None:
        """Should create paths structure if not present in config."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64},
            }
        )
        prepared = prepare_config_for_remote(config, run_id="test")
        assert "paths" in prepared
        assert prepared["paths"]["output_dir"] is None
        assert prepared["paths"]["results_dir"] is None


class TestResolveExecutionPaths:
    """Tests for _resolve_execution_paths() function."""

    def test_uses_tmgg_output_base_env(self) -> None:
        """Should use TMGG_OUTPUT_BASE env var if set."""
        with patch.dict(os.environ, {"TMGG_OUTPUT_BASE": "/custom/output"}):
            output_dir, results_dir = _resolve_execution_paths("run123")

        assert output_dir == "/custom/output/run123"
        assert results_dir == "/custom/output/run123/results"

    def test_modal_default_when_data_exists(self) -> None:
        """Should use /data/outputs when /data exists (Modal container)."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", return_value=True),
        ):
            # Clear TMGG_OUTPUT_BASE to test default
            os.environ.pop("TMGG_OUTPUT_BASE", None)
            output_dir, results_dir = _resolve_execution_paths("run123")

        assert output_dir == "/data/outputs/run123"
        assert results_dir == "/data/outputs/run123/results"

    def test_local_default_when_data_missing(self) -> None:
        """Should use ./outputs when /data doesn't exist (local)."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.path.exists", return_value=False),
        ):
            os.environ.pop("TMGG_OUTPUT_BASE", None)
            output_dir, results_dir = _resolve_execution_paths("run123")

        assert output_dir == "./outputs/run123"
        assert results_dir == "./outputs/run123/results"


class TestExecuteTask:
    """Tests for execute_task() function.

    These tests patch the loaded module directly rather than using string paths,
    since we loaded the module via importlib to avoid package init issues.
    """

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create a mock storage backend."""
        storage = MagicMock()
        storage.upload_checkpoint.return_value = "s3://bucket/ckpt/run123/best.ckpt"
        storage.upload_metrics.return_value = "s3://bucket/metrics/run123.json"
        return storage

    @pytest.fixture
    def minimal_task(self) -> TaskInput:
        """Create a minimal task input for testing."""
        return TaskInput(
            config={
                "paths": {"output_dir": None, "results_dir": None},
                "model": {"hidden_dim": 64},
                "seed": 42,
            },
            run_id="test123",
            gpu_tier="debug",
            timeout_seconds=600,
        )

    def test_successful_execution_uploads_metrics(
        self, minimal_task: TaskInput, mock_storage: MagicMock
    ) -> None:
        """Successful execution should upload metrics to storage."""
        mock_result = {
            "best_val_loss": 0.5,
            "best_model_path": "/data/outputs/test123/checkpoints/best.ckpt",
        }

        # Patch run_experiment on the module (storage is now a parameter)
        original_run_experiment = _task_module.run_experiment
        _task_module.run_experiment = lambda: (lambda cfg: mock_result)  # type: ignore[attr-defined]

        try:
            with patch("pathlib.Path.exists", return_value=True):
                output = execute_task(minimal_task, get_storage=lambda: mock_storage)
        finally:
            _task_module.run_experiment = original_run_experiment  # type: ignore[attr-defined]

        assert output.status == "completed"
        assert output.metrics["best_val_loss"] == 0.5
        mock_storage.upload_metrics.assert_called_once()

    def test_failed_execution_uploads_failure_then_raises(
        self, minimal_task: TaskInput, mock_storage: MagicMock
    ) -> None:
        """Failed execution should upload failure record, then re-raise."""
        original_run_experiment = _task_module.run_experiment

        def failing_run(_cfg: Any) -> None:
            raise RuntimeError("Training crashed")

        _task_module.run_experiment = lambda: failing_run  # type: ignore[attr-defined]

        try:
            with pytest.raises(RuntimeError, match="Training crashed"):
                execute_task(minimal_task, get_storage=lambda: mock_storage)
        finally:
            _task_module.run_experiment = original_run_experiment  # type: ignore[attr-defined]

        # Metrics should still be uploaded with failed status
        mock_storage.upload_metrics.assert_called_once()
        call_args = mock_storage.upload_metrics.call_args
        uploaded_data = call_args[0][0]
        assert uploaded_data["status"] == "failed"
        assert "Training crashed" in uploaded_data["error_message"]

    def test_execution_without_storage(self, minimal_task: TaskInput) -> None:
        """Execution should work without storage configured."""
        mock_result = {"best_val_loss": 0.5, "best_model_path": None}

        original_run_experiment = _task_module.run_experiment
        _task_module.run_experiment = lambda: (lambda cfg: mock_result)  # type: ignore[attr-defined]

        try:
            # Pass None for get_storage (no storage configured)
            output = execute_task(minimal_task, get_storage=None)
        finally:
            _task_module.run_experiment = original_run_experiment  # type: ignore[attr-defined]

        assert output.status == "completed"
        assert output.checkpoint_uri is None

    def test_execution_sets_paths_from_environment(
        self, minimal_task: TaskInput, mock_storage: MagicMock
    ) -> None:
        """Execution should resolve paths from TMGG_OUTPUT_BASE."""
        mock_result = {"best_val_loss": 0.5, "best_model_path": None}
        captured_config: dict[str, Any] = {}

        def capture_config(config: DictConfig) -> dict[str, Any]:
            captured_config["paths"] = OmegaConf.to_container(config.paths)
            return mock_result

        original_run_experiment = _task_module.run_experiment
        _task_module.run_experiment = lambda: capture_config  # type: ignore[attr-defined]

        try:
            with patch.dict(os.environ, {"TMGG_OUTPUT_BASE": "/custom/base"}):
                execute_task(minimal_task, get_storage=lambda: mock_storage)
        finally:
            _task_module.run_experiment = original_run_experiment  # type: ignore[attr-defined]

        assert captured_config["paths"]["output_dir"] == "/custom/base/test123"
        assert captured_config["paths"]["results_dir"] == "/custom/base/test123/results"
