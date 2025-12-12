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

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys

import pytest

# Mock modal before importing tmgg_modal modules to avoid image creation at import time
sys.modules["modal"] = MagicMock()


class TestResultStatus:
    """Tests for ResultStatus enum and check_result_status function."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return MagicMock()

    def test_missing_when_result_not_found(self, mock_storage):
        """Should return MISSING when storage.exists returns False."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

        mock_storage.exists.return_value = False

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.MISSING

    def test_complete_with_valid_metrics(self, mock_storage):
        """Should return COMPLETE when all required metrics present."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.123},
            "completed_at": datetime.now().isoformat(),
        }

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.COMPLETE

    def test_partial_when_metric_missing(self, mock_storage):
        """Should return PARTIAL when required metric is None."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"other_metric": 0.5},  # Missing best_val_loss
        }

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.PARTIAL

    def test_partial_when_metric_is_inf(self, mock_storage):
        """Should return PARTIAL when metric is infinity (failed run)."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": float("inf")},
        }

        status = check_result_status(mock_storage, "run-123")

        assert status == ResultStatus.PARTIAL

    def test_stale_when_older_than_threshold(self, mock_storage):
        """Should return STALE when result is older than max_age_hours."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

        mock_storage.exists.return_value = True
        old_time = datetime.now() - timedelta(hours=48)
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.123},
            "completed_at": old_time.isoformat(),
        }

        status = check_result_status(
            mock_storage, "run-123", max_age_hours=24  # 24 hour threshold
        )

        assert status == ResultStatus.STALE

    def test_complete_when_within_age_threshold(self, mock_storage):
        """Should return COMPLETE when result is within max_age_hours."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

        mock_storage.exists.return_value = True
        recent_time = datetime.now() - timedelta(hours=12)
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.123},
            "completed_at": recent_time.isoformat(),
        }

        status = check_result_status(
            mock_storage, "run-123", max_age_hours=24  # 24 hour threshold
        )

        assert status == ResultStatus.COMPLETE

    def test_custom_required_metrics(self, mock_storage):
        """Should check custom required_metrics list."""
        from tmgg_modal.result_status import ResultStatus, check_result_status

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
        from tmgg_modal.result_status import ResultStatus, filter_configs_by_status

        filtered, status_map = filter_configs_by_status(None, sample_configs)

        assert len(filtered) == 3
        assert all(s == ResultStatus.MISSING for s in status_map.values())

    def test_filters_complete_results_by_default(self, mock_storage, sample_configs):
        """By default, should skip COMPLETE results."""
        from tmgg_modal.result_status import ResultStatus, filter_configs_by_status

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
        from tmgg_modal.result_status import ResultStatus, filter_configs_by_status

        # All are complete
        mock_storage.exists.return_value = True
        mock_storage.download_metrics.return_value = {
            "metrics": {"best_val_loss": 0.1}
        }

        # Don't skip anything
        filtered, _ = filter_configs_by_status(
            mock_storage, sample_configs, skip_statuses=set()
        )

        assert len(filtered) == 3

    def test_status_map_contains_all_run_ids(self, mock_storage, sample_configs):
        """Status map should contain entries for all configs."""
        from tmgg_modal.result_status import filter_configs_by_status

        mock_storage.exists.return_value = False

        _, status_map = filter_configs_by_status(mock_storage, sample_configs)

        assert set(status_map.keys()) == {"run-1", "run-2", "run-3"}


class TestSummarizeStatusMap:
    """Tests for summarize_status_map function."""

    def test_summarizes_all_statuses(self):
        """Should include counts for all present statuses."""
        from tmgg_modal.result_status import ResultStatus, summarize_status_map

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
        from tmgg_modal.result_status import summarize_status_map

        summary = summarize_status_map({})

        assert summary == "no results"


class TestModalRunner:
    """Tests for ModalRunner experiment execution."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        return MagicMock()

    def test_runner_initialization(self, mock_storage):
        """ModalRunner should initialize with gpu_type and storage."""
        from tmgg_modal.runner import ModalRunner

        runner = ModalRunner(gpu_type="fast", storage=mock_storage)

        assert runner.gpu_type == "fast"
        assert runner.storage is mock_storage

    def test_create_runner_factory(self):
        """create_runner should create ModalRunner with default settings."""
        with patch.dict("os.environ", {}, clear=True):
            from tmgg_modal.runner import create_runner

            runner = create_runner(gpu_type="standard")

            assert runner.gpu_type == "standard"
            # Storage should be None when env not configured
            assert runner.storage is None


class TestExperimentResult:
    """Tests for ExperimentResult dataclass structure."""

    def test_result_from_dict(self):
        """ExperimentResult should be constructable from result dict."""
        from tmgg_modal.runner import ExperimentResult

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
        from tmgg_modal.runner import ExperimentResult

        result = ExperimentResult(
            run_id="failed-run",
            config={},
            metrics={},
            status="failed",
            error_message="CUDA out of memory",
            duration_seconds=120.0,
        )

        assert result.status == "failed"
        assert "CUDA" in result.error_message


class TestGPUConfigs:
    """Tests for GPU configuration constants."""

    def test_gpu_configs_defined(self):
        """GPU_CONFIGS should define expected tiers."""
        from tmgg_modal.app import GPU_CONFIGS

        assert "standard" in GPU_CONFIGS
        assert "fast" in GPU_CONFIGS

    def test_default_timeouts_match_gpu_tiers(self):
        """DEFAULT_TIMEOUTS should have entries for GPU tiers."""
        from tmgg_modal.app import DEFAULT_TIMEOUTS, GPU_CONFIGS

        # All GPU configs should have corresponding timeouts
        for tier in GPU_CONFIGS:
            assert tier in DEFAULT_TIMEOUTS, f"Missing timeout for {tier}"

    def test_timeouts_are_reasonable(self):
        """Timeouts should be in reasonable range (10 min to 24 hours)."""
        from tmgg_modal.app import DEFAULT_TIMEOUTS

        for tier, timeout in DEFAULT_TIMEOUTS.items():
            assert timeout >= 600, f"Timeout for {tier} too short"
            assert timeout <= 86400, f"Timeout for {tier} too long"
