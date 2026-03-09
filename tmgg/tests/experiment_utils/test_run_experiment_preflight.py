"""Tests for run_experiment preflight utilities.

These test the run_id auto-generation and W&B duplicate detection
that replaced the config_builder pipeline. The run_id generator
produces human-readable names from resolved Hydra config values,
matching the format previously handled by ExperimentConfigBuilder.generate_run_id().
The W&B check queries the API for an existing run with the same display name.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from tmgg.experiments._shared_utils.orchestration.run_experiment import (
    check_wandb_run_exists,
    generate_run_id,
)


class TestGenerateRunId:
    """Tests for generate_run_id().

    The function produces a compact, human-readable identifier from
    the resolved config, using short prefixes and scientific notation
    for small values (matching the old run_id_template convention).
    """

    def test_spectral_arch_config(self):
        """Full spectral arch config produces expected format."""
        config = OmegaConf.create(
            {
                "experiment_name": "stage1",
                "model": {"_target_": "tmgg.models.SpectralArch", "k": 8},
                "learning_rate": 1e-4,
                "weight_decay": 1e-3,
                "seed": 1,
            }
        )
        result = generate_run_id(config)
        assert result == "stage1_SpectralArch_lr1e-4_wd1e-3_k8_s1"

    def test_scientific_notation_small_values(self):
        """Values < 0.01 use scientific notation without zero-padded exponent."""
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "learning_rate": 5e-5,
                "weight_decay": 1e-3,
                "seed": 2,
            }
        )
        result = generate_run_id(config)
        assert "lr5e-5" in result
        assert "wd1e-3" in result
        assert result.endswith("_s2")

    def test_large_values_no_scientific(self):
        """Values >= 0.01 use plain string representation."""
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "learning_rate": 0.01,
                "weight_decay": 0.1,
                "seed": 1,
            }
        )
        result = generate_run_id(config)
        assert "lr0.01" in result
        assert "wd0.1" in result

    def test_missing_optional_fields(self):
        """Works with minimal config (only seed required)."""
        config = OmegaConf.create({"seed": 42})
        result = generate_run_id(config)
        assert result == "s42"

    def test_diffusion_steps_included(self):
        """model.diffusion_steps appears as T prefix."""
        config = OmegaConf.create(
            {
                "model": {"_target_": "tmgg.models.Foo", "diffusion_steps": 500},
                "seed": 1,
            }
        )
        result = generate_run_id(config)
        assert "T500" in result

    def test_existing_run_id_not_overwritten(self):
        """If config already has run_id, generate_run_id returns it unchanged."""
        config = OmegaConf.create(
            {
                "run_id": "my_custom_id",
                "seed": 1,
            }
        )
        result = generate_run_id(config)
        assert result == "my_custom_id"


class TestCheckWandbRunExists:
    """Tests for check_wandb_run_exists().

    Uses mocked W&B API to avoid real network calls. The function
    queries by displayName filter to check for a single run efficiently.
    """

    def test_returns_true_when_run_found(self, monkeypatch):
        """Returns True when W&B API returns a matching run."""
        mock_api_cls = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "stage1_linear_pe_lr1e-4_s1"
        mock_api_cls.return_value.runs.return_value = [mock_run]
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        assert check_wandb_run_exists("team", "project", "stage1_linear_pe_lr1e-4_s1")
        mock_api_cls.return_value.runs.assert_called_once_with(
            "team/project",
            filters={"displayName": "stage1_linear_pe_lr1e-4_s1"},
        )

    def test_returns_false_when_no_match(self, monkeypatch):
        """Returns False when W&B API returns empty list."""
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.runs.return_value = []
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        assert not check_wandb_run_exists("team", "project", "nonexistent")

    def test_returns_false_on_comm_error(self, monkeypatch):
        """Returns False (not crash) on CommError -- transient network issues
        should not block experiment execution."""
        from wandb.errors import CommError

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.runs.side_effect = CommError("network error")
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        assert not check_wandb_run_exists("team", "project", "run_name")

    def test_reraises_non_comm_errors(self, monkeypatch):
        """Non-communication errors (auth, programming) must propagate, not silently return False.

        Rationale: the bare ``except Exception`` previously in check_wandb_run_exists
        swallowed authentication failures, programming bugs, and anything else. A
        swallowed auth error would cause the dedup check to silently return False,
        potentially duplicating hours of GPU work. Only CommError (network issues)
        should be caught; everything else must propagate so the caller sees the real
        failure.
        """
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.runs.side_effect = ValueError(
            "unexpected internal error"
        )
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        with pytest.raises(ValueError, match="unexpected internal error"):
            check_wandb_run_exists("entity", "project", "run_name")
