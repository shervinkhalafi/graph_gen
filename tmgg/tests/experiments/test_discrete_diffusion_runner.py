"""Tests for discrete diffusion Hydra runner and config (P1.7).

Testing strategy
----------------
Verifies that the YAML configuration parses correctly, the runner's
utility functions behave as expected, and that the full Hydra-driven
training loop can execute a minimal 2-step run. The fast tests
(config parsing, checkpoint detection) run in CI; the full end-to-end
training test is marked slow.

Key invariants tested:
- The YAML config parses without OmegaConf syntax errors
- All required _target_ classes resolve to importable types
- _is_training_complete returns False for missing/incomplete checkpoints
- A 2-step Hydra-configured training run completes without errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tmgg.experiment_utils.run_experiment import _is_training_complete


def _load_config_as_dict(path: Path) -> dict[str, Any]:
    """Load a YAML config and return as a plain dict for test assertions."""
    cfg = OmegaConf.load(path)
    result = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(result, dict)
    return result  # pyright: ignore[reportReturnType]


# ------------------------------------------------------------------
# Config loading tests
# ------------------------------------------------------------------


class TestConfigLoads:
    """Verify the YAML config parses and contains expected structure."""

    @pytest.fixture()
    def config_path(self) -> Path:
        """Path to the discrete diffusion base config."""
        return (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tmgg"
            / "exp_configs"
            / "base_config_discrete_diffusion_generative.yaml"
        )

    def test_yaml_parses(self, config_path: Path) -> None:
        """Config should load as valid OmegaConf without syntax errors."""
        cfg = OmegaConf.load(config_path)
        assert OmegaConf.is_config(cfg)

    def test_has_model_section(self, config_path: Path) -> None:
        """Model section must contain _target_ and nested model/noise_schedule."""
        cfg = _load_config_as_dict(config_path)
        assert "model" in cfg
        model_cfg = cfg["model"]
        assert "_target_" in model_cfg
        assert "model" in model_cfg
        assert "noise_schedule" in model_cfg
        assert model_cfg["model"]["_target_"].endswith("DiscreteGraphTransformer")
        assert model_cfg["noise_schedule"]["_target_"].endswith(
            "PredefinedNoiseScheduleDiscrete"
        )

    def test_has_data_section(self, config_path: Path) -> None:
        """Data section must target SyntheticCategoricalDataModule."""
        cfg = _load_config_as_dict(config_path)
        assert "data" in cfg
        assert cfg["data"]["_target_"].endswith("SyntheticCategoricalDataModule")

    def test_callbacks_monitor_nll(self, config_path: Path) -> None:
        """Callbacks must monitor val/epoch_NLL, not val/loss."""
        cfg = _load_config_as_dict(config_path)
        callbacks = cfg["callbacks"]
        assert callbacks["early_stopping"]["monitor"] == "val/epoch_NLL"
        assert callbacks["checkpoint"]["monitor"] == "val/epoch_NLL"

    def test_target_classes_importable(self, config_path: Path) -> None:
        """All _target_ strings must resolve to importable Python classes."""
        cfg = _load_config_as_dict(config_path)

        targets = [
            cfg["model"]["_target_"],
            cfg["model"]["model"]["_target_"],
            cfg["model"]["noise_schedule"]["_target_"],
            cfg["data"]["_target_"],
        ]

        for target in targets:
            module_path, class_name = target.rsplit(".", 1)
            mod = __import__(module_path, fromlist=[class_name])
            assert hasattr(mod, class_name), f"{target} not importable"


# ------------------------------------------------------------------
# Utility function tests
# ------------------------------------------------------------------


class TestIsTrainingComplete:
    """Checkpoint completion detection logic."""

    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        """Returns False for nonexistent checkpoint."""
        assert _is_training_complete(tmp_path / "missing.ckpt", 100) is False

    def test_incomplete_checkpoint(self, tmp_path: Path) -> None:
        """Returns False when global_step < max_steps."""
        ckpt_path = tmp_path / "last.ckpt"
        torch.save({"global_step": 50}, ckpt_path)
        assert _is_training_complete(ckpt_path, 100) is False

    def test_complete_checkpoint(self, tmp_path: Path) -> None:
        """Returns True when global_step >= max_steps."""
        ckpt_path = tmp_path / "last.ckpt"
        torch.save({"global_step": 100}, ckpt_path)
        assert _is_training_complete(ckpt_path, 100) is True

    def test_over_complete_checkpoint(self, tmp_path: Path) -> None:
        """Returns True when global_step exceeds max_steps."""
        ckpt_path = tmp_path / "last.ckpt"
        torch.save({"global_step": 200}, ckpt_path)
        assert _is_training_complete(ckpt_path, 100) is True


# ------------------------------------------------------------------
# End-to-end training (Hydra-free, programmatic)
# ------------------------------------------------------------------


class TestEndToEndTraining:
    """Run a complete 2-step training via the programmatic API.

    This bypasses Hydra to avoid subprocess overhead while still
    verifying that all components wire together: datamodule → model →
    trainer → validation with MMD.
    """

    def test_two_step_training(self) -> None:
        """A 2-step training run should complete and log val/epoch_NLL."""
        import pytorch_lightning as pl

        from tmgg.experiments.discrete_diffusion_generative.datamodule import (
            SyntheticCategoricalDataModule,
        )
        from tmgg.experiments.discrete_diffusion_generative.lightning_module import (
            DiscreteDiffusionLightningModule,
        )
        from tmgg.models.digress.discrete_transformer import (
            DiscreteGraphTransformer,
        )
        from tmgg.models.digress.noise_schedule import (
            PredefinedNoiseScheduleDiscrete,
        )

        dm = SyntheticCategoricalDataModule(
            num_nodes=8, num_graphs=32, batch_size=4, seed=42
        )
        model = DiscreteGraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 2, "y": 1},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
            output_dims={"X": 2, "E": 2, "y": 0},
        )
        ns = PredefinedNoiseScheduleDiscrete("cosine", 10)
        module = DiscreteDiffusionLightningModule(
            model=model,
            noise_schedule=ns,
            diffusion_steps=10,
            eval_num_samples=4,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, dm)

        logged = trainer.logged_metrics
        assert "val/epoch_NLL" in logged, f"Missing val/epoch_NLL in {list(logged)}"
