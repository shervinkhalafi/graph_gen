"""Tests for DenoisingLightningModule base class.

This module tests the core training module that handles:
- Noise application
- Training/validation/test step execution
- Optimizer and scheduler configuration
- Spectral delta logging

Testing Strategy:
- Create minimal mock model implementing _DenoisingModelProtocol
- Create concrete subclass of DenoisingLightningModule for testing
- Test core functionality without full training loops
- Mock trainer/datamodule dependencies where needed

Key Invariants:
- Noise is applied fresh each call
- Noise levels sourced exclusively from datamodule (fails if not attached)
- eval_noise_levels can override noise_levels for evaluation
- Scheduler configuration respects declared types
- Loss type maps to correct criterion
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.experiment_utils.exceptions import ConfigurationError


class MockDenoisingModel(nn.Module):
    """Minimal model implementing _DenoisingModelProtocol for testing."""

    def __init__(self, input_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size * input_size, input_size * input_size)

    def parameter_count(self) -> dict[str, Any]:
        return {"total": sum(p.numel() for p in self.parameters())}

    def get_config(self) -> dict[str, Any]:
        return {"input_size": self.input_size}

    def transform_for_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identity transform for testing."""
        return output, target

    def predict(self, logits: torch.Tensor, zero_diag: bool = True) -> torch.Tensor:
        """Apply sigmoid for predictions."""
        preds = torch.sigmoid(logits)
        if zero_diag and preds.dim() >= 2:
            preds = preds.clone()
            for i in range(preds.shape[-1]):
                preds[..., i, i] = 0
        return preds

    def logits_to_graph(self, logits: torch.Tensor) -> torch.Tensor:
        """Threshold at 0.5 for binary prediction."""
        return (torch.sigmoid(logits) > 0.5).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        B, N, _ = x.shape
        flat = x.reshape(B, -1)
        out = self.linear(flat)
        return out.reshape(B, N, N)


class ConcreteDenoisingModule(DenoisingLightningModule):
    """Concrete implementation of DenoisingLightningModule for testing."""

    def _make_model(self, *args: Any, **kwargs: Any) -> MockDenoisingModel:
        return MockDenoisingModel(input_size=kwargs.get("input_size", 10))


class TestDenoisingLightningModuleInit:
    """Tests for module initialization."""

    def test_default_initialization(self) -> None:
        """Should initialize with minimal required parameters.

        Rationale: Default values should provide sensible behavior without
        requiring explicit configuration.
        """
        module = ConcreteDenoisingModule()

        assert module.learning_rate == 0.001
        assert module.weight_decay == 0.0
        assert module.optimizer_type == "adam"
        assert module.noise_type == "Digress"
        assert module.visualization_interval == 5000

    def test_loss_type_mse(self) -> None:
        """loss_type='MSE' should use MSELoss.

        Rationale: MSE loss is standard for regression-like denoising tasks.
        """
        module = ConcreteDenoisingModule(loss_type="MSE")

        assert isinstance(module.criterion, nn.MSELoss)

    def test_loss_type_bce(self) -> None:
        """loss_type='BCEWithLogits' should use BCEWithLogitsLoss.

        Rationale: BCE is appropriate for binary adjacency prediction.
        """
        module = ConcreteDenoisingModule(loss_type="BCEWithLogits")

        assert isinstance(module.criterion, nn.BCEWithLogitsLoss)

    def test_invalid_loss_type_raises(self) -> None:
        """Invalid loss_type should raise ConfigurationError.

        Rationale: Clear error helps users discover valid options.
        """
        with pytest.raises(ConfigurationError, match="Unknown loss_type"):
            ConcreteDenoisingModule(loss_type="invalid")

    def test_noise_generator_created(self) -> None:
        """Should create noise generator from noise_type.

        Rationale: Noise generator is required for applying noise during training.
        """
        module = ConcreteDenoisingModule(noise_type="Gaussian")

        assert module.noise_generator is not None
        assert hasattr(module.noise_generator, "add_noise")


class TestNoiseLevelsProperty:
    """Tests for noise_levels property behavior."""

    def test_raises_without_trainer(self) -> None:
        """Should raise RuntimeError when no trainer attached.

        Rationale: noise_levels is sourced exclusively from the datamodule,
        so accessing it without a trainer should fail loudly.
        """
        module = ConcreteDenoisingModule()

        # Lightning raises its own error when accessing trainer property without attachment
        with pytest.raises(RuntimeError, match="not attached to a"):
            _ = module.noise_levels

    def test_raises_without_datamodule(self) -> None:
        """Should raise RuntimeError when trainer has no datamodule.

        Rationale: Even with a trainer, noise_levels requires a datamodule.
        """
        module = ConcreteDenoisingModule()

        # Mock trainer without datamodule
        mock_trainer = MagicMock()
        mock_trainer.datamodule = None
        module._trainer = mock_trainer

        with pytest.raises(
            RuntimeError, match="not attached to trainer with datamodule"
        ):
            _ = module.noise_levels

    def test_raises_when_datamodule_lacks_attribute(self) -> None:
        """Should raise RuntimeError when datamodule lacks noise_levels.

        Rationale: Clear error when datamodule doesn't provide noise_levels.
        """
        module = ConcreteDenoisingModule()

        # Mock datamodule without noise_levels
        mock_dm = MagicMock(spec=[])  # Empty spec means no attributes
        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer

        with pytest.raises(RuntimeError, match="does not have noise_levels"):
            _ = module.noise_levels

    def test_returns_datamodule_levels(self) -> None:
        """Should return noise_levels from datamodule when available.

        Rationale: Datamodule is the authoritative source for noise_levels.
        """
        module = ConcreteDenoisingModule()

        # Mock trainer and datamodule
        mock_dm = MagicMock()
        mock_dm.noise_levels = [0.01, 0.02, 0.03]

        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm

        # Attach mock trainer
        module._trainer = mock_trainer

        assert module.noise_levels == [0.01, 0.02, 0.03]

    def test_eval_noise_levels_override(self) -> None:
        """Should use eval_noise_levels when explicitly set.

        Rationale: Evaluation may use different noise levels than training.
        """
        module = ConcreteDenoisingModule(eval_noise_levels=[0.05, 0.15, 0.25])

        # eval_noise_levels can be accessed without trainer if explicitly set
        assert module.eval_noise_levels == [0.05, 0.15, 0.25]

    def test_eval_noise_levels_fallback_to_noise_levels(self) -> None:
        """Should fall back to noise_levels when eval_noise_levels not set.

        Rationale: By default, use same levels for training and evaluation.
        """
        module = ConcreteDenoisingModule()

        # Mock trainer and datamodule
        mock_dm = MagicMock()
        mock_dm.noise_levels = [0.1, 0.2, 0.3]

        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer

        # Without eval_noise_levels override, falls back to noise_levels
        assert module.eval_noise_levels == [0.1, 0.2, 0.3]


class TestApplyNoise:
    """Tests for _apply_noise method."""

    def test_applies_fresh_noise_each_call(self) -> None:
        """Each call should apply fresh noise.

        Rationale: Training needs diverse noise for generalization.
        """
        module = ConcreteDenoisingModule()

        batch = torch.eye(8).unsqueeze(0).expand(4, -1, -1)
        eps = 0.1

        # Apply noise twice
        noisy1 = module._apply_noise(batch, eps)
        noisy2 = module._apply_noise(batch, eps)

        # Should be different (with very high probability)
        assert not torch.allclose(noisy1, noisy2)


class TestTrainingStep:
    """Tests for training_step method."""

    def _create_module_with_trainer(self) -> ConcreteDenoisingModule:
        """Create module with mock trainer and datamodule attached."""
        module = ConcreteDenoisingModule()

        # Mock trainer and datamodule with noise_levels
        mock_dm = MagicMock()
        mock_dm.noise_levels = [0.1, 0.2, 0.3]

        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer

        return module

    def test_returns_loss_dict(self) -> None:
        """training_step should return dict with 'loss' key.

        Rationale: Lightning requires 'loss' key for optimization.
        """
        module = self._create_module_with_trainer()
        batch = torch.randn(4, 10, 10)

        result = module.training_step(batch, batch_idx=0)

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)

    def test_returns_logits(self) -> None:
        """training_step should return logits for debugging callbacks.

        Rationale: Callbacks may need access to model outputs.
        """
        module = self._create_module_with_trainer()
        batch = torch.randn(4, 10, 10)

        result = module.training_step(batch, batch_idx=0)

        assert "logits" in result
        assert result["logits"].shape == batch.shape


class TestSchedulerConfiguration:
    """Tests for configure_optimizers scheduler options."""

    def test_no_scheduler_returns_optimizer_only(self) -> None:
        """Without scheduler_config, should return just optimizer.

        Rationale: Simple setups don't need learning rate scheduling.
        """
        module = ConcreteDenoisingModule(scheduler_config=None)

        result = module.configure_optimizers()

        assert isinstance(result, torch.optim.Optimizer)

    def test_adam_optimizer_default(self) -> None:
        """Default optimizer should be Adam.

        Rationale: Adam is standard for deep learning.
        """
        module = ConcreteDenoisingModule(optimizer_type="adam")

        result = module.configure_optimizers()

        assert isinstance(result, torch.optim.Adam)

    def test_adamw_optimizer(self) -> None:
        """optimizer_type='adamw' should use AdamW.

        Rationale: AdamW provides better weight decay handling.
        """
        module = ConcreteDenoisingModule(optimizer_type="adamw", weight_decay=0.01)

        result = module.configure_optimizers()

        assert isinstance(result, torch.optim.AdamW)

    def test_cosine_scheduler_creation(self) -> None:
        """scheduler_type='cosine' should create CosineAnnealingWarmRestarts.

        Rationale: Cosine schedule is common for gradual LR decay.
        """
        module = ConcreteDenoisingModule(scheduler_config={"type": "cosine", "T_0": 10})

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "lr_scheduler" in result
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        )

    def test_step_scheduler_creation(self) -> None:
        """scheduler_type='step' should create StepLR.

        Rationale: Step scheduler provides discrete LR drops.
        """
        module = ConcreteDenoisingModule(
            scheduler_config={"type": "step", "step_size": 10, "gamma": 0.5}
        )

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "lr_scheduler" in result
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_cosine_warmup_scheduler_creation(self) -> None:
        """scheduler_type='cosine_warmup' should create LambdaLR with warmup.

        Rationale: Warmup helps training stability in early steps.
        """
        module = ConcreteDenoisingModule(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 100,
                "T_max": 1000,
            }
        )

        # Mock trainer to avoid "not attached" error
        mock_trainer = MagicMock()
        mock_trainer.max_steps = 0  # Force fallback path
        module._trainer = mock_trainer

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "lr_scheduler" in result
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_cosine_warmup_fraction_config(self) -> None:
        """Fraction-based cosine_warmup config should work.

        Rationale: Fraction-based config automatically scales with training.
        """
        module = ConcreteDenoisingModule(
            scheduler_config={
                "type": "cosine_warmup",
                "warmup_fraction": 0.02,
                "decay_fraction": 0.8,
            }
        )

        # Need trainer mock for fraction-based config
        mock_trainer = MagicMock()
        mock_trainer.max_steps = 10000
        module._trainer = mock_trainer

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "lr_scheduler" in result

    def test_t_max_greater_than_t_warmup_validation(self) -> None:
        """T_max must be > T_warmup or raise ValueError.

        Rationale: Invalid config should fail fast with clear error.
        """
        module = ConcreteDenoisingModule(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 1000,
                "T_max": 500,  # Invalid: T_max < T_warmup
            }
        )

        # Mock trainer to avoid "not attached" error
        mock_trainer = MagicMock()
        mock_trainer.max_steps = 0
        module._trainer = mock_trainer

        with pytest.raises(ValueError, match="T_max.*must be > T_warmup"):
            module.configure_optimizers()


class TestValTestStep:
    """Tests for validation and test step execution."""

    def test_val_step_iterates_noise_levels(self) -> None:
        """validation_step should evaluate across all eval_noise_levels.

        Rationale: Comprehensive evaluation at multiple noise levels.
        """
        levels = [0.1, 0.2, 0.3]
        # Use eval_noise_levels to avoid needing trainer
        module = ConcreteDenoisingModule(eval_noise_levels=levels)

        with patch.object(module, "log") as mock_log:
            batch = torch.randn(4, 10, 10)
            module.validation_step(batch, batch_idx=0)

            # Should log for each noise level
            log_calls = [call[0][0] for call in mock_log.call_args_list]
            for eps in levels:
                assert any(f"val_{eps}/loss" in call for call in log_calls)

    def test_test_step_returns_metrics(self) -> None:
        """test_step should return dict with metrics.

        Rationale: Test metrics needed for evaluation.
        """
        # Need to set up trainer with datamodule for test_step to work
        module = ConcreteDenoisingModule(eval_noise_levels=[0.1])
        batch = torch.randn(4, 10, 10)

        result = module.test_step(batch, batch_idx=0)

        assert isinstance(result, dict)
        assert "test_loss" in result


class TestSpectralDeltaLogging:
    """Tests for spectral delta metric logging."""

    def test_spectral_deltas_disabled_by_default(self) -> None:
        """log_spectral_deltas should be False by default.

        Rationale: Spectral computation is expensive; opt-in only.
        """
        module = ConcreteDenoisingModule()

        assert module.log_spectral_deltas is False

    def test_spectral_deltas_enabled(self) -> None:
        """log_spectral_deltas=True should enable logging.

        Rationale: User can opt into spectral analysis.
        """
        module = ConcreteDenoisingModule(log_spectral_deltas=True, spectral_k=6)

        assert module.log_spectral_deltas is True
        assert module.spectral_k == 6

    def test_spectral_deltas_logged_when_enabled(self) -> None:
        """Should log spectral deltas during validation when enabled.

        Rationale: Verify integration with compute_spectral_deltas.
        """
        module = ConcreteDenoisingModule(
            log_spectral_deltas=True,
            eval_noise_levels=[0.1],  # Use eval_noise_levels to avoid needing trainer
        )

        with patch.object(module, "log") as mock_log:
            batch = torch.randn(4, 10, 10)
            module.validation_step(batch, batch_idx=0)

            # Should have spectral delta logs
            log_calls = [call[0][0] for call in mock_log.call_args_list]
            spectral_calls = [c for c in log_calls if "noisy_" in c or "denoised_" in c]
            assert len(spectral_calls) > 0


class TestSetup:
    """Tests for setup method validation."""

    def test_setup_validates_datamodule_attrs(self) -> None:
        """setup should validate datamodule has required attributes.

        Rationale: Early validation catches configuration errors.
        """
        module = ConcreteDenoisingModule()

        # Mock datamodule without noise_levels
        mock_dm = MagicMock(spec=[])  # Empty spec means no attributes
        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer

        with pytest.raises(ValueError, match="missing required attributes"):
            module.setup("fit")

    def test_setup_succeeds_with_valid_datamodule(self) -> None:
        """setup should pass with valid datamodule.

        Rationale: Normal case should work without errors.
        """
        module = ConcreteDenoisingModule()

        mock_dm = MagicMock()
        mock_dm.noise_levels = [0.1, 0.2]
        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer

        # Should not raise
        module.setup("fit")

    def test_setup_handles_no_datamodule(self) -> None:
        """setup should handle case when trainer has no datamodule.

        Rationale: Skip validation when datamodule is None.
        """
        module = ConcreteDenoisingModule()

        # Mock trainer without datamodule
        mock_trainer = MagicMock()
        mock_trainer.datamodule = None
        module._trainer = mock_trainer

        # Should not raise when datamodule is None
        module.setup("fit")


class TestGetModelConfig:
    """Tests for get_model_config method."""

    def test_returns_model_config(self) -> None:
        """get_model_config should return model's configuration.

        Rationale: Logging and reproducibility require access to config.
        """
        module = ConcreteDenoisingModule(input_size=20)

        config = module.get_model_config()

        assert isinstance(config, dict)
        assert config["input_size"] == 20
