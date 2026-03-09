"""Tests for SingleStepDenoisingModule.

SingleStepDenoisingModule extends BaseGraphModule for T=1 denoising: sample
a noise level, corrupt, predict clean in one pass.  These tests verify
instantiation, training_step, forward bridge, per-noise-level evaluation,
property fallback behaviour, and both loss types.

The tests use a GNN model registered as ``"gnn"`` via the model factory,
matching production usage.  Batches are built with
``GraphData.from_adjacency`` on random symmetric binary matrices.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import GraphData
from tmgg.experiments._shared_utils.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

# -----------------------------------------------------------------------
# Shared constants and helpers
# -----------------------------------------------------------------------

_MODEL_TYPE = "gnn"
_MODEL_CONFIG: dict[str, Any] = {
    "num_layers": 2,
    "num_terms": 2,
    "feature_dim_in": 10,
    "feature_dim_out": 10,
}
_NOISE_LEVELS = [0.1, 0.3, 0.5]
_NUM_NODES = 10
_BATCH_SIZE = 2


def _make_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> GraphData:
    """Create a synthetic GraphData batch with random symmetric binary adjacency."""
    adj = torch.zeros(bs, n, n)
    for i in range(bs):
        upper = (torch.rand(n, n) > 0.5).float()
        sym = upper.triu(diagonal=1)
        adj[i] = sym + sym.t()
    return GraphData.from_adjacency(adj)


def _make_module(
    loss_type: str = "BCEWithLogits",
    noise_levels: list[float] | None = None,
    eval_noise_levels: list[float] | None = None,
    **overrides: Any,
) -> SingleStepDenoisingModule:
    """Build a SingleStepDenoisingModule with sensible test defaults."""
    kwargs: dict[str, Any] = {
        "model_type": _MODEL_TYPE,
        "model_config": _MODEL_CONFIG,
        "noise_type": "digress",
        "noise_levels": noise_levels if noise_levels is not None else _NOISE_LEVELS,
        "eval_noise_levels": eval_noise_levels,
        "loss_type": loss_type,
        "seed": 42,
    }
    kwargs.update(overrides)
    return SingleStepDenoisingModule(**kwargs)


# -----------------------------------------------------------------------
# 1. Instantiation
# -----------------------------------------------------------------------


class TestInstantiation:
    """Verify the module constructs correctly with all parameter variants."""

    def test_default_params(self) -> None:
        """The module should initialise with the specified defaults and
        set the criterion to BCEWithLogitsLoss.
        """
        m = _make_module()
        assert isinstance(m.criterion, nn.BCEWithLogitsLoss)
        assert m.noise_type == "digress"
        assert m.learning_rate == 0.001

    def test_is_diffusion_module_subclass(self) -> None:
        """SingleStepDenoisingModule must be a DiffusionModule subclass,
        per the design doc's requirement that it semantically hardcodes
        T=1 and sampler=None within the DiffusionModule hierarchy.
        """
        from tmgg.experiments._shared_utils.lightning_modules.diffusion_module import (
            DiffusionModule,
        )

        m = _make_module()
        assert isinstance(m, DiffusionModule)

    def test_mse_criterion(self) -> None:
        """When loss_type='MSE', the criterion should be nn.MSELoss."""
        m = _make_module(loss_type="MSE")
        assert isinstance(m.criterion, nn.MSELoss)

    def test_bce_criterion(self) -> None:
        """When loss_type='BCEWithLogits', the criterion should be BCEWithLogitsLoss."""
        m = _make_module(loss_type="BCEWithLogits")
        assert isinstance(m.criterion, nn.BCEWithLogitsLoss)

    def test_unknown_loss_type_raises(self) -> None:
        """An unrecognised loss_type should raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            _make_module(loss_type="huber")


# -----------------------------------------------------------------------
# 2. training_step
# -----------------------------------------------------------------------


class TestTrainingStep:
    """Verify that training_step produces finite loss with gradients."""

    def test_finite_loss_with_gradient(self) -> None:
        """training_step should return a scalar tensor with a grad_fn
        attached, and its value should be finite.
        """
        m = _make_module()
        batch = _make_batch()
        loss = m.training_step(batch, batch_idx=0)
        assert loss.grad_fn is not None, "loss has no grad_fn"
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    def test_logs_metrics(self) -> None:
        """training_step should call self.log for train/loss,
        train/accuracy, and train/noise_level.

        We patch self.log and inspect the call args to confirm the
        expected metric keys are present.
        """
        m = _make_module()
        batch = _make_batch()

        logged: dict[str, Any] = {}

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = value

        m.log = capture_log  # type: ignore[assignment]
        _ = m.training_step(batch, batch_idx=0)

        assert (
            "train/loss" in logged
        ), f"Missing train/loss. Logged keys: {list(logged)}"
        assert "train/accuracy" in logged
        assert "train/noise_level" in logged

    def test_mse_loss_type_works(self) -> None:
        """training_step should work correctly with MSE loss, producing
        a finite loss with gradient.
        """
        m = _make_module(loss_type="MSE")
        batch = _make_batch()
        loss = m.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.grad_fn is not None

    def test_bce_loss_type_works(self) -> None:
        """training_step should work correctly with BCEWithLogits loss,
        producing a finite loss with gradient.
        """
        m = _make_module(loss_type="BCEWithLogits")
        batch = _make_batch()
        loss = m.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.grad_fn is not None


# -----------------------------------------------------------------------
# 3. forward
# -----------------------------------------------------------------------


class TestForward:
    """Verify that forward() bridges raw tensors to/from GraphData."""

    def test_tensor_in_tensor_out(self) -> None:
        """forward() should accept a raw (B, N, N) tensor and return a
        raw (B, N, N) tensor of the same batch and spatial dimensions.
        """
        m = _make_module()
        x = torch.rand(_BATCH_SIZE, _NUM_NODES, _NUM_NODES)
        out = m.forward(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (_BATCH_SIZE, _NUM_NODES, _NUM_NODES)

    def test_forward_with_timestep(self) -> None:
        """forward() should accept an optional timestep tensor without
        error, even though single-step models typically ignore it.
        """
        m = _make_module()
        x = torch.rand(_BATCH_SIZE, _NUM_NODES, _NUM_NODES)
        t = torch.zeros(_BATCH_SIZE)
        out = m.forward(x, t=t)
        assert out.shape == (_BATCH_SIZE, _NUM_NODES, _NUM_NODES)


# -----------------------------------------------------------------------
# 4. _val_or_test
# -----------------------------------------------------------------------


class TestValOrTest:
    """Verify per-noise-level evaluation and averaged metric logging."""

    def test_evaluates_all_noise_levels(self) -> None:
        """_val_or_test should log per-noise-level metrics for every
        level in eval_noise_levels (3 levels => 3 per-level loss keys).
        """
        levels = [0.1, 0.3, 0.5]
        m = _make_module(noise_levels=levels, eval_noise_levels=levels)

        logged: dict[str, Any] = {}

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = value

        m.log = capture_log  # type: ignore[assignment]
        batch = _make_batch()
        _ = m._val_or_test("val", batch)

        for eps in levels:
            key = f"val_{eps}/loss"
            assert key in logged, (
                f"Expected per-level key '{key}' not found. "
                f"Logged keys: {sorted(logged)}"
            )

    def test_logs_averaged_metrics(self) -> None:
        """_val_or_test should log noise-level-averaged val/loss and
        val/accuracy alongside the per-level keys.
        """
        m = _make_module()
        logged: dict[str, Any] = {}

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = value

        m.log = capture_log  # type: ignore[assignment]
        batch = _make_batch()
        _ = m._val_or_test("val", batch)

        assert "val/loss" in logged
        assert "val/accuracy" in logged

    def test_validation_step_delegates(self) -> None:
        """validation_step should delegate to _val_or_test('val', batch)
        and return a dict.
        """
        m = _make_module()
        m.log = lambda *a, **kw: None  # type: ignore[assignment]
        batch = _make_batch()
        result = m.validation_step(batch, batch_idx=0)
        assert isinstance(result, dict)
        assert "val_loss" in result

    def test_test_step_delegates(self) -> None:
        """test_step should delegate to _val_or_test('test', batch)
        and return a dict with 'test_loss'.
        """
        m = _make_module()
        m.log = lambda *a, **kw: None  # type: ignore[assignment]
        batch = _make_batch()
        result = m.test_step(batch, batch_idx=0)
        assert isinstance(result, dict)
        assert "test_loss" in result


# -----------------------------------------------------------------------
# 5. noise_levels property
# -----------------------------------------------------------------------


class TestNoiseLevelsProperty:
    """Verify noise_levels sourcing from constructor and datamodule."""

    def test_explicit_noise_levels(self) -> None:
        """When noise_levels are provided at construction, the property
        should return them directly without needing a datamodule.
        """
        levels = [0.2, 0.4]
        m = _make_module(noise_levels=levels)
        assert m.noise_levels == levels

    def test_reads_from_datamodule(self) -> None:
        """When noise_levels is None, the property should read from
        trainer.datamodule.noise_levels.
        """
        m = SingleStepDenoisingModule(
            model_type=_MODEL_TYPE,
            model_config=_MODEL_CONFIG,
            noise_levels=None,
            seed=42,
        )

        mock_dm = MagicMock()
        mock_dm.noise_levels = [0.05, 0.15, 0.25]
        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm

        # Attach the mock trainer
        m._trainer = mock_trainer  # type: ignore[attr-defined]
        # LightningModule.trainer is a property; patch it
        with patch.object(
            type(m), "trainer", new_callable=PropertyMock, return_value=mock_trainer
        ):
            assert m.noise_levels == [0.05, 0.15, 0.25]

    def test_no_datamodule_raises(self) -> None:
        """When noise_levels is None and no trainer is attached, accessing
        the property should raise RuntimeError.
        """
        m = SingleStepDenoisingModule(
            model_type=_MODEL_TYPE,
            model_config=_MODEL_CONFIG,
            noise_levels=None,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="noise_levels"):
            _ = m.noise_levels


# -----------------------------------------------------------------------
# 6. eval_noise_levels fallback
# -----------------------------------------------------------------------


class TestEvalNoiseLevelsFallback:
    """Verify eval_noise_levels falls back to noise_levels when unset."""

    def test_explicit_eval_levels(self) -> None:
        """When eval_noise_levels is provided, the property should return
        those levels, independent of noise_levels.
        """
        m = _make_module(
            noise_levels=[0.1, 0.3, 0.5],
            eval_noise_levels=[0.2, 0.4],
        )
        assert m.eval_noise_levels == [0.2, 0.4]

    def test_falls_back_to_noise_levels(self) -> None:
        """When eval_noise_levels is None, the property should return
        the same list as noise_levels.
        """
        m = _make_module(
            noise_levels=[0.1, 0.3, 0.5],
            eval_noise_levels=None,
        )
        assert m.eval_noise_levels == [0.1, 0.3, 0.5]
