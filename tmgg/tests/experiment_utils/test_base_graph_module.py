"""Tests for BaseGraphModule shared Lightning infrastructure.

BaseGraphModule provides common plumbing for all graph learning experiments:
model creation via ModelRegistry, optimizer/scheduler setup, GraphData batch
transfer, and parameter-count logging. It carries no training logic.

Testing strategy: a concrete subclass ``_ConcreteModule`` supplies trivial
``training_step`` and ``forward`` implementations so we can instantiate the
ABC and test the infrastructure methods in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.experiments._shared_utils.lightning_modules.base_graph_module import (
    BaseGraphModule,
)
from tmgg.models.base import GraphModel

# -----------------------------------------------------------------------
# Concrete subclass for testing
# -----------------------------------------------------------------------


class _ConcreteModule(BaseGraphModule):
    """Minimal concrete subclass that satisfies the ABC contract."""

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def forward(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        pass


# Default model_type and config used throughout the tests.  "gnn" is
# registered in the factory and produces a GraphModel.
_DEFAULT_MODEL_TYPE = "gnn"
_DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "num_layers": 2,
    "num_terms": 2,
    "feature_dim_in": 10,
    "feature_dim_out": 10,
}


def _make_module(**overrides: Any) -> _ConcreteModule:
    """Convenience factory that merges *overrides* into sensible defaults."""
    kwargs: dict[str, Any] = {
        "model_type": _DEFAULT_MODEL_TYPE,
        "model_config": _DEFAULT_MODEL_CONFIG,
    }
    kwargs.update(overrides)
    return _ConcreteModule(**kwargs)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestBaseGraphModuleIsAbstract:
    """Verify that BaseGraphModule cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseGraphModule is abstract; it must be subclassed.

        The ABC mechanism prevents direct construction because
        ``training_step`` and ``forward`` are not implemented.
        """
        with pytest.raises(TypeError):
            BaseGraphModule(  # type: ignore[abstract]
                model_type=_DEFAULT_MODEL_TYPE,
                model_config=_DEFAULT_MODEL_CONFIG,
            )


class TestMakeParametrizedModel:
    """Tests for _make_parametrized_model (model creation via registry)."""

    def test_creates_graph_model_from_registry(self) -> None:
        """The factory should return a GraphModel subclass for a valid model_type.

        Uses the ``"gnn"`` factory registered in ``tmgg.models.factory``.
        """
        module = _make_module()
        assert isinstance(module.model, GraphModel)

    def test_model_config_forwarded(self) -> None:
        """Config values should reach the created model.

        We verify by checking the GNN's ``num_layers`` attribute, which is
        set from the config dict.
        """
        module = _make_module(
            model_config={
                "num_layers": 5,
                "num_terms": 3,
                "feature_dim_in": 16,
                "feature_dim_out": 16,
            }
        )
        assert module.model.num_layers == 5  # pyright: ignore[reportAttributeAccessIssue]

    def test_rejects_non_graph_model(self) -> None:
        """If the registry factory returns a non-GraphModel, a TypeError is raised.

        We patch the registry to return a plain nn.Module and confirm the
        guard clause fires.
        """
        with (
            patch(
                "tmgg.models.factory.ModelRegistry.create",
                return_value=torch.nn.Linear(4, 4),
            ),
            pytest.raises(TypeError, match="expected GraphModel subclass"),
        ):
            _make_module()

    def test_unknown_model_type_raises(self) -> None:
        """An unregistered model_type should surface a ValueError from the registry."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            _make_module(model_type="nonexistent_model_xyz")


class TestConfigureOptimizers:
    """Tests for optimizer and scheduler construction."""

    def test_no_scheduler_returns_optimizer(self) -> None:
        """Without scheduler_config the method should return a bare optimizer.

        This is the simplest configuration path.
        """
        module = _make_module(scheduler_config=None)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)

    def test_adam_default(self) -> None:
        """The default optimizer_type 'adam' should produce an Adam instance."""
        module = _make_module(optimizer_type="adam", scheduler_config=None)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)

    def test_adamw(self) -> None:
        """optimizer_type='adamw' should produce AdamW."""
        module = _make_module(
            optimizer_type="adamw", weight_decay=0.01, scheduler_config=None
        )
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_cosine_scheduler(self) -> None:
        """scheduler_type='cosine' should wrap the optimizer in an lr_scheduler dict."""
        module = _make_module(scheduler_config={"type": "cosine", "T_0": 10})
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "lr_scheduler" in result

    def test_step_scheduler(self) -> None:
        """scheduler_type='step' should produce a StepLR scheduler."""
        module = _make_module(
            scheduler_config={"type": "step", "step_size": 10, "gamma": 0.5}
        )
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        scheduler = result["lr_scheduler"]["scheduler"]  # type: ignore[index]
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_scheduler_info_stored_for_cosine_warmup(self) -> None:
        """cosine_warmup should populate _scheduler_info with SchedulerInfo."""
        module = _make_module(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 100,
                "T_max": 1000,
            }
        )
        # cosine_warmup needs a trainer with max_steps
        mock_trainer = MagicMock()
        mock_trainer.max_steps = 5000
        module._trainer = mock_trainer

        _ = module.configure_optimizers()
        assert module._scheduler_info is not None
        assert module._scheduler_info.T_warmup == 100


class TestTransferBatchToDevice:
    """Tests for GraphData device transfer."""

    def test_moves_graph_data_to_device(self) -> None:
        """GraphData batches should be transferred via GraphData.to().

        We verify that all constituent tensors land on the target device.
        """
        module = _make_module()
        batch = GraphData.from_adjacency(torch.randn(2, 5, 5))
        device = torch.device("cpu")

        result = module.transfer_batch_to_device(batch, device, dataloader_idx=0)

        assert isinstance(result, GraphData)
        assert result.X.device == device
        assert result.E.device == device
        assert result.y.device == device
        assert result.node_mask.device == device

    def test_non_graph_data_falls_through(self) -> None:
        """Non-GraphData batches should delegate to the default Lightning impl.

        This confirms the isinstance guard works correctly.
        """
        module = _make_module()
        plain_tensor = torch.randn(4, 8)
        device = torch.device("cpu")

        result = module.transfer_batch_to_device(plain_tensor, device, dataloader_idx=0)  # type: ignore[arg-type]
        assert isinstance(result, torch.Tensor)


class TestOnFitStart:
    """Tests for parameter-count logging at training start."""

    def test_calls_log_parameter_count(self) -> None:
        """on_fit_start should invoke _log_parameter_count with the model and logger.

        We patch the logging function and check it receives the expected
        arguments: the model, the model name, and the current logger.
        """
        module = _make_module()
        mock_logger = MagicMock()
        module._logger = mock_logger  # type: ignore[assignment]

        with patch(
            "tmgg.experiments._shared_utils.lightning_modules.base_graph_module._log_parameter_count"
        ) as mock_log:
            module.on_fit_start()

            mock_log.assert_called_once_with(
                module.model,
                module.get_model_name(),
                module.logger,
            )


class TestGetModelName:
    """Tests for get_model_name()."""

    def test_returns_model_type(self) -> None:
        """get_model_name should return the model_type stored in hparams.

        This is the default behaviour; subclasses can override for custom names.
        """
        module = _make_module(model_type="gnn")
        assert module.get_model_name() == "gnn"


class TestGetModelConfig:
    """Tests for get_model_config() delegation."""

    def test_delegates_to_model(self) -> None:
        """get_model_config should return whatever model.get_config() produces.

        We verify against the known config shape of the GNN model.
        """
        module = _make_module()
        config = module.get_model_config()

        assert isinstance(config, dict)
        # GNN.get_config() includes these keys
        assert "num_layers" in config
        assert config["num_layers"] == 2

    def test_matches_model_get_config(self) -> None:
        """The return value should be identical to calling model.get_config() directly."""
        module = _make_module()
        assert module.get_model_config() == module.model.get_config()
