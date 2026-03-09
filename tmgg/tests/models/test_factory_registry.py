"""Tests for the ModelRegistry singleton."""

import pytest
import torch.nn as nn

from tmgg.models.base import BaseModel
from tmgg.models.factory import ModelRegistry, create_model, register_model


class TestModelRegistryAPI:
    """Core class-method API: register, deregister, create, keys, __contains__."""

    def test_keys_returns_registered_names(self) -> None:
        expected = {
            "bilinear",
            "bilinear_mlp",
            "multilayer_bilinear",
            "self_attention",
            "linear_pe",
            "filter_bank",
            "gnn",
            "gnn_sym",
            "node_var_gnn",
            "hybrid",
            "hybrid_sequential",
            "linear",
            "mlp",
            "graph_transformer",
        }
        assert expected.issubset(set(ModelRegistry.keys()))

    def test_contains(self) -> None:
        assert "linear_pe" in ModelRegistry
        assert "nonexistent_xyz_123" not in ModelRegistry

    def test_register_and_deregister(self) -> None:
        name = "__test_register_deregister__"

        def _factory(config: dict) -> nn.Module:
            from tmgg.models.baselines import LinearBaseline

            return LinearBaseline(max_nodes=10)

        ModelRegistry.register(name, _factory)
        assert name in ModelRegistry

        ModelRegistry.deregister(name)
        assert name not in ModelRegistry

    def test_register_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            ModelRegistry.register("linear_pe", lambda cfg: nn.Linear(1, 1))

    def test_deregister_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            ModelRegistry.deregister("nonexistent_xyz_123")

    def test_create_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            ModelRegistry.create("nonexistent_model_xyz", {})

    def test_create_enforces_base_model_contract(self) -> None:
        """Factory that returns a plain nn.Module (not BaseModel) is rejected."""
        name = "__test_bad_factory__"
        ModelRegistry.register(name, lambda cfg: nn.Linear(1, 1))
        try:
            with pytest.raises(TypeError, match="expected BaseModel subclass"):
                ModelRegistry.create(name, {})
        finally:
            ModelRegistry.deregister(name)

    def test_create_returns_base_model(self) -> None:
        model = ModelRegistry.create("linear_pe", {})
        assert isinstance(model, BaseModel)


class TestBackwardCompatWrappers:
    """Module-level create_model and register_model still work."""

    def test_create_model_delegates(self) -> None:
        model = create_model("linear_pe", {})
        assert isinstance(model, BaseModel)

    def test_register_model_decorator(self) -> None:
        name = "__test_decorator__"

        @register_model(name)
        def _make_test(config: dict) -> nn.Module:
            from tmgg.models.baselines import LinearBaseline

            return LinearBaseline(max_nodes=10)

        assert name in ModelRegistry
        ModelRegistry.deregister(name)
