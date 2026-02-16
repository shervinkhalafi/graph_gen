"""Test the model registry pattern."""

from tmgg.models.factory import MODEL_REGISTRY, create_model, register_model


def test_registry_is_dict():
    assert isinstance(MODEL_REGISTRY, dict)


def test_all_known_types_registered():
    expected = {
        "bilinear",
        "bilinear_mlp",
        "self_attention",
        "linear_pe",
        "filter_bank",
        "gnn",
        "GNN",
        "gnn_sym",
        "GNNSymmetric",
        "NodeVarGNN",
        "hybrid",
        "hybrid_sequential",
        "linear",
        "mlp",
    }
    assert expected.issubset(set(MODEL_REGISTRY.keys()))


def test_register_model_decorator():
    @register_model("test_dummy")
    def _make_test(config):
        return "dummy"

    assert "test_dummy" in MODEL_REGISTRY
    result = create_model("test_dummy", {})
    assert result == "dummy"
    del MODEL_REGISTRY["test_dummy"]


def test_unknown_type_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown model_type"):
        create_model("nonexistent_model_xyz", {})
