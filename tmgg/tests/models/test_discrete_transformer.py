"""Tests for DiscreteGraphTransformer wrapper.

Test rationale: The discrete transformer is the core neural network in the
diffusion pipeline. Shape mismatches between its output and the loss/sampling
modules cause silent dimension errors that are hard to trace. These tests
verify shape correctness, output finiteness, and config serialization.
"""

import pytest
import torch

from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
from tmgg.models.digress.transformer_model import GraphFeatures

BS = 4
N = 10
DX_IN = 2
DE_IN = 2
DY_IN = 1
# Output dims must be <= input dims due to the skip connection in
# _GraphTransformer (X_to_out = X[..., :out_dim_X]).
DX_OUT = 2
DE_OUT = 2
DY_OUT = 1


@pytest.fixture()
def model() -> DiscreteGraphTransformer:
    """Small DiscreteGraphTransformer for testing."""
    return DiscreteGraphTransformer(
        n_layers=2,
        input_dims={"X": DX_IN, "E": DE_IN, "y": DY_IN},
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
        output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
    )


@pytest.fixture()
def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random categorical input tensors."""
    X = torch.randn(BS, N, DX_IN)
    E = torch.randn(BS, N, N, DE_IN)
    y = torch.randn(BS, DY_IN)
    node_mask = torch.ones(BS, N)
    return X, E, y, node_mask


class TestForwardPass:
    """Verify forward pass produces correctly shaped, finite outputs."""

    def test_output_is_graph_features(
        self,
        model: DiscreteGraphTransformer,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Forward should return a GraphFeatures named tuple."""
        X, E, y, node_mask = inputs
        out = model(X, E, y, node_mask)
        assert isinstance(out, GraphFeatures)

    def test_output_shapes(
        self,
        model: DiscreteGraphTransformer,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """X, E, y output shapes must match the configured output_dims."""
        X, E, y, node_mask = inputs
        out = model(X, E, y, node_mask)
        assert out.X.shape == (BS, N, DX_OUT)
        assert out.E.shape == (BS, N, N, DE_OUT)
        assert out.y.shape == (BS, DY_OUT)

    def test_outputs_finite(
        self,
        model: DiscreteGraphTransformer,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """All output tensors must contain finite values (no NaN or Inf)."""
        X, E, y, node_mask = inputs
        out = model(X, E, y, node_mask)
        assert torch.isfinite(out.X).all(), "X contains NaN or Inf"
        assert torch.isfinite(out.E).all(), "E contains NaN or Inf"
        assert torch.isfinite(out.y).all(), "y contains NaN or Inf"

    def test_partial_node_mask(
        self,
        model: DiscreteGraphTransformer,
    ) -> None:
        """Forward pass works when some nodes are masked out."""
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        y = torch.randn(BS, DY_IN)
        node_mask = torch.ones(BS, N)
        node_mask[:, 7:] = 0  # mask out last 3 nodes
        out = model(X, E, y, node_mask)
        assert out.X.shape == (BS, N, DX_OUT)
        assert torch.isfinite(out.X).all()


class TestGetConfig:
    """Verify get_config returns a complete, correct configuration dict."""

    def test_expected_keys(self, model: DiscreteGraphTransformer) -> None:
        config = model.get_config()
        expected_keys = {
            "model_class",
            "n_layers",
            "input_dims",
            "hidden_mlp_dims",
            "hidden_dims",
            "output_dims",
            "use_eigenvectors",
            "k",
        }
        assert set(config.keys()) == expected_keys

    def test_model_class_value(self, model: DiscreteGraphTransformer) -> None:
        config = model.get_config()
        assert config["model_class"] == "DiscreteGraphTransformer"

    def test_n_layers_value(self, model: DiscreteGraphTransformer) -> None:
        config = model.get_config()
        assert config["n_layers"] == 2

    def test_eigenvector_defaults(self, model: DiscreteGraphTransformer) -> None:
        """Vanilla model reports use_eigenvectors=False and k=None."""
        config = model.get_config()
        assert config["use_eigenvectors"] is False
        assert config["k"] is None


class TestEigenvectorMode:
    """Verify the eigenvector embedding path through DiscreteGraphTransformer.

    When use_eigenvectors=True, the transformer extracts top-k eigenvectors
    from the noisy adjacency (E.argmax(-1) > 0) and concatenates them with X
    before the inner transformer. input_dims["X"] must account for the extra k
    dimensions (categorical_dim + k).
    """

    K = 5
    DX_EIGEN = DX_IN + K  # 2 one-hot + 5 eigenvectors

    @pytest.fixture()
    def eigen_model(self) -> DiscreteGraphTransformer:
        return DiscreteGraphTransformer(
            n_layers=2,
            input_dims={"X": self.DX_EIGEN, "E": DE_IN, "y": DY_IN},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
            use_eigenvectors=True,
            k=self.K,
        )

    def test_forward_shape(self, eigen_model: DiscreteGraphTransformer) -> None:
        """Eigenvector path produces correct output shapes."""
        X = torch.randn(
            BS, N, DX_IN
        )  # only categorical dims, eigenvecs added internally
        E = torch.randn(BS, N, N, DE_IN)
        y = torch.randn(BS, DY_IN)
        node_mask = torch.ones(BS, N)
        out = eigen_model(X, E, y, node_mask)
        assert out.X.shape == (BS, N, DX_OUT)
        assert out.E.shape == (BS, N, N, DE_OUT)

    def test_config_reports_eigenvectors(
        self, eigen_model: DiscreteGraphTransformer
    ) -> None:
        config = eigen_model.get_config()
        assert config["use_eigenvectors"] is True
        assert config["k"] == self.K

    def test_k_required_when_enabled(self) -> None:
        """Omitting k with use_eigenvectors=True must raise ValueError."""
        with pytest.raises(ValueError, match="k must be specified"):
            DiscreteGraphTransformer(
                n_layers=2,
                input_dims={"X": 7, "E": 2, "y": 1},
                hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
                hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
                output_dims={"X": 2, "E": 2, "y": 1},
                use_eigenvectors=True,
                # k omitted
            )
