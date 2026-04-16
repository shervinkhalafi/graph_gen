"""Tests for GraphTransformer model.

Test rationale: The graph transformer is the core neural network in the
diffusion pipeline. Shape mismatches between its output and the loss/sampling
modules cause silent dimension errors that are hard to trace. These tests
verify shape correctness, output finiteness, and config serialization.
"""

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.digress.extra_features import EigenvectorAugmentation
from tmgg.models.digress.transformer_model import GraphTransformer

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
def model() -> GraphTransformer:
    """Small GraphTransformer for testing."""
    return GraphTransformer(
        n_layers=2,
        input_dims={"X": DX_IN, "E": DE_IN, "y": DY_IN},
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
        output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
    )


@pytest.fixture()
def graph_data() -> GraphData:
    """Random categorical input as GraphData."""
    X = torch.randn(BS, N, DX_IN)
    E = torch.randn(BS, N, N, DE_IN)
    return GraphData(
        y=torch.randn(BS, DY_IN),
        node_mask=torch.ones(BS, N),
        X_class=X,
        E_class=E,
    )


class TestForwardPass:
    """Verify forward pass produces correctly shaped, finite outputs."""

    def test_output_is_graph_data(
        self,
        model: GraphTransformer,
        graph_data: GraphData,
    ) -> None:
        """Forward should return a GraphData instance."""
        out = model(graph_data)
        assert isinstance(out, GraphData)

    def test_output_shapes(
        self,
        model: GraphTransformer,
        graph_data: GraphData,
    ) -> None:
        """X, E, y output shapes must match the configured output_dims."""
        out = model(graph_data)
        assert out.X_class.shape == (BS, N, DX_OUT)
        assert out.E_class.shape == (BS, N, N, DE_OUT)
        assert out.y.shape == (BS, DY_OUT)

    def test_outputs_finite(
        self,
        model: GraphTransformer,
        graph_data: GraphData,
    ) -> None:
        """All output tensors must contain finite values (no NaN or Inf)."""
        out = model(graph_data)
        assert torch.isfinite(out.X_class).all(), "X contains NaN or Inf"
        assert torch.isfinite(out.E_class).all(), "E contains NaN or Inf"
        assert torch.isfinite(out.y).all(), "y contains NaN or Inf"

    def test_partial_node_mask(
        self,
        model: GraphTransformer,
    ) -> None:
        """Forward pass works when some nodes are masked out."""
        node_mask = torch.ones(BS, N)
        node_mask[:, 7:] = 0  # mask out last 3 nodes
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        data = GraphData(
            y=torch.randn(BS, DY_IN),
            node_mask=node_mask,
            X_class=X,
            E_class=E,
        )
        out = model(data)
        assert out.X_class.shape == (BS, N, DX_OUT)
        assert torch.isfinite(out.X_class).all()


class TestGetConfig:
    """Verify get_config returns a complete, correct configuration dict."""

    def test_expected_keys(self, model: GraphTransformer) -> None:
        config = model.get_config()
        expected_keys = {
            "n_layers",
            "input_dims",
            "hidden_mlp_dims",
            "hidden_dims",
            "output_dims",
            "extra_features",
            "use_timestep",
        }
        assert set(config.keys()) == expected_keys

    def test_n_layers_value(self, model: GraphTransformer) -> None:
        config = model.get_config()
        assert config["n_layers"] == 2

    def test_augmentation_defaults(self, model: GraphTransformer) -> None:
        """Vanilla model reports no extra_features and use_timestep=False."""
        config = model.get_config()
        assert config["extra_features"] is None
        assert config["use_timestep"] is False


class TestEigenvectorMode:
    """Verify the eigenvector augmentation path through GraphTransformer.

    When an EigenvectorAugmentation is passed as extra_features, the
    transformer extracts top-k eigenvectors from the noisy adjacency and
    concatenates them with X before the inner transformer. The augmentation
    handles dim adjustment via adjust_dims, so input_dims uses base dims only.
    """

    K = 5

    @pytest.fixture()
    def eigen_model(self) -> GraphTransformer:
        return GraphTransformer(
            n_layers=2,
            input_dims={"X": DX_IN, "E": DE_IN, "y": DY_IN},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
            extra_features=EigenvectorAugmentation(k=self.K),
        )

    def test_forward_shape(self, eigen_model: GraphTransformer) -> None:
        """Eigenvector path produces correct output shapes."""
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        data = GraphData(
            y=torch.randn(BS, DY_IN),
            node_mask=torch.ones(BS, N),
            X_class=X,
            E_class=E,
        )
        out = eigen_model(data)
        assert out.X_class.shape == (BS, N, DX_OUT)
        assert out.E_class.shape == (BS, N, N, DE_OUT)

    def test_config_reports_augmentation(self, eigen_model: GraphTransformer) -> None:
        config = eigen_model.get_config()
        assert config["extra_features"] == "EigenvectorAugmentation"
        assert config["use_timestep"] is False


class TestTimestepMode:
    """Verify the timestep injection path through GraphTransformer.

    When use_timestep=True, the transformer appends the normalised diffusion
    timestep to y before the inner transformer, adding one dimension to y.
    """

    @pytest.fixture()
    def timestep_model(self) -> GraphTransformer:
        return GraphTransformer(
            n_layers=2,
            input_dims={"X": DX_IN, "E": DE_IN, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX_OUT, "E": DE_OUT, "y": 0},
            use_timestep=True,
        )

    def test_forward_with_timestep(self, timestep_model: GraphTransformer) -> None:
        """Forward pass with a timestep tensor produces correct output shapes."""
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        data = GraphData(
            y=torch.zeros(BS, 0),
            node_mask=torch.ones(BS, N),
            X_class=X,
            E_class=E,
        )
        t = torch.rand(BS)
        out = timestep_model(data, t=t)
        assert out.X_class.shape == (BS, N, DX_OUT)

    def test_config_reports_timestep(self, timestep_model: GraphTransformer) -> None:
        config = timestep_model.get_config()
        assert config["use_timestep"] is True
