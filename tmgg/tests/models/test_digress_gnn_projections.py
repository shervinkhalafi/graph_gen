"""Tests for GNN-based Q/K/V projections in DiGress NodeEdgeBlock.

Test Rationale
--------------
These tests verify that GNN projections can be independently enabled for the
query, key, and value projections in NodeEdgeBlock. The goal is to replace
standard linear projections with polynomial graph convolutions that incorporate
graph structure into the attention mechanism.

Key invariants:
- Backwards compatibility: default params produce same architecture as original
- Per-projection control: each of Q/K/V can independently use GNN or Linear
- Gradient flow: gradients propagate through GNN convolutions
- Shape preservation: output shapes match regardless of projection type
"""

import pytest
import torch

from tmgg.models.digress.transformer_model import (
    GraphTransformer,
    NodeEdgeBlock,
    XEyTransformerLayer,
)
from tmgg.models.layers import BareGraphConvolutionLayer


class TestBareGraphConvolutionLayer:
    """Tests for the BareGraphConvolutionLayer used in attention projections."""

    def test_output_shape(self):
        """Output shape matches input shape (batch, n, channels)."""
        layer = BareGraphConvolutionLayer(num_terms=2, num_channels=32)
        A = torch.rand(4, 16, 16)
        A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
        X = torch.rand(4, 16, 32)

        Y = layer(A, X)
        assert Y.shape == X.shape

    def test_gradient_flow(self):
        """Gradients flow through the layer to inputs and parameters."""
        layer = BareGraphConvolutionLayer(num_terms=2, num_channels=32)
        A = torch.rand(4, 16, 16, requires_grad=True)
        X = torch.rand(4, 16, 32, requires_grad=True)

        Y = layer(A, X)
        loss = Y.sum()
        loss.backward()

        assert X.grad is not None
        assert layer.H.grad is not None

    def test_polynomial_terms(self):
        """Parameter shape matches num_terms + 1 (including identity)."""
        layer = BareGraphConvolutionLayer(num_terms=3, num_channels=64)
        # H should have shape (num_terms+1, channels, channels)
        assert layer.H.shape == (4, 64, 64)

    def test_zero_adjacency_identity(self):
        """With zero adjacency, only the identity term (H[0]) applies."""
        layer = BareGraphConvolutionLayer(num_terms=2, num_channels=16)
        A = torch.zeros(2, 8, 8)  # No edges
        X = torch.rand(2, 8, 16)

        Y = layer(A, X)
        # Should just be X @ H[0]
        expected = X @ layer.H[0]
        assert torch.allclose(Y, expected, atol=1e-5)


class TestNodeEdgeBlockGNN:
    """Tests for NodeEdgeBlock with GNN projections."""

    def test_backwards_compatibility(self):
        """Default params produce same architecture as original (all Linear)."""
        block = NodeEdgeBlock(dx=128, de=32, dy=64, n_head=4)

        # All projections should be Linear
        assert isinstance(block.q, torch.nn.Linear)
        assert isinstance(block.k, torch.nn.Linear)
        assert isinstance(block.v, torch.nn.Linear)

    @pytest.mark.parametrize(
        "use_gnn_q,use_gnn_k,use_gnn_v",
        [
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ],
    )
    def test_selective_gnn_projections(self, use_gnn_q, use_gnn_k, use_gnn_v):
        """GNN projections can be enabled independently for Q, K, V."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_gnn_q=use_gnn_q,
            use_gnn_k=use_gnn_k,
            use_gnn_v=use_gnn_v,
        )

        expected_q = BareGraphConvolutionLayer if use_gnn_q else torch.nn.Linear
        expected_k = BareGraphConvolutionLayer if use_gnn_k else torch.nn.Linear
        expected_v = BareGraphConvolutionLayer if use_gnn_v else torch.nn.Linear

        assert isinstance(block.q, expected_q)
        assert isinstance(block.k, expected_k)
        assert isinstance(block.v, expected_v)

    def test_forward_with_gnn_projections(self):
        """Forward pass works with GNN projections enabled."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=True,
            use_gnn_v=True,
        )

        bs, n = 2, 16
        X = torch.rand(bs, n, 128)
        E = torch.rand(bs, n, n, 32)
        y = torch.rand(bs, 64)
        node_mask = torch.ones(bs, n)

        newX, newE, new_y = block(X, E, y, node_mask)

        assert newX.shape == X.shape
        assert newE.shape == E.shape
        assert new_y.shape == y.shape

    def test_forward_mixed_projections(self):
        """Forward pass works with mixed Linear and GNN projections."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=False,  # Linear
            use_gnn_v=True,
        )

        bs, n = 2, 12
        X = torch.rand(bs, n, 64)
        E = torch.rand(bs, n, n, 16)
        y = torch.rand(bs, 32)
        node_mask = torch.ones(bs, n)

        newX, newE, new_y = block(X, E, y, node_mask)

        assert newX.shape == X.shape
        assert newE.shape == E.shape
        assert new_y.shape == y.shape

    def test_gradient_flow_through_gnn(self):
        """Gradients flow through GNN projections to input and parameters."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=True,
            use_gnn_v=True,
        )

        bs, n = 2, 8
        X = torch.rand(bs, n, 64, requires_grad=True)
        E = torch.rand(bs, n, n, 16)
        y = torch.rand(bs, 32)
        node_mask = torch.ones(bs, n)

        newX, newE, new_y = block(X, E, y, node_mask)
        loss = newX.sum() + newE.sum() + new_y.sum()
        loss.backward()

        # Check gradient flows to input
        assert X.grad is not None

        # Check gradient flows to GNN parameters
        assert block.q.H.grad is not None  # pyright: ignore[reportAttributeAccessIssue]
        assert block.k.H.grad is not None  # pyright: ignore[reportAttributeAccessIssue]
        assert block.v.H.grad is not None  # pyright: ignore[reportAttributeAccessIssue]

    def test_configurable_num_terms(self):
        """gnn_num_terms parameter controls polynomial degree."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=True,
            use_gnn_v=True,
            gnn_num_terms=3,
        )

        # H should have shape (num_terms+1, channels, channels)
        assert block.q.H.shape == (4, 64, 64)  # pyright: ignore[reportAttributeAccessIssue]
        assert block.k.H.shape == (4, 64, 64)  # pyright: ignore[reportAttributeAccessIssue]
        assert block.v.H.shape == (4, 64, 64)  # pyright: ignore[reportAttributeAccessIssue]


class TestXEyTransformerLayerGNN:
    """Tests for XEyTransformerLayer with GNN configuration."""

    def test_gnn_config_passed_to_block(self):
        """GNN config is correctly passed to NodeEdgeBlock."""
        layer = XEyTransformerLayer(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=False,
            use_gnn_v=True,
        )

        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert isinstance(layer.self_attn.k, torch.nn.Linear)
        assert isinstance(layer.self_attn.v, BareGraphConvolutionLayer)


class TestGraphTransformerGNN:
    """Tests for full GraphTransformer with GNN config in hidden_dims."""

    def test_gnn_config_in_hidden_dims(self):
        """GNN config flows through hidden_dims to transformer layers."""
        hidden_dims = {
            "dx": 128,
            "de": 32,
            "dy": 64,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": True,
            "use_gnn_v": False,
            "gnn_num_terms": 3,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 64, "E": 32, "y": 64},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        # Check first layer's attention block
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert isinstance(layer.self_attn.k, BareGraphConvolutionLayer)
        assert isinstance(layer.self_attn.v, torch.nn.Linear)

        # Verify num_terms
        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert layer.self_attn.q.H.shape[0] == 4  # num_terms + 1

    def test_backwards_compatibility_no_gnn_keys(self):
        """GraphTransformer works without GNN keys in hidden_dims."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        # All projections should be Linear
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, torch.nn.Linear)
        assert isinstance(layer.self_attn.k, torch.nn.Linear)
        assert isinstance(layer.self_attn.v, torch.nn.Linear)

    def test_forward_pass_with_gnn(self):
        """Full forward pass works with GNN projections."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": True,
            "use_gnn_v": True,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        x = torch.rand(2, 12, 12)  # Adjacency matrix
        out = model(x)

        assert out.shape == (2, 12, 12)

    def test_forward_pass_with_eigenvectors_and_gnn(self):
        """Forward pass works with both eigenvectors and GNN projections."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": False,
            "use_gnn_v": True,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 16, "E": 1, "y": 0},  # X dim matches k
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
            use_eigenvectors=True,
            k=16,
        )

        x = torch.rand(2, 20, 20)  # Adjacency matrix
        x = (x + x.transpose(-1, -2)) / 2  # Symmetrize for eigenvector extraction
        out = model(x)

        assert out.shape == (2, 20, 20)
