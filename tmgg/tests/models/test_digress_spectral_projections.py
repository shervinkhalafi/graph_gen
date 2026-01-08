"""Tests for spectral Q/K/V projections in DiGress NodeEdgeBlock.

Test Rationale
--------------
These tests verify that spectral projections can be independently enabled for
query, key, and value projections in NodeEdgeBlock. Spectral projections use
eigenvalue polynomial filters (V @ W where W = sum_l Lambda^l * H[l]) instead
of linear or GNN-based projections.

Key invariants:
- Backwards compatibility: default params produce linear architecture
- Per-projection control: Q/K/V can independently use spectral projections
- Mutual exclusivity: GNN and spectral cannot both be enabled for same projection
- Gradient flow: gradients propagate through spectral polynomial filters
- Shape preservation: output shapes match regardless of projection type
"""

import pytest
import torch

from tmgg.models.digress.transformer_model import (
    GraphTransformer,
    NodeEdgeBlock,
    XEyTransformerLayer,
)
from tmgg.models.layers import SpectralProjectionLayer


class TestNodeEdgeBlockSpectral:
    """Tests for NodeEdgeBlock with spectral projections."""

    def test_backwards_compatibility(self):
        """Default params produce same architecture as original (all Linear)."""
        block = NodeEdgeBlock(dx=128, de=32, dy=64, n_head=4)

        # All projections should be Linear by default
        assert isinstance(block.q, torch.nn.Linear)
        assert isinstance(block.k, torch.nn.Linear)
        assert isinstance(block.v, torch.nn.Linear)

    def test_mutual_exclusivity(self):
        """GNN and spectral cannot both be enabled for the same projection."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeEdgeBlock(
                dx=128,
                de=32,
                dy=64,
                n_head=4,
                use_gnn_q=True,
                use_spectral_q=True,
            )

        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeEdgeBlock(
                dx=128,
                de=32,
                dy=64,
                n_head=4,
                use_gnn_k=True,
                use_spectral_k=True,
            )

        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeEdgeBlock(
                dx=128,
                de=32,
                dy=64,
                n_head=4,
                use_gnn_v=True,
                use_spectral_v=True,
            )

    @pytest.mark.parametrize(
        "use_spectral_q,use_spectral_k,use_spectral_v",
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
    def test_selective_spectral_projections(
        self, use_spectral_q, use_spectral_k, use_spectral_v
    ):
        """Spectral projections can be enabled independently for Q, K, V."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_spectral_q=use_spectral_q,
            use_spectral_k=use_spectral_k,
            use_spectral_v=use_spectral_v,
            spectral_k=16,
        )

        expected_q = SpectralProjectionLayer if use_spectral_q else torch.nn.Linear
        expected_k = SpectralProjectionLayer if use_spectral_k else torch.nn.Linear
        expected_v = SpectralProjectionLayer if use_spectral_v else torch.nn.Linear

        assert isinstance(block.q, expected_q)
        assert isinstance(block.k, expected_k)
        assert isinstance(block.v, expected_v)

    def test_forward_with_spectral_projections(self):
        """Forward pass works with spectral projections enabled."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=True,
            use_spectral_v=True,
            spectral_k=16,
        )

        bs, n = 2, 20
        X = torch.rand(bs, n, 128)
        E = torch.rand(bs, n, n, 32)
        y = torch.rand(bs, 64)
        node_mask = torch.ones(bs, n)

        # Create eigenvector inputs for spectral projections
        V = torch.rand(bs, n, 16)
        Lambda = torch.rand(bs, 16)

        newX, newE, new_y = block(X, E, y, node_mask, V=V, Lambda=Lambda)

        assert newX.shape == X.shape
        assert newE.shape == E.shape
        assert new_y.shape == y.shape

    def test_forward_mixed_projections(self):
        """Forward pass works with mixed Linear and Spectral projections."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=False,  # Linear
            use_spectral_v=True,
            spectral_k=16,
        )

        bs, n = 2, 16
        X = torch.rand(bs, n, 64)
        E = torch.rand(bs, n, n, 16)
        y = torch.rand(bs, 32)
        node_mask = torch.ones(bs, n)

        V = torch.rand(bs, n, 16)
        Lambda = torch.rand(bs, 16)

        newX, newE, new_y = block(X, E, y, node_mask, V=V, Lambda=Lambda)

        assert newX.shape == X.shape
        assert newE.shape == E.shape
        assert new_y.shape == y.shape

    def test_gradient_flow_through_spectral(self):
        """Gradients flow through spectral projections to input and parameters."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=True,
            use_spectral_v=True,
            spectral_k=8,
        )

        bs, n = 2, 12
        X = torch.rand(bs, n, 64, requires_grad=True)
        E = torch.rand(bs, n, n, 16)
        y = torch.rand(bs, 32)
        node_mask = torch.ones(bs, n)

        V = torch.rand(bs, n, 8, requires_grad=True)
        Lambda = torch.rand(bs, 8, requires_grad=True)

        newX, newE, new_y = block(X, E, y, node_mask, V=V, Lambda=Lambda)
        loss = newX.sum() + newE.sum() + new_y.sum()
        loss.backward()

        # Check gradient flows to input
        assert X.grad is not None

        # Check gradient flows to spectral parameters
        assert isinstance(block.q, SpectralProjectionLayer)
        assert isinstance(block.k, SpectralProjectionLayer)
        assert isinstance(block.v, SpectralProjectionLayer)
        assert any(p.grad is not None for p in block.q.parameters())
        assert any(p.grad is not None for p in block.k.parameters())
        assert any(p.grad is not None for p in block.v.parameters())

    def test_configurable_spectral_params(self):
        """spectral_k and spectral_num_terms parameters control layer config."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=True,
            use_spectral_v=True,
            spectral_k=24,
            spectral_num_terms=5,
        )

        assert isinstance(block.q, SpectralProjectionLayer)
        assert block.q.k == 24
        assert block.q.num_terms == 5
        assert len(block.q.H) == 5


class TestXEyTransformerLayerSpectral:
    """Tests for XEyTransformerLayer with spectral configuration."""

    def test_spectral_config_passed_to_block(self):
        """Spectral config is correctly passed to NodeEdgeBlock."""
        layer = XEyTransformerLayer(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=False,
            use_spectral_v=True,
            spectral_k=16,
        )

        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert isinstance(layer.self_attn.k, torch.nn.Linear)
        assert isinstance(layer.self_attn.v, SpectralProjectionLayer)


class TestGraphTransformerSpectral:
    """Tests for full GraphTransformer with spectral projections."""

    def test_spectral_config_in_hidden_dims(self):
        """Spectral config flows through hidden_dims to transformer layers."""
        hidden_dims = {
            "dx": 128,
            "de": 32,
            "dy": 64,
            "n_head": 4,
            "use_spectral_q": True,
            "use_spectral_k": True,
            "use_spectral_v": False,
            "spectral_k": 16,
            "spectral_num_terms": 3,
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
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert isinstance(layer.self_attn.k, SpectralProjectionLayer)
        assert isinstance(layer.self_attn.v, torch.nn.Linear)

        # Verify spectral params
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert layer.self_attn.q.k == 16
        assert layer.self_attn.q.num_terms == 3

    def test_backwards_compatibility_no_spectral_keys(self):
        """GraphTransformer works without spectral keys in hidden_dims."""
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

    def test_forward_pass_with_spectral(self):
        """Full forward pass works with spectral projections."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_spectral_q": True,
            "use_spectral_k": True,
            "use_spectral_v": True,
            "spectral_k": 16,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        # Forward pass with adjacency matrix
        x = torch.rand(2, 20, 20)
        x = (x + x.transpose(-1, -2)) / 2  # Symmetrize for eigendecomposition
        out = model(x)

        assert out.shape == (2, 20, 20)

    def test_eigenvector_extraction_for_spectral(self):
        """Eigenvectors are computed when spectral projections are used."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_spectral_q": True,
            "use_spectral_k": False,
            "use_spectral_v": False,
            "spectral_k": 16,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        # Verify model has spectral projection flag set
        assert model.transformer._use_spectral_projections is True

        # Forward pass should work
        x = torch.rand(2, 20, 20)
        x = (x + x.transpose(-1, -2)) / 2
        out = model(x)

        assert out.shape == (2, 20, 20)

    def test_gradient_flow_full_model(self):
        """Gradients flow through spectral projections in full model."""
        hidden_dims = {
            "dx": 32,
            "de": 8,
            "dy": 16,
            "n_head": 2,
            "use_spectral_q": True,
            "use_spectral_k": True,
            "use_spectral_v": True,
            "spectral_k": 8,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 8, "y": 16},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        x = torch.rand(2, 12, 12, requires_grad=True)
        x = (x + x.transpose(-1, -2)) / 2

        out = model(x)
        loss = out.sum()
        loss.backward()

        # Verify gradients flow to spectral projection parameters
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.self_attn.q.parameters()
        )
