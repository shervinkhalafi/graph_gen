"""Tests for the correct self-attention denoiser (Vaswani et al. 2017).

Rationale: The audit found the old 'SelfAttentionDenoiser' was bilinear
similarity (QK^T) without softmax or values. The replacement must have:
1. Three projections (Q, K, V)
2. Softmax-normalized attention weights (rows sum to 1)
3. Value aggregation (attention-weighted sum of V)
4. A readout layer that maps node embeddings -> adjacency logits
"""

import torch

from tmgg.data.datasets.graph_types import GraphData


class TestSelfAttentionDenoiser:
    def test_importable(self):
        # Must NOT be an alias for BilinearDenoiser anymore
        from tmgg.models.spectral_denoisers import (
            BilinearDenoiser,
            SelfAttentionDenoiser,
        )

        assert SelfAttentionDenoiser is not BilinearDenoiser

    def test_output_shape(self):
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=8, d_k=16)
        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2
        result = model(GraphData.from_adjacency(A))
        assert isinstance(result, GraphData)
        assert result.to_adjacency().shape == (2, 20, 20)

    def test_unbatched_input(self):
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)
        A = torch.randn(10, 10)
        A = (A + A.T) / 2
        result = model(GraphData.from_adjacency(A))
        assert isinstance(result, GraphData)
        # Unbatched: from_adjacency squeezes batch, but model may add it back
        adj = result.to_adjacency()
        assert adj.shape[-2:] == (10, 10)

    def test_has_value_projection(self):
        """Must have W_V parameter (not just W_Q and W_K)."""
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=8, d_k=16)
        param_names = {name for name, _ in model.named_parameters()}
        assert any(
            "W_V" in n or "W_v" in n for n in param_names
        ), f"No value projection found. Parameters: {param_names}"

    def test_attention_weights_sum_to_one(self):
        """Internal attention weights must be softmax-normalized (rows sum to 1)."""
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)

        assert hasattr(model, "get_attention_weights"), (
            "SelfAttentionDenoiser must expose get_attention_weights() "
            "for interpretability"
        )
        A = torch.randn(2, 10, 10)
        A = (A + A.transpose(-1, -2)) / 2
        data = GraphData.from_adjacency(A)
        attn = model.get_attention_weights(data)
        assert attn.shape == (2, 10, 10)

        # Rows must sum to ~1.0 (softmax property)
        row_sums = attn.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        )
        # All values non-negative (softmax property)
        assert (attn >= 0).all()

    def test_factory_creates_correct_self_attention(self):
        """Factory key 'self_attention' must return the new correct class."""
        from tmgg.models.factory import create_model

        model = create_model("self_attention", {"k": 4, "d_k": 8})
        assert type(model).__name__ == "SelfAttentionDenoiser"

    def test_factory_bilinear_still_works(self):
        """Factory key 'bilinear' must still return BilinearDenoiser."""
        from tmgg.models.factory import create_model

        model = create_model("bilinear", {"k": 4, "d_k": 8})
        assert type(model).__name__ == "BilinearDenoiser"

    def test_gradient_flows_to_parameters(self):
        """Verify gradients propagate to model parameters through forward pass."""
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)
        A = torch.randn(2, 10, 10)
        A_sym = (A + A.transpose(-1, -2)) / 2
        data = GraphData.from_adjacency(A_sym)
        result = model(data)
        # Use only the edge-probability channel for loss. Summing both channels
        # gives a constant (E[...,0] + E[...,1] = 1 per position) with zero gradient.
        loss = result.E[..., 1].sum()
        loss.backward()
        # Check parameter gradients (not input gradients, since from_adjacency is non-differentiable)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "No parameter received gradients"

    def test_with_pearl_embeddings(self):
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(
            k=8,
            d_k=16,
            embedding_source="pearl_random",
            pearl_num_layers=2,
        )
        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2
        result = model(GraphData.from_adjacency(A))
        assert result.to_adjacency().shape == (2, 20, 20)
