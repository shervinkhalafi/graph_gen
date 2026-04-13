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
        result = model(GraphData.from_edge_state(A))
        assert isinstance(result, GraphData)
        assert result.to_edge_state().shape == (2, 20, 20)

    def test_forward_avoids_binary_projection(self, monkeypatch):
        """Spectral denoising should stay in edge-state space."""
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        def _raise(*_args, **_kwargs):
            raise AssertionError("binary topology should not be used here")

        monkeypatch.setattr(GraphData, "to_binary_adjacency", _raise)

        model = SelfAttentionDenoiser(k=4, d_k=8)
        data = GraphData.from_edge_state(torch.randn(1, 10, 10))
        result = model(data)
        assert result.to_edge_state().shape == (1, 10, 10)

    def test_unbatched_input(self):
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)
        A = torch.randn(10, 10)
        A = (A + A.T) / 2
        result = model(GraphData.from_edge_state(A))
        assert isinstance(result, GraphData)
        # Unbatched edge-state input may stay unbatched or be rebatched.
        adj = result.to_edge_state()
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
        data = GraphData.from_edge_state(A)
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

    def test_gradient_flows_to_parameters(self):
        """Verify gradients propagate to model parameters through forward pass."""
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)
        A = torch.randn(2, 10, 10)
        A_sym = (A + A.transpose(-1, -2)) / 2
        data = GraphData.from_edge_state(A_sym)
        result = model(data)
        loss = result.to_edge_state().sum()
        loss.backward()
        # Check parameter gradients (not input gradients, since GraphData wrapping
        # is not the quantity we are differentiating through here).
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
        result = model(GraphData.from_edge_state(A))
        assert result.to_edge_state().shape == (2, 20, 20)
