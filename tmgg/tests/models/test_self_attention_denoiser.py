"""Tests for the correct self-attention denoiser (Vaswani et al. 2017).

Rationale: The audit found the old 'SelfAttentionDenoiser' was bilinear
similarity (QK^T) without softmax or values. The replacement must have:
1. Three projections (Q, K, V)
2. Softmax-normalized attention weights (rows sum to 1)
3. Value aggregation (attention-weighted sum of V)
4. A readout layer that maps node embeddings -> adjacency logits
"""

import torch


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
        out = model(A)
        assert out.shape == (2, 20, 20)

    def test_unbatched_input(self):
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)
        A = torch.randn(10, 10)
        A = (A + A.T) / 2
        out = model(A)
        assert out.shape == (10, 10)

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
        attn = model.get_attention_weights(A)
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

    def test_gradient_flows(self):
        """Verify gradients propagate through the full forward pass."""
        from tmgg.models.spectral_denoisers import SelfAttentionDenoiser

        model = SelfAttentionDenoiser(k=4, d_k=8)
        A = torch.randn(2, 10, 10, requires_grad=True)
        A_sym = (A + A.transpose(-1, -2)) / 2
        out = model(A_sym)
        loss = out.sum()
        loss.backward()
        assert A.grad is not None
        assert A.grad.abs().sum() > 0

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
        out = model(A)
        assert out.shape == (2, 20, 20)
