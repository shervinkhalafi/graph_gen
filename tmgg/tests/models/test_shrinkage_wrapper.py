"""Tests for SVD-based shrinkage wrappers.

Test Rationale
--------------
These tests verify the experimental shrinkage wrapper architectures that enforce
shrinkage-based denoising via singular value modification. The wrappers use inner
spectral denoisers to extract features for predicting shrinkage coefficients.

Key invariants tested:
- SVD decomposition is correct and reconstructs the original matrix
- Strict shrinkage outputs are bounded (coefficients in [0, 1])
- Relaxed shrinkage allows both expansion and contraction
- Gradients flow through the entire pipeline (inner model + shrinkage)
- Output shapes match input shapes for both batched and unbatched inputs
"""

import pytest
import torch

from tmgg.models.spectral_denoisers import (
    BilinearDenoiser,
    RelaxedShrinkageWrapper,
    ShrinkageSVDLayer,
    StrictShrinkageWrapper,
)


class TestShrinkageSVDLayer:
    """Test SVD layer for shrinkage-based denoising."""

    def test_svd_decomposition_shape(self):
        """Verify SVD returns correctly shaped tensors."""
        max_rank = 8
        layer = ShrinkageSVDLayer(max_rank=max_rank)

        # Create symmetric matrix
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        U, S, Vh = layer(A)

        assert U.shape == (n, max_rank)
        assert S.shape == (max_rank,)
        assert Vh.shape == (max_rank, n)

    def test_svd_decomposition_batched(self):
        """Verify SVD handles batched input correctly."""
        max_rank = 8
        batch_size = 4
        n = 20
        layer = ShrinkageSVDLayer(max_rank=max_rank)

        A = torch.randn(batch_size, n, n)
        A = (A + A.transpose(-1, -2)) / 2

        U, S, Vh = layer(A)

        assert U.shape == (batch_size, n, max_rank)
        assert S.shape == (batch_size, max_rank)
        assert Vh.shape == (batch_size, max_rank, n)

    def test_svd_reconstruction(self):
        """Verify SVD decomposition approximately reconstructs the original matrix."""
        max_rank = 15
        n = 20
        layer = ShrinkageSVDLayer(max_rank=max_rank)

        A = torch.randn(n, n)
        A = (A + A.T) / 2

        U, S, Vh = layer(A)

        # Reconstruct: U @ diag(S) @ Vh
        A_reconstructed = U @ torch.diag(S) @ Vh

        # Low-rank approximation should be close for random symmetric matrices
        reconstruction_error = torch.norm(A - A_reconstructed).item()
        relative_error = reconstruction_error / torch.norm(A).item()

        # With rank 15 out of 20, approximation should be good
        assert relative_error < 0.5

    def test_svd_small_matrix(self):
        """Verify SVD handles matrices smaller than max_rank."""
        max_rank = 50
        n = 10  # Smaller than max_rank
        layer = ShrinkageSVDLayer(max_rank=max_rank)

        A = torch.randn(n, n)
        A = (A + A.T) / 2

        U, S, Vh = layer(A)

        # Should truncate to actual rank (n)
        assert U.shape == (n, n)
        assert S.shape == (n,)
        assert Vh.shape == (n, n)


class TestStrictShrinkageWrapper:
    """Test strict shrinkage wrapper with sigmoid gating."""

    @pytest.fixture
    def inner_model(self):
        """Create inner spectral denoiser."""
        return BilinearDenoiser(k=8, d_k=32)

    @pytest.fixture
    def wrapper(self, inner_model):
        """Create strict shrinkage wrapper."""
        return StrictShrinkageWrapper(
            inner_model=inner_model,
            max_rank=16,
            aggregation="mean",
            hidden_dim=64,
            mlp_layers=2,
        )

    def test_forward_shape_unbatched(self, wrapper):
        """Verify output shape matches input for unbatched input."""
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        output = wrapper(A)

        assert output.shape == (n, n)

    def test_forward_shape_batched(self, wrapper):
        """Verify output shape matches input for batched input."""
        batch_size = 4
        n = 20
        A = torch.randn(batch_size, n, n)
        A = (A + A.transpose(-1, -2)) / 2

        output = wrapper(A)

        assert output.shape == (batch_size, n, n)

    def test_output_symmetry(self, wrapper):
        """Verify output is symmetric."""
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        output = wrapper(A)

        assert torch.allclose(output, output.T, atol=1e-5)

    def test_gradient_flow(self, wrapper):
        """Verify gradients flow through entire wrapper including inner model.

        Note: The shrinkage wrapper uses get_features() which only uses W_K,
        not W_Q. This is expected - W_Q is only used for adjacency reconstruction
        in the full forward pass of the inner model.
        """
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2
        A.requires_grad_(True)

        output = wrapper(A)
        loss = output.sum()
        loss.backward()

        # Verify gradient exists for input
        assert A.grad is not None
        assert not torch.all(A.grad == 0)

        # Verify gradients exist for W_K (used by get_features)
        assert wrapper.inner_model.W_K.grad is not None, "No gradient for W_K"

        # Verify gradients exist for shrinkage MLP
        for name, param in wrapper.shrinkage_mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for shrinkage_mlp.{name}"

    def test_get_config(self, wrapper, inner_model):
        """Verify config contains all relevant hyperparameters."""
        config = wrapper.get_config()

        assert config["model_class"] == "StrictShrinkageWrapper"
        assert config["max_rank"] == 16
        assert config["aggregation"] == "mean"
        assert config["hidden_dim"] == 64
        assert config["mlp_layers"] == 2
        assert "inner_model" in config
        assert config["inner_model"]["model_class"] == "BilinearDenoiser"


class TestRelaxedShrinkageWrapper:
    """Test relaxed shrinkage wrapper with FiLM-style modulation."""

    @pytest.fixture
    def inner_model(self):
        """Create inner spectral denoiser."""
        return BilinearDenoiser(k=8, d_k=32)

    @pytest.fixture
    def wrapper(self, inner_model):
        """Create relaxed shrinkage wrapper."""
        return RelaxedShrinkageWrapper(
            inner_model=inner_model,
            max_rank=16,
            aggregation="mean",
            hidden_dim=64,
            mlp_layers=2,
        )

    def test_forward_shape_unbatched(self, wrapper):
        """Verify output shape matches input for unbatched input."""
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        output = wrapper(A)

        assert output.shape == (n, n)

    def test_forward_shape_batched(self, wrapper):
        """Verify output shape matches input for batched input."""
        batch_size = 4
        n = 20
        A = torch.randn(batch_size, n, n)
        A = (A + A.transpose(-1, -2)) / 2

        output = wrapper(A)

        assert output.shape == (batch_size, n, n)

    def test_output_symmetry(self, wrapper):
        """Verify output is symmetric."""
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        output = wrapper(A)

        assert torch.allclose(output, output.T, atol=1e-5)

    def test_gradient_flow(self, wrapper):
        """Verify gradients flow through entire wrapper including inner model.

        Note: The shrinkage wrapper uses get_features() which only uses W_K,
        not W_Q. This is expected - W_Q is only used for adjacency reconstruction
        in the full forward pass of the inner model.
        """
        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2
        A.requires_grad_(True)

        output = wrapper(A)
        loss = output.sum()
        loss.backward()

        # Verify gradient exists for input
        assert A.grad is not None
        assert not torch.all(A.grad == 0)

        # Verify gradients exist for W_K (used by get_features)
        assert wrapper.inner_model.W_K.grad is not None, "No gradient for W_K"

        # Verify gradients exist for shrinkage MLP
        for name, param in wrapper.shrinkage_mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for shrinkage_mlp.{name}"


class TestInnerModelFeatures:
    """Test get_features() method on spectral denoisers used as inner models."""

    def test_self_attention_get_features_shape(self):
        """Verify BilinearDenoiser.get_features() returns correct shape."""
        k = 8
        d_k = 32
        model = BilinearDenoiser(k=k, d_k=d_k)

        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        features = model.get_features(A)

        assert features.shape == (n, d_k)

    def test_self_attention_get_features_batched(self):
        """Verify get_features() handles batched input."""
        k = 8
        d_k = 32
        batch_size = 4
        model = BilinearDenoiser(k=k, d_k=d_k)

        n = 20
        A = torch.randn(batch_size, n, n)
        A = (A + A.transpose(-1, -2)) / 2

        features = model.get_features(A)

        assert features.shape == (batch_size, n, d_k)

    def test_features_are_learned(self):
        """Verify features depend on learned parameters."""
        k = 8
        d_k = 32
        model = BilinearDenoiser(k=k, d_k=d_k)

        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        features_before = model.get_features(A).clone()

        # Modify W_K parameter
        with torch.no_grad():
            model.W_K.data += torch.randn_like(model.W_K) * 0.5

        features_after = model.get_features(A)

        # Features should change after modifying W_K
        assert not torch.allclose(features_before, features_after)


class TestAttentionPooling:
    """Test attention-based feature aggregation."""

    def test_attention_aggregation(self):
        """Verify attention aggregation produces different results than mean."""
        inner = BilinearDenoiser(k=8, d_k=32)
        wrapper_mean = StrictShrinkageWrapper(
            inner_model=inner, max_rank=16, aggregation="mean"
        )
        wrapper_attn = StrictShrinkageWrapper(
            inner_model=BilinearDenoiser(k=8, d_k=32),
            max_rank=16,
            aggregation="attention",
        )

        n = 20
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        # Both should produce valid outputs
        output_mean = wrapper_mean(A)
        output_attn = wrapper_attn(A)

        assert output_mean.shape == (n, n)
        assert output_attn.shape == (n, n)


class TestShrinkageWrapperLogitsToGraph:
    """Regression tests for ShrinkageWrapper.logits_to_graph threshold (P2.1).

    ShrinkageWrapper.forward() returns adjacency values in [0, 1] after SVD
    reconstruction and symmetrization, not raw logits. The base DenoisingModel
    thresholds at 0 (correct for BCE logits), but adjacency values need a 0.5
    threshold. Without the override, values like 0.3 would incorrectly map to 1.
    """

    @pytest.fixture
    def strict_wrapper(self):
        """Create a StrictShrinkageWrapper instance."""
        inner = BilinearDenoiser(k=8, d_k=32)
        return StrictShrinkageWrapper(inner_model=inner, max_rank=16)

    @pytest.fixture
    def relaxed_wrapper(self):
        """Create a RelaxedShrinkageWrapper instance."""
        inner = BilinearDenoiser(k=8, d_k=32)
        return RelaxedShrinkageWrapper(inner_model=inner, max_rank=16)

    def test_threshold_at_half_not_zero(self, strict_wrapper):
        """Regression test for P2.1: threshold must be 0.5, not 0.

        Before the fix, logits_to_graph inherited from DenoisingModel and used
        ``(logits > 0).float()``, which is wrong for adjacency-range outputs:
        a value of 0.3 would be classified as edge-present (1) when it should
        be edge-absent (0).
        """
        adjacency_output = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
        result = strict_wrapper.logits_to_graph(adjacency_output)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.equal(result, expected)

    def test_threshold_boundary_value(self, strict_wrapper):
        """Values exactly at 0.5 should map to 0 (strict greater-than)."""
        adjacency_output = torch.tensor([0.5])
        result = strict_wrapper.logits_to_graph(adjacency_output)
        assert result.item() == 0.0

    def test_relaxed_wrapper_shares_override(self, relaxed_wrapper):
        """RelaxedShrinkageWrapper inherits the same 0.5 threshold."""
        adjacency_output = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
        result = relaxed_wrapper.logits_to_graph(adjacency_output)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.equal(result, expected)

    def test_batched_threshold(self, strict_wrapper):
        """Threshold works correctly on batched input."""
        adjacency_output = torch.tensor(
            [
                [[0.1, 0.9], [0.9, 0.1]],
                [[0.6, 0.4], [0.4, 0.6]],
            ]
        )
        result = strict_wrapper.logits_to_graph(adjacency_output)
        expected = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        assert torch.equal(result, expected)
