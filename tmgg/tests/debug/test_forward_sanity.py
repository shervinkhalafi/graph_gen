"""Forward pass sanity tests for spectral denoising models.

These tests verify that models produce non-zero logits and that gradients
flow to parameters. A model that outputs near-zero logits will produce
sigmoid(0) = 0.5 predictions, indicating training failure.

Test Rationale
--------------
- test_model_produces_nonzero_logits: Verifies untrained model doesn't start
  at a degenerate point where logits are exactly zero.
- test_gradients_flow_to_parameters: Confirms BCEWithLogitsLoss gradients
  actually reach model parameters through the forward pass.
- test_baseline_produces_nonzero_logits: Same check for baseline models.
"""

import pytest
import torch
import torch.nn.functional as F

from tmgg.models.spectral_denoisers import LinearPE, GraphFilterBank, SelfAttentionDenoiser
from tmgg.models.baselines import LinearBaseline, MLPBaseline


class TestSpectralModelsForwardSanity:
    """Verify spectral denoising models produce valid outputs."""

    @pytest.fixture
    def identity_adjacency(self):
        """Create identity adjacency matrix batch."""
        return torch.eye(32).unsqueeze(0)

    @pytest.fixture
    def random_adjacency(self):
        """Create random symmetric adjacency matrix batch."""
        A = torch.rand(4, 32, 32)
        A = (A + A.transpose(-2, -1)) / 2  # Symmetrize
        return A

    def test_linear_pe_produces_nonzero_logits(self, identity_adjacency):
        """Verify LinearPE doesn't output exactly zero logits."""
        model = LinearPE(k=8, max_nodes=32)
        logits = model(identity_adjacency)

        # Logits should not be exactly zero
        assert logits.abs().mean() > 1e-6, (
            f"LinearPE outputs near-zero logits: mean={logits.mean():.2e}, "
            f"std={logits.std():.2e}"
        )

    def test_filter_bank_produces_nonzero_logits(self, identity_adjacency):
        """Verify GraphFilterBank doesn't output exactly zero logits."""
        model = GraphFilterBank(k=8, polynomial_degree=5)
        logits = model(identity_adjacency)

        assert logits.abs().mean() > 1e-6, (
            f"GraphFilterBank outputs near-zero logits: mean={logits.mean():.2e}"
        )

    def test_self_attention_produces_nonzero_logits(self, identity_adjacency):
        """Verify SelfAttentionDenoiser doesn't output exactly zero logits."""
        model = SelfAttentionDenoiser(k=8, d_k=64)
        logits = model(identity_adjacency)

        assert logits.abs().mean() > 1e-6, (
            f"SelfAttentionDenoiser outputs near-zero logits: mean={logits.mean():.2e}"
        )

    @pytest.mark.parametrize("model_class,kwargs", [
        (LinearPE, {"k": 8, "max_nodes": 32}),
        (GraphFilterBank, {"k": 8, "polynomial_degree": 5}),
        (SelfAttentionDenoiser, {"k": 8, "d_k": 64}),
    ])
    def test_gradients_flow_to_parameters(self, model_class, kwargs, identity_adjacency):
        """Verify BCEWithLogitsLoss gradients reach model parameters."""
        model = model_class(**kwargs)

        # Forward pass
        logits = model(identity_adjacency)
        target = torch.ones_like(logits) * 0.5  # Uniform target

        # Compute loss and backward
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        max_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                max_grad = max(max_grad, grad_norm)
                if grad_norm > 1e-10:
                    has_grad = True

        assert has_grad, (
            f"{model_class.__name__} has no meaningful gradients. "
            f"Max gradient: {max_grad:.2e}"
        )


class TestBaselineModelsForwardSanity:
    """Verify baseline models produce valid outputs."""

    @pytest.fixture
    def random_adjacency(self):
        """Create random symmetric adjacency matrix batch."""
        A = torch.rand(4, 32, 32)
        A = (A + A.transpose(-2, -1)) / 2
        return A

    def test_linear_baseline_produces_nonzero_logits(self, random_adjacency):
        """Verify LinearBaseline doesn't output exactly zero logits.

        Note: LinearBaseline starts at identity, so initial output ≈ input.
        With random input, output should be non-zero.
        """
        model = LinearBaseline(max_nodes=32)
        logits = model(random_adjacency)

        # With identity initialization, output ≈ input (which is non-zero)
        assert logits.abs().mean() > 0.1, (
            f"LinearBaseline outputs near-zero logits: mean={logits.mean():.2e}"
        )

    def test_mlp_baseline_produces_nonzero_logits(self, random_adjacency):
        """Verify MLPBaseline doesn't output exactly zero logits."""
        model = MLPBaseline(max_nodes=32, hidden_dim=256)
        logits = model(random_adjacency)

        # MLP should produce some output from random initialization
        assert logits.abs().mean() > 1e-6, (
            f"MLPBaseline outputs near-zero logits: mean={logits.mean():.2e}"
        )

    @pytest.mark.parametrize("model_class,kwargs", [
        (LinearBaseline, {"max_nodes": 32}),
        (MLPBaseline, {"max_nodes": 32, "hidden_dim": 256}),
    ])
    def test_baseline_gradients_flow(self, model_class, kwargs, random_adjacency):
        """Verify baseline model gradients flow properly."""
        model = model_class(**kwargs)

        logits = model(random_adjacency)
        target = (random_adjacency > 0.5).float()

        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        # Check gradients exist and are non-trivial
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().max() > 1e-10:
                has_grad = True
                break

        assert has_grad, f"{model_class.__name__} has no meaningful gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
