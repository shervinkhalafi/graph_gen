"""Tests for DenoisingModel base class.

Test rationale:
    After removing the domain transformation system, models now apply sigmoid
    explicitly in their architectures. The base class provides minimal
    functionality: transform_for_loss returns output and target unchanged.

Invariants:
    - transform_for_loss: output and target are returned unchanged (identity)
    - Models with explicit sigmoid produce output in [0, 1]
"""

import pytest
import torch

from tmgg.models.base import DenoisingModel


class ConcreteDenoisingModel(DenoisingModel):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, apply_sigmoid: bool = True):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten, process, reshape
        batch_size, n, m = x.shape
        x = x.view(batch_size * n, m)
        x = self.linear(x)
        x = x.view(batch_size, n, m)
        if self.apply_sigmoid:
            return torch.sigmoid(x)
        return x

    def get_config(self):
        return {"apply_sigmoid": self.apply_sigmoid}


class TestTransformForLoss:
    """Test that transform_for_loss is identity (no domain transforms)."""

    @pytest.fixture
    def model(self):
        return ConcreteDenoisingModel()

    @pytest.fixture
    def sample_adjacency(self):
        """Typical adjacency matrix with values in [0, 1]."""
        torch.manual_seed(42)
        A = torch.rand(4, 10, 10)
        A = (A + A.transpose(-1, -2)) / 2
        A.diagonal(dim1=-2, dim2=-1).zero_()
        return A

    def test_transform_for_loss_is_identity(self, model, sample_adjacency):
        """transform_for_loss should return tensors unchanged."""
        output = sample_adjacency.clone()
        target = sample_adjacency.clone()

        out_transformed, tgt_transformed = model.transform_for_loss(output, target)

        assert torch.allclose(out_transformed, output)
        assert torch.allclose(tgt_transformed, target)

    def test_transform_for_loss_eval_mode(self, model, sample_adjacency):
        """transform_for_loss should be identity in eval mode too."""
        model.eval()
        output = sample_adjacency.clone()
        target = sample_adjacency.clone()

        out_transformed, tgt_transformed = model.transform_for_loss(output, target)

        assert torch.allclose(out_transformed, output)
        assert torch.allclose(tgt_transformed, target)


class TestOutputBounds:
    """Test that models with sigmoid produce bounded output."""

    def test_model_with_sigmoid_produces_bounded_output(self):
        """Model with explicit sigmoid should produce output in [0, 1]."""
        model = ConcreteDenoisingModel(apply_sigmoid=True)
        model.eval()

        input_data = torch.randn(4, 10, 10) * 2  # Unbounded input

        with torch.no_grad():
            output = model(input_data)

        assert (output >= 0).all(), "Output has values < 0"
        assert (output <= 1).all(), "Output has values > 1"

    def test_model_without_sigmoid_can_produce_unbounded(self):
        """Model without sigmoid can produce unbounded output."""
        model = ConcreteDenoisingModel(apply_sigmoid=False)
        model.eval()

        input_data = torch.randn(4, 10, 10) * 2

        with torch.no_grad():
            output = model(input_data)

        # Should have values outside [0, 1]
        has_negative = (output < 0).any()
        has_above_one = (output > 1).any()
        assert has_negative or has_above_one, "Expected unbounded output"


class TestGetConfig:
    """Test model configuration retrieval."""

    def test_get_config_returns_dict(self):
        """get_config should return a dictionary."""
        model = ConcreteDenoisingModel()
        config = model.get_config()

        assert isinstance(config, dict)
        assert "apply_sigmoid" in config
