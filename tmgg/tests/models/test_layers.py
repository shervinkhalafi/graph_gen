"""Tests for utility layers."""

import pytest
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import strategies as st

from tmgg.models.layers import Etoy, Xtoy, masked_softmax


class TestXtoy:
    """Test Xtoy layer."""

    def test_init(self):
        """Test initialization."""
        dx = 16
        dy = 32

        layer = Xtoy(dx, dy)

        assert isinstance(layer.lin, nn.Linear)
        assert layer.lin.in_features == 4 * dx
        assert layer.lin.out_features == dy

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        num_nodes = 10
        dx = 16
        dy = 32

        layer = Xtoy(dx, dy)
        X = torch.randn(batch_size, num_nodes, dx)

        output = layer(X)

        assert output.shape == (batch_size, dy)

    def test_aggregation_components(self):
        """Test that all aggregation components are computed correctly."""
        batch_size = 2
        dx = 8
        dy = 16

        layer = Xtoy(dx, dy)

        # Create specific input to test aggregations
        X = torch.tensor(
            [
                [[1.0] * dx, [2.0] * dx, [3.0] * dx, [4.0] * dx, [5.0] * dx],
                [[0.0] * dx, [1.0] * dx, [2.0] * dx, [3.0] * dx, [4.0] * dx],
            ]
        )

        output = layer(X)

        # Check output shape
        assert output.shape == (batch_size, dy)

        # Verify the linear layer receives correct sized input
        expected_mean = X.mean(dim=1)
        expected_min = X.min(dim=1)[0]
        expected_max = X.max(dim=1)[0]
        expected_std = X.std(dim=1)

        assert expected_mean.shape == (batch_size, dx)
        assert expected_min.shape == (batch_size, dx)
        assert expected_max.shape == (batch_size, dx)
        assert expected_std.shape == (batch_size, dx)

    @given(
        batch_size=st.integers(min_value=1, max_value=10),
        num_nodes=st.integers(min_value=2, max_value=20),
        dx=st.integers(min_value=4, max_value=32),
        dy=st.integers(min_value=4, max_value=32),
    )
    def test_property_shapes(self, batch_size, num_nodes, dx, dy):
        """Property test for shape consistency."""
        layer = Xtoy(dx, dy)
        X = torch.randn(batch_size, num_nodes, dx)

        output = layer(X)

        assert output.shape == (batch_size, dy)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestEtoy:
    """Test Etoy layer."""

    def test_init(self):
        """Test initialization."""
        de = 16
        dy = 32

        layer = Etoy(de, dy)

        assert isinstance(layer.lin, nn.Linear)
        assert layer.lin.in_features == 4 * de
        assert layer.lin.out_features == dy

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        num_nodes = 10
        de = 16
        dy = 32

        layer = Etoy(de, dy)
        E = torch.randn(batch_size, num_nodes, num_nodes, de)

        output = layer(E)

        assert output.shape == (batch_size, dy)

    def test_aggregation_components(self):
        """Test that all aggregation components are computed correctly."""
        batch_size = 2
        num_nodes = 3
        de = 8
        dy = 16

        layer = Etoy(de, dy)

        # Create specific input to test aggregations
        E = torch.zeros(batch_size, num_nodes, num_nodes, de)
        # Set different values for each edge
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    E[b, i, j] = (b + 1) * (i + 1) * (j + 1)

        output = layer(E)

        # Check output shape
        assert output.shape == (batch_size, dy)

        # Verify aggregations work correctly
        expected_mean = E.mean(dim=(1, 2))
        expected_min = E.min(dim=2)[0].min(dim=1)[0]
        expected_max = E.max(dim=2)[0].max(dim=1)[0]
        expected_std = torch.std(E, dim=(1, 2))

        assert expected_mean.shape == (batch_size, de)
        assert expected_min.shape == (batch_size, de)
        assert expected_max.shape == (batch_size, de)
        assert expected_std.shape == (batch_size, de)

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        num_nodes=st.integers(min_value=2, max_value=10),
        de=st.integers(min_value=4, max_value=32),
        dy=st.integers(min_value=4, max_value=32),
    )
    def test_property_shapes(self, batch_size, num_nodes, de, dy):
        """Property test for shape consistency."""
        layer = Etoy(de, dy)
        E = torch.randn(batch_size, num_nodes, num_nodes, de)

        output = layer(E)

        assert output.shape == (batch_size, dy)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMaskedSoftmax:
    """Test masked_softmax function."""

    def test_no_mask(self):
        """Test softmax without mask."""
        x = torch.randn(2, 5, 5)

        output = masked_softmax(x, None, dim=-1)
        expected = torch.softmax(x, dim=-1)

        assert torch.allclose(output, expected)

    def test_full_mask(self):
        """Test with all positions valid."""
        x = torch.randn(2, 5, 5)
        mask = torch.ones_like(x)

        output = masked_softmax(x, mask, dim=-1)
        expected = torch.softmax(x, dim=-1)

        assert torch.allclose(output, expected)

    def test_partial_mask(self):
        """Test with some positions masked."""
        x = torch.randn(1, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)

        output = masked_softmax(x, mask, dim=-1)

        # Check masked positions have near-zero probability
        assert output[0, 3] < 1e-6
        assert output[0, 4] < 1e-6

        # Check unmasked positions sum to 1
        assert torch.allclose(output[0, :3].sum(), torch.tensor(1.0))

    def test_empty_mask(self):
        """Test with empty mask (all zeros)."""
        x = torch.randn(1, 5)
        mask = torch.zeros_like(x)

        output = masked_softmax(x, mask, dim=-1)

        # When mask.sum() == 0, should return zeros
        assert output.shape == x.shape
        assert torch.all(output == 0)

    def test_2d_mask_on_3d_tensor(self):
        """Test 2D mask applied to 3D tensor."""
        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, seq_len)

        # Mask that masks out last position
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -1] = 0

        output = masked_softmax(x, mask, dim=-1)

        # Check last column has near-zero values
        assert torch.all(output[:, :, -1] < 1e-6)

        # Check each row sums to 1
        row_sums = output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    @given(
        batch_size=st.integers(min_value=1, max_value=5),
        seq_len=st.integers(min_value=2, max_value=10),
    )
    def test_property_probability_distribution(self, batch_size, seq_len):
        """Property test: masked softmax output is a valid probability distribution."""
        x = torch.randn(batch_size, seq_len)

        # Generate random mask ensuring each row has at least one unmasked position
        mask = torch.zeros((batch_size, seq_len))
        for i in range(batch_size):
            # Randomly select positions to unmask, ensuring at least one
            num_unmasked = torch.randint(1, seq_len + 1, (1,)).item()
            indices = torch.randperm(seq_len)[:num_unmasked]
            mask[i, indices] = 1

        output = masked_softmax(x, mask, dim=-1)

        # Check all values are non-negative
        assert torch.all(output >= 0)

        # Check all values are <= 1
        assert torch.all(output <= 1)

        # Check rows sum to 1 (allowing for numerical errors)
        row_sums = output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

        # Check masked positions have near-zero probability
        assert torch.all(output[mask == 0] < 1e-6)

    @given(
        shape=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=4),
        dim=st.integers(
            min_value=-1, max_value=-1
        ),  # Only test last dimension for simplicity
    )
    def test_property_shape_preservation(self, shape, dim):
        """Property test: output shape matches input shape."""
        x = torch.randn(*shape)
        mask = torch.ones_like(x)

        output = masked_softmax(x, mask, dim=dim)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
