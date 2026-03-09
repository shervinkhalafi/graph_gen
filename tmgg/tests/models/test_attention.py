"""Tests for attention models."""

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.layers import MultiHeadAttention


class TestMultiHeadAttention:
    """Test MultiHeadAttention module."""

    def test_init(self):
        """Test initialization."""
        d_model = 64
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)

        assert attention.d_model == d_model
        assert attention.num_heads == num_heads
        assert attention.d_k == d_model // num_heads
        assert attention.d_v == d_model // num_heads

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 64
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_mask_functionality(self):
        """Test that masking works correctly."""
        batch_size = 1
        seq_len = 5
        d_model = 32
        num_heads = 4

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create mask that masks out last position
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -1] = 0

        output = attention(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)


class TestMultiLayerAttention:
    """Test MultiLayerAttention module.

    MultiLayerAttention is a denoising model that processes adjacency matrices.
    It returns a single tensor (reconstructed adjacency matrix), not a tuple.
    """

    def test_init(self):
        """Test initialization."""
        d_model = 64
        num_heads = 8
        num_layers = 4

        model = MultiLayerAttention(d_model, num_heads, num_layers)

        assert len(model.layers) == num_layers
        assert model.d_model == d_model
        assert model.num_heads == num_heads
        assert model.num_layers == num_layers

    def test_forward_shape(self):
        """Test forward pass output shapes.

        MultiLayerAttention takes GraphData and returns GraphData.
        The adjacency is extracted/reconstructed via to_adjacency()/from_adjacency().
        """
        batch_size = 2
        num_nodes = 10
        d_model = num_nodes  # d_model should match adjacency matrix dimension

        model = MultiLayerAttention(d_model, num_heads=2, num_layers=2)

        # Input: batch of adjacency matrices wrapped as GraphData
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        A = A + torch.randn_like(A) * 0.1  # Add some noise
        data = GraphData.from_adjacency(A)

        result = model(data)
        assert isinstance(result, GraphData)
        assert result.to_adjacency().shape == (batch_size, num_nodes, num_nodes)

    def test_get_config(self):
        """Test configuration retrieval."""
        d_model = 64
        num_heads = 8
        num_layers = 4
        dropout = 0.1
        bias = True

        model = MultiLayerAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
        )

        config = model.get_config()

        assert config["d_model"] == d_model
        assert config["num_heads"] == num_heads
        assert config["num_layers"] == num_layers
        assert config["dropout"] == dropout
        assert config["bias"] == bias


if __name__ == "__main__":
    pytest.main([__file__])
