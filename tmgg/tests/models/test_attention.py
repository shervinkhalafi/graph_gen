"""Tests for attention models."""

import pytest
import torch

from tmgg.models.attention import MultiHeadAttention, MultiLayerAttention


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
        
        output, scores = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert scores.shape == (batch_size, seq_len, seq_len)
    
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
        
        output, scores = attention(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        # With masking, attention to last position should be minimal
        assert torch.all(scores[:, :, -1] < 0.1)


class TestMultiLayerAttention:
    """Test MultiLayerAttention module."""
    
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
        """Test forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 64
        num_heads = 8
        num_layers = 4
        
        model = MultiLayerAttention(d_model, num_heads, num_layers)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_scores = model(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert len(attention_scores) == num_layers
        for scores in attention_scores:
            assert scores.shape == (batch_size, seq_len, seq_len)
    
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
            bias=bias
        )
        
        config = model.get_config()
        
        assert config["d_model"] == d_model
        assert config["num_heads"] == num_heads
        assert config["num_layers"] == num_layers
        assert config["dropout"] == dropout
        assert config["bias"] == bias


if __name__ == "__main__":
    pytest.main([__file__])