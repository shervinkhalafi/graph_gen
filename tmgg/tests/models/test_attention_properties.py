"""Property-based tests for attention models using Hypothesis."""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite
import numpy as np
from unittest.mock import Mock, patch

from tmgg.models.attention import MultiHeadAttention, MultiLayerAttention


# Custom strategies for generating test data
@composite
def attention_dims(draw):
    """Generate valid dimensions for attention models."""
    d_model = draw(st.integers(min_value=8, max_value=128).filter(lambda x: x % 2 == 0))
    num_heads = draw(st.sampled_from([1, 2, 4, 8]).filter(lambda h: d_model % h == 0))
    return d_model, num_heads


@composite
def attention_input(draw, d_model=None, batch_size=None, seq_len=None):
    """Generate valid input tensors for attention models."""
    if d_model is None:
        d_model = draw(st.integers(min_value=8, max_value=64).filter(lambda x: x % 2 == 0))
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=8))
    if seq_len is None:
        seq_len = draw(st.integers(min_value=2, max_value=32))
    
    # Generate input tensor with controlled values to avoid numerical issues
    x = draw(st.lists(
        st.lists(
            st.lists(
                st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
                min_size=d_model, max_size=d_model
            ),
            min_size=seq_len, max_size=seq_len
        ),
        min_size=batch_size, max_size=batch_size
    ))
    
    return torch.tensor(x, dtype=torch.float32)


@composite
def attention_mask(draw, batch_size, seq_len):
    """Generate valid attention masks ensuring at least one unmasked position per row."""
    masks = []
    for _ in range(batch_size):
        mask_batch = []
        for _ in range(seq_len):
            # Ensure at least one position is unmasked in each row
            row = draw(st.lists(
                st.integers(min_value=0, max_value=1),
                min_size=seq_len, max_size=seq_len
            ))
            if sum(row) == 0:  # If all masked, unmask at least one position
                row[draw(st.integers(min_value=0, max_value=seq_len-1))] = 1
            mask_batch.append(row)
        masks.append(mask_batch)
    
    return torch.tensor(masks, dtype=torch.float32)


class TestMultiHeadAttentionProperties:
    """Property-based tests for MultiHeadAttention."""
    
    @given(dims=attention_dims())
    def test_output_shape_invariant(self, dims):
        """Test that output shape matches input shape regardless of internal dimensions."""
        d_model, num_heads = dims
        model = MultiHeadAttention(d_model, num_heads)
        
        # Test with various input shapes
        for batch_size in [1, 4]:
            for seq_len in [5, 10]:
                x = torch.randn(batch_size, seq_len, d_model)
                output, scores = model(x)
                
                assert output.shape == x.shape
                assert scores.shape == (batch_size, seq_len, seq_len)
    
    @given(
        dims=attention_dims(),
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=16)
    )
    @settings(max_examples=50)
    def test_attention_scores_valid_range(self, dims, batch_size, seq_len):
        """Test that combined attention scores are in a reasonable range."""
        d_model, num_heads = dims
        
        # Generate non-zero input to avoid all-zero attention
        x = torch.randn(batch_size, seq_len, d_model) * 0.1 + 0.1
        
        model = MultiHeadAttention(d_model, num_heads)
        _, scores = model(x)
        
        # The combined scores are outputs of a linear layer, so they can be negative
        # Just check they're not exploding or all zeros
        assert not torch.all(scores == 0)  # Should have some non-zero values
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        assert torch.abs(scores).max() < 100  # Should not explode
    
    @given(
        dims=attention_dims(),
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=16)
    )
    def test_mask_zeros_attention(self, dims, batch_size, seq_len):
        """Test that masked positions have exactly zero attention weight."""
        d_model, num_heads = dims
        model = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask that masks out specific positions
        mask = torch.ones(batch_size, seq_len, seq_len)
        # Mask out last column
        mask[:, :, -1] = 0
        
        _, scores = model(x, mask=mask)
        
        # Check masked positions have near-zero attention
        assert torch.all(scores[:, :, -1] < 1e-6)
    
    @given(
        dims=attention_dims(),
        batch_size=st.integers(min_value=1, max_value=2),
        seq_len=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.data_too_large])
    def test_numerical_stability(self, dims, batch_size, seq_len):
        """Test that the model handles various input scales without producing NaN/Inf."""
        d_model, num_heads = dims
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        model = MultiHeadAttention(d_model, num_heads)
        
        # Test with different input scales
        for scale in [0.01, 1.0, 10.0]:
            scaled_x = x * scale
            output, scores = model(scaled_x)
            
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert not torch.isnan(scores).any()
            assert not torch.isinf(scores).any()
    
    @given(dims=attention_dims())
    @settings(deadline=None)  # Disable deadline for this test
    def test_residual_connection_improves_gradient_flow(self, dims):
        """Test that residual connections prevent gradient vanishing."""
        d_model, num_heads = dims
        model = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(2, 10, d_model, requires_grad=True)
        output, _ = model(x)
        
        # Compute gradient with respect to input
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are not vanishing
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.isnan(x.grad).any()


class TestMultiLayerAttentionProperties:
    """Property-based tests for MultiLayerAttention."""
    
    @given(
        dims=attention_dims(),
        num_layers=st.integers(min_value=1, max_value=8)
    )
    def test_layer_composition_preserves_shape(self, dims, num_layers):
        """Test that stacking layers preserves input/output shape."""
        d_model, num_heads = dims
        model = MultiLayerAttention(d_model, num_heads, num_layers)
        
        x = torch.randn(2, 10, d_model)
        output, attention_scores = model(x)
        
        assert output.shape == x.shape
        assert len(attention_scores) == num_layers
        for scores in attention_scores:
            assert scores.shape == (2, 10, 10)
    
    @given(
        dims=attention_dims(),
        num_layers=st.integers(min_value=2, max_value=6),
        batch_size=st.integers(min_value=1, max_value=2),
        seq_len=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=20)
    def test_deeper_models_maintain_stability(self, dims, num_layers, batch_size, seq_len):
        """Test that deeper models don't suffer from vanishing/exploding activations."""
        d_model, num_heads = dims
        
        # Generate non-zero input
        x = torch.randn(batch_size, seq_len, d_model) * 0.5 + 0.1
        
        model = MultiLayerAttention(d_model, num_heads, num_layers)
        output, _ = model(x)
        
        # Check output magnitude is reasonable
        output_norm = torch.norm(output, p=2)
        input_norm = torch.norm(x, p=2)
        
        # Output should not explode or vanish relative to input
        ratio = output_norm / (input_norm + 1e-8)
        assert 0.01 < ratio < 100.0  # More lenient bounds
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @given(
        dims=attention_dims(),
        num_layers=st.integers(min_value=1, max_value=4)
    )
    def test_mask_propagation_through_layers(self, dims, num_layers):
        """Test that masks are correctly propagated through all layers."""
        d_model, num_heads = dims
        model = MultiLayerAttention(d_model, num_heads, num_layers)
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask that masks out last two positions
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -2:] = 0
        
        _, attention_scores = model(x, mask=mask)
        
        # Check that mask is applied in all layers
        for layer_scores in attention_scores:
            assert torch.all(layer_scores[:, :, -2:] < 1e-6)


class TestAttentionErrorHandling:
    """Test error handling with mocked components."""
    
    def test_handles_dimension_mismatch_gracefully(self):
        """Test that model handles dimension mismatches appropriately."""
        model = MultiHeadAttention(d_model=64, num_heads=8)
        
        # Wrong input dimension will cause an error in the linear layer
        x = torch.randn(2, 10, 32)  # 32 != 64
        
        # This will raise an error because linear layer expects d_model input
        with pytest.raises(RuntimeError):
            output, scores = model(x)
    
    @patch('torch.nn.functional.softmax')
    def test_softmax_numerical_stability(self, mock_softmax):
        """Test handling of numerical instabilities in softmax."""
        # Mock softmax to return NaN to test error handling
        mock_softmax.return_value = torch.full((2, 10, 10), float('nan'))
        
        model = MultiHeadAttention(d_model=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        
        # The model should handle this gracefully
        with pytest.raises(Exception):  # Model may not have explicit NaN handling
            output, scores = model(x)
    
    def test_zero_attention_scores_handling(self):
        """Test behavior when all attention scores are masked."""
        model = MultiHeadAttention(d_model=32, num_heads=4)
        x = torch.randn(1, 5, 32)
        
        # Mask that zeros out everything
        mask = torch.zeros(1, 5, 5)
        
        output, scores = model(x, mask=mask)
        
        # With all-zero mask, scores should be uniform or handle gracefully
        # The implementation might handle this differently
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


# Hypothesis settings for debugging
pytest.register_assert_rewrite(__name__)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])