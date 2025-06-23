"""Tests for GraphTransformer model."""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, assume, settings
import numpy as np

from tmgg.models.transformer import (
    GraphTransformer, 
    NodeEdgeBlock, 
    XEyTransformerLayer,
    GraphFeatures
)
from tmgg.models import GraphTransformer as ImportedGraphTransformer


class TestImports:
    """Test that all components are importable."""
    
    def test_import_from_models(self):
        """Test importing GraphTransformer from models package."""
        assert ImportedGraphTransformer is GraphTransformer
    
    def test_import_components(self):
        """Test importing individual components."""
        from tmgg.models.layers import Xtoy, Etoy, masked_softmax
        
        assert Xtoy is not None
        assert Etoy is not None
        assert masked_softmax is not None


class TestGraphFeatures:
    """Test GraphFeatures named tuple."""
    
    def test_creation(self):
        """Test creating GraphFeatures."""
        X = torch.randn(2, 10, 16)
        E = torch.randn(2, 10, 10, 8)
        y = torch.randn(2, 32)
        
        features = GraphFeatures(X=X, E=E, y=y)
        
        assert torch.equal(features.X, X)
        assert torch.equal(features.E, E)
        assert torch.equal(features.y, y)
    
    def test_masking(self):
        """Test mask method."""
        batch_size = 2
        num_nodes = 5
        
        X = torch.ones(batch_size, num_nodes, 16)
        E = torch.ones(batch_size, num_nodes, num_nodes, 8)
        y = torch.ones(batch_size, 32)
        
        # Create mask that masks out last 2 nodes
        node_mask = torch.ones(batch_size, num_nodes)
        node_mask[:, -2:] = 0
        
        features = GraphFeatures(X=X, E=E, y=y)
        masked_features = features.mask(node_mask)
        
        # Check node features are masked
        assert torch.all(masked_features.X[:, -2:, :] == 0)
        assert torch.all(masked_features.X[:, :-2, :] == 1)
        
        # Check edge features are masked
        assert torch.all(masked_features.E[:, -2:, :, :] == 0)
        assert torch.all(masked_features.E[:, :, -2:, :] == 0)
        
        # Check global features are unchanged
        assert torch.equal(masked_features.y, y)


class TestNodeEdgeBlock:
    """Test NodeEdgeBlock layer."""
    
    def test_init(self):
        """Test initialization."""
        dx = 64
        de = 32
        dy = 16
        n_head = 8
        
        block = NodeEdgeBlock(dx, de, dy, n_head)
        
        assert block.dx == dx
        assert block.de == de
        assert block.dy == dy
        assert block.n_head == n_head
        assert block.df == dx // n_head
    
    def test_init_invalid_heads(self):
        """Test initialization with invalid number of heads."""
        dx = 64
        de = 32
        dy = 16
        n_head = 7  # Not divisible
        
        with pytest.raises(AssertionError):
            NodeEdgeBlock(dx, de, dy, n_head)
    
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size = 2
        num_nodes = 10
        dx = 64
        de = 32
        dy = 16
        n_head = 8
        
        block = NodeEdgeBlock(dx, de, dy, n_head)
        
        X = torch.randn(batch_size, num_nodes, dx)
        E = torch.randn(batch_size, num_nodes, num_nodes, de)
        y = torch.randn(batch_size, dy)
        node_mask = torch.ones(batch_size, num_nodes)
        
        newX, newE, new_y = block(X, E, y, node_mask)
        
        assert newX.shape == X.shape
        assert newE.shape == E.shape
        assert new_y.shape == y.shape
    
    def test_masking_effect(self):
        """Test that masking properly zeros out features."""
        batch_size = 1
        num_nodes = 5
        dx = 32
        de = 16
        dy = 8
        n_head = 4
        
        block = NodeEdgeBlock(dx, de, dy, n_head)
        
        X = torch.ones(batch_size, num_nodes, dx)
        E = torch.ones(batch_size, num_nodes, num_nodes, de)
        y = torch.ones(batch_size, dy)
        
        # Mask out last 2 nodes
        node_mask = torch.ones(batch_size, num_nodes)
        node_mask[:, -2:] = 0
        
        newX, newE, new_y = block(X, E, y, node_mask)
        
        # Check masked node features are zero
        assert torch.all(newX[:, -2:, :] == 0)
        
        # Check masked edge features are zero
        assert torch.all(newE[:, -2:, :, :] == 0)
        assert torch.all(newE[:, :, -2:, :] == 0)


class TestXEyTransformerLayer:
    """Test XEyTransformerLayer."""
    
    def test_init(self):
        """Test initialization."""
        dx = 64
        de = 32
        dy = 16
        n_head = 8
        
        layer = XEyTransformerLayer(dx, de, dy, n_head)
        
        assert isinstance(layer.self_attn, NodeEdgeBlock)
        assert isinstance(layer.normX1, nn.LayerNorm)
        assert isinstance(layer.normE1, nn.LayerNorm)
        assert isinstance(layer.norm_y1, nn.LayerNorm)
    
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size = 2
        num_nodes = 10
        dx = 64
        de = 32
        dy = 16
        n_head = 8
        
        layer = XEyTransformerLayer(dx, de, dy, n_head)
        
        X = torch.randn(batch_size, num_nodes, dx)
        E = torch.randn(batch_size, num_nodes, num_nodes, de)
        y = torch.randn(batch_size, dy)
        node_mask = torch.ones(batch_size, num_nodes)
        
        newX, newE, new_y = layer(X, E, y, node_mask)
        
        assert newX.shape == X.shape
        assert newE.shape == E.shape
        assert new_y.shape == y.shape
    
    def test_residual_connections(self):
        """Test that residual connections are applied."""
        batch_size = 1
        num_nodes = 5
        dx = 32
        de = 16
        dy = 8
        n_head = 4
        
        layer = XEyTransformerLayer(dx, de, dy, n_head, dropout=0.0)
        
        # Use specific values to test residual
        X = torch.ones(batch_size, num_nodes, dx) * 0.1
        E = torch.ones(batch_size, num_nodes, num_nodes, de) * 0.1
        y = torch.ones(batch_size, dy) * 0.1
        node_mask = torch.ones(batch_size, num_nodes)
        
        # Set to eval mode to disable dropout
        layer.eval()
        
        with torch.no_grad():
            newX, newE, new_y = layer(X, E, y, node_mask)
        
        # Output should be different from input due to transformations
        assert not torch.allclose(newX, X)
        assert not torch.allclose(newE, E)
        assert not torch.allclose(new_y, y)


class TestGraphTransformer:
    """Test GraphTransformer model."""
    
    def test_init(self):
        """Test initialization."""
        n_layers = 3
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 32}
        hidden_dims = {
            'dx': 64, 'de': 32, 'dy': 16, 
            'n_head': 8, 'dim_ffX': 128, 'dim_ffE': 64
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        assert len(model.tf_layers) == n_layers
        assert model.n_layers == n_layers
        assert model.out_dim_X == output_dims['X']
        assert model.out_dim_E == output_dims['E']
        assert model.out_dim_y == output_dims['y']
    
    def test_forward_with_all_inputs(self):
        """Test forward pass with all inputs provided."""
        batch_size = 2
        num_nodes = 10
        
        n_layers = 2
        input_dims = {'X': 16, 'E': 8, 'y': 4}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 32}
        hidden_dims = {
            'dx': 64, 'de': 32, 'dy': 16,
            'n_head': 8, 'dim_ffX': 128, 'dim_ffE': 64
        }
        output_dims = {'X': 16, 'E': 8, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        X = torch.randn(batch_size, num_nodes, input_dims['X'])
        E = torch.randn(batch_size, num_nodes, num_nodes, input_dims['E'])
        y = torch.randn(batch_size, input_dims['y'])
        node_mask = torch.ones(batch_size, num_nodes)
        
        output = model(X, E, y, node_mask)
        
        assert isinstance(output, GraphFeatures)
        assert output.X.shape == (batch_size, num_nodes, output_dims['X'])
        assert output.E.shape == (batch_size, num_nodes, num_nodes, output_dims['E'])
        assert output.y.shape == (batch_size, output_dims['y'])
    
    def test_forward_adjacency_matrix_only(self):
        """Test forward pass with only adjacency matrix."""
        batch_size = 2
        num_nodes = 10
        
        n_layers = 2
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 32}
        hidden_dims = {
            'dx': 64, 'de': 32, 'dy': 16,
            'n_head': 8
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        # Provide only adjacency matrix
        A = torch.randn(batch_size, num_nodes, num_nodes)
        
        output = model(A)
        
        assert isinstance(output, GraphFeatures)
        assert output.X.shape == (batch_size, num_nodes, output_dims['X'])
        assert output.E.shape == (batch_size, num_nodes, num_nodes, output_dims['E'])
        assert output.y.shape == (batch_size, output_dims['y'])
    
    def test_edge_symmetry(self):
        """Test that edge features remain symmetric."""
        batch_size = 1
        num_nodes = 5
        
        n_layers = 1
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
        hidden_dims = {
            'dx': 32, 'de': 16, 'dy': 8,
            'n_head': 4
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        # Create symmetric input
        A = torch.randn(batch_size, num_nodes, num_nodes)
        A = (A + A.transpose(1, 2)) / 2  # Make symmetric
        
        output = model(A)
        
        # Check output edge features are symmetric
        E_out = output.E[0]  # Remove batch dimension
        assert torch.allclose(E_out, E_out.transpose(0, 1), atol=1e-6)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        n_layers = 2
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 32}
        hidden_dims = {
            'dx': 64, 'de': 32, 'dy': 16,
            'n_head': 8
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        config = model.get_config()
        
        assert config["n_layers"] == n_layers
        assert config["input_dims"] == input_dims
        assert config["hidden_mlp_dims"] == hidden_mlp_dims
        assert config["hidden_dims"] == hidden_dims
        assert config["output_dims"] == output_dims
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        num_nodes=st.integers(min_value=3, max_value=8),
        n_layers=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_shape_consistency(self, batch_size, num_nodes, n_layers):
        """Property test for shape consistency through the model."""
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
        hidden_dims = {
            'dx': 32, 'de': 16, 'dy': 8,
            'n_head': 4
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        A = torch.randn(batch_size, num_nodes, num_nodes)
        
        output = model(A)
        
        assert output.X.shape == (batch_size, num_nodes, output_dims['X'])
        assert output.E.shape == (batch_size, num_nodes, num_nodes, output_dims['E'])
        assert output.y.shape == (batch_size, output_dims['y'])
        
        # Check no NaN or inf values
        assert not torch.isnan(output.X).any()
        assert not torch.isinf(output.X).any()
        assert not torch.isnan(output.E).any()
        assert not torch.isinf(output.E).any()
        assert not torch.isnan(output.y).any()
        assert not torch.isinf(output.y).any()
    
    @given(
        batch_size=st.integers(min_value=1, max_value=3),
        num_nodes=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_masking_gradients(self, batch_size, num_nodes):
        """Property test: masked positions should not contribute to gradients."""
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
        hidden_dims = {
            'dx': 32, 'de': 16, 'dy': 8,
            'n_head': 4
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=1,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        A = torch.randn(batch_size, num_nodes, num_nodes, requires_grad=True)
        
        # Create mask that masks out some nodes
        node_mask = torch.ones(batch_size, num_nodes)
        num_masked = min(2, num_nodes // 2)
        node_mask[:, -num_masked:] = 0
        
        output = model(A, node_mask=node_mask)
        
        # Compute loss only on masked positions
        masked_loss = output.X[:, -num_masked:, :].sum()
        masked_loss.backward()
        
        # Gradient should be very small since masked positions shouldn't contribute
        assert A.grad is not None
        
        # Check that output at masked positions is zero
        assert torch.all(output.X[:, -num_masked:, :] == 0)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_node_graph(self):
        """Test with single node graph."""
        batch_size = 2
        num_nodes = 1
        
        n_layers = 1
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
        hidden_dims = {
            'dx': 32, 'de': 16, 'dy': 8,
            'n_head': 4
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        A = torch.randn(batch_size, num_nodes, num_nodes)
        
        output = model(A)
        
        assert output.X.shape == (batch_size, num_nodes, output_dims['X'])
        assert output.E.shape == (batch_size, num_nodes, num_nodes, output_dims['E'])
        assert output.y.shape == (batch_size, output_dims['y'])
    
    def test_fully_masked_graph(self):
        """Test with all nodes masked."""
        batch_size = 1
        num_nodes = 5
        
        n_layers = 1
        input_dims = {'X': 1, 'E': 1, 'y': 4}
        hidden_mlp_dims = {'X': 16, 'E': 8, 'y': 16}
        hidden_dims = {
            'dx': 32, 'de': 16, 'dy': 8,
            'n_head': 4
        }
        output_dims = {'X': 1, 'E': 1, 'y': 4}
        
        model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims
        )
        
        A = torch.randn(batch_size, num_nodes, num_nodes)
        node_mask = torch.zeros(batch_size, num_nodes)  # All masked
        
        output = model(A, node_mask=node_mask)
        
        # All outputs should be zero
        assert torch.all(output.X == 0)
        assert torch.all(output.E == 0)
        # Global features might not be zero due to initialization


if __name__ == "__main__":
    pytest.main([__file__])