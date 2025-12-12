"""Tests for hybrid models combining GNN and attention mechanisms."""

import pytest
import torch

from tmgg.models.hybrid import SequentialDenoisingModel, create_sequential_model
from tmgg.models.gnn import GNN
from tmgg.models.attention import MultiLayerAttention


class TestSequentialDenoisingModel:
    """Test SequentialDenoisingModel class."""
    
    def test_init_with_denoising(self):
        """Test initialization with both embedding and denoising models."""
        # Create embedding model
        embedding_model = GNN(
            num_layers=2,
            num_terms=3,
            feature_dim_in=20,
            feature_dim_out=5
        )
        
        # Create denoising model
        denoising_model = MultiLayerAttention(
            d_model=10,  # 2 * feature_dim_out
            num_heads=2,
            num_layers=2
        )
        
        # Create sequential model
        model = SequentialDenoisingModel(embedding_model, denoising_model)
        
        assert model.embedding_model is embedding_model
        assert model.denoising_model is denoising_model
    
    def test_init_without_denoising(self):
        """Test initialization with only embedding model."""
        embedding_model = GNN(
            num_layers=2,
            num_terms=3,
            feature_dim_in=20,
            feature_dim_out=5
        )
        
        model = SequentialDenoisingModel(embedding_model, None)
        
        assert model.embedding_model is embedding_model
        assert model.denoising_model is None
    
    def test_forward_with_denoising(self):
        """Test forward pass with denoising model.

        forward() returns raw logits; predict() returns probabilities in [0, 1].
        """
        batch_size = 2
        num_nodes = 10
        feature_dim_out = 5

        # Create models
        embedding_model = GNN(
            num_layers=1,
            num_terms=2,
            feature_dim_in=num_nodes,
            feature_dim_out=feature_dim_out
        )

        denoising_model = MultiLayerAttention(
            d_model=2 * feature_dim_out,
            num_heads=2,
            num_layers=1
        )

        model = SequentialDenoisingModel(embedding_model, denoising_model)

        # Create input
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        # Forward pass returns logits
        logits = model(A)
        assert logits.shape == (batch_size, num_nodes, num_nodes)

        # predict() applies sigmoid to get probabilities in [0, 1]
        probs = model.predict(logits)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_forward_without_denoising(self):
        """Test forward pass without denoising model.

        forward() returns raw logits; predict() returns probabilities in [0, 1].
        """
        batch_size = 2
        num_nodes = 10
        feature_dim_out = 5

        # Create embedding model only
        embedding_model = GNN(
            num_layers=1,
            num_terms=2,
            feature_dim_in=num_nodes,
            feature_dim_out=feature_dim_out
        )

        model = SequentialDenoisingModel(embedding_model, None)

        # Create input
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        # Forward pass returns logits
        logits = model(A)
        assert logits.shape == (batch_size, num_nodes, num_nodes)

        # predict() applies sigmoid to get probabilities in [0, 1]
        probs = model.predict(logits)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        embedding_model = GNN(
            num_layers=2,
            num_terms=3,
            feature_dim_in=20,
            feature_dim_out=5
        )
        
        denoising_model = MultiLayerAttention(
            d_model=10,
            num_heads=2,
            num_layers=2
        )
        
        model = SequentialDenoisingModel(embedding_model, denoising_model)
        
        config = model.get_config()
        
        assert "embedding_model" in config
        assert "has_denoising" in config
        assert config["has_denoising"] is True
        assert "denoising_model" in config
        assert config["model_type"] == "SequentialDenoisingModel"


class TestCreateSequentialModel:
    """Test factory function for creating sequential models."""
    
    def test_create_with_transformer(self):
        """Test creating model with transformer denoising."""
        gnn_config = {
            "num_layers": 2,
            "num_terms": 3,
            "feature_dim_in": 20,
            "feature_dim_out": 5
        }
        
        transformer_config = {
            "num_heads": 4,
            "num_layers": 2,
            "d_k": 10,
            "d_v": 10,
            "dropout": 0.1,
            "bias": True
        }
        
        model = create_sequential_model(gnn_config, transformer_config)
        
        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert isinstance(model.denoising_model, MultiLayerAttention)
        
        # Check GNN config
        assert model.embedding_model.num_layers == 2
        assert model.embedding_model.feature_dim_out == 5
        
        # Check transformer config
        assert model.denoising_model.d_model == 10  # 2 * feature_dim_out
        assert model.denoising_model.num_heads == 4
        assert model.denoising_model.num_layers == 2
    
    def test_create_without_transformer(self):
        """Test creating model without transformer denoising."""
        gnn_config = {
            "num_layers": 2,
            "num_terms": 3,
            "feature_dim_in": 20,
            "feature_dim_out": 5
        }
        
        model = create_sequential_model(gnn_config, None)
        
        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert model.denoising_model is None
    
    def test_create_with_defaults(self):
        """Test creating model with default configurations."""
        model = create_sequential_model({}, {})
        
        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert isinstance(model.denoising_model, MultiLayerAttention)
        
        # Check defaults
        assert model.embedding_model.num_layers == 2
        assert model.embedding_model.feature_dim_out == 5
        assert model.denoising_model.num_heads == 4
        assert model.denoising_model.num_layers == 4
    
    def test_forward_pass_integration(self):
        """Test full forward pass through created model.

        forward() returns raw logits; predict() returns probabilities in [0, 1].
        """
        batch_size = 2
        num_nodes = 10

        gnn_config = {
            "num_layers": 1,
            "feature_dim_in": num_nodes,
            "feature_dim_out": 5
        }

        transformer_config = {
            "num_heads": 2,
            "num_layers": 1
        }

        model = create_sequential_model(gnn_config, transformer_config)

        # Create input
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        # Forward pass returns logits
        logits = model(A)
        assert logits.shape == (batch_size, num_nodes, num_nodes)

        # predict() applies sigmoid to get probabilities in [0, 1]
        probs = model.predict(logits)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)


if __name__ == "__main__":
    pytest.main([__file__])