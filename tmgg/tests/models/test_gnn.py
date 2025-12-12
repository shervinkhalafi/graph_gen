"""Tests for GNN models."""

import pytest
import torch

from tmgg.models import GNN, GNNSymmetric
from tmgg.models.gnn import NodeVarGNN
from tmgg.models.layers import (
    EigenEmbedding,
    GaussianEmbedding,
    GraphConvolutionLayer,
)


class TestEmbeddings:
    """Test embedding layers."""

    def test_eigen_embedding(self):
        """Test EigenEmbedding."""
        batch_size = 2
        num_nodes = 5

        embedding = EigenEmbedding()
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        embeddings = embedding(A)

        assert embeddings.shape == (batch_size, num_nodes, num_nodes)

    def test_gaussian_embedding(self):
        """Test GaussianEmbedding."""
        batch_size = 2
        num_nodes = 5
        num_terms = 3
        num_channels = 10

        embedding = GaussianEmbedding(num_terms, num_channels)
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        embeddings = embedding(A)

        assert embeddings.shape == (batch_size, num_nodes, num_channels)


class TestGNNModels:
    """Test GNN model classes."""

    def test_gnn_forward(self):
        """Test standard GNN forward pass."""
        batch_size = 2
        num_nodes = 5
        num_layers = 2
        num_terms = 3
        feature_dim_in = 5
        feature_dim_out = 10

        model = GNN(
            num_layers=num_layers,
            num_terms=num_terms,
            feature_dim_in=feature_dim_in,
            feature_dim_out=feature_dim_out,
        )

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        X, Y = model(A)

        assert X.shape == (batch_size, num_nodes, feature_dim_out)
        assert Y.shape == (batch_size, num_nodes, feature_dim_out)

    def test_gnn_symmetric_forward(self):
        """Test symmetric GNN forward pass.

        forward() returns raw logits; predict() returns probabilities in [0, 1].
        """
        batch_size = 2
        num_nodes = 5
        num_layers = 2
        feature_dim_out = 10

        model = GNNSymmetric(num_layers=num_layers, feature_dim_out=feature_dim_out)

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        logits, X = model(A)

        assert logits.shape == (batch_size, num_nodes, num_nodes)
        assert X.shape == (batch_size, num_nodes, feature_dim_out)

        # predict() applies sigmoid to get probabilities in [0, 1]
        probs = model.predict(logits)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_nodevar_gnn_forward(self):
        """Test NodeVarGNN forward pass.

        forward() returns raw logits; predict() returns probabilities in [0, 1].
        """
        batch_size = 2
        num_nodes = 5
        num_layers = 1
        feature_dim = 10

        model = NodeVarGNN(num_layers=num_layers, feature_dim=feature_dim)

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        logits = model(A)

        assert logits.shape == (batch_size, num_nodes, num_nodes)

        # predict() applies sigmoid to get probabilities in [0, 1]
        probs = model.predict(logits)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_gnn_get_config(self):
        """Test GNN configuration retrieval."""
        num_layers = 3
        num_terms = 4
        feature_dim_in = 15
        feature_dim_out = 20

        model = GNN(
            num_layers=num_layers,
            num_terms=num_terms,
            feature_dim_in=feature_dim_in,
            feature_dim_out=feature_dim_out,
        )

        config = model.get_config()

        assert config["num_layers"] == num_layers
        assert config["num_terms"] == num_terms
        assert config["feature_dim_in"] == feature_dim_in
        assert config["feature_dim_out"] == feature_dim_out


class TestGraphConvolutionLayer:
    """Test GraphConvolutionLayer."""

    def test_forward_shape(self):
        """Test forward pass shapes."""
        batch_size = 2
        num_nodes = 5
        num_terms = 3
        num_channels = 10

        layer = GraphConvolutionLayer(num_terms, num_channels)

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        X = torch.randn(batch_size, num_nodes, num_channels)

        Y = layer(A, X)

        assert Y.shape == (batch_size, num_nodes, num_channels)


if __name__ == "__main__":
    pytest.main([__file__])
