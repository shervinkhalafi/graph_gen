from typing import Any, Dict

import torch
import torch.nn as nn

from tmgg.models.layers.eigen_embedding import EigenEmbedding
from tmgg.models.layers.nvgcn_layer import NodeVarGraphConvolutionLayer

from ..base import DenoisingModel


class NodeVarGNN(DenoisingModel):
    """Node-variant Graph Neural Network."""

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim: int = 10,
        eigenvalue_reg: float = 0.0,
    ):
        """
        Initialize Node-variant GNN.

        Args:
            num_layers: Number of layers
            num_terms: Number of polynomial terms
            feature_dim: Feature dimension
            eigenvalue_reg: Diagonal regularization for eigendecomposition stability
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim = feature_dim
        self.eigenvalue_reg = eigenvalue_reg

        self.embedding_layer = EigenEmbedding(eigenvalue_reg=eigenvalue_reg)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                NodeVarGraphConvolutionLayer(num_terms, feature_dim, feature_dim)
            )

        self.out_x = nn.Linear(feature_dim, feature_dim)
        self.out_y = nn.Linear(feature_dim, feature_dim)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning reconstructed adjacency matrix.

        Args:
            A: Input adjacency matrix

        Returns:
            Reconstructed adjacency matrix
        """
        Z = self.embedding_layer(A)
        # Take only the first feature_dim columns from eigenvectors
        # But ensure we don't exceed the available columns
        actual_feature_dim = min(Z.shape[2], self.feature_dim)
        Z = Z[:, :, :actual_feature_dim]

        # If we have fewer features than expected, pad with zeros
        if actual_feature_dim < self.feature_dim:
            padding = torch.zeros(
                Z.shape[0],
                Z.shape[1],
                self.feature_dim - actual_feature_dim,
                device=Z.device,
                dtype=Z.dtype,
            )
            Z = torch.cat([Z, padding], dim=2)

        # NOTE: Dynamic layer recreation removed. The redesigned
        # NodeVarGraphConvolutionLayer uses node-agnostic parameters,
        # supporting any graph size without re-creating layers.

        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)
        Y = self.out_y(Z)
        outer = torch.bmm(X, Y.transpose(1, 2))
        # Return raw logits per base class contract; use predict() for probabilities
        return outer

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim": self.feature_dim,
            "eigenvalue_reg": self.eigenvalue_reg,
        }
