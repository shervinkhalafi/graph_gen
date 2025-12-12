"""Graph Neural Network models for graph denoising."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from tmgg.models.layers.eigen_embedding import EigenEmbedding
from tmgg.models.layers.gcn import GraphConvolutionLayer

from ..base import DenoisingModel


class GNNSymmetric(DenoisingModel):
    """Symmetric GNN using same embedding for both X and Y."""

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim_in: int = 10,
        feature_dim_out: int = 10,
        eigenvalue_reg: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim_in = feature_dim_in
        self.feature_dim_out = feature_dim_out
        self.eigenvalue_reg = eigenvalue_reg

        self.embedding_layer = EigenEmbedding(eigenvalue_reg=eigenvalue_reg)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolutionLayer(num_terms, feature_dim_in))

        self.out_x = nn.Linear(feature_dim_in, feature_dim_out)

    def forward(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with symmetric embeddings.

        Args:
            A: Adjacency matrix

        Returns:
            Tuple of (reconstructed_adjacency, X_embeddings)
        """
        Z = self.embedding_layer(A)
        # Take only the first feature_dim_in columns from eigenvectors
        # But ensure we don't exceed the available columns
        actual_feature_dim = min(Z.shape[2], self.feature_dim_in)
        Z = Z[:, :, :actual_feature_dim]

        # If we have fewer features than expected, pad with zeros
        if actual_feature_dim < self.feature_dim_in:
            padding = torch.zeros(
                Z.shape[0],
                Z.shape[1],
                self.feature_dim_in - actual_feature_dim,
                device=Z.device,
                dtype=Z.dtype,
            )
            Z = torch.cat([Z, padding], dim=2)
        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)

        outer = torch.bmm(X, X.transpose(1, 2))
        # Return raw logits per base class contract; use predict() for probabilities
        return outer, X

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
            "eigenvalue_reg": self.eigenvalue_reg,
        }
