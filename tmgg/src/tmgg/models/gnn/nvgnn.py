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
        domain: str = "standard",
    ):
        """
        Initialize Node-variant GNN.

        Args:
            num_layers: Number of layers
            num_terms: Number of polynomial terms
            feature_dim: Feature dimension
            domain: Domain for adjacency matrix processing ("standard" or "inv-sigmoid")
        """
        super(NodeVarGNN, self).__init__(domain=domain)

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim = feature_dim

        self.embedding_layer = EigenEmbedding()

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
        # Apply domain transformation to input
        A_transformed = self._apply_domain_transform(A)

        Z = self.embedding_layer(A_transformed)
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

        # Dynamically create layers if needed with correct num_nodes
        if len(self.layers) == 0 or self.layers[0].num_nodes != Z.shape[1]:
            self.layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.layers.append(
                    NodeVarGraphConvolutionLayer(
                        self.num_terms, self.feature_dim, Z.shape[1]
                    )
                )

        for layer in self.layers:
            Z = layer(A_transformed, Z)
        X = self.out_x(Z)
        Y = self.out_y(Z)
        outer = torch.bmm(X, Y.transpose(1, 2))
        # Apply output transformation based on domain and training mode
        return self._apply_output_transform(outer)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim": self.feature_dim,
            "domain": self.domain,
        }
