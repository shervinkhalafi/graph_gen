"""Graph Neural Network models for graph denoising."""

from typing import Any, override

import torch
import torch.nn as nn

from tmgg.models.layers.eigen_embedding import EigenEmbedding
from tmgg.models.layers.gcn import GraphConvolutionLayer

from ..base import DenoisingModel


class GNN(DenoisingModel):
    """Graph Neural Network for adjacency matrix reconstruction.

    Returns adjacency logits (pre-sigmoid) directly. Uses asymmetric embeddings
    (separate X and Y) with reconstruction via X @ Y.T.
    """

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim_in: int = 10,
        feature_dim_out: int = 10,
        eigenvalue_reg: float = 0.0,
    ):
        """Initialize GNN.

        Parameters
        ----------
        num_layers
            Number of graph convolution layers.
        num_terms
            Number of terms in polynomial filters.
        feature_dim_in
            Input feature dimension (truncated eigenvector dimension).
        feature_dim_out
            Output embedding dimension for adjacency reconstruction.
        eigenvalue_reg
            Diagonal regularization for eigendecomposition stability.
        """
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
        self.out_y = nn.Linear(feature_dim_in, feature_dim_out)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adjacency logits from input adjacency matrix.

        Parameters
        ----------
        x
            Input adjacency matrix of shape (batch, n, n).

        Returns
        -------
        torch.Tensor
            Adjacency logits (pre-sigmoid) of shape (batch, n, n).
        """
        z = self.embedding_layer(x)
        # Take only the first feature_dim_in columns from eigenvectors
        actual_feature_dim = min(z.shape[2], self.feature_dim_in)
        z = z[:, :, :actual_feature_dim]

        # Pad with zeros if we have fewer features than expected
        if actual_feature_dim < self.feature_dim_in:
            padding = torch.zeros(
                z.shape[0],
                z.shape[1],
                self.feature_dim_in - actual_feature_dim,
                device=z.device,
                dtype=z.dtype,
            )
            z = torch.cat([z, padding], dim=2)

        for layer in self.layers:
            z = layer(x, z)

        emb_x = self.out_x(z)
        emb_y = self.out_y(z)

        # Reconstruct adjacency logits via asymmetric outer product
        return torch.bmm(emb_x, emb_y.transpose(1, 2))

    def embeddings(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute node embeddings without adjacency reconstruction.

        Used by hybrid models that need embeddings for further processing.

        Parameters
        ----------
        x
            Input adjacency matrix of shape (batch, n, n).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (X, Y) embeddings, each of shape (batch, n, feature_dim_out).
        """
        z = self.embedding_layer(x)
        actual_feature_dim = min(z.shape[2], self.feature_dim_in)
        z = z[:, :, :actual_feature_dim]

        if actual_feature_dim < self.feature_dim_in:
            padding = torch.zeros(
                z.shape[0],
                z.shape[1],
                self.feature_dim_in - actual_feature_dim,
                device=z.device,
                dtype=z.dtype,
            )
            z = torch.cat([z, padding], dim=2)

        for layer in self.layers:
            z = layer(x, z)

        emb_x = self.out_x(z)
        emb_y = self.out_y(z)
        return emb_x, emb_y

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
            "eigenvalue_reg": self.eigenvalue_reg,
        }
