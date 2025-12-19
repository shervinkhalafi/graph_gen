"""Hybrid models combining GNN embeddings with transformer denoising."""

from typing import Any

import torch

from ..attention import MultiLayerAttention
from ..base import DenoisingModel, EmbeddingModel
from ..gnn import GNN


class SequentialDenoisingModel(DenoisingModel):
    """
    Sequential model combining GNN embeddings with attention-based denoising.

    This model first generates embeddings using a GNN, then applies
    a transformer to denoise these embeddings, and finally reconstructs
    the adjacency matrix.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        denoising_model: DenoisingModel | None = None,
    ):
        """
        Initialize the sequential denoising model.

        Args:
            embedding_model: GNN model for generating embeddings
            denoising_model: Optional transformer model for denoising embeddings
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.denoising_model = denoising_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining GNN embedding and transformer denoising.

        Args:
            x: Input adjacency matrix

        Returns:
            Reconstructed adjacency matrix
        """
        # Generate embeddings using GNN
        if hasattr(self.embedding_model, "forward") and callable(
            self.embedding_model.forward
        ):
            # For GNN models that return (X, Y) embeddings
            X, Y = self.embedding_model(x)
        else:
            raise ValueError("Embedding model must return (X, Y) embeddings")

        # Concatenate embeddings
        Z = torch.cat([X, Y], dim=2)  # Shape: (batch_size, num_nodes, 2*feature_dim)

        # Apply denoising if available
        if self.denoising_model is not None:
            # Apply transformer denoising and add residual connection
            Z_denoised = self.denoising_model(Z)
            # Validate shape match for residual connection
            assert Z_denoised.shape == Z.shape, (
                f"Denoising model output shape {Z_denoised.shape} != input shape {Z.shape}. "
                "Check d_model matches 2 * feature_dim_out."
            )
            Z_pred = Z + Z_denoised
        else:
            Z_pred = Z

        # Split back into X and Y embeddings
        feature_dim = X.shape[2]
        X_pred = Z_pred[:, :, :feature_dim]
        Y_pred = Z_pred[:, :, feature_dim:]

        # Reconstruct adjacency matrix - return raw logits per base class contract
        A_recon = self._reconstruct_adjacency_from_embeddings(X_pred, Y_pred)
        return A_recon

    def _reconstruct_adjacency_from_embeddings(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct adjacency matrix from X and Y embeddings.

        Args:
            X: First set of embeddings
            Y: Second set of embeddings

        Returns:
            Reconstructed adjacency matrix
        """
        # Compute outer product and apply output transformation based on domain and training mode
        A_recon = torch.bmm(X, Y.transpose(1, 2))
        # Note: This method doesn't have access to domain/training info, so return raw logits
        # The calling model should handle the transformation
        return A_recon

    def get_config(self) -> dict[str, Any]:
        """Get configuration for both embedding and denoising components."""
        config = {
            "model_type": "SequentialDenoisingModel",
            "embedding_model": self.embedding_model.get_config()
            if hasattr(self.embedding_model, "get_config")
            else str(type(self.embedding_model)),
            "has_denoising": self.denoising_model is not None,
        }
        if self.denoising_model is not None and hasattr(
            self.denoising_model, "get_config"
        ):
            config["denoising_model"] = self.denoising_model.get_config()
        return config


def create_sequential_model(
    gnn_config: dict[str, Any], transformer_config: dict[str, Any] | None = None
) -> SequentialDenoisingModel:
    """
    Factory function to create a sequential denoising model.

    Args:
        gnn_config: Configuration for the GNN embedding model
        transformer_config: Optional configuration for the transformer denoising model

    Returns:
        Configured SequentialDenoisingModel
    """
    # Create GNN embedding model
    embedding_model = GNN(
        num_layers=gnn_config.get("num_layers", 2),
        num_terms=gnn_config.get("num_terms", 2),
        feature_dim_in=gnn_config.get("feature_dim_in", 20),
        feature_dim_out=gnn_config.get("feature_dim_out", 5),
        eigenvalue_reg=gnn_config.get("eigenvalue_reg", 0.0),
    )

    # Create transformer denoising model if config provided
    denoising_model = None
    if transformer_config is not None:
        # The input dimension for the transformer is 2 * feature_dim_out (concatenated X, Y)
        d_model = 2 * gnn_config.get("feature_dim_out", 5)

        denoising_model = MultiLayerAttention(
            d_model=d_model,
            num_heads=transformer_config.get("num_heads", 4),
            num_layers=transformer_config.get("num_layers", 4),
            d_k=transformer_config.get("d_k", None),
            d_v=transformer_config.get("d_v", None),
            dropout=transformer_config.get("dropout", 0.0),
            bias=transformer_config.get("bias", True),
        )

    # GNN produces (X, Y) embeddings and implements both EmbeddingModel and DenoisingModel
    return SequentialDenoisingModel(embedding_model, denoising_model)
