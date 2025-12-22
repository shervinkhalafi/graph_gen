"""Hybrid models combining GNN embeddings with transformer denoising."""

from typing import Any, override

import torch
import torch.nn as nn

from ..attention import MultiLayerAttention
from ..base import DenoisingModel
from ..gnn import GNN


class SequentialDenoisingModel(DenoisingModel):
    """Sequential model combining GNN embeddings with attention-based denoising.

    First generates embeddings using a GNN, then applies a transformer
    to denoise these embeddings, and finally reconstructs the adjacency matrix.
    """

    embedding_model: nn.Module
    denoising_model: DenoisingModel | None

    def __init__(
        self,
        embedding_model: nn.Module,
        denoising_model: DenoisingModel | None = None,
    ) -> None:
        """Initialize the sequential denoising model.

        Parameters
        ----------
        embedding_model
            Model for generating embeddings (must have embeddings() method
            returning (X, Y) tuple of tensors).
        denoising_model
            Optional transformer model for denoising embeddings.
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.denoising_model = denoising_model

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine GNN embedding and transformer denoising.

        Parameters
        ----------
        x
            Input adjacency matrix of shape (batch, n, n).

        Returns
        -------
        torch.Tensor
            Adjacency logits (pre-sigmoid) of shape (batch, n, n).
        """
        # Generate embeddings using GNN's embeddings() method
        if not hasattr(self.embedding_model, "embeddings"):
            raise ValueError(
                "Embedding model must have embeddings() method returning (X, Y) tuple"
            )
        # Duck-typed access to embeddings method - hasattr check above ensures safety
        X, Y = self.embedding_model.embeddings(x)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

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

    @override
    def get_config(self) -> dict[str, object]:
        """Get configuration for both embedding and denoising components."""
        embedding_config: object
        if hasattr(self.embedding_model, "get_config"):
            # Duck-typed access - hasattr check ensures safety
            embedding_config = self.embedding_model.get_config()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        else:
            embedding_config = str(type(self.embedding_model))

        config: dict[str, object] = {
            "model_type": "SequentialDenoisingModel",
            "embedding_model": embedding_config,
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
