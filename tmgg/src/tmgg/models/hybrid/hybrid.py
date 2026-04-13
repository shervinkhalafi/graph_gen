"""Hybrid models combining GNN embeddings with transformer denoising."""

from typing import Any, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import GraphData

from ..attention import MultiLayerAttention
from ..base import BaseModel, EmbeddingProvider, GraphModel
from ..gnn import GNN


class SequentialDenoisingModel(GraphModel):
    """Sequential model combining GNN embeddings with attention-based denoising.

    First generates embeddings using a GNN, then applies a transformer
    to denoise these embeddings, and finally reconstructs the adjacency matrix.
    """

    embedding_model: EmbeddingProvider
    denoising_model: nn.Module | None

    def __init__(
        self,
        embedding_model: EmbeddingProvider,
        denoising_model: nn.Module | None = None,
    ) -> None:
        """Initialize the sequential denoising model.

        Parameters
        ----------
        embedding_model
            Model satisfying ``EmbeddingProvider``: must expose
            ``embeddings(GraphData) -> (X, Y)`` and ``get_config()``.
        denoising_model
            Optional model for denoising concatenated embeddings. If it is a
            ``MultiLayerAttention``, the raw ``apply_attention()`` path is
            used; otherwise it is called directly on the feature tensor.
        """
        super().__init__()
        if not isinstance(embedding_model, EmbeddingProvider):
            raise TypeError(
                f"embedding_model must satisfy EmbeddingProvider protocol "
                f"(requires embeddings() and get_config() methods), "
                f"got {type(embedding_model).__name__}"
            )
        self.embedding_model = embedding_model  # pyright: ignore[reportIncompatibleVariableOverride]
        self.denoising_model = denoising_model

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Combine GNN embedding and transformer denoising.

        Parameters
        ----------
        data
            Graph features. The dense edge state is extracted via
            ``data.to_edge_state()`` and passed to the embedding model.
        t
            Diffusion timestep tensor, or None. Currently unused.

        Returns
        -------
        GraphData
            Denoised graph with 2-class edge features.
        """
        # Generate embeddings via the EmbeddingProvider protocol
        X, Y = self.embedding_model.embeddings(data)

        # Concatenate embeddings
        Z = torch.cat([X, Y], dim=2)  # Shape: (batch_size, num_nodes, 2*feature_dim)

        # Apply denoising if available
        if self.denoising_model is not None:
            # Use raw tensor path for MultiLayerAttention; fall back to direct call
            if isinstance(self.denoising_model, MultiLayerAttention):
                Z_denoised = self.denoising_model.apply_attention(Z)
            else:
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

        # Reconstruct adjacency matrix
        A_recon = torch.bmm(X_pred, Y_pred.transpose(1, 2))
        return GraphData.from_edge_state(A_recon, node_mask=data.node_mask)

    @override
    def get_config(self) -> dict[str, object]:
        """Get configuration for both embedding and denoising components."""
        embedding_config = self.embedding_model.get_config()

        config: dict[str, object] = {
            "model_type": "SequentialDenoisingModel",
            "embedding_model": embedding_config,
            "has_denoising": self.denoising_model is not None,
        }
        if self.denoising_model is not None and isinstance(
            self.denoising_model, BaseModel
        ):
            config["denoising_model"] = self.denoising_model.get_config()
        return config


def create_sequential_model(
    gnn_config: dict[str, Any], transformer_config: dict[str, Any] | None = None
) -> SequentialDenoisingModel:
    """Create a sequential denoising model from GNN + optional attention config.

    Parameters
    ----------
    gnn_config
        Configuration for the GNN embedding model.
    transformer_config
        Optional configuration for the attention denoising model.

    Returns
    -------
    SequentialDenoisingModel
        Configured model combining GNN embeddings with optional attention.
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

    return SequentialDenoisingModel(embedding_model, denoising_model)
