"""PyTorch Lightning module for GNN-based denoising."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.gnn import GNN, GNNSymmetric, NodeVarGNN


class GNNDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for GNN-based graph denoising."""

    def __init__(
        self,
        model_type: str = "GNN",
        num_layers: int = 1,
        num_terms: int = 4,
        feature_dim_in: int = 20,
        feature_dim_out: int = 20,
        learning_rate: float = 0.001,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Lightning module.

        Args:
            model_type: Type of GNN model ("GNN", "NodeVarGNN", "GNNSymmetric")
            num_layers: Number of GNN layers
            num_terms: Number of polynomial terms
            feature_dim_in: Input feature dimension
            feature_dim_out: Output feature dimension
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration
            noise_levels: List of noise levels to sample from during training
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
            rotation_k: Dimension for rotation noise skew matrix
            seed: Random seed for reproducible noise generation
        """
        super().__init__(
            model_type=model_type,
            num_layers=num_layers,
            num_terms=num_terms,
            feature_dim_in=feature_dim_in,
            feature_dim_out=feature_dim_out,
            learning_rate=learning_rate,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            noise_levels=noise_levels,
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed,
        )

    def _make_model(
        self,
        *args,
        model_type: str = "GNN",
        num_layers: int = 1,
        num_terms: int = 4,
        feature_dim_in: int = 20,
        feature_dim_out: int = 20,
        **kwargs,
    ) -> nn.Module:
        # Model selection
        if model_type == "GNN":
            model = GNN(
                num_layers=num_layers,
                num_terms=num_terms,
                feature_dim_in=feature_dim_in,
                feature_dim_out=feature_dim_out,
            )
        elif model_type == "NodeVarGNN":
            model = NodeVarGNN(
                num_layers=num_layers,
                num_terms=num_terms,
                feature_dim=feature_dim_out,  # NodeVarGNN uses single feature_dim
            )
        elif model_type == "GNNSymmetric":
            model = GNNSymmetric(
                num_layers=num_layers,
                num_terms=num_terms,
                feature_dim_in=feature_dim_in,
                feature_dim_out=feature_dim_out,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model

    def get_model_name(self) -> str:
        """Get the name of the model for visualization purposes."""
        return "GNN"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN model.

        Args:
            x: Input adjacency matrix

        Returns:
            Reconstructed adjacency matrix
        """
        model_output = self.model(x)
        
        # Handle different return types from different GNN models
        if isinstance(model_output, tuple):
            # Standard GNN and GNNSymmetric return (embeddings, adjacency) or (x_embed, y_embed)
            # For standard GNN, we need to reconstruct adjacency from embeddings
            if hasattr(self.model, 'model_type') and self.model.model_type == "GNNSymmetric":
                # GNNSymmetric returns (reconstructed_adjacency, embeddings)
                return model_output[0]
            else:
                # Standard GNN returns (X_embeddings, Y_embeddings)
                # We need to reconstruct adjacency matrix from these embeddings
                X, Y = model_output
                # Simple reconstruction: A_reconstructed = X @ Y.T
                return torch.bmm(X, Y.transpose(-2, -1))
        else:
            # NodeVarGNN returns adjacency matrix directly
            return model_output
