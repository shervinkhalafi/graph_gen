"""PyTorch Lightning module for GNN-based denoising."""

from typing import Any

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
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: dict[str, Any] | None = None,
        noise_levels: list[float] | None = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: int | None = None,
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
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
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
        """Compute adjacency logits from input.

        Parameters
        ----------
        x
            Input adjacency matrix of shape (batch, n, n).

        Returns
        -------
        torch.Tensor
            Adjacency logits (pre-sigmoid) of shape (batch, n, n).
        """
        # All GNN models now return adjacency logits directly
        return self.model(x)
