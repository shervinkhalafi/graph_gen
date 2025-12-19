"""PyTorch Lightning module for hybrid GNN+Transformer denoising."""

from typing import Any

import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.hybrid import create_sequential_model


class HybridDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for hybrid GNN+Transformer graph denoising."""

    def __init__(
        self,
        # GNN configuration
        gnn_num_layers: int = 2,
        gnn_num_terms: int = 2,
        gnn_feature_dim_in: int = 20,
        gnn_feature_dim_out: int = 5,
        # Transformer configuration
        use_transformer: bool = True,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 4,
        transformer_d_k: int | None = None,
        transformer_d_v: int | None = None,
        transformer_dropout: float = 0.0,
        transformer_bias: bool = True,
        # Training configuration
        learning_rate: float = 0.005,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "BCE",
        scheduler_config: dict[str, Any] | None = None,
        noise_levels: list[float] | None = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: int | None = None,
    ):
        """
        Initialize the Lightning module.

        Args:
            gnn_num_layers: Number of GNN layers
            gnn_num_terms: Number of polynomial terms in GNN
            gnn_feature_dim_in: GNN input feature dimension
            gnn_feature_dim_out: GNN output feature dimension
            use_transformer: Whether to use transformer denoising
            transformer_num_layers: Number of transformer layers
            transformer_num_heads: Number of attention heads
            transformer_d_k: Key dimension for transformer
            transformer_d_v: Value dimension for transformer
            transformer_dropout: Transformer dropout rate
            transformer_bias: Whether to use bias in transformer
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration
            noise_levels: List of noise levels for evaluation
            noise_type: Type of noise to apply ("gaussian", "digress", "rotation")
            rotation_k: Dimension for rotation noise skew matrix
            seed: Random seed for reproducible noise generation
        """
        super().__init__(
            gnn_num_layers=gnn_num_layers,
            gnn_num_terms=gnn_num_terms,
            gnn_feature_dim_in=gnn_feature_dim_in,
            gnn_feature_dim_out=gnn_feature_dim_out,
            use_transformer=use_transformer,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_d_k=transformer_d_k,
            transformer_d_v=transformer_d_v,
            transformer_dropout=transformer_dropout,
            transformer_bias=transformer_bias,
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
        # GNN configuration
        gnn_num_layers: int = 2,
        gnn_num_terms: int = 2,
        gnn_feature_dim_in: int = 20,
        gnn_feature_dim_out: int = 5,
        # Transformer configuration
        use_transformer: bool = True,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 4,
        transformer_d_k: int | None = None,
        transformer_d_v: int | None = None,
        transformer_dropout: float = 0.0,
        transformer_bias: bool = True,
        **kwargs,
    ) -> nn.Module:
        # GNN configuration
        gnn_config = {
            "num_layers": gnn_num_layers,
            "num_terms": gnn_num_terms,
            "feature_dim_in": gnn_feature_dim_in,
            "feature_dim_out": gnn_feature_dim_out,
        }

        # Transformer configuration
        transformer_config = None
        if use_transformer:
            transformer_config = {
                "num_layers": transformer_num_layers,
                "num_heads": transformer_num_heads,
                "d_k": transformer_d_k,
                "d_v": transformer_d_v,
                "dropout": transformer_dropout,
                "bias": transformer_bias,
            }

        # Create hybrid model
        model = create_sequential_model(gnn_config, transformer_config)
        return model

    def get_model_name(self) -> str:
        """Get the name of the model for visualization purposes."""
        return "Hybrid"
