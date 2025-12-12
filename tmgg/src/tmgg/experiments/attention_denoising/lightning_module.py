"""PyTorch Lightning module for attention-based denoising."""

from typing import Any, Dict, List, Optional, override

import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.attention import MultiLayerAttention


class AttentionDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for attention-based graph denoising."""

    def __init__(
        self,
        d_model: int = 20,
        num_heads: int = 8,
        num_layers: int = 8,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: Optional[int] = None,
        domain: str = "standard",
    ):
        """
        Initialize the Lightning module.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            d_k: Key dimension (default: d_model // num_heads)
            d_v: Value dimension (default: d_model // num_heads)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration
            noise_levels: List of noise levels to sample from during training
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
            rotation_k: Dimension for rotation noise skew matrix (number of eigenvectors)
            seed: Random seed for reproducible noise generation
            domain: Domain for adjacency matrix processing ("standard" or "inv-sigmoid")
        """
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            bias=bias,
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
            domain=domain,
        )

    @override
    def _make_model(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_k: int,
        d_v: int,
        dropout: float,
        bias: bool,
        domain: str,
        *args: Any,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> nn.Module:
        return MultiLayerAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            bias=bias,
            domain=domain,
        )

    def get_model_name(self) -> str:
        """Get the name of the model for visualization purposes."""
        return "Attention"
