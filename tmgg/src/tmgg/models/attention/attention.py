"""Attention-based models for graph denoising."""

from typing import Any, override

import torch
import torch.nn as nn

from tmgg.models.layers.mha_layer import MultiHeadAttention

from ..base import DenoisingModel


class MultiLayerAttention(DenoisingModel):
    """Multi-layer attention model for graph denoising."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Initialize the Multi-Layer Attention module.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            d_k: Dimension of keys (default: d_model // num_heads)
            d_v: Dimension of values (default: d_model // num_heads)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        self.dropout = dropout
        self.bias = bias

        # Create stack of attention layers
        self.layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model,
                    num_heads=num_heads,
                    d_k=self.d_k,
                    d_v=self.d_v,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, A: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through all attention layers.

        Args:
            A: Input adjacency matrix
            mask: Optional attention mask

        Returns:
            Reconstructed adjacency matrix
        """
        x = A
        # Pass through each attention layer sequentially
        for layer in self.layers:
            x, _ = layer(x, mask=mask)

        # Return raw logits per base class contract; use predict() for probabilities
        return x

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "dropout": self.dropout,
            "bias": self.bias,
        }
