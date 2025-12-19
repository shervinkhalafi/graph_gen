"""Linear baseline model for graph denoising.

A minimal model that learns a direct linear transformation of the adjacency matrix.
Initialized at identity to enable learning from a reasonable starting point.
"""

from typing import Any

import torch
import torch.nn as nn

from tmgg.models.base import DenoisingModel


class LinearBaseline(DenoisingModel):
    """Linear transformation baseline: A_pred = W @ A @ W.T + b.

    This model applies a learnable linear transformation to the input adjacency
    matrix. Initialized with W=I (identity) so the initial output equals the
    input, providing a stable starting point for gradient descent.

    Use this model to verify the training pipeline works correctly. If this
    simple model cannot learn, the issue lies in the training loop or data,
    not in model architecture complexity.

    Parameters
    ----------
    max_nodes
        Maximum number of nodes in the graph. Determines parameter dimensions.

    Attributes
    ----------
    W : nn.Parameter
        Learnable weight matrix of shape (max_nodes, max_nodes).
    b : nn.Parameter
        Learnable bias matrix of shape (max_nodes, max_nodes).

    Examples
    --------
    >>> model = LinearBaseline(max_nodes=32)
    >>> A = torch.eye(32).unsqueeze(0)  # batch of 1
    >>> logits = model(A)  # Returns raw logits
    >>> probs = model.predict(logits)  # Apply sigmoid for [0, 1]
    """

    def __init__(self, max_nodes: int):
        super().__init__()
        self.max_nodes = max_nodes

        # Initialize W at identity for stable starting point
        self.W = nn.Parameter(torch.eye(max_nodes))
        self.b = nn.Parameter(torch.zeros(max_nodes, max_nodes))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation to input adjacency matrix.

        Parameters
        ----------
        A
            Input adjacency matrix of shape (batch, N, N) where N <= max_nodes.

        Returns
        -------
        torch.Tensor
            Raw logits of shape (batch, N, N). Use predict() for probabilities.
        """
        B, N, _ = A.shape

        # Handle variable-size graphs by slicing parameters
        W = self.W[:N, :N]
        b = self.b[:N, :N]

        # A_pred = W @ A @ W.T + b
        # Expand W for batch matmul: (1, N, N) -> broadcast with (B, N, N)
        W_expanded = W.unsqueeze(0).expand(B, -1, -1)
        out = torch.bmm(torch.bmm(W_expanded, A), W_expanded.transpose(-2, -1))
        out = out + b.unsqueeze(0)

        return out

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        return {
            "model_type": "LinearBaseline",
            "max_nodes": self.max_nodes,
        }
