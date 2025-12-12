"""MLP baseline model for graph denoising.

A simple multi-layer perceptron that flattens the adjacency matrix, processes
it through hidden layers, and reshapes back. Tests whether the training
pipeline can train any neural network at all.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from tmgg.models.base import DenoisingModel


class MLPBaseline(DenoisingModel):
    """MLP baseline: flatten -> MLP -> reshape.

    This model treats the adjacency matrix as a flat vector, processes it
    through a standard MLP, and reshapes back to a matrix. It ignores graph
    structure entirely but should be able to learn if the training loop works.

    Use this model as a sanity check: if an MLP cannot learn to denoise,
    the training pipeline itself has issues.

    Parameters
    ----------
    max_nodes
        Maximum number of nodes in the graph.
    hidden_dim
        Size of hidden layers in the MLP. Default 256.
    num_layers
        Number of hidden layers. Default 2.

    Attributes
    ----------
    mlp : nn.Sequential
        The MLP architecture with ReLU activations.

    Examples
    --------
    >>> model = MLPBaseline(max_nodes=32, hidden_dim=256)
    >>> A = torch.rand(4, 32, 32)  # batch of 4
    >>> logits = model(A)  # Returns raw logits
    >>> probs = model.predict(logits)  # Apply sigmoid for [0, 1]
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.flatten_dim = max_nodes * max_nodes

        # Build MLP
        layers = []
        in_dim = self.flatten_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, self.flatten_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Apply MLP to flattened adjacency matrix.

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

        # Pad to max_nodes if needed
        if N < self.max_nodes:
            A_padded = torch.zeros(B, self.max_nodes, self.max_nodes, device=A.device)
            A_padded[:, :N, :N] = A
        else:
            A_padded = A

        # Flatten -> MLP -> reshape
        flat = A_padded.view(B, -1)
        out = self.mlp(flat)
        out = out.view(B, self.max_nodes, self.max_nodes)

        # Slice back to original size
        if N < self.max_nodes:
            out = out[:, :N, :N]

        return out

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "model_type": "MLPBaseline",
            "max_nodes": self.max_nodes,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "flatten_dim": self.flatten_dim,
        }
