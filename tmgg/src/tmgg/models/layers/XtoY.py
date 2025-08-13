"""Utility layers for digress style graph neural networks."""

import torch
import torch.nn as nn


class Xtoy(nn.Module):
    """Aggregate node features to graph-level features via statistics.

    Computes mean, min, max, and std across nodes, then projects to output dimension.
    """

    def __init__(self, dx: int, dy: int):
        """
        Parameters
        ----------
        dx : int
            Node feature dimension
        dy : int
            Output global feature dimension
        """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : torch.Tensor
            Node features of shape (batch_size, n_nodes, dx)

        Returns
        -------
        torch.Tensor
            Global features of shape (batch_size, dy)
        """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out
