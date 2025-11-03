"""Utility layers for digress style graph neural networks."""

import torch
import torch.nn as nn


class Etoy(nn.Module):
    """Aggregate edge features to graph-level features via statistics.

    Computes mean, min, max, and std across all edges, then projects to output dimension.
    """

    def __init__(self, de: int, dy: int):
        """
        Parameters
        ----------
        de : int
            Edge feature dimension
        dy : int
            Output global feature dimension
        """
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        E : torch.Tensor
            Edge features of shape (batch_size, n_nodes, n_nodes, de)

        Returns
        -------
        torch.Tensor
            Global features of shape (batch_size, dy)
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out
