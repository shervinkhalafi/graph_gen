"""Utility layers for graph neural networks."""

import torch
import torch.nn as nn
from typing import Optional


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


def masked_softmax(x: torch.Tensor, mask: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
    """Compute softmax with optional masking.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    mask : torch.Tensor, optional
        Binary mask (1 for valid positions, 0 for masked)
    **kwargs
        Additional arguments for torch.softmax (e.g., dim)
        
    Returns
    -------
    torch.Tensor
        Softmax output with masked positions set to 0
        
    Notes
    -----
    When all positions are masked, returns zeros instead of NaN.
    """
    if mask is None:
        return torch.softmax(x, **kwargs)
    
    if mask.sum() == 0:
        return torch.zeros_like(x)
    
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)