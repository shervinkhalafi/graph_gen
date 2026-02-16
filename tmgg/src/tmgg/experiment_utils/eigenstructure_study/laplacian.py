"""Laplacian matrix computation utilities.

Provides functions for computing the combinatorial Laplacian L = D - A
where D is the degree matrix and A is the adjacency matrix.
"""

import torch


def compute_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the combinatorial Laplacian L = D - A.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape (n, n) for a single graph,
        or (batch, n, n) for a batch of graphs.

    Returns
    -------
    torch.Tensor
        Laplacian matrix with same shape as input.

    Notes
    -----
    The combinatorial Laplacian has eigenvalues in [0, 2*d_max] for
    simple graphs. For connected graphs, the smallest eigenvalue is 0
    with multiplicity equal to the number of connected components.
    """
    if A.dim() == 2:
        degrees = A.sum(dim=1)
        D = torch.diag(degrees)
        return D - A
    elif A.dim() == 3:
        degrees = A.sum(dim=-1)
        D = torch.diag_embed(degrees)
        return D - A
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {A.shape}")


def compute_normalized_laplacian(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape (n, n) or (batch, n, n).
    eps : float
        Small value added to degrees to avoid division by zero
        for isolated nodes.

    Returns
    -------
    torch.Tensor
        Normalized Laplacian with same shape as input.

    Notes
    -----
    The normalized Laplacian has eigenvalues in [0, 2] for simple graphs.
    This formulation is symmetric and thus has real eigenvalues.
    """
    if A.dim() == 2:
        degrees = A.sum(dim=1)
        deg_inv_sqrt = (degrees + eps).pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        identity = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return identity - D_inv_sqrt @ A @ D_inv_sqrt
    elif A.dim() == 3:
        degrees = A.sum(dim=-1)
        deg_inv_sqrt = (degrees + eps).pow(-0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        n = A.shape[-1]
        identity = (
            torch.eye(n, device=A.device, dtype=A.dtype)
            .unsqueeze(0)
            .expand(A.shape[0], -1, -1)
        )
        return identity - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {A.shape}")
