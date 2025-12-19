"""Eigenvalue and eigenvector computation utilities."""

import numpy as np
import torch


def compute_eigendecomposition(
    A: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix.

    Args:
        A: Input matrix (symmetric)

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvalues are sorted
        in ascending order
    """
    # Convert to tensor if needed
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return eigenvalues, eigenvectors


def compute_top_k_eigendecomposition(
    A: torch.Tensor | np.ndarray, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k eigenvalues and eigenvectors of a symmetric matrix.

    Args:
        A: Input matrix (symmetric)
        k: Number of top eigenvalues/eigenvectors to compute

    Returns:
        Tuple of (top_k_eigenvalues, top_k_eigenvectors) sorted by magnitude
        in descending order
    """
    eigenvalues, eigenvectors = compute_eigendecomposition(A)

    # Convert to tensor if needed for ndim check
    A_tensor = torch.tensor(A) if isinstance(A, np.ndarray) else A

    # Get indices of top-k eigenvalues by magnitude
    if A_tensor.dim() == 2:
        # Single matrix case
        _, indices = torch.topk(torch.abs(eigenvalues), k)
        top_eigenvalues = eigenvalues[indices]
        top_eigenvectors = eigenvectors[:, indices]
    else:
        # Batch case
        _, indices = torch.topk(torch.abs(eigenvalues), k, dim=-1)
        batch_size = eigenvalues.shape[0]
        batch_indices = torch.arange(batch_size).unsqueeze(1)
        top_eigenvalues = eigenvalues[batch_indices, indices]
        top_eigenvectors = eigenvectors[
            batch_indices.unsqueeze(2), :, indices.unsqueeze(1)
        ]
        top_eigenvectors = top_eigenvectors.transpose(1, 2)

    return top_eigenvalues, top_eigenvectors


def compute_spectral_distance(
    A1: torch.Tensor | np.ndarray,
    A2: torch.Tensor | np.ndarray,
    metric: str = "frobenius",
) -> torch.Tensor:
    """
    Compute distance between two matrices based on their eigendecompositions.

    Args:
        A1: First matrix
        A2: Second matrix
        metric: Distance metric to use ("frobenius", "eigenvalue", "subspace")

    Returns:
        Distance value
    """
    eigenvalues1, eigenvectors1 = compute_eigendecomposition(A1)
    eigenvalues2, eigenvectors2 = compute_eigendecomposition(A2)

    if metric == "frobenius":
        # Frobenius norm of the difference
        return torch.norm(A1 - A2, p="fro")
    elif metric == "eigenvalue":
        # L2 distance between eigenvalue vectors
        return torch.norm(eigenvalues1 - eigenvalues2, p=2)
    elif metric == "subspace":
        # Subspace distance based on eigenvector alignment
        # Using the Frobenius norm of I - V1^T V2 V2^T V1
        A1_tensor = torch.tensor(A1) if isinstance(A1, np.ndarray) else A1
        if A1_tensor.dim() == 2:
            projection = eigenvectors1.T @ eigenvectors2
            return torch.norm(
                torch.eye(projection.shape[0]) - projection @ projection.T, p="fro"
            )
        else:
            # Batch case
            projection = torch.bmm(eigenvectors1.transpose(-2, -1), eigenvectors2)
            identity = (
                torch.eye(projection.shape[-1]).unsqueeze(0).expand_as(projection)
            )
            return torch.norm(
                identity - torch.bmm(projection, projection.transpose(-2, -1)),
                p="fro",
                dim=(-2, -1),
            )
    else:
        raise ValueError(f"Unknown metric: {metric}")
