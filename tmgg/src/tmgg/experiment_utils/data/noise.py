"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.linalg import expm
from typing import List, Tuple, Optional, Union
import random


def random_skew_symmetric_matrix(n: int) -> np.ndarray:
    """
    Create a random n×n skew-symmetric matrix.

    Args:
        n: Matrix dimension

    Returns:
        Skew-symmetric matrix
    """
    A = np.random.rand(n, n)
    return (A - A.T) / 2


def add_rotation_noise(
    A: Union[torch.Tensor, np.ndarray], eps: float, skew: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add rotation noise to adjacency matrix by rotating eigenvectors.

    Args:
        A: Input adjacency matrix
        eps: Noise level
        skew: Skew-symmetric matrix for rotation

    Returns:
        Tuple of (noisy_adjacency, rotated_eigenvectors, eigenvalues)
    """
    # Convert to tensor if needed
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)
    else:
        A = A.float()

    # Handle both single matrix and batch cases
    if A.dim() == 2:
        # Single matrix case
        l, V = torch.linalg.eigh(A)
        R = expm(eps * skew)
        R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
        V_rot = V @ R_tensor
        A_noisy = V_rot @ torch.diag(l) @ V_rot.T
    else:
        # Batch case
        l, V = torch.linalg.eigh(A)
        R = expm(eps * skew)
        R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
        V_rot = V @ R_tensor

        # Initialize tensor of appropriate size
        l_diag = torch.zeros(
            l.shape[0], l.shape[1], l.shape[1], device=l.device, dtype=l.dtype
        )

        # Fill diagonal elements for each matrix in the batch
        batch_indices = torch.arange(l.shape[0])
        diag_indices = torch.arange(l.shape[1])
        l_diag[batch_indices[:, None], diag_indices, diag_indices] = l

        A_noisy = torch.matmul(
            torch.matmul(V_rot, l_diag), torch.transpose(V_rot, 1, 2)
        )

    return torch.real(A_noisy), torch.real(V_rot), torch.real(l)


def add_gaussian_noise(
    A: Union[torch.Tensor, np.ndarray], eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise to adjacency matrix.

    Args:
        A: Input adjacency matrix
        eps: Noise level

    Returns:
        Tuple of (noisy_adjacency, eigenvectors, eigenvalues)
    """
    # Convert to tensor if needed
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)
    else:
        A = A.float()

    A_noisy = A + eps * torch.randn_like(A)
    l, V = torch.linalg.eigh(A_noisy)
    return A_noisy, V, l


def add_digress_noise(
    A: Union[torch.Tensor, np.ndarray],
    p: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add noise to an adjacency matrix by flipping edges with probability p.

    Args:
        A: Input adjacency matrix (0s and 1s)
        p: Probability of flipping each element
        rng: Random number generator (optional)

    Returns:
        Tuple of (noisy_adjacency, eigenvectors, eigenvalues)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert to tensor if needed
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)
    else:
        A = A.float()

    # Generate random values for each element
    random_values = torch.rand_like(A)

    # Create masks for upper triangular part (excluding diagonal)
    if A.dim() == 2:
        upper_mask = torch.triu(torch.ones_like(A, dtype=torch.bool), diagonal=1)
    else:
        # Batch case
        batch_size = A.shape[0]
        num_nodes = A.shape[1]
        upper_mask = torch.triu(
            torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=A.device),
            diagonal=1,
        )
        upper_mask = upper_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Create flip mask only for upper triangle
    flip_mask_upper = (random_values < p) & upper_mask

    # Make symmetric flip mask by mirroring upper triangle to lower
    if A.dim() == 2:
        flip_mask = flip_mask_upper | flip_mask_upper.T
    else:
        flip_mask = flip_mask_upper | flip_mask_upper.transpose(-2, -1)

    # Flip the elements where the mask is True (using XOR operation)
    A_noisy = torch.where(flip_mask, 1 - A, A)

    l, V = torch.linalg.eigh(A_noisy)

    return A_noisy, V, l
