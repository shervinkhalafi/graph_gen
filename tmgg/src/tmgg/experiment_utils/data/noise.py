"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np
import torch
from scipy.linalg import expm


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
    A: torch.Tensor | np.ndarray, eps: float, skew: np.ndarray
) -> torch.Tensor:
    """
    Add rotation noise to adjacency matrix by rotating eigenvectors.

    Args:
        A: Input adjacency matrix
        eps: Noise level
        skew: Skew-symmetric matrix for rotation

    Returns:
        Noisy adjacency matrix
    """
    # Convert to tensor if needed
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    # Handle both single matrix and batch cases
    if A.dim() == 2:
        # Single matrix case
        eigenvalues, V = torch.linalg.eigh(A)
        R = expm(eps * skew)
        R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
        V_rot = V @ R_tensor
        A_noisy = V_rot @ torch.diag(eigenvalues) @ V_rot.T
    else:
        # Batch case
        eigenvalues, V = torch.linalg.eigh(A)
        R = expm(eps * skew)
        R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
        V_rot = V @ R_tensor

        # Initialize tensor of appropriate size
        eig_diag = torch.zeros(
            eigenvalues.shape[0],
            eigenvalues.shape[1],
            eigenvalues.shape[1],
            device=eigenvalues.device,
            dtype=eigenvalues.dtype,
        )

        # Fill diagonal elements for each matrix in the batch
        batch_indices = torch.arange(eigenvalues.shape[0])
        diag_indices = torch.arange(eigenvalues.shape[1])
        eig_diag[batch_indices[:, None], diag_indices, diag_indices] = eigenvalues

        A_noisy = torch.matmul(
            torch.matmul(V_rot, eig_diag), torch.transpose(V_rot, 1, 2)
        )

    return torch.real(A_noisy)


def add_gaussian_noise(A: torch.Tensor | np.ndarray, eps: float) -> torch.Tensor:
    """
    Add Gaussian noise to adjacency matrix.

    Args:
        A: Input adjacency matrix
        eps: Noise level

    Returns:
        Noisy adjacency matrix
    """
    # Convert to tensor if needed
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    A_noisy = A + eps * torch.randn_like(A)
    return A_noisy


def add_logit_noise(
    A: torch.Tensor | np.ndarray,
    sigma: float,
    clamp_eps: float = 1e-6,
) -> torch.Tensor:
    """Add Gaussian noise in logit space, returning soft adjacency in (0, 1).

    This noise process operates in the logit (log-odds) space, which provides
    natural bounds for the resulting noisy adjacency values. The transformation
    pipeline is:

        A ∈ {0,1}  →  clamp to [ε, 1-ε]  →  logit: L = log(A/(1-A))
            →  L_noisy = L + N(0, σ²)  →  sigmoid: P = 1/(1+e^{-L_noisy})

    Parameters
    ----------
    A
        Input adjacency matrix (binary or soft). Can be a single matrix (n, n)
        or a batch (batch_size, n, n).
    sigma
        Standard deviation of Gaussian noise in logit space. Typical range is
        0.5 to 5.0. Higher values cause more aggressive edge perturbation.
    clamp_eps
        Small epsilon for clamping to avoid log(0) or log(inf). Default 1e-6.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix with values in (0, 1). The output is symmetric
        if the input was symmetric, since noise is applied symmetrically.

    Notes
    -----
    - Unlike flip-based noise (DiGress), this produces soft/continuous outputs
    - The sigma parameter controls noise severity in log-odds space:
      - sigma=0.5: mild perturbation, most edges retain original polarity
      - sigma=2.0: moderate perturbation
      - sigma=5.0: aggressive perturbation, significant edge flipping

    Examples
    --------
    >>> A = torch.eye(5)
    >>> A_noisy = add_logit_noise(A, sigma=1.0)
    >>> assert A_noisy.min() > 0 and A_noisy.max() < 1
    >>> assert torch.allclose(A_noisy, A_noisy.T)  # Symmetric
    """
    # Convert to tensor if needed
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    # Clamp for numerical stability (avoid log(0) and log(inf))
    A_clamped = A.clamp(clamp_eps, 1 - clamp_eps)

    # Transform to logit space: L = log(A / (1 - A))
    L = torch.logit(A_clamped)

    # Generate symmetric noise (upper triangle mirrored to lower)
    noise = torch.randn_like(L) * sigma

    if L.dim() == 2:
        # Single matrix case
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.T
    else:
        # Batch case
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.transpose(-2, -1)

    # Add noise in logit space and transform back via sigmoid
    L_noisy = L + noise
    return torch.sigmoid(L_noisy)


def add_digress_noise(
    A: torch.Tensor | np.ndarray,
    p: float,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Add noise to an adjacency matrix by flipping edges with probability p.

    Args:
        A: Input adjacency matrix (0s and 1s)
        p: Probability of flipping each element
        rng: Random number generator (optional)

    Returns:
        Noisy adjacency matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert to tensor if needed
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

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

    return A_noisy
