"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np
import torch
from scipy.linalg import expm


def random_skew_symmetric_matrix(
    n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Create a random n x n skew-symmetric matrix.

    Parameters
    ----------
    n
        Matrix dimension.
    rng
        NumPy random generator for reproducibility. When ``None``, a fresh
        ``np.random.default_rng()`` is created (independent of the global
        numpy RNG state).

    Returns
    -------
    np.ndarray
        Skew-symmetric matrix of shape ``(n, n)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    A = rng.random((n, n))
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
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()  # pyright: ignore[reportConstantRedefinition]  # math notation

    # Handle both single matrix and batch cases
    if A.dim() == 2:
        # Single matrix case
        eigenvalues, V = torch.linalg.eigh(A)  # pyright: ignore[reportConstantRedefinition]  # math notation
        R = expm(eps * skew)  # pyright: ignore[reportConstantRedefinition]  # math notation
        R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
        V_rot = V @ R_tensor
        A_noisy = V_rot @ torch.diag(eigenvalues) @ V_rot.T
    else:
        # Batch case
        eigenvalues, V = torch.linalg.eigh(A)  # pyright: ignore[reportConstantRedefinition]  # math notation
        R = expm(eps * skew)  # pyright: ignore[reportConstantRedefinition]  # math notation
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
    """Add symmetric Gaussian noise to adjacency matrix.

    Generates noise for the upper triangle and mirrors it to the lower
    triangle, matching the symmetrization convention of all other noise
    types (edge_flip, logit, digress). Diagonal entries receive no noise.

    Parameters
    ----------
    A
        Input adjacency matrix, single (n, n) or batched (batch_size, n, n).
    eps
        Noise standard deviation.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix with symmetric noise.
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()  # pyright: ignore[reportConstantRedefinition]  # math notation

    noise = eps * torch.randn_like(A)

    if A.dim() == 2:
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.T
    else:
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.transpose(-2, -1)

    return A + noise


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
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()  # pyright: ignore[reportConstantRedefinition]  # math notation

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
    alpha_bar: float,
) -> torch.Tensor:
    """Apply DiGress-style categorical noise to a binary adjacency matrix.

    Implements the forward process from Vignac et al. (2023) for binary edges
    (K=2 classes). The transition matrix is:

        Q_bar_t = alpha_bar * I + (1 - alpha_bar) * U

    where U is the 2x2 uniform matrix. For binary edges, each entry flips
    with probability (1 - alpha_bar) / 2.

    Parameters
    ----------
    A
        Binary adjacency matrix (0s and 1s), single (n, n) or batched
        (batch_size, n, n).
    alpha_bar
        Cumulative noise schedule parameter in [0, 1]. At alpha_bar=1.0,
        no noise (clean). At alpha_bar=0.0, output is uniform random
        (fully noisy). Related to the noise schedule by
        alpha_bar_t = prod_{s=1}^{t} (1 - beta_s).

    Returns
    -------
    torch.Tensor
        Noisy binary adjacency matrix, same shape as input. Symmetric
        if input was symmetric.
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    # Flip probability from DiGress transition: (1 - alpha_bar) / 2
    flip_prob = (1.0 - alpha_bar) / 2.0

    # Delegate to edge-flip noise with the derived probability
    return add_edge_flip_noise(A, p=flip_prob)


def add_edge_flip_noise(
    A: torch.Tensor | np.ndarray,
    p: float,
) -> torch.Tensor:
    """Flip edges with probability p (symmetric Bernoulli noise).

    Each upper-triangular entry is independently flipped with probability
    ``p``, then mirrored to the lower triangle to preserve symmetry.
    Noise is sampled via ``torch.rand_like``, so reproducibility is
    controlled by PyTorch's global random state (``torch.manual_seed``).

    Parameters
    ----------
    A
        Input adjacency matrix (0s and 1s), single ``(n, n)`` or
        batched ``(batch_size, n, n)``.
    p
        Probability of flipping each edge.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix with same shape as input.
    """
    # Convert to tensor if needed
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()  # pyright: ignore[reportConstantRedefinition]  # math notation

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
