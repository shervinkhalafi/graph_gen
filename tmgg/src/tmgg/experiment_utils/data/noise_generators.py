"""Noise generator classes for graph denoising experiments.

This module provides a clean, object-oriented interface for adding different types
of noise to graphs, with proper state management for noise types that require it.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from .noise import (
    add_digress_noise,
    add_edge_flip_noise,
    add_gaussian_noise,
    add_logit_noise,
    add_rotation_noise,
)


class NoiseGenerator(ABC):
    """Abstract base class for noise generators."""

    @abstractmethod
    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Add noise to an adjacency matrix.

        Args:
            A: Input adjacency matrix (can be batched)
            eps: Noise level/intensity

        Returns:
            Noisy adjacency matrix
        """
        pass

    @property
    @abstractmethod
    def requires_state(self) -> bool:
        """Whether this noise generator maintains internal state."""
        pass


class GaussianNoiseGenerator(NoiseGenerator):
    """Gaussian noise generator (stateless)."""

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        """Add Gaussian noise to adjacency matrix."""
        return add_gaussian_noise(A, eps)

    @property
    def requires_state(self) -> bool:
        return False


class EdgeFlipNoiseGenerator(NoiseGenerator):
    """Symmetric Bernoulli edge-flip noise generator (stateless)."""

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        """Flip edges with probability eps."""
        return add_edge_flip_noise(A, eps)

    @property
    def requires_state(self) -> bool:
        return False


class DigressNoiseGenerator(NoiseGenerator):
    """DiGress categorical noise generator (Vignac et al. 2023).

    Implements the forward diffusion process from DiGress for binary
    adjacency matrices. The eps parameter from the NoiseGenerator interface
    is converted to alpha_bar = 1 - eps before applying noise:
    - eps=0 -> alpha_bar=1 -> no noise (clean)
    - eps=1 -> alpha_bar=0 -> fully random (uniform)

    At a given eps, this produces half the flip rate of EdgeFlipNoiseGenerator,
    because DiGress models noise as a categorical transition where the maximum
    flip probability (at alpha_bar=0) is 0.5, not 1.0.
    """

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        alpha_bar = 1.0 - eps
        return add_digress_noise(A, alpha_bar=alpha_bar)

    @property
    def requires_state(self) -> bool:
        return False


class RotationNoiseGenerator(NoiseGenerator):
    """Rotation noise generator with skew matrix management.

    Maintains a skew-symmetric matrix used to rotate eigenvectors,
    providing a consistent rotation throughout the experiment.

    Notes
    -----
    The skew matrix dimension is fixed at construction and must match
    the input graph size exactly. Variable-size graphs or padded batches
    will raise ``AssertionError``. Create separate generators for each
    graph size, or use ``GaussianNoiseGenerator`` for variable-size inputs.
    """

    def __init__(self, k: int, seed: int | None = None):
        """
        Initialize rotation noise generator.

        Args:
            k: Dimension of the skew matrix (number of eigenvectors)
            seed: Random seed for reproducible skew matrix generation
        """
        self.k = k
        self.seed = seed
        self.skew = self._generate_skew_matrix(k, seed)

    def _generate_skew_matrix(self, k: int, seed: int | None) -> np.ndarray:
        """Generate a random skew-symmetric matrix."""
        rng = np.random.default_rng(seed)
        A = rng.random((k, k))
        return (A - A.T) / 2

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        """Add rotation noise by rotating eigenvectors."""
        assert A.shape[-1] == self.skew.shape[0], (
            f"Graph dimension {A.shape[-1]} != skew matrix dimension {self.skew.shape[0]}. "
            f"RotationNoiseGenerator was created with k={self.skew.shape[0]}."
        )
        return add_rotation_noise(A, eps, self.skew)

    @property
    def requires_state(self) -> bool:
        return True

    def get_config(self) -> dict[str, int | None]:
        """Get configuration for this generator."""
        return {
            "k": self.k,
            "seed": self.seed,
        }


class LogitNoiseGenerator(NoiseGenerator):
    """Logit-space noise generator.

    This generator operates in the log-odds space, providing naturally bounded
    outputs. The noise is added in logit space and transformed back via sigmoid,
    ensuring outputs remain in (0, 1).

    Note: This noise type is experimental and not yet used in standard sweeps.
    The eps parameter for this generator represents the standard deviation in
    logit space, not a flip probability. Typical values range from 0.5 to 5.0.

    Parameters
    ----------
    clamp_eps
        Small epsilon for clamping input values to avoid numerical issues
        with log(0) or log(inf). Default 1e-6.

    See Also
    --------
    add_logit_noise : The underlying noise function with full documentation.
    """

    def __init__(self, clamp_eps: float = 1e-6):
        self.clamp_eps = clamp_eps

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        """Add noise in logit space.

        Parameters
        ----------
        A
            Input adjacency matrix (binary or soft).
        eps
            Standard deviation of Gaussian noise in logit space.
            Note: This differs from DiGress where eps is flip probability.
            Typical range: 0.5 to 5.0.

        Returns
        -------
        torch.Tensor
            Noisy adjacency with values in (0, 1).
        """
        return add_logit_noise(A, sigma=eps, clamp_eps=self.clamp_eps)

    @property
    def requires_state(self) -> bool:
        return False


def create_noise_generator(
    noise_type: str, rotation_k: int | None = None, seed: int | None = None, **kwargs
) -> NoiseGenerator:
    """
    Factory function to create noise generators.

    Args:
        noise_type: Type of noise ("gaussian", "edge_flip", "digress", "rotation", or "logit")
        rotation_k: Dimension for rotation noise skew matrix
        seed: Random seed for reproducible noise generation
        **kwargs: Additional arguments (for future extensibility)
            - clamp_eps: For logit noise, epsilon for numerical stability (default 1e-6)

    Returns:
        NoiseGenerator instance

    Raises:
        ValueError: If noise_type is unknown or required parameters are missing
    """
    noise_type = noise_type.lower()

    if noise_type == "gaussian":
        return GaussianNoiseGenerator()
    elif noise_type == "edge_flip":
        return EdgeFlipNoiseGenerator()
    elif noise_type == "digress":
        return DigressNoiseGenerator()
    elif noise_type == "rotation":
        if rotation_k is None:
            raise ValueError("rotation_k parameter is required for rotation noise")
        return RotationNoiseGenerator(k=rotation_k, seed=seed)
    elif noise_type == "logit":
        clamp_eps = kwargs.get("clamp_eps", 1e-6)
        return LogitNoiseGenerator(clamp_eps=clamp_eps)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
