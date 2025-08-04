"""Noise generator classes for graph denoising experiments.

This module provides a clean, object-oriented interface for adding different types
of noise to graphs, with proper state management for noise types that require it.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import torch

from .noise import (
    add_gaussian_noise,
    add_digress_noise,
    add_rotation_noise,
    random_skew_symmetric_matrix,
)


class NoiseGenerator(ABC):
    """Abstract base class for noise generators."""
    
    @abstractmethod
    def add_noise(
        self, A: torch.Tensor, eps: float
    ) -> torch.Tensor:
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
    
    def add_noise(
        self, A: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Add Gaussian noise to adjacency matrix."""
        return add_gaussian_noise(A, eps)
    
    @property
    def requires_state(self) -> bool:
        return False


class DigressNoiseGenerator(NoiseGenerator):
    """Digress (edge flipping) noise generator (stateless)."""
    
    def add_noise(
        self, A: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Add digress noise by flipping edges with probability eps."""
        return add_digress_noise(A, eps)
    
    @property
    def requires_state(self) -> bool:
        return False


class RotationNoiseGenerator(NoiseGenerator):
    """
    Rotation noise generator with skew matrix management.
    
    This generator maintains a skew-symmetric matrix that is used to
    rotate eigenvectors, providing a consistent rotation throughout
    the experiment.
    """
    
    def __init__(self, k: int, seed: Optional[int] = None):
        """
        Initialize rotation noise generator.
        
        Args:
            k: Dimension of the skew matrix (number of eigenvectors)
            seed: Random seed for reproducible skew matrix generation
        """
        self.k = k
        self.seed = seed
        self.skew = self._generate_skew_matrix(k, seed)
    
    def _generate_skew_matrix(self, k: int, seed: Optional[int]) -> np.ndarray:
        """Generate a random skew-symmetric matrix."""
        if seed is not None:
            rng = np.random.RandomState(seed)
            A = rng.rand(k, k)
        else:
            A = np.random.rand(k, k)
        return (A - A.T) / 2
    
    def add_noise(
        self, A: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Add rotation noise by rotating eigenvectors."""
        return add_rotation_noise(A, eps, self.skew)
    
    @property
    def requires_state(self) -> bool:
        return True
    
    def get_config(self) -> dict:
        """Get configuration for this generator."""
        return {
            "k": self.k,
            "seed": self.seed,
        }


def create_noise_generator(
    noise_type: str, 
    rotation_k: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> NoiseGenerator:
    """
    Factory function to create noise generators.
    
    Args:
        noise_type: Type of noise ("gaussian", "digress", or "rotation")
        rotation_k: Dimension for rotation noise skew matrix
        seed: Random seed for reproducible noise generation
        **kwargs: Additional arguments (for future extensibility)
        
    Returns:
        NoiseGenerator instance
        
    Raises:
        ValueError: If noise_type is unknown or required parameters are missing
    """
    noise_type = noise_type.lower()
    
    if noise_type == "gaussian":
        return GaussianNoiseGenerator()
    elif noise_type == "digress":
        return DigressNoiseGenerator()
    elif noise_type == "rotation":
        if rotation_k is None:
            raise ValueError("rotation_k parameter is required for rotation noise")
        return RotationNoiseGenerator(k=rotation_k, seed=seed)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")