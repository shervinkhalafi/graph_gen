"""Base class for spectral graph denoising models.

All spectral denoisers share a common pattern: extract top-k eigenvectors from
the noisy adjacency matrix, process them through an architecture-specific
transformation, and reconstruct the denoised adjacency matrix.
"""

from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch

from tmgg.models.base import DenoisingModel
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer


class SpectralDenoiser(DenoisingModel):
    """Abstract base class for spectral graph denoising models.

    Spectral denoisers operate on the eigenspace of the noisy adjacency matrix.
    They extract the top-k eigenvectors, apply a learnable transformation, and
    reconstruct the denoised adjacency.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use. Capped by graph size at runtime.

    Attributes
    ----------
    eigen_layer : TopKEigenLayer
        Layer for extracting top-k eigenvectors with sign normalization.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.eigen_layer = TopKEigenLayer(k=k)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Denoise adjacency matrix via spectral transformation.

        Parameters
        ----------
        A : torch.Tensor
            Noisy adjacency matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        torch.Tensor
            Denoised adjacency matrix of same shape as input.
        """
        # Extract top-k eigenvectors
        V, Lambda = self.eigen_layer(A)

        # Architecture-specific processing
        return self._spectral_forward(V, Lambda, A)

    @abstractmethod
    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Architecture-specific spectral processing.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Corresponding eigenvalues of shape (batch, k) or (k,).
        A : torch.Tensor
            The (possibly transformed) input adjacency matrix, provided for
            architectures that need access to the full matrix.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency matrix (raw, before output transform).
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for logging/saving.

        Returns
        -------
        dict
            Dictionary containing model hyperparameters.
        """
        return {
            "model_class": self.__class__.__name__,
            "k": self.k,
        }


