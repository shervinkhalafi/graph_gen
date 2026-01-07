"""Base class for spectral graph denoising models.

All spectral denoisers share a common pattern: extract top-k eigenvectors from
the noisy adjacency matrix, process them through an architecture-specific
transformation, and reconstruct the denoised adjacency matrix.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from tmgg.models.base import DenoisingModel
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer


class SpectralDenoiser(DenoisingModel, ABC):
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

        # Pad V and Lambda to k columns if graph is smaller than k
        # This ensures models with k-dimensional weights work on any graph size
        actual_k = V.shape[-1]
        if actual_k < self.k:
            pad_size = self.k - actual_k
            if V.ndim == 2:
                # Unbatched: (n, actual_k) -> (n, k)
                V = torch.nn.functional.pad(V, (0, pad_size))
                Lambda = torch.nn.functional.pad(Lambda, (0, pad_size))
            else:
                # Batched: (batch, n, actual_k) -> (batch, n, k)
                V = torch.nn.functional.pad(V, (0, pad_size))
                Lambda = torch.nn.functional.pad(Lambda, (0, pad_size))

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

    def get_config(self) -> dict[str, Any]:
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

    def get_features(self, A: torch.Tensor) -> torch.Tensor:
        """Extract learned features for each node.

        This method provides access to internal representations learned by
        the model, intended for use by wrapper architectures like
        ShrinkageWrapper that need to aggregate features for graph-level
        predictions.

        The default implementation extracts eigenvectors, which may be
        overridden by subclasses to return richer learned representations
        (e.g., Q/K projections from attention).

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        torch.Tensor
            Node features of shape (batch, n, feature_dim) or (n, feature_dim).
            The feature_dim depends on the model architecture.

        Notes
        -----
        Subclasses with learnable projections (e.g., SelfAttentionDenoiser)
        should override this to return the projected features, enabling
        wrapper architectures to leverage learned representations.
        """
        # Default: return eigenvectors
        V, _Lambda = self.eigen_layer(A)

        # Pad to k if needed
        actual_k = V.shape[-1]
        if actual_k < self.k:
            pad_size = self.k - actual_k
            V = torch.nn.functional.pad(V, (0, pad_size))

        return V
