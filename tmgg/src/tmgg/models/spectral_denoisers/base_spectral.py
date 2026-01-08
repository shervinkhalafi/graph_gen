"""Base class for spectral graph denoising models.

All spectral denoisers share a common pattern: extract embeddings (either
eigenvectors or learned positional encodings like PEARL) from the noisy
adjacency matrix, process them through an architecture-specific transformation,
and reconstruct the denoised adjacency matrix.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

from tmgg.models.base import DenoisingModel
from tmgg.models.layers.pearl_embedding import PEARLEmbedding
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer

EmbeddingSource = Literal["eigenvector", "pearl_random", "pearl_basis"]


class SpectralDenoiser(DenoisingModel, ABC):
    """Abstract base class for spectral graph denoising models.

    Spectral denoisers operate on node embeddings extracted from the adjacency
    matrix. They support multiple embedding sources:

    - "eigenvector": Top-k eigenvectors via eigendecomposition (default)
    - "pearl_random": R-PEARL positional encodings (random init + GNN)
    - "pearl_basis": B-PEARL positional encodings (basis vectors + GNN)

    Parameters
    ----------
    k : int
        Embedding dimension. For eigenvectors, this is the number of top-k
        eigenvectors. For PEARL, this is the output dimension.
    embedding_source : {"eigenvector", "pearl_random", "pearl_basis"}
        Source of node embeddings. PEARL variants use learned GNN message
        passing instead of eigendecomposition. Default is "eigenvector".
    pearl_num_layers : int
        Number of GNN layers for PEARL embeddings. Ignored if embedding_source
        is "eigenvector". Default is 3.
    pearl_hidden_dim : int
        Hidden dimension for PEARL GNN layers. Default is 64.
    pearl_input_samples : int
        Number of random input samples for R-PEARL. Default is 32.
    pearl_max_nodes : int
        Maximum graph size for B-PEARL basis vectors. Default is 200.

    Attributes
    ----------
    embedding_layer : TopKEigenLayer | PEARLEmbedding
        Layer for extracting node embeddings.
    embedding_source : str
        The embedding source being used.

    Notes
    -----
    PEARL reference: E. Hejin et al., "PEARL: A Scalable and Effective Random
    Positional Encoding." ICLR 2025. https://github.com/ehejin/Pearl-PE
    """

    def __init__(
        self,
        k: int,
        embedding_source: EmbeddingSource = "eigenvector",
        pearl_num_layers: int = 3,
        pearl_hidden_dim: int = 64,
        pearl_input_samples: int = 32,
        pearl_max_nodes: int = 200,
    ):
        super().__init__()
        self.k = k
        self.embedding_source = embedding_source

        if embedding_source == "eigenvector":
            self.embedding_layer: TopKEigenLayer | PEARLEmbedding = TopKEigenLayer(k=k)
            # Backward compat: subclasses may reference eigen_layer directly
            self.eigen_layer = self.embedding_layer
        elif embedding_source in ("pearl_random", "pearl_basis"):
            mode = "random" if embedding_source == "pearl_random" else "basis"
            self.embedding_layer = PEARLEmbedding(
                output_dim=k,
                num_layers=pearl_num_layers,
                mode=mode,
                hidden_dim=pearl_hidden_dim,
                input_samples=pearl_input_samples,
                max_nodes=pearl_max_nodes,
            )
            # PEARL has no eigen_layer - subclasses using eigen_layer will fail
            # with a clear error
            self.eigen_layer = None  # type: ignore[assignment]
        else:
            raise ValueError(
                f"embedding_source must be 'eigenvector', 'pearl_random', or "
                f"'pearl_basis', got {embedding_source!r}"
            )

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
        if self.embedding_source == "eigenvector":
            # Extract top-k eigenvectors
            assert isinstance(self.embedding_layer, TopKEigenLayer)
            V, Lambda = self.embedding_layer(A)

            # Pad V and Lambda to k columns if graph is smaller than k
            # This ensures models with k-dimensional weights work on any graph size
            actual_k = V.shape[-1]
            if actual_k < self.k:
                pad_size = self.k - actual_k
                V = torch.nn.functional.pad(V, (0, pad_size))
                Lambda = torch.nn.functional.pad(Lambda, (0, pad_size))
        else:
            # PEARL embeddings
            assert isinstance(self.embedding_layer, PEARLEmbedding)
            V = self.embedding_layer(A)  # (batch, n, k) or (n, k)
            # PEARL has no eigenvalues; pass zeros as placeholder
            unbatched = V.ndim == 2
            if unbatched:
                Lambda = torch.zeros(self.k, device=V.device, dtype=V.dtype)
            else:
                Lambda = torch.zeros(V.shape[0], self.k, device=V.device, dtype=V.dtype)

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
            "embedding_source": self.embedding_source,
        }

    def get_features(self, A: torch.Tensor) -> torch.Tensor:
        """Extract learned features for each node.

        This method provides access to internal representations learned by
        the model, intended for use by wrapper architectures like
        ShrinkageWrapper that need to aggregate features for graph-level
        predictions.

        The default implementation extracts eigenvectors (or PEARL embeddings),
        which may be overridden by subclasses to return richer learned
        representations (e.g., Q/K projections from attention).

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
        if self.embedding_source == "eigenvector":
            assert isinstance(self.embedding_layer, TopKEigenLayer)
            V, _Lambda = self.embedding_layer(A)

            # Pad to k if needed
            actual_k = V.shape[-1]
            if actual_k < self.k:
                pad_size = self.k - actual_k
                V = torch.nn.functional.pad(V, (0, pad_size))
        else:
            assert isinstance(self.embedding_layer, PEARLEmbedding)
            V = self.embedding_layer(A)

        return V
