"""Self-Attention denoiser using query/key projections on eigenvectors.

Implements a scaled dot-product attention mechanism operating on the
eigenspace of the noisy adjacency matrix.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser


class SelfAttentionDenoiser(SpectralDenoiser):
    """Self-Attention denoiser with query/key projections.

    Reconstructs the adjacency matrix using scaled dot-product attention on
    the eigenvector embeddings:
        Q = V W_Q
        K = V W_K
        Â = Q K^T / √d_k

    where V ∈ R^{n×k} are the top-k eigenvectors, W_Q, W_K ∈ R^{k×d_k} are
    learnable projection matrices, and d_k is the key dimension.

    The 1/√d_k scaling stabilizes gradients following standard practice in
    transformer architectures.

    Outputs raw logits. Use model.predict(logits) for [0,1] probabilities.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use as input dimension.
    d_k : int, optional
        Key/query dimension for attention. Default is 64.

    Notes
    -----
    Unlike transformer attention which uses softmax over the sequence
    dimension, this produces a symmetric attention matrix representing edge
    logits. The model.predict() method applies sigmoid to get probabilities.

    Examples
    --------
    >>> model = SelfAttentionDenoiser(k=8, d_k=32)
    >>> A_noisy = torch.randn(4, 50, 50)
    >>> A_noisy = (A_noisy + A_noisy.transpose(-1, -2)) / 2
    >>> logits = model(A_noisy)
    >>> predictions = model.predict(logits)  # apply sigmoid
    >>> predictions.shape
    torch.Size([4, 50, 50])
    """

    def __init__(
        self,
        k: int,
        d_k: int = 64,
    ):
        super().__init__(k=k)
        self.d_k = d_k
        self.scale = d_k ** -0.5  # 1/√d_k

        # Query and Key projection matrices
        self.W_Q = nn.Parameter(torch.empty(k, d_k))
        self.W_K = nn.Parameter(torch.empty(k, d_k))

        # Initialize with Xavier for better gradient flow
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused in this model).
        A : torch.Tensor
            Input adjacency (unused).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency logits.
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        # Compute Q = V W_Q and K = V W_K
        # V: (batch, n, k), W_Q: (k, d_k) -> Q: (batch, n, d_k)
        Q = torch.matmul(V, self.W_Q)
        K = torch.matmul(V, self.W_K)

        # Scaled dot-product: Q K^T / √d_k
        # (batch, n, d_k) @ (batch, d_k, n) -> (batch, n, n)
        A_reconstructed = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["d_k"] = self.d_k
        return config
