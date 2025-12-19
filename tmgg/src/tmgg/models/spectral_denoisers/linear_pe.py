"""Linear Positional Encoding denoiser for graph denoising.

Implements the simplest spectral denoiser: a linear transformation in
eigenspace with optional node-specific bias correction.
"""

from typing import Any

import torch
import torch.nn as nn

from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser


class LinearPE(SpectralDenoiser):
    """Linear Positional Encoding denoiser.

    Reconstructs the adjacency matrix as:
        Â = V W V^T + 1 b^T + b 1^T

    where V ∈ R^{n×k} are the top-k eigenvectors of the noisy adjacency,
    W ∈ R^{k×k} is a learnable weight matrix, and b ∈ R^{max_n} is a
    learnable bias vector capturing node-specific degree corrections.

    Outputs raw logits. Use model.predict(logits) for [0,1] probabilities.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use.
    max_nodes : int, optional
        Maximum number of nodes in any graph. Required if use_bias is True.
        Default is 200.
    use_bias : bool, optional
        Whether to use the node-specific bias term. For multi-graph settings
        with varying sizes, consider setting this to False. Default is True.

    Notes
    -----
    The bias term adds a rank-2 correction to the low-rank reconstruction.
    Entry (i, j) receives bias b_i + b_j, which captures node degree effects
    not captured by the top-k eigenspace.

    Examples
    --------
    >>> model = LinearPE(k=8, max_nodes=50)
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
        max_nodes: int = 200,
        use_bias: bool = True,
    ):
        super().__init__(k=k)
        self.max_nodes = max_nodes
        self.use_bias = use_bias

        # Learnable weight matrix W ∈ R^{k×k}
        self.W = nn.Parameter(torch.empty(k, k))
        nn.init.xavier_uniform_(self.W)

        # Learnable bias vector b ∈ R^{max_nodes}
        if use_bias:
            self.b = nn.Parameter(torch.zeros(max_nodes))
        else:
            self.register_parameter("b", None)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute V W V^T + bias.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused in this model).
        A : torch.Tensor
            Input adjacency (unused, available for other architectures).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency logits.
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        batch_size, n, k = V.shape

        # Core reconstruction: V W V^T
        # V @ W: (batch, n, k)
        VW = torch.matmul(V, self.W)
        # VW @ V^T: (batch, n, n)
        A_reconstructed = torch.matmul(VW, V.transpose(-1, -2))

        # Add bias term if enabled
        if self.use_bias and self.b is not None:
            # Get bias for actual graph size
            b = self.b[:n]  # (n,)
            # Compute 1 b^T + b 1^T
            # This adds b_i + b_j to entry (i, j)
            ones = torch.ones(n, device=V.device, dtype=V.dtype)
            # outer products: (n, 1) @ (1, n) -> (n, n)
            bias_term = b.unsqueeze(1) @ ones.unsqueeze(0) + ones.unsqueeze(
                1
            ) @ b.unsqueeze(0)
            # Broadcast over batch
            A_reconstructed = A_reconstructed + bias_term.unsqueeze(0)

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "max_nodes": self.max_nodes,
                "use_bias": self.use_bias,
            }
        )
        return config
