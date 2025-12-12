"""Node-variant graph convolution layer.

REDESIGNED: The original implementation used node-specific parameters with shape
(num_terms+1, num_channels, num_nodes), which caused:
- Dynamic layer recreation breaking optimizer state
- Parameter truncation/padding losing learned weights
- Incompatibility with variable graph sizes

The new design uses node-agnostic parameters with shape
(num_terms+1, num_channels_in, num_channels_out), supporting any graph size.
"""

import torch
import torch.nn as nn


class NodeVarGraphConvolutionLayer(nn.Module):
    """Graph convolution layer with polynomial filters.

    This layer computes a polynomial graph convolution:
        Y = sum_{i=0}^{num_terms} A^i @ X @ H[i]

    where A is the (normalized) adjacency matrix and H[i] are learnable weight
    matrices. Unlike the original "node-variant" design, parameters are shared
    across all nodes, enabling support for variable graph sizes.

    Parameters
    ----------
    num_terms : int
        Number of polynomial terms (excluding the identity term).
    num_channels_in : int
        Input feature dimension.
    num_channels_out : int
        Output feature dimension.

    Notes
    -----
    The `num_nodes` parameter from the original API is kept for backwards
    compatibility but is now ignored - the layer works with any graph size.
    """

    def __init__(
        self, num_terms: int, num_channels_in: int, num_channels_out: int = None
    ):
        """Initialize the graph convolution layer.

        Parameters
        ----------
        num_terms : int
            Number of polynomial terms.
        num_channels_in : int
            Input feature dimension.
        num_channels_out : int, optional
            Output feature dimension. If None, defaults to num_channels_in.
            (For backwards compatibility with the old 3-arg signature where
            the third arg was num_nodes, this is ignored if it seems too large.)
        """
        super().__init__()
        self.num_terms = num_terms
        self.num_channels_in = num_channels_in

        # Backwards compatibility: old signature was (num_terms, num_channels, num_nodes)
        # where num_nodes was typically >> num_channels. Detect this and ignore.
        if num_channels_out is None or num_channels_out > 100:
            # Likely old usage with num_nodes as third arg, use num_channels_in
            self.num_channels_out = num_channels_in
        else:
            self.num_channels_out = num_channels_out

        # Node-agnostic parameters: (num_terms+1, in_channels, out_channels)
        self.H = nn.Parameter(
            torch.randn(num_terms + 1, self.num_channels_in, self.num_channels_out)
        )
        nn.init.xavier_uniform_(self.H)

        # GELU provides better nonlinearity than Tanh after LayerNorm
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.num_channels_out)

    @property
    def num_nodes(self) -> int:
        """Backwards compatibility: return a placeholder value.

        The new design doesn't store num_nodes since it's node-agnostic.
        Returns -1 to indicate this, which will always differ from actual
        node counts, preventing the old dynamic recreation logic from triggering.
        """
        return -1

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of graph convolution.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, num_nodes, num_nodes).
        X : torch.Tensor
            Input features of shape (batch, num_nodes, num_channels_in).

        Returns
        -------
        torch.Tensor
            Output features of shape (batch, num_nodes, num_channels_out).
        """
        # Ensure consistent dtype
        X = X.to(self.H.dtype)
        A = A.to(self.H.dtype)

        # Symmetric normalization to bound spectral radius
        D = A.sum(dim=-1)
        D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt_mat, A), D_inv_sqrt_mat)

        # Polynomial convolution: Y = sum_i A^i @ X @ H[i]
        Y = X @ self.H[0]  # Identity term (A^0 = I)

        A_power = A_norm  # Start with A^1
        for i in range(1, self.num_terms + 1):
            Y = Y + torch.bmm(A_power, X) @ self.H[i]
            if i < self.num_terms:
                A_power = torch.bmm(A_power, A_norm)  # A^{i+1}

        Y = self.layer_norm(Y)
        Y = self.activation(Y)
        return Y
