"""Top-k eigenvector extraction layer for spectral graph denoising.

This module extracts the top-k eigenvectors (by eigenvalue magnitude) from
symmetric adjacency matrices, with sign normalization to resolve eigenvector
sign ambiguity.
"""

import torch
import torch.nn as nn


class TopKEigenLayer(nn.Module):
    """Extract top-k eigenvectors from symmetric adjacency matrices.

    Given a batch of symmetric adjacency matrices, computes eigendecomposition
    and returns the top-k eigenvectors ordered by eigenvalue magnitude. Sign
    ambiguity is resolved by enforcing the first nonzero entry of each
    eigenvector to be positive.

    Parameters
    ----------
    k : int
        Number of eigenvectors to extract. If k exceeds the matrix dimension n,
        all n eigenvectors are returned.

    Examples
    --------
    >>> layer = TopKEigenLayer(k=4)
    >>> A = torch.randn(8, 20, 20)
    >>> A = (A + A.transpose(-1, -2)) / 2  # symmetrize
    >>> V, Lambda = layer(A)
    >>> V.shape
    torch.Size([8, 20, 4])
    >>> Lambda.shape
    torch.Size([8, 4])
    """

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def forward(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract top-k eigenvectors from adjacency matrices.

        Parameters
        ----------
        A : torch.Tensor
            Batch of symmetric adjacency matrices of shape (batch, n, n).
            Can also accept unbatched input of shape (n, n).

        Returns
        -------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k) if unbatched.
            Each column is an eigenvector, ordered by decreasing eigenvalue
            magnitude. Sign-normalized so first nonzero entry is positive.
        Lambda : torch.Tensor
            Corresponding eigenvalues of shape (batch, k) or (k,) if unbatched.
            Ordered by decreasing magnitude.
        """
        unbatched = A.ndim == 2
        if unbatched:
            A = A.unsqueeze(0)

        batch_size, n, _ = A.shape
        k = min(self.k, n)

        # Enforce symmetry before eigh: floating-point operations can
        # introduce small asymmetries, and eigh silently uses only the
        # lower triangle, which can cause inconsistent results
        A_sym = (A + A.transpose(-1, -2)) / 2

        # Compute full eigendecomposition (eigh returns ascending eigenvalues)
        eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
        # eigenvalues: (batch, n), eigenvectors: (batch, n, n)

        # Select top-k by magnitude (descending order)
        abs_eigenvalues = eigenvalues.abs()
        _, topk_indices = torch.topk(abs_eigenvalues, k, dim=-1)
        # topk_indices: (batch, k)

        # Gather eigenvalues
        Lambda = torch.gather(eigenvalues, dim=-1, index=topk_indices)

        # Gather eigenvectors: need to expand indices for the n dimension
        # eigenvectors has shape (batch, n, n), we want to select k columns
        expanded_indices = topk_indices.unsqueeze(1).expand(-1, n, -1)
        V = torch.gather(eigenvectors, dim=-1, index=expanded_indices)
        # V: (batch, n, k)

        # Sign normalization: make first nonzero entry of each eigenvector positive
        V = self._normalize_signs(V)

        if unbatched:
            V = V.squeeze(0)
            Lambda = Lambda.squeeze(0)

        return V, Lambda

    def _normalize_signs(self, V: torch.Tensor) -> torch.Tensor:
        """Normalize eigenvector signs so first nonzero entry is positive.

        Parameters
        ----------
        V : torch.Tensor
            Eigenvectors of shape (batch, n, k).

        Returns
        -------
        torch.Tensor
            Sign-normalized eigenvectors of same shape.
        """
        # For each eigenvector column, find the first nonzero entry
        # and flip sign if it's negative
        batch, n, k = V.shape

        # Find first nonzero entry per eigenvector
        # Use a small threshold to handle numerical zeros
        eps = 1e-10
        abs_V = V.abs()

        # Create mask of nonzero entries
        nonzero_mask = abs_V > eps  # (batch, n, k)

        # For each column, find the index of first nonzero entry
        # We do this by finding argmax of cumsum of the mask (first True becomes 1)
        cumsum_mask = nonzero_mask.cumsum(dim=1)  # (batch, n, k)
        first_nonzero_mask = (cumsum_mask == 1) & nonzero_mask

        # Get the sign of the first nonzero entry
        # Where mask is True, take the sign; elsewhere 0
        first_signs = torch.where(
            first_nonzero_mask, torch.sign(V), torch.zeros_like(V)
        )
        # Sum over n dimension to get the sign for each eigenvector
        sign_multipliers = first_signs.sum(dim=1)  # (batch, k)

        # Handle case where eigenvector is all zeros (shouldn't happen in practice)
        # This can indicate k > rank(A) or numerical issues
        zero_eigenvector_mask = sign_multipliers == 0
        if zero_eigenvector_mask.any():
            import warnings

            warnings.warn(
                "Zero eigenvector detected in sign normalization; may indicate k > rank(A) "
                "or numerical issues in eigendecomposition.",
                RuntimeWarning,
                stacklevel=2,
            )
        sign_multipliers = torch.where(
            zero_eigenvector_mask, torch.ones_like(sign_multipliers), sign_multipliers
        )

        # Apply sign correction
        return V * sign_multipliers.unsqueeze(1)

    def extra_repr(self) -> str:
        return f"k={self.k}"
