"""Graph Filter Bank denoiser with spectral polynomial parameterization.

Implements a learnable spectral filter as a polynomial in the eigenvalues,
allowing the model to learn frequency-dependent transformations.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser


class GraphFilterBank(SpectralDenoiser):
    """Graph Filter Bank denoiser with spectral polynomial.

    Reconstructs the adjacency matrix using a spectral polynomial filter:
        W = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H^{(ℓ)}
        Â = V W V^T

    where V ∈ R^{n×k} are top-k eigenvectors, Λ = diag(λ_1,...,λ_k) contains
    the corresponding eigenvalues, and H^{(ℓ)} ∈ R^{k×k} are learnable
    coefficient matrices.

    The polynomial filter allows learning frequency-dependent transformations:
    different eigenvalue magnitudes (frequencies) can be amplified or
    attenuated differently.

    Outputs raw logits. Use model.predict(logits) for [0,1] probabilities.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use.
    polynomial_degree : int, optional
        Degree of the spectral polynomial (K in the formula). Higher degrees
        allow more complex frequency responses. Default is 5.

    Notes
    -----
    The spectral polynomial is computed as:
        W_ij = Σ_ℓ (λ_i^ℓ) * H^{(ℓ)}_ij

    This can be seen as a bank of filters, each H^{(ℓ)} scaled by the ℓ-th
    power of eigenvalues, capturing different spectral characteristics.

    Examples
    --------
    >>> model = GraphFilterBank(k=8, polynomial_degree=5)
    >>> A_noisy = torch.randn(4, 50, 50)
    >>> A_noisy = (A_noisy + A_noisy.transpose(-1, -2)) / 2
    >>> logits = model(A_noisy)
    >>> predictions = model.predict(logits)  # apply sigmoid
    >>> predictions.shape
    torch.Size([4, 50, 50])
    """

    def __init__(self, k: int, polynomial_degree: int = 5):
        super().__init__(k=k)
        self.polynomial_degree = polynomial_degree

        # Learnable coefficient matrices H^{(ℓ)} ∈ R^{k×k} for ℓ = 0, ..., K-1
        self.H = nn.ParameterList([
            nn.Parameter(torch.empty(k, k))
            for _ in range(polynomial_degree)
        ])
        for h in self.H:
            nn.init.xavier_uniform_(h)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute V W V^T with polynomial W.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues of shape (batch, k) or (k,).
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
            Lambda = Lambda.unsqueeze(0)

        batch_size, n, k = V.shape

        # Normalize eigenvalues to [-1, 1] before polynomial computation.
        # Without normalization, Lambda^ell can explode (e.g., 10^5 for λ=10, ℓ=5)
        # causing gradient imbalance across polynomial terms.
        Lambda_max = Lambda.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        Lambda_normalized = Lambda / Lambda_max

        # Compute W = Σ Λ^ℓ ⊙ H^{(ℓ)}
        # Λ^ℓ is broadcast to (batch, k, k) via outer product of eigenvalue powers
        W = torch.zeros(batch_size, k, k, device=V.device, dtype=V.dtype)

        Lambda_power = torch.ones_like(Lambda_normalized)  # (batch, k), starts as Λ^0 = 1
        for ell in range(self.polynomial_degree):
            # Create scaling matrix from eigenvalue powers
            # Λ^ℓ_ij = λ_i^ℓ (same value across columns j)
            Lambda_matrix = Lambda_power.unsqueeze(-1).expand(-1, -1, k)  # (batch, k, k)
            # Element-wise multiply with H^{(ℓ)}
            W = W + Lambda_matrix * self.H[ell].unsqueeze(0)
            # Update for next power (using normalized eigenvalues)
            Lambda_power = Lambda_power * Lambda_normalized

        # Compute V W V^T
        # V @ W: (batch, n, k)
        VW = torch.matmul(V, W)
        # VW @ V^T: (batch, n, n)
        A_reconstructed = torch.matmul(VW, V.transpose(-1, -2))

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["polynomial_degree"] = self.polynomial_degree
        return config
