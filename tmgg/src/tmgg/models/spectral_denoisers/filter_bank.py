"""Graph Filter Bank denoiser with spectral polynomial parameterization.

Implements a learnable spectral filter as a polynomial in the eigenvalues,
allowing the model to learn frequency-dependent transformations. Supports
both symmetric (VWV^T) and asymmetric (XY^T) reconstruction modes.
"""

from typing import Any

import torch
import torch.nn as nn

from tmgg.models.spectral_denoisers.base_spectral import (
    EmbeddingSource,
    SpectralDenoiser,
)


class GraphFilterBank(SpectralDenoiser):
    """Graph Filter Bank denoiser with spectral polynomial.

    Reconstructs the adjacency matrix using spectral polynomial filters with
    optional asymmetric mode:

    Symmetric mode (default):
        W = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H^{(ℓ)}
        Â = V W V^T

    Asymmetric mode:
        W_X = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H_X^{(ℓ)},  W_Y = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H_Y^{(ℓ)}
        X = V W_X,  Y = V W_Y
        Â = X Y^T

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
    asymmetric : bool, optional
        If True, use separate polynomial coefficient matrices H_X and H_Y
        for asymmetric reconstruction X @ Y.T. Default is False.

    Notes
    -----
    The spectral polynomial is computed as:
        W_ij = Σ_ℓ (λ_i^ℓ) * H^{(ℓ)}_ij

    This can be seen as a bank of filters, each H^{(ℓ)} scaled by the ℓ-th
    power of eigenvalues, capturing different spectral characteristics.

    Asymmetric mode doubles the parameter count but allows the model to
    learn different row and column spectral transformations.

    Examples
    --------
    >>> model = GraphFilterBank(k=8, polynomial_degree=5)
    >>> A_noisy = torch.randn(4, 50, 50)
    >>> A_noisy = (A_noisy + A_noisy.transpose(-1, -2)) / 2
    >>> logits = model(A_noisy)
    >>> predictions = model.predict(logits)  # apply sigmoid
    >>> predictions.shape
    torch.Size([4, 50, 50])

    >>> # Asymmetric mode
    >>> model_asym = GraphFilterBank(k=8, polynomial_degree=5, asymmetric=True)
    >>> logits_asym = model_asym(A_noisy)
    """

    def __init__(
        self,
        k: int,
        polynomial_degree: int = 5,
        asymmetric: bool = False,
        embedding_source: EmbeddingSource = "eigenvector",
        pearl_num_layers: int = 3,
        pearl_hidden_dim: int = 64,
        pearl_input_samples: int = 32,
        pearl_max_nodes: int = 200,
    ):
        super().__init__(
            k=k,
            embedding_source=embedding_source,
            pearl_num_layers=pearl_num_layers,
            pearl_hidden_dim=pearl_hidden_dim,
            pearl_input_samples=pearl_input_samples,
            pearl_max_nodes=pearl_max_nodes,
        )
        self.polynomial_degree = polynomial_degree
        self.asymmetric = asymmetric

        if asymmetric:
            # Separate coefficient matrices for X and Y branches
            self.H_X = nn.ParameterList(
                [nn.Parameter(torch.empty(k, k)) for _ in range(polynomial_degree)]
            )
            self.H_Y = nn.ParameterList(
                [nn.Parameter(torch.empty(k, k)) for _ in range(polynomial_degree)]
            )
            for h in self.H_X:
                nn.init.xavier_uniform_(h)
            for h in self.H_Y:
                nn.init.xavier_uniform_(h)
            self.H = None  # type: ignore[assignment]
        else:
            # Learnable coefficient matrices H^{(ℓ)} ∈ R^{k×k} for ℓ = 0, ..., K-1
            self.H = nn.ParameterList(
                [nn.Parameter(torch.empty(k, k)) for _ in range(polynomial_degree)]
            )
            for h in self.H:
                nn.init.xavier_uniform_(h)
            self.H_X = None  # type: ignore[assignment]
            self.H_Y = None  # type: ignore[assignment]

    def _compute_spectral_polynomial(
        self,
        Lambda_normalized: torch.Tensor,
        H_list: nn.ParameterList,
        k: int,
    ) -> torch.Tensor:
        """Compute spectral polynomial W = Σ Λ^ℓ ⊙ H^{(ℓ)}.

        Parameters
        ----------
        Lambda_normalized : torch.Tensor
            Normalized eigenvalues of shape (batch, k).
        H_list : nn.ParameterList
            List of coefficient matrices H^{(ℓ)}.
        k : int
            Number of eigenvectors.

        Returns
        -------
        torch.Tensor
            Polynomial result W of shape (batch, k, k).
        """
        batch_size = Lambda_normalized.shape[0]
        W = torch.zeros(
            batch_size,
            k,
            k,
            device=Lambda_normalized.device,
            dtype=Lambda_normalized.dtype,
        )

        Lambda_power = torch.ones_like(
            Lambda_normalized
        )  # (batch, k), starts as Λ^0 = 1
        for ell in range(self.polynomial_degree):
            # Create scaling matrix from eigenvalue powers
            Lambda_matrix = Lambda_power.unsqueeze(-1).expand(
                -1, -1, k
            )  # (batch, k, k)
            W = W + Lambda_matrix * H_list[ell].unsqueeze(0)
            Lambda_power = Lambda_power * Lambda_normalized

        return W

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VWV^T (symmetric) or XY^T (asymmetric) with polynomial W.

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

        if self.asymmetric:
            # Compute separate polynomials for X and Y
            assert self.H_X is not None and self.H_Y is not None
            W_X = self._compute_spectral_polynomial(Lambda_normalized, self.H_X, k)
            W_Y = self._compute_spectral_polynomial(Lambda_normalized, self.H_Y, k)

            # X = V @ W_X, Y = V @ W_Y
            X = torch.matmul(V, W_X)  # (batch, n, k)
            Y = torch.matmul(V, W_Y)  # (batch, n, k)
            A_reconstructed = torch.matmul(X, Y.transpose(-1, -2))  # (batch, n, n)
        else:
            # Symmetric: compute single polynomial and do V W V^T
            assert self.H is not None
            W = self._compute_spectral_polynomial(Lambda_normalized, self.H, k)
            VW = torch.matmul(V, W)  # (batch, n, k)
            A_reconstructed = torch.matmul(VW, V.transpose(-1, -2))  # (batch, n, n)

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["polynomial_degree"] = self.polynomial_degree
        config["asymmetric"] = self.asymmetric
        return config
