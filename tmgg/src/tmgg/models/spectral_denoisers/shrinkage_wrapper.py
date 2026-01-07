"""SVD-based shrinkage wrappers for spectral denoising models.

This module provides wrapper architectures that enforce shrinkage-based denoising
via singular value modification. The wrappers use an inner model (e.g.,
SelfAttentionDenoiser) to extract learned features, which are then aggregated
and used to predict shrinkage coefficients for the SVD singular values.

Two variants are provided:

- **StrictShrinkageWrapper**: Sigmoid gating ensures coefficients in [0, 1],
  enforcing a strict shrinkage (values can only decrease).
- **RelaxedShrinkageWrapper**: FiLM-style modulation (scale * S + shift) allows
  both shrinkage and expansion of singular values.

Note
----
This module is experimental and not yet used in standard experiment sweeps.
The classes are registered and documented for future use in spectral denoising
research.

Mathematical Formulation
------------------------
Given noisy adjacency A_noisy:

1. SVD: U, S, V^T = torch.linalg.svd(A_noisy)
2. Features: F = inner_model.get_features(A_noisy)
3. Aggregate: h = aggregate(F)  # Mean or attention pooling
4. Predict: raw_α = MLP(h)

For strict shrinkage:
    α = sigmoid(raw_α)
    S_mod = α * S

For relaxed (FiLM) modulation:
    (scale, shift) = split(raw_α)
    S_mod = scale * S + shift

5. Reconstruct: Â = U @ diag(S_mod) @ V^T
6. Symmetrize: Â = (Â + Â^T) / 2

Examples
--------
>>> from tmgg.models.spectral_denoisers import SelfAttentionDenoiser
>>> inner = SelfAttentionDenoiser(k=8, d_k=32)
>>> model = StrictShrinkageWrapper(inner, max_rank=16)
>>> A_noisy = torch.randn(4, 50, 50)
>>> A_noisy = (A_noisy + A_noisy.T) / 2
>>> A_denoised = model(A_noisy)
"""

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from tmgg.models.base import DenoisingModel
from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser


class ShrinkageSVDLayer(nn.Module):
    """Compute truncated SVD of adjacency matrices.

    Extracts the top-r singular values and corresponding left/right
    singular vectors, suitable for low-rank approximation or shrinkage.

    Parameters
    ----------
    max_rank : int
        Maximum number of singular values/vectors to retain.
    """

    def __init__(self, max_rank: int):
        super().__init__()
        self.max_rank = max_rank

    def forward(
        self, A: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute truncated SVD.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        U : torch.Tensor
            Left singular vectors, shape (batch, n, r) or (n, r).
        S : torch.Tensor
            Singular values, shape (batch, r) or (r,).
        Vh : torch.Tensor
            Right singular vectors (transposed), shape (batch, r, n) or (r, n).
        """
        unbatched = A.ndim == 2
        if unbatched:
            A = A.unsqueeze(0)

        # Full SVD then truncate (more numerically stable than randomized)
        U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)

        # Truncate to max_rank (or actual rank if smaller)
        r = min(self.max_rank, S_full.shape[-1])
        U = U_full[..., :r]
        S = S_full[..., :r]
        Vh = Vh_full[..., :r, :]

        if unbatched:
            U = U.squeeze(0)
            S = S.squeeze(0)
            Vh = Vh.squeeze(0)

        return U, S, Vh


class ShrinkageWrapper(DenoisingModel):
    """Base wrapper for shrinkage-based spectral denoising.

    Wraps an inner SpectralDenoiser model to extract learned features,
    which are aggregated and used to predict shrinkage coefficients
    for SVD singular values.

    Subclasses implement `_apply_shrinkage` to define the shrinkage function.

    Parameters
    ----------
    inner_model : SpectralDenoiser
        Inner model used to extract features. Must implement `get_features()`.
    max_rank : int
        Maximum number of singular values to use. Default is 50.
    aggregation : str
        Feature aggregation method: "mean" or "attention". Default is "mean".
    hidden_dim : int
        Hidden dimension for the shrinkage MLP. Default is 128.
    mlp_layers : int
        Number of layers in the shrinkage MLP. Default is 2.

    Raises
    ------
    ValueError
        If inner_model does not implement get_features().
    """

    def __init__(
        self,
        inner_model: SpectralDenoiser,
        max_rank: int = 50,
        aggregation: str = "mean",
        hidden_dim: int = 128,
        mlp_layers: int = 2,
    ):
        super().__init__()

        if not hasattr(inner_model, "get_features"):
            raise ValueError(
                f"Inner model {type(inner_model).__name__} must implement get_features() "
                "to be used with ShrinkageWrapper"
            )

        self.inner_model = inner_model
        self.max_rank = max_rank
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim
        self.mlp_layers = mlp_layers

        # SVD layer
        self.svd_layer = ShrinkageSVDLayer(max_rank=max_rank)

        # Feature dimension from inner model
        self.feature_dim = self._get_inner_feature_dim()

        # Aggregation (for attention pooling)
        if aggregation == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(self.feature_dim, 1),
                nn.Softmax(dim=-2),  # Softmax over nodes
            )

        # Shrinkage coefficient prediction MLP
        self.shrinkage_mlp = self._build_shrinkage_mlp()

    def _get_inner_feature_dim(self) -> int:
        """Get feature dimension from inner model.

        Returns the dimension of features returned by inner_model.get_features().
        """
        # Try to infer from model structure
        if hasattr(self.inner_model, "d_k"):
            return int(self.inner_model.d_k)  # type: ignore[arg-type]
        if hasattr(self.inner_model, "d_model"):
            return int(self.inner_model.d_model)  # type: ignore[arg-type]
        # Fallback: assume k-dimensional features
        return int(self.inner_model.k)

    @abstractmethod
    def _get_output_dim(self) -> int:
        """Return the output dimension for the shrinkage MLP.

        Subclasses return 1 per singular value (strict) or 2 (relaxed).
        """
        pass

    def _build_shrinkage_mlp(self) -> nn.Module:
        """Build the MLP that predicts shrinkage coefficients."""
        layers: list[nn.Module] = []
        in_dim = self.feature_dim
        out_dim = self._get_output_dim() * self.max_rank

        for _ in range(self.mlp_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def _aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate node features to graph-level representation.

        Parameters
        ----------
        features : torch.Tensor
            Node features of shape (batch, n, feature_dim) or (n, feature_dim).

        Returns
        -------
        torch.Tensor
            Graph-level features of shape (batch, feature_dim) or (feature_dim,).
        """
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)

        if self.aggregation == "mean":
            h = features.mean(dim=1)  # (batch, feature_dim)
        elif self.aggregation == "attention":
            # Attention-weighted pooling
            attn_weights = self.attn_pool(features)  # (batch, n, 1)
            h = (features * attn_weights).sum(dim=1)  # (batch, feature_dim)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if unbatched:
            h = h.squeeze(0)

        return h

    def _predict_shrinkage(self, h: torch.Tensor, num_sv: int) -> torch.Tensor:
        """Predict shrinkage parameters from aggregated features.

        Parameters
        ----------
        h : torch.Tensor
            Graph-level features of shape (batch, feature_dim) or (feature_dim,).
        num_sv : int
            Number of singular values (may be less than max_rank for small graphs).

        Returns
        -------
        torch.Tensor
            Raw shrinkage parameters before applying the shrinkage function.
        """
        raw_params = self.shrinkage_mlp(
            h
        )  # (batch, output_dim * max_rank) or (output_dim * max_rank,)
        return raw_params

    @abstractmethod
    def _apply_shrinkage(
        self, raw_params: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """Apply shrinkage to singular values.

        Parameters
        ----------
        raw_params : torch.Tensor
            Raw parameters from the shrinkage MLP.
        S : torch.Tensor
            Original singular values.

        Returns
        -------
        torch.Tensor
            Modified singular values.
        """
        pass

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Denoise adjacency via shrinkage.

        Parameters
        ----------
        A : torch.Tensor
            Noisy adjacency matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        torch.Tensor
            Denoised adjacency matrix.
        """
        unbatched = A.ndim == 2
        if unbatched:
            A = A.unsqueeze(0)

        # SVD decomposition
        U, S, Vh = self.svd_layer(A)

        # Get features from inner model
        features = self.inner_model.get_features(A)  # (batch, n, feature_dim)

        # Aggregate to graph-level
        h = self._aggregate(features)  # (batch, feature_dim)

        # Predict shrinkage coefficients
        num_sv = S.shape[-1]
        raw_params = self._predict_shrinkage(h, num_sv)

        # Apply shrinkage to singular values
        S_mod = self._apply_shrinkage(raw_params, S)

        # Reconstruct: U @ diag(S_mod) @ Vh
        # (batch, n, r) @ diag(r) @ (batch, r, n) -> (batch, n, n)
        S_diag = torch.diag_embed(S_mod)  # (batch, r, r)
        A_reconstructed = torch.matmul(torch.matmul(U, S_diag), Vh)

        # Symmetrize
        A_reconstructed = (A_reconstructed + A_reconstructed.transpose(-1, -2)) / 2

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        """Get model configuration for logging/saving."""
        return {
            "model_class": self.__class__.__name__,
            "inner_model": self.inner_model.get_config(),
            "max_rank": self.max_rank,
            "aggregation": self.aggregation,
            "hidden_dim": self.hidden_dim,
            "mlp_layers": self.mlp_layers,
        }


class StrictShrinkageWrapper(ShrinkageWrapper):
    """Shrinkage wrapper with sigmoid gating.

    Enforces strict shrinkage where each singular value is multiplied by
    a coefficient α ∈ [0, 1]:

        S_mod = sigmoid(raw_α) * S

    This guarantees that singular values can only decrease, never increase,
    which corresponds to a denoising interpretation where noise adds energy
    that should be removed.

    See Also
    --------
    ShrinkageWrapper : Base class with full parameter documentation.
    RelaxedShrinkageWrapper : Alternative with unconstrained modulation.
    """

    def _get_output_dim(self) -> int:
        return 1  # One coefficient per singular value

    def _apply_shrinkage(
        self, raw_params: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """Apply sigmoid-gated shrinkage.

        Parameters
        ----------
        raw_params : torch.Tensor
            Raw shrinkage logits of shape (batch, max_rank) or (max_rank,).
        S : torch.Tensor
            Singular values of shape (batch, r) or (r,).

        Returns
        -------
        torch.Tensor
            Modified singular values in [0, S].
        """
        # Reshape raw_params to match S
        unbatched = S.ndim == 1
        if unbatched:
            raw_params = raw_params.unsqueeze(0)
            S = S.unsqueeze(0)

        # Truncate to actual number of singular values
        r = S.shape[-1]
        alpha = torch.sigmoid(raw_params[..., :r])  # (batch, r) in [0, 1]

        S_mod = alpha * S

        if unbatched:
            S_mod = S_mod.squeeze(0)

        return S_mod


class RelaxedShrinkageWrapper(ShrinkageWrapper):
    """Shrinkage wrapper with FiLM-style modulation.

    Applies affine transformation to singular values, allowing both
    shrinkage and expansion:

        S_mod = scale * S + shift

    where scale and shift are predicted from the learned features. This is
    more expressive than strict shrinkage but loses the guarantee that
    energy only decreases.

    The scale is passed through softplus to ensure positivity.

    See Also
    --------
    ShrinkageWrapper : Base class with full parameter documentation.
    StrictShrinkageWrapper : Alternative with guaranteed shrinkage.
    """

    def _get_output_dim(self) -> int:
        return 2  # Scale and shift per singular value

    def _apply_shrinkage(
        self, raw_params: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM-style affine modulation.

        Parameters
        ----------
        raw_params : torch.Tensor
            Raw parameters of shape (batch, 2*max_rank) or (2*max_rank,).
            First half is scale (pre-softplus), second half is shift.
        S : torch.Tensor
            Singular values of shape (batch, r) or (r,).

        Returns
        -------
        torch.Tensor
            Affinely transformed singular values.
        """
        unbatched = S.ndim == 1
        if unbatched:
            raw_params = raw_params.unsqueeze(0)
            S = S.unsqueeze(0)

        r = S.shape[-1]

        # Split into scale (first half) and shift (second half)
        scale_raw = raw_params[..., : self.max_rank]
        shift = raw_params[..., self.max_rank : 2 * self.max_rank]

        # Truncate to actual rank
        scale_raw = scale_raw[..., :r]
        shift = shift[..., :r]

        # Softplus for positive scale
        scale = nn.functional.softplus(scale_raw)

        S_mod = scale * S + shift

        if unbatched:
            S_mod = S_mod.squeeze(0)

        return S_mod
