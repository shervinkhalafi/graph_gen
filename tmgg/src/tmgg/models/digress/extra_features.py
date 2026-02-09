"""Extra feature computation for discrete diffusion models."""

from __future__ import annotations

import torch
from torch import Tensor


class DummyExtraFeatures:
    """Returns zero-width tensors for extra features.

    Placeholder for the synthetic graph MVP. Replace with actual feature
    computation (eigenvalues, cycle counts, etc.) for molecular graphs.
    """

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (zero-width) extra features.

        Parameters
        ----------
        X
            Node features, shape ``(bs, n, dx)``.
        E
            Edge features, shape ``(bs, n, n, de)``.
        y
            Global features, shape ``(bs, dy)``.
        node_mask
            Boolean mask for valid nodes, shape ``(bs, n)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(extra_X, extra_E, extra_y)`` all with zero-width last
            dimension: shapes ``(bs, n, 0)``, ``(bs, n, n, 0)``,
            ``(bs, 0)``.
        """
        bs, n, _ = X.shape
        return (
            torch.zeros(bs, n, 0, device=X.device, dtype=X.dtype),
            torch.zeros(bs, n, n, 0, device=X.device, dtype=X.dtype),
            torch.zeros(bs, 0, device=X.device, dtype=X.dtype),
        )
