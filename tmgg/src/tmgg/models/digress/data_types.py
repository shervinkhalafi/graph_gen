"""Transformer-internal subclass for DiGress's spectral attention context.

``DenseGraphTransformerData`` extends ``DenseGraphDistribution`` with frozen
spectral attention context (eigvec / eigval). The transformer's body is
distribution-internal (every (i, j) slot has a learned hidden value),
so the base type is ``DenseGraphDistribution``. eigvec / eigval are
read-only context that survives layer iteration unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from tmgg.data.datasets.graph_types import DenseGraphDistribution


@dataclass(frozen=True)
class DenseGraphTransformerData(DenseGraphDistribution):
    """DenseGraphDistribution + frozen spectral attention context."""

    eigvec: Tensor | None = None  # (B, n_max, k)
    eigval: Tensor | None = None  # (B, k)

    @classmethod
    def from_base(
        cls,
        base: DenseGraphDistribution,
        *,
        eigvec: Tensor | None = None,
        eigval: Tensor | None = None,
    ) -> DenseGraphTransformerData:
        return cls(
            num_nodes_per_graph=base.num_nodes_per_graph,
            y=base.y,
            X_class=base.X_class,
            X_feat=base.X_feat,
            E_class=base.E_class,
            E_feat=base.E_feat,
            eigvec=eigvec,
            eigval=eigval,
        )

    def to_base(self) -> DenseGraphDistribution:
        """Drop transformer-specific fields. Used at the forward output."""
        return DenseGraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=self.X_class,
            X_feat=self.X_feat,
            E_class=self.E_class,
            E_feat=self.E_feat,
        )
