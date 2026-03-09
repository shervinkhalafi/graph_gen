"""Symmetric GNN variant — reconstruction via X @ X.T instead of X @ Y.T."""

from typing import Any, override

import torch

from tmgg.data.datasets.graph_types import GraphData

from .gnn import GNN


class GNNSymmetric(GNN):
    """Symmetric GNN for adjacency matrix reconstruction.

    Inherits all GNN infrastructure (embedding, convolution layers, output
    projections) but uses a single projection ``out_x`` for symmetric
    reconstruction via ``X @ X.T`` instead of asymmetric ``X @ Y.T``.

    The ``out_y`` projection inherited from ``GNN`` is removed in ``__init__``.
    """

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim_in: int = 10,
        feature_dim_out: int = 10,
        eigenvalue_reg: float = 0.0,
    ):
        super().__init__(
            num_layers=num_layers,
            num_terms=num_terms,
            feature_dim_in=feature_dim_in,
            feature_dim_out=feature_dim_out,
            eigenvalue_reg=eigenvalue_reg,
        )
        # Symmetric: only one output projection needed
        del self.out_y

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Compute denoised graph via symmetric reconstruction X @ X.T.

        Parameters
        ----------
        data
            Graph features. The adjacency is extracted via
            ``data.to_adjacency()``.
        t
            Diffusion timestep tensor, or None. Currently unused.

        Returns
        -------
        GraphData
            Denoised graph with 2-class edge features.
        """
        A = data.to_adjacency()
        z = self.embedding_layer(A)

        for layer in self.layers:
            z = layer(A, z)

        emb = self.out_x(z)
        result_adj = torch.bmm(emb, emb.transpose(1, 2))
        return GraphData.from_adjacency(result_adj)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
            "eigenvalue_reg": self.eigenvalue_reg,
        }
