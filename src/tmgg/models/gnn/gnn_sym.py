"""Symmetric GNN variant — reconstruction via X @ X.T instead of X @ Y.T."""

from typing import Any, override

import torch

from tmgg.data.datasets.graph_types import GraphData

from ..base import EdgeSource, read_edge_scalar, write_edge_scalar
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
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__(
            num_layers=num_layers,
            num_terms=num_terms,
            feature_dim_in=feature_dim_in,
            feature_dim_out=feature_dim_out,
            eigenvalue_reg=eigenvalue_reg,
            edge_source=edge_source,
            output_dims_x_class=output_dims_x_class,
            output_dims_x_feat=output_dims_x_feat,
            output_dims_e_class=output_dims_e_class,
            output_dims_e_feat=output_dims_e_feat,
        )
        # Symmetric: only one output projection needed
        del self.out_y

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Compute denoised graph via symmetric reconstruction X @ X.T.

        Reads the dense scalar adjacency through ``read_edge_scalar`` (which
        respects ``self.edge_source``) and writes the prediction to the
        configured split edge field via ``write_edge_scalar``.
        """
        A = read_edge_scalar(data, self.edge_source)
        z = self.embedding_layer(A)

        for layer in self.layers:
            z = layer(A, z)

        emb = self.out_x(z)
        result_adj = torch.bmm(emb, emb.transpose(1, 2))
        out = write_edge_scalar(
            data, edge_scalar=result_adj, target=self._output_target
        )
        if t is not None:
            new_y = torch.cat([out.y, t.unsqueeze(-1)], dim=-1)
            out = out.replace(y=new_y)
        return out

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
            "eigenvalue_reg": self.eigenvalue_reg,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
