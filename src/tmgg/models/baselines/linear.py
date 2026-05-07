"""Linear baseline model for graph denoising.

A minimal model that learns a direct linear transformation of the adjacency matrix.
Initialized at identity to enable learning from a reasonable starting point.
"""

from typing import Any

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import GraphData

from ..base import EdgeSource, GraphModel, read_edge_scalar, write_edge_scalar


class LinearBaseline(GraphModel):
    """Linear transformation baseline: A_pred = W @ A @ W.T + b.

    This model applies a learnable linear transformation to the input adjacency
    matrix. Initialized with W=I (identity) so the initial output equals the
    input, providing a stable starting point for gradient descent.

    Use this model to verify the training pipeline works correctly. If this
    simple model cannot learn, the issue lies in the training loop or data,
    not in model architecture complexity.

    Parameters
    ----------
    max_nodes
        Maximum number of nodes in the graph. Determines parameter dimensions.

    Attributes
    ----------
    W : nn.Parameter
        Learnable weight matrix of shape (max_nodes, max_nodes).
    b : nn.Parameter
        Learnable bias matrix of shape (max_nodes, max_nodes).

    Examples
    --------
    >>> model = LinearBaseline(max_nodes=32)
    >>> A = torch.eye(32).unsqueeze(0)  # batch of 1
    >>> logits = model(A)  # Returns raw logits
    >>> probs = model.predict(logits)  # Apply sigmoid for [0, 1]
    """

    def __init__(
        self,
        max_nodes: int,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        # Initialize W at identity for stable starting point
        self.W = nn.Parameter(torch.eye(max_nodes))
        self.b = nn.Parameter(torch.zeros(max_nodes, max_nodes))

    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Apply linear transformation to input graph.

        Reads the dense scalar adjacency through ``read_edge_scalar`` (which
        respects ``self.edge_source``) and writes the prediction to the
        configured split edge field via ``write_edge_scalar``. When ``t`` is
        provided it is appended to ``data.y`` following the spec's two-line
        pattern.
        """
        A = read_edge_scalar(data, self.edge_source)
        B, N, _ = A.shape

        W = self.W[:N, :N]
        b = self.b[:N, :N]

        W_expanded = W.unsqueeze(0).expand(B, -1, -1)
        out_adj = torch.bmm(torch.bmm(W_expanded, A), W_expanded.transpose(-2, -1))
        out_adj = out_adj + b.unsqueeze(0)

        out = write_edge_scalar(data, edge_scalar=out_adj, target=self._output_target)
        if t is not None:
            new_y = torch.cat([out.y, t.unsqueeze(-1)], dim=-1)
            out = out.replace(y=new_y)
        return out

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        return {
            "model_type": "LinearBaseline",
            "max_nodes": self.max_nodes,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
