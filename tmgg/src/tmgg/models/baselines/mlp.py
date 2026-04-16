"""MLP baseline model for graph denoising.

A simple multi-layer perceptron that flattens the adjacency matrix, processes
it through hidden layers, and reshapes back. Tests whether the training
pipeline can train any neural network at all.
"""

from typing import Any

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import GraphData

from ..base import EdgeSource, GraphModel, read_edge_scalar, write_edge_scalar


class MLPBaseline(GraphModel):
    """MLP baseline: flatten -> MLP -> reshape.

    This model treats the adjacency matrix as a flat vector, processes it
    through a standard MLP, and reshapes back to a matrix. It ignores graph
    structure entirely but should be able to learn if the training loop works.

    Use this model as a sanity check: if an MLP cannot learn to denoise,
    the training pipeline itself has issues.

    Parameters
    ----------
    max_nodes
        Maximum number of nodes in the graph.
    hidden_dim
        Size of hidden layers in the MLP. Default 256.
    num_layers
        Number of hidden layers. Default 2.

    Attributes
    ----------
    mlp : nn.Sequential
        The MLP architecture with ReLU activations.

    Examples
    --------
    >>> model = MLPBaseline(max_nodes=32, hidden_dim=256)
    >>> A = torch.rand(4, 32, 32)  # batch of 4
    >>> logits = model(A)  # Returns raw logits
    >>> probs = model.predict(logits)  # Apply sigmoid for [0, 1]
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.flatten_dim = max_nodes * max_nodes
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        # Build MLP
        layers = []
        in_dim = self.flatten_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, self.flatten_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Apply MLP to flattened adjacency matrix.

        Reads the dense scalar adjacency through ``read_edge_scalar`` and
        writes the prediction via ``write_edge_scalar``; ``t`` is appended
        to ``data.y`` when supplied, matching the spec's two-line pattern.
        """
        A = read_edge_scalar(data, self.edge_source)
        B, N, _ = A.shape

        if self.max_nodes > N:
            A_padded = torch.zeros(B, self.max_nodes, self.max_nodes, device=A.device)
            A_padded[:, :N, :N] = A
        else:
            A_padded = A

        flat = A_padded.view(B, -1)
        out_flat = self.mlp(flat)
        out_adj = out_flat.view(B, self.max_nodes, self.max_nodes)

        if self.max_nodes > N:
            out_adj = out_adj[:, :N, :N]

        out = write_edge_scalar(data, edge_scalar=out_adj, target=self._output_target)
        if t is not None:
            new_y = torch.cat([out.y, t.unsqueeze(-1)], dim=-1)
            out = out.replace(y=new_y)
        return out

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        return {
            "model_type": "MLPBaseline",
            "max_nodes": self.max_nodes,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "flatten_dim": self.flatten_dim,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
