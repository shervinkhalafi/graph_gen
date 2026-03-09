from typing import Any, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.layers.eigen_embedding import TruncatedEigenEmbedding
from tmgg.models.layers.nvgcn_layer import NodeVarGraphConvolutionLayer

from ..base import GraphModel


class NodeVarGNN(GraphModel):
    """Node-variant Graph Neural Network."""

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim: int = 10,
        eigenvalue_reg: float = 0.0,
    ):
        """
        Initialize Node-variant GNN.

        Args:
            num_layers: Number of layers
            num_terms: Number of polynomial terms
            feature_dim: Feature dimension
            eigenvalue_reg: Diagonal regularization for eigendecomposition stability
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim = feature_dim
        self.eigenvalue_reg = eigenvalue_reg

        self.embedding_layer = TruncatedEigenEmbedding(
            target_dim=feature_dim, eigenvalue_reg=eigenvalue_reg
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                NodeVarGraphConvolutionLayer(num_terms, feature_dim, feature_dim)
            )

        self.out_x = nn.Linear(feature_dim, feature_dim)
        self.out_y = nn.Linear(feature_dim, feature_dim)

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Compute denoised graph from input graph data.

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
        x = data.to_adjacency()
        z = self.embedding_layer(x)

        for layer in self.layers:
            z = layer(x, z)
        emb_x = self.out_x(z)
        emb_y = self.out_y(z)
        result_adj = torch.bmm(emb_x, emb_y.transpose(1, 2))
        return GraphData.from_adjacency(result_adj)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim": self.feature_dim,
            "eigenvalue_reg": self.eigenvalue_reg,
        }
