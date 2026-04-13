"""Graph Neural Network models for graph denoising."""

from typing import Any, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.layers.eigen_embedding import TruncatedEigenEmbedding
from tmgg.models.layers.gcn import GraphConvolutionLayer

from ..base import GraphModel


class GNN(GraphModel):
    """Graph Neural Network for adjacency matrix reconstruction.

    Computes separate X and Y node embeddings and reconstructs the
    adjacency via ``X @ Y.T``. By default the output is symmetrized
    with ``(A + A.T) / 2``; set ``symmetrized_output=False`` to keep
    the raw asymmetric product.
    """

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim_in: int = 10,
        feature_dim_out: int = 10,
        eigenvalue_reg: float = 0.0,
        symmetrized_output: bool = True,
    ):
        """Initialize GNN.

        Parameters
        ----------
        num_layers
            Number of graph convolution layers.
        num_terms
            Number of terms in polynomial filters.
        feature_dim_in
            Input feature dimension (truncated eigenvector dimension).
        feature_dim_out
            Output embedding dimension for adjacency reconstruction.
        eigenvalue_reg
            Diagonal regularization for eigendecomposition stability.
        symmetrized_output
            If True (default), symmetrize the reconstructed adjacency
            via ``(A + A.T) / 2`` after the dot product.
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim_in = feature_dim_in
        self.feature_dim_out = feature_dim_out
        self.eigenvalue_reg = eigenvalue_reg
        self.symmetrized_output = symmetrized_output

        self.embedding_layer = TruncatedEigenEmbedding(
            target_dim=feature_dim_in, eigenvalue_reg=eigenvalue_reg
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolutionLayer(num_terms, feature_dim_in))

        self.out_x = nn.Linear(feature_dim_in, feature_dim_out)
        self.out_y = nn.Linear(feature_dim_in, feature_dim_out)

    def _embed(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (X, Y) node embeddings from adjacency.

        Shared between ``forward()`` (which reconstructs the adjacency)
        and ``embeddings()`` (used by hybrid models).
        """
        z = self.embedding_layer(A)

        for layer in self.layers:
            z = layer(A, z)

        return self.out_x(z), self.out_y(z)

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Compute denoised graph from input graph data.

        Parameters
        ----------
        data
            Graph features. The dense edge state is extracted via
            ``data.to_edge_state()``.
        t
            Diffusion timestep tensor, or None. Currently unused.

        Returns
        -------
        GraphData
            Denoised graph with 2-class edge features.
        """
        A = data.to_edge_state()
        emb_x, emb_y = self._embed(A)
        result_adj = torch.bmm(emb_x, emb_y.transpose(1, 2))
        if self.symmetrized_output:
            result_adj = (result_adj + result_adj.transpose(1, 2)) / 2
        return GraphData.from_edge_state(result_adj, node_mask=data.node_mask)

    def embeddings(self, data: GraphData) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute node embeddings without adjacency reconstruction.

        Used by hybrid models that need embeddings for further processing.

        Parameters
        ----------
        data
            Graph features. The dense edge state is extracted internally.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (X, Y) embeddings, each of shape
            ``(batch, n, feature_dim_out)``.
        """
        A = data.to_edge_state()
        return self._embed(A)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
            "eigenvalue_reg": self.eigenvalue_reg,
            "symmetrized_output": self.symmetrized_output,
        }
