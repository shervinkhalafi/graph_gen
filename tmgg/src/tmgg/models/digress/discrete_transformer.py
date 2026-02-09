"""Discrete diffusion wrapper for the DiGress _GraphTransformer."""

from __future__ import annotations

from typing import Any, override

import torch.nn as nn
from torch import Tensor

from tmgg.models.digress.transformer_model import GraphFeatures, _GraphTransformer


class DiscreteGraphTransformer(nn.Module):
    """Wrapper for _GraphTransformer accepting categorical (X, E, y, node_mask).

    This thin wrapper instantiates ``_GraphTransformer`` with
    ``assume_adjacency_input=False`` so the transformer expects pre-encoded
    categorical node and edge features rather than raw adjacency matrices.

    Parameters
    ----------
    n_layers
        Number of transformer layers.
    input_dims
        Input feature dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    hidden_mlp_dims
        Hidden MLP dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    hidden_dims
        Transformer hidden dimensions with keys ``"dx"``, ``"de"``, ``"dy"``,
        ``"n_head"``.
    output_dims
        Output dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    """

    n_layers: int
    input_dims: dict[str, int]
    hidden_mlp_dims: dict[str, int]
    hidden_dims: dict[str, int]
    output_dims: dict[str, int]
    transformer: _GraphTransformer

    def __init__(
        self,
        n_layers: int,
        input_dims: dict[str, int],
        hidden_mlp_dims: dict[str, int],
        hidden_dims: dict[str, int],
        output_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_dims = input_dims
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.transformer = _GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            assume_adjacency_input=False,
        )

    @override
    def forward(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> GraphFeatures:
        """Forward pass through the graph transformer.

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
        GraphFeatures
            Named tuple of ``(X, E, y)`` with output dimensions.
        """
        return self.transformer(X, E, y, node_mask)

    def get_config(self) -> dict[str, Any]:
        """Return model configuration for serialization and logging."""
        return {
            "model_class": "DiscreteGraphTransformer",
            "n_layers": self.n_layers,
            "input_dims": self.input_dims,
            "hidden_mlp_dims": self.hidden_mlp_dims,
            "hidden_dims": self.hidden_dims,
            "output_dims": self.output_dims,
        }
