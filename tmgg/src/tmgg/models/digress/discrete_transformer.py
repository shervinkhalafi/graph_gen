"""Discrete diffusion wrapper for the DiGress _GraphTransformer."""

from __future__ import annotations

from typing import Any, override

import torch
import torch.nn as nn
from torch import Tensor

from tmgg.models.digress.transformer_model import GraphFeatures, _GraphTransformer


class DiscreteGraphTransformer(nn.Module):
    """Wrapper for _GraphTransformer accepting categorical (X, E, y, node_mask).

    This thin wrapper instantiates ``_GraphTransformer`` with
    ``assume_adjacency_input=False`` so the transformer expects pre-encoded
    categorical node and edge features rather than raw adjacency matrices.

    Optionally extracts top-k eigenvectors from the noisy adjacency matrix
    and concatenates them with the one-hot node features, giving the
    transformer richer structural information while preserving the
    categorical diffusion mathematics (transition matrices still operate
    on the 2-class edge representation).

    Parameters
    ----------
    n_layers
        Number of transformer layers.
    input_dims
        Input feature dimensions with keys ``"X"``, ``"E"``, ``"y"``.
        When ``use_eigenvectors=True``, ``X`` must equal the categorical
        node dimension plus ``k`` (e.g. ``2 + 50 = 52``).
    hidden_mlp_dims
        Hidden MLP dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    hidden_dims
        Transformer hidden dimensions with keys ``"dx"``, ``"de"``, ``"dy"``,
        ``"n_head"``.
    output_dims
        Output dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    use_eigenvectors
        If True, extract top-k eigenvectors from the noisy adjacency
        (derived from ``E`` via argmax) and concatenate with ``X`` before
        feeding the transformer.
    k
        Number of eigenvectors to extract. Required when
        ``use_eigenvectors=True``.
    """

    n_layers: int
    input_dims: dict[str, int]
    hidden_mlp_dims: dict[str, int]
    hidden_dims: dict[str, int]
    output_dims: dict[str, int]
    transformer: _GraphTransformer
    eigen_layer: nn.Module | None

    def __init__(
        self,
        n_layers: int,
        input_dims: dict[str, int],
        hidden_mlp_dims: dict[str, int],
        hidden_dims: dict[str, int],
        output_dims: dict[str, int],
        use_eigenvectors: bool = False,
        k: int | None = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_dims = input_dims
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self._use_eigenvectors = use_eigenvectors
        self._k = k

        if use_eigenvectors:
            if k is None:
                raise ValueError("k must be specified when use_eigenvectors=True")
            from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer

            self.eigen_layer = TopKEigenLayer(k=k)
        else:
            self.eigen_layer = None

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

        When ``use_eigenvectors=True``, extracts eigenvectors from the
        noisy adjacency (``E.argmax(-1)``) and concatenates them with
        ``X`` before the transformer. The transformer's ``input_dims["X"]``
        must account for the extra ``k`` dimensions.

        Parameters
        ----------
        X
            Node features, shape ``(bs, n, dx_cat)``.
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
        if self._use_eigenvectors and self.eigen_layer is not None:
            # Derive adjacency from one-hot edges: class > 0 means edge present
            adj = (E.argmax(dim=-1) > 0).float()

            # Extract eigenvectors, pad if graph smaller than k
            V, _ = self.eigen_layer(adj)  # (bs, n, actual_k)
            actual_k = V.shape[-1]
            if self._k is not None and actual_k < self._k:
                V = torch.nn.functional.pad(V, (0, self._k - actual_k))

            # Zero out eigenvectors for masked nodes
            V = V * node_mask.unsqueeze(-1).float()

            # Concatenate: [one-hot node features | eigenvectors]
            X = torch.cat([X, V], dim=-1)

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
            "use_eigenvectors": self._use_eigenvectors,
            "k": self._k,
        }
