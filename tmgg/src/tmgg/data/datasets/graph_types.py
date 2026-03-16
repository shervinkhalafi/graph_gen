"""Typed containers for categorical graph data and structural context.

``GraphData`` is the universal batch type for all experiments. It holds
batched one-hot node/edge/global features alongside a ``node_mask`` that
tracks which node positions are real versus padding. Conversion methods
bridge the categorical representation and the simpler adjacency format
used by denoising experiments.

``GraphStructure`` bundles pre-computed topological features (adjacency,
eigenvectors, eigenvalues) derived from a ``GraphData`` instance before
any learned transformations. These remain constant across transformer
layer iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch_geometric.data import Batch, Data


@dataclass(frozen=True, slots=True)
class GraphData:
    """Batched categorical graph features with node validity mask.

    Parameters
    ----------
    X : Tensor
        Node features. Batched: ``(bs, n, dx)``; single: ``(n, dx)``.
    E : Tensor
        Edge features. Batched: ``(bs, n, n, de)``; single: ``(n, n, de)``.
    y : Tensor
        Global features. Batched: ``(bs, dy)``; single: ``(dy,)``.
    node_mask : Tensor
        Boolean or float mask indicating real (vs padded) nodes.
        Batched: ``(bs, n)``; single: ``(n,)``.
    """

    X: Tensor
    E: Tensor
    y: Tensor
    node_mask: Tensor

    def mask(self) -> GraphData:
        """Zero out features at masked node positions and assert E symmetry.

        Returns
        -------
        GraphData
            New instance with masked values zeroed.

        Raises
        ------
        AssertionError
            If edge features are not symmetric after masking.
        """
        x_mask = self.node_mask.unsqueeze(-1)  # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)  # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)  # (bs, 1, n, 1)

        X_masked = self.X * x_mask
        E_masked = self.E * e_mask1 * e_mask2
        assert torch.allclose(E_masked, torch.transpose(E_masked, 1, 2)), (
            f"Edge features E must be symmetric after masking. "
            f"Max asymmetry: {(E_masked - torch.transpose(E_masked, 1, 2)).abs().max().item():.2e}"
        )
        return GraphData(X=X_masked, E=E_masked, y=self.y, node_mask=self.node_mask)

    def mask_zero_diag(self) -> GraphData:
        """Zero masked positions and the diagonal of E.

        Used inside the transformer where self-loops are excluded from
        the attention mechanism.
        """
        bs, n = self.node_mask.size()
        mask_diag = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        mask_diag = mask_diag * (
            ~torch.eye(n, device=self.node_mask.device, dtype=torch.bool).unsqueeze(0)
        )

        X_masked = self.X * self.node_mask.unsqueeze(-1)
        E_masked = self.E * mask_diag.unsqueeze(-1)

        return GraphData(X=X_masked, E=E_masked, y=self.y, node_mask=self.node_mask)

    def type_as(self, x: Tensor) -> GraphData:
        """Return a new instance with feature tensors cast to match ``x``.

        The ``node_mask`` is left unchanged (it is boolean, not a feature).
        """
        return GraphData(
            X=self.X.type_as(x),
            E=self.E.type_as(x),
            y=self.y.type_as(x),
            node_mask=self.node_mask,
        )

    def to(self, device: torch.device | str) -> GraphData:
        """Move all tensors to ``device``, returning a new instance.

        Needed for Lightning's ``transfer_batch_to_device`` since frozen
        dataclasses are not supported by ``apply_to_collection``.
        """
        return GraphData(
            X=self.X.to(device),
            E=self.E.to(device),
            y=self.y.to(device),
            node_mask=self.node_mask.to(device),
        )

    # ---- Conversion classmethods / methods ----------------------------

    @classmethod
    def from_adjacency(cls, adj: Tensor) -> GraphData:
        """Convert binary adjacency matrices to one-hot categorical features.

        Produces ``dx=2`` (no-node / node) and ``de=2`` (no-edge / edge)
        encodings. All node positions are marked valid; padding is handled
        separately by `collate`.

        Parameters
        ----------
        adj
            Binary adjacency, shape ``(n, n)`` for a single graph or
            ``(bs, n, n)`` for a batch. Values should be 0 or 1.

        Returns
        -------
        GraphData
            One-hot encoded graph. Single graphs have no batch dimension;
            batched inputs produce batched outputs. Creates an all-ones
            ``node_mask``, treating every position as real. For variable-size
            batches with zero-padded graphs, callers must construct the mask
            separately (see ``collate``).
        """
        single = adj.dim() == 2
        if single:
            adj = adj.unsqueeze(0)

        bs, n, _ = adj.shape
        adj = adj.float()

        # Nodes: all are real (padding handled by collate)
        x_out = torch.zeros(bs, n, 2, device=adj.device, dtype=adj.dtype)
        x_out[:, :, 1] = 1.0  # category 1 = real node

        # Edges: one-hot encode the adjacency
        e_out = torch.zeros(bs, n, n, 2, device=adj.device, dtype=adj.dtype)
        e_out[:, :, :, 0] = 1.0 - adj  # category 0 = no edge
        e_out[:, :, :, 1] = adj  # category 1 = edge

        # Zero out diagonal (no self-loops)
        diag_idx = torch.arange(n, device=adj.device)
        e_out[:, diag_idx, diag_idx, :] = 0
        e_out[:, diag_idx, diag_idx, 0] = 1.0  # diagonal = "no edge"

        y_out = torch.zeros(bs, 0, device=adj.device, dtype=adj.dtype)
        node_mask = torch.ones(bs, n, device=adj.device, dtype=torch.bool)

        if single:
            return cls(
                X=x_out.squeeze(0),
                E=e_out.squeeze(0),
                y=y_out.squeeze(0),
                node_mask=node_mask.squeeze(0),
            )
        return cls(X=x_out, E=e_out, y=y_out, node_mask=node_mask)

    @classmethod
    def from_pyg_batch(cls, batch: Batch) -> GraphData:
        """Convert a PyG Batch to dense GraphData.

        Parameters
        ----------
        batch
            PyG Batch from ``Batch.from_data_list()``.

        Returns
        -------
        GraphData
            Dense batched one-hot representation with ``dx=2`` (no-node /
            node) and ``de=2`` (no-edge / edge) encodings. ``node_mask``
            reflects actual node counts per graph; padded positions are
            marked ``False``.
        """
        from torch_geometric.utils import to_dense_adj

        bs = int(batch.num_graphs)
        edge_index: Tensor = getattr(batch, "edge_index")  # noqa: B009
        batch_vec: Tensor = getattr(batch, "batch")  # noqa: B009
        adj = to_dense_adj(edge_index, batch_vec)
        n_max = adj.shape[1]

        # Node mask from batch vector
        node_counts = torch.bincount(batch_vec, minlength=bs)
        arange = torch.arange(n_max, device=adj.device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < node_counts.unsqueeze(1)

        # Zero diagonal, symmetrise
        diag = torch.arange(n_max, device=adj.device)
        adj[:, diag, diag] = 0.0
        adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)

        # One-hot edge features (bs, n, n, 2): [no-edge, edge]
        E = torch.stack([1.0 - adj, adj], dim=-1)

        # One-hot node features (bs, n, 2): [no-node, node]
        node_ind = node_mask.float()
        X = torch.stack([1.0 - node_ind, node_ind], dim=-1)

        y = torch.zeros(bs, 0, device=adj.device)

        return cls(X=X, E=E, y=y, node_mask=node_mask)

    @staticmethod
    def edge_features_to_adjacency(
        E: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Extract binary adjacency from categorical edge features.

        Inverse of the encoding in ``from_adjacency``: channel 0 is
        "no-edge", so ``argmax > 0`` recovers the adjacency. Single-channel
        ``E`` is treated as a raw adjacency (legacy format).

        Parameters
        ----------
        E
            Edge features, shape ``(..., de)``.
        node_mask
            Boolean or float mask, shape ``(bs, n)`` or broadcastable.
            When provided, masked node positions are zeroed in the output.

        Returns
        -------
        Tensor
            Binary adjacency, shape ``(...)`` (last dim collapsed).
        """
        adj = (E.argmax(dim=-1) > 0).float() if E.shape[-1] > 1 else E[..., 0].clone()

        if node_mask is not None:
            mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
            adj = adj * mask_2d.float()

        return adj

    def to_adjacency(self) -> Tensor:
        """Convert categorical edge features to a binary adjacency matrix.

        Class 0 is "no edge"; any other class is treated as an edge.
        Masked node positions are zeroed.
        """
        return GraphData.edge_features_to_adjacency(self.E, self.node_mask)

    def to_pyg(self) -> Data:
        """Convert this (unbatched) GraphData to a PyG Data object.

        Returns
        -------
        torch_geometric.data.Data
            Sparse COO representation with ``edge_index`` and ``num_nodes``.

        Raises
        ------
        ValueError
            If the GraphData has batch size > 1.
        """
        from torch_geometric.data import Data
        from torch_geometric.utils import dense_to_sparse

        adj = self.to_adjacency()
        if adj.ndim == 3:
            if adj.shape[0] != 1:
                raise ValueError(
                    f"to_pyg() requires unbatched or batch-size-1 GraphData, "
                    f"got batch size {adj.shape[0]}"
                )
            adj = adj.squeeze(0)

        n = (
            int(self.node_mask.sum().item())
            if self.node_mask.ndim == 1
            else adj.shape[0]
        )
        adj_trimmed = adj[:n, :n]
        edge_index, _ = dense_to_sparse(adj_trimmed)
        return Data(edge_index=edge_index, num_nodes=n)

    @staticmethod
    def collate(graphs: list[GraphData]) -> GraphData:
        """Collate variable-size graphs into a padded batch.

        Pads all graphs to the maximum node count. Padded node positions
        receive the "no-node" class (one-hot index 0), padded edge positions
        receive "no-edge" (one-hot index 0), and ``node_mask`` is ``False``
        for padded slots.

        Parameters
        ----------
        graphs
            Individual (unbatched) ``GraphData`` instances.

        Returns
        -------
        GraphData
            Batched instance with shapes ``(bs, n_max, ...)``.
        """
        bs = len(graphs)
        ns = [g.node_mask.shape[0] for g in graphs]
        n_max = max(ns)
        dx = graphs[0].X.shape[-1]
        de = graphs[0].E.shape[-1]
        dy = graphs[0].y.shape[0]

        x_batch = torch.zeros(bs, n_max, dx)
        e_batch = torch.zeros(bs, n_max, n_max, de)
        y_batch = torch.zeros(bs, dy)
        mask_batch = torch.zeros(bs, n_max, dtype=torch.bool)

        for i, (g, ni) in enumerate(zip(graphs, ns, strict=False)):
            x_batch[i, :ni] = g.X
            e_batch[i, :ni, :ni] = g.E
            if dy > 0:
                y_batch[i] = g.y
            mask_batch[i, :ni] = True

            # Padded edge positions -> "no edge" (class 0)
            e_batch[i, :ni, ni:, 0] = 1.0
            e_batch[i, ni:, :, 0] = 1.0
            # Padded node positions -> "no node" (class 0)
            x_batch[i, ni:, 0] = 1.0

        return GraphData(X=x_batch, E=e_batch, y=y_batch, node_mask=mask_batch)


def collapse_to_indices(data: GraphData) -> GraphData:
    """Argmax X and E to class indices, with -1 for masked positions.

    Converts one-hot encoded features to integer class labels. Positions
    where ``node_mask`` is zero are set to -1 as a sentinel.

    Parameters
    ----------
    data
        One-hot encoded graph features with ``node_mask``.

    Returns
    -------
    GraphData
        X shape ``(bs, n)``, E shape ``(bs, n, n)`` — integer indices.
    """
    node_mask = data.node_mask
    x_mask = node_mask.unsqueeze(-1)  # (bs, n, 1)
    e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)  # (bs, n, n, 1)

    X = torch.argmax(data.X, dim=-1)
    E = torch.argmax(data.E, dim=-1)

    X[node_mask == 0] = -1
    E[e_mask.squeeze(-1) == 0] = -1

    return GraphData(X=X, E=E, y=data.y, node_mask=node_mask)


@dataclass(frozen=True, slots=True)
class GraphStructure:
    """Pre-computed structural features of a graph's topology.

    Derived from the categorical edge features of a ``GraphData`` instance
    before any learned transformations. These tensors remain constant
    across transformer layer iterations — they describe the input graph's
    structure, not the evolving hidden state.

    Parameters
    ----------
    adjacency
        Binary adjacency matrix, shape ``(bs, n, n)``. Populated when
        GNN projections need it; ``None`` otherwise.
    eigenvectors
        Top-k eigenvectors of the adjacency, shape ``(bs, n, k)``.
        Populated for spectral projections; ``None`` otherwise.
    eigenvalues
        Corresponding eigenvalues, shape ``(bs, k)``.
    """

    adjacency: Tensor | None = None
    eigenvectors: Tensor | None = None
    eigenvalues: Tensor | None = None
