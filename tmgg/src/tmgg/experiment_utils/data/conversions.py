"""Conversion utilities between adjacency and categorical graph representations.

These pure functions convert between the binary adjacency format used by
spectral denoisers and the categorical (X, E, y, node_mask) format used
by the discrete diffusion pipeline. For synthetic graphs, the encoding
uses dx=2 (no-node / node) and de=2 (no-edge / edge).
"""

from __future__ import annotations

import torch
from torch import Tensor


def adjacency_to_categorical(
    adj: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert binary adjacency matrices to one-hot categorical features.

    Parameters
    ----------
    adj
        Binary adjacency matrices of shape ``(bs, n, n)`` or ``(n, n)``
        for a single graph. Values should be 0 or 1.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(X, E, y, node_mask)`` where:

        - ``X``: Node features ``(bs, n, 2)`` with one-hot ``[no-node, node]``.
          All real nodes are encoded as ``[0, 1]``.
        - ``E``: Edge features ``(bs, n, n, 2)`` with one-hot ``[no-edge, edge]``.
          Diagonal entries are set to ``[1, 0]`` (no self-loops).
        - ``y``: Global features ``(bs, 0)`` (empty).
        - ``node_mask``: Boolean mask ``(bs, n)`` with all entries True.
    """
    single = adj.dim() == 2
    if single:
        adj = adj.unsqueeze(0)

    bs, n, _ = adj.shape
    adj = adj.float()

    # Nodes: all are real (padding is handled by collate_categorical)
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
        return (
            x_out.squeeze(0),
            e_out.squeeze(0),
            y_out.squeeze(0),
            node_mask.squeeze(0),
        )
    return x_out, e_out, y_out, node_mask


def categorical_to_adjacency(
    edge_features: Tensor,
    node_mask: Tensor | None = None,
) -> Tensor:
    """Convert categorical edge features to binary adjacency matrices.

    Takes argmax over the edge type dimension. Class 0 is interpreted as
    "no edge"; any other class is treated as an edge.

    Parameters
    ----------
    edge_features
        Edge features of shape ``(bs, n, n, de)`` or ``(n, n, de)``.
    node_mask
        Optional boolean mask ``(bs, n)`` or ``(n,)``. When provided,
        adjacency entries for masked nodes are zeroed.

    Returns
    -------
    Tensor
        Binary adjacency matrices of shape ``(bs, n, n)`` or ``(n, n)``.
    """
    edge_types = edge_features.argmax(dim=-1)  # (..., n, n)
    adj = (edge_types > 0).float()

    if node_mask is not None:
        mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        adj = adj * mask_2d.float()

    return adj


def collate_categorical(
    batch: list[tuple[Tensor, Tensor, Tensor, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate variable-size categorical graphs into a padded batch.

    Pads all graphs to the maximum node count in the batch. Padded node
    positions receive the "no-node" class (index 0) and padded edge
    positions receive the "no-edge" class (index 0).

    Parameters
    ----------
    batch
        List of ``(X, E, y, n)`` tuples, where ``X`` has shape ``(n_i, dx)``,
        ``E`` has shape ``(n_i, n_i, de)``, ``y`` has shape ``(dy,)``,
        and ``n`` is the number of real nodes.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(X, E, y, node_mask)`` where:

        - ``X``: ``(bs, n_max, dx)``
        - ``E``: ``(bs, n_max, n_max, de)``
        - ``y``: ``(bs, dy)``
        - ``node_mask``: ``(bs, n_max)`` boolean
    """
    n_max = max(n for _, _, _, n in batch)
    bs = len(batch)
    dx = batch[0][0].shape[-1]
    de = batch[0][1].shape[-1]
    dy = batch[0][2].shape[0]

    x_batch = torch.zeros(bs, n_max, dx)
    e_batch = torch.zeros(bs, n_max, n_max, de)
    y_batch = torch.zeros(bs, dy)
    node_mask = torch.zeros(bs, n_max, dtype=torch.bool)

    for i, (xi, ei, yi, ni) in enumerate(batch):
        x_batch[i, :ni] = xi
        e_batch[i, :ni, :ni] = ei
        if dy > 0:
            y_batch[i] = yi
        node_mask[i, :ni] = True

        # Set padded edge positions to "no edge" (class 0)
        e_batch[i, :ni, ni:, 0] = 1.0
        e_batch[i, ni:, :, 0] = 1.0
        # Set padded node positions to "no node" (class 0)
        x_batch[i, ni:, 0] = 1.0

    return x_batch, e_batch, y_batch, node_mask
