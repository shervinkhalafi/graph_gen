"""Test-local GraphData constructors replacing the removed legacy helpers.

The Wave 9.2 removal deletes ``GraphData.from_edge_state`` and
``GraphData.from_binary_adjacency`` from production code (see
``docs/specs/2026-04-15-unified-graph-features-spec.md §"Removed fields"``).
Many test fixtures depended on those shapes — a dense adjacency producing a
two-channel ``E_class`` with a degenerate "node-or-not" ``X_class`` — and
replicating that inline at every call site would bloat diffs. This module
provides shared builders that tests import directly.

None of these helpers are production surface: they exist only so test
authors keep a concise way to construct the exact tensor layouts the legacy
helpers produced, without resurrecting the helpers themselves.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData


def legacy_edge_scalar(data: GraphData) -> Tensor:
    """Return a dense scalar adjacency regardless of which edge field is populated.

    Replaces the removed ``GraphData.to_edge_state``. Delegates to
    :meth:`GraphData.to_edge_scalar` using whichever split edge field is
    populated, preferring ``E_feat`` when both are present (matching the
    legacy precedence).
    """
    if data.E_feat is not None:
        return data.to_edge_scalar(source="feat")
    return data.to_edge_scalar(source="class")


def binary_graphdata(adj: Tensor) -> GraphData:
    """Build a GraphData from a binary adjacency matching the legacy helper layout.

    Replaces the removed ``GraphData.from_binary_adjacency``. Produces the
    same two-channel ``E_class`` (``[no-edge, edge]``) with zero diagonal and
    a degenerate ``X_class`` marking every position as a "real" node. All
    node positions are valid (``node_mask`` is all-True); padding is
    handled separately by ``GraphData.collate``.

    Parameters
    ----------
    adj
        Binary adjacency, shape ``(n, n)`` for a single graph or
        ``(bs, n, n)`` for a batch. Values should be 0 or 1.

    Returns
    -------
    GraphData
        One-hot encoded graph with ``dx_class=2`` and ``de_class=2``.
    """
    single = adj.dim() == 2
    if single:
        adj = adj.unsqueeze(0)

    bs, n, _ = adj.shape
    adj = adj.float()

    x_class = torch.zeros(bs, n, 2, device=adj.device, dtype=adj.dtype)
    x_class[:, :, 1] = 1.0

    e_class = torch.zeros(bs, n, n, 2, device=adj.device, dtype=adj.dtype)
    e_class[:, :, :, 0] = 1.0 - adj
    e_class[:, :, :, 1] = adj

    diag_idx = torch.arange(n, device=adj.device)
    e_class[:, diag_idx, diag_idx, :] = 0
    e_class[:, diag_idx, diag_idx, 0] = 1.0

    y_out = torch.zeros(bs, 0, device=adj.device, dtype=adj.dtype)
    node_mask = torch.ones(bs, n, device=adj.device, dtype=torch.bool)

    if single:
        return GraphData(
            y=y_out.squeeze(0),
            node_mask=node_mask.squeeze(0),
            X_class=x_class.squeeze(0),
            E_class=e_class.squeeze(0),
        )
    return GraphData(
        y=y_out,
        node_mask=node_mask,
        X_class=x_class,
        E_class=e_class,
    )


def edge_scalar_graphdata(
    edge_state: Tensor, *, node_mask: Tensor | None = None
) -> GraphData:
    """Build a structure-only GraphData from a dense scalar edge tensor.

    Replaces the removed ``GraphData.from_edge_state``. Wraps a dense scalar
    adjacency as ``E_feat`` with shape ``(..., 1)`` and no categorical
    fields. When ``node_mask`` is not provided, every position is marked
    valid.

    Parameters
    ----------
    edge_state
        Dense scalar edges, ``(n, n)`` or ``(bs, n, n)``; a trailing
        single-channel axis is accepted and squeezed.
    node_mask
        Optional node-validity mask matching the batch shape of
        ``edge_state``. When ``None``, an all-True mask is synthesised.

    Returns
    -------
    GraphData
        Instance with ``E_feat`` populated.
    """
    single = edge_state.dim() == 2
    if single:
        edge_state = edge_state.unsqueeze(0)

    if edge_state.dim() == 4 and edge_state.shape[-1] == 1:
        edge_state = edge_state[..., 0]

    if edge_state.dim() != 3:
        raise ValueError(
            "edge_scalar_graphdata() expects a 2D or 3D edge-state tensor, "
            f"got shape {tuple(edge_state.shape)}"
        )

    bs, n, _ = edge_state.shape

    if node_mask is None:
        node_mask = torch.ones(bs, n, device=edge_state.device, dtype=torch.bool)
    elif node_mask.dim() == 1:
        node_mask = node_mask.unsqueeze(0)

    if node_mask.shape != (bs, n):
        raise ValueError(
            "node_mask must have shape (bs, n) for edge_scalar_graphdata(), "
            f"got {tuple(node_mask.shape)}"
        )

    if single:
        return GraphData.from_structure_only(
            node_mask.squeeze(0), edge_state.squeeze(0)
        )
    return GraphData.from_structure_only(node_mask, edge_state)
