"""Roundtrip and structural property tests for the 4-type GraphData grid.

Covers carrier conversions (sparse↔dense), content conversions
(state↔distribution), and the cell-to-cell composition matrix.
Verifies invariants and that lossless paths commute bit-for-bit.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import (
    GraphState,
)


def _no_edge_fill(d_ec: int) -> torch.Tensor:
    fill = torch.zeros(d_ec, dtype=torch.float32)
    fill[0] = 1.0
    return fill


@pytest.fixture
def small_state() -> GraphState:
    """Two graphs: 2 nodes / 1 edge, 3 nodes / 2 edges. Edge_class is one-hot
    over 2 channels (no-edge / edge)."""
    num_nodes = torch.tensor([2, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 3, 4],
            [1, 0, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )
    edge_class = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],  # graph 0 edge 0-1 (both directions)
            [0.0, 1.0],
            [0.0, 1.0],  # graph 1 edge 2-3
            [0.0, 1.0],
            [0.0, 1.0],  # graph 1 edge 3-4
        ]
    )
    return GraphState(
        num_nodes_per_graph=num_nodes,
        y=torch.zeros(2, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=edge_class,
        edge_feat=None,
    )
