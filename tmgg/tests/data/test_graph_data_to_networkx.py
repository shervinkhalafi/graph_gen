"""Tests for ``GraphData.to_networkx`` and ``to_networkx_list``.

Pinned behaviours:

* Padding rows/columns (``node_mask == 0``) are dropped from the
  resulting graph.
* ``E_class`` argmax > 0 defines edges (channel 0 = "no edge").
* ``X_class`` argmax indices ride along as ``x_class`` node attrs.
* ``E_class`` argmax indices ride along as ``e_class`` edge attrs,
  but only for edges that survived the binary threshold.
* Batched ``GraphData`` requires an explicit ``batch_index``.
* Unbatched ``GraphData`` rejects a ``batch_index``.
* ``to_networkx_list`` expands the whole batch.

These were carved out of the GraphData-as-universal-transport
refactor (spec 2026-05-01); the categorical-index attributes are the
hook molecular post-hoc analysis uses to reconstruct atom/bond types
without re-loading the codec.
"""

from __future__ import annotations

import torch

from tmgg.data.datasets.graph_types import GraphData


def _two_node_one_edge_unbatched() -> GraphData:
    """Hand-crafted 2-node graph with one edge, both nodes valid."""
    return GraphData(
        node_mask=torch.tensor([1, 1], dtype=torch.float32),
        # X_class: node 0 -> class 1, node 1 -> class 2.
        X_class=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        # E_class: 2-channel "no-edge / edge".
        # 0-1 edge present (channel 1), 0-0 / 1-1 / 1-0 self-or-symmetric.
        E_class=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        y=torch.zeros(0),
    )


def _padded_batched() -> GraphData:
    """Batched GraphData with one valid graph (n=2) padded to n=3."""
    return GraphData(
        node_mask=torch.tensor([[1, 1, 0]], dtype=torch.float32),
        X_class=torch.tensor(
            [
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],  # padded position
                ]
            ],
            dtype=torch.float32,
        ),
        E_class=torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        y=torch.zeros(1, 0),
    )


def test_unbatched_to_networkx_topology_and_attrs() -> None:
    gd = _two_node_one_edge_unbatched()
    g = gd.to_networkx()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    # Node attrs: x_class indices land on each node.
    assert g.nodes[0]["x_class"] == 1
    assert g.nodes[1]["x_class"] == 2
    # Edge attrs: only on surviving edges; e_class is the argmax of
    # the 2-channel E_class slot, which is 1 for the present edge.
    assert g.edges[0, 1]["e_class"] == 1


def test_batched_to_networkx_drops_padding_rows() -> None:
    gd = _padded_batched()
    g = gd.to_networkx(batch_index=0)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert g.nodes[0]["x_class"] == 1
    assert g.nodes[1]["x_class"] == 0


def test_batched_requires_batch_index() -> None:
    gd = _padded_batched()
    raised = False
    try:
        gd.to_networkx()
    except ValueError as e:
        raised = True
        assert "batch_index" in str(e)
    assert raised, "batched GraphData with no batch_index should raise"


def test_unbatched_rejects_batch_index() -> None:
    gd = _two_node_one_edge_unbatched()
    raised = False
    try:
        gd.to_networkx(batch_index=0)
    except ValueError as e:
        raised = True
        assert "not allowed" in str(e)
    assert raised, "unbatched GraphData with a batch_index should raise"


def test_to_networkx_list_expands_full_batch() -> None:
    gd = _padded_batched()
    graphs = gd.to_networkx_list()
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 2


def test_to_networkx_list_unbatched_returns_singleton() -> None:
    gd = _two_node_one_edge_unbatched()
    graphs = gd.to_networkx_list()
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 2


def test_no_x_class_means_no_node_attrs() -> None:
    """When X_class is None the per-node x_class attr must be absent."""
    gd = GraphData(
        node_mask=torch.tensor([1, 1], dtype=torch.float32),
        X_feat=torch.zeros(2, 1),  # populate the X side via X_feat instead
        E_class=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        y=torch.zeros(0),
    )
    g = gd.to_networkx()
    assert "x_class" not in g.nodes[0]
    assert "x_class" not in g.nodes[1]
    # E_class still drives edge attrs.
    assert g.edges[0, 1]["e_class"] == 1
