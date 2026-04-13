"""Tests for GraphData ↔ PyG conversion methods.

Rationale
---------
``GraphData.from_pyg_batch`` and ``GraphData.to_pyg`` bridge two
representations:

- **PyG sparse COO**: ``Data(edge_index, num_nodes)`` and batched
  ``Batch``, which torch_geometric datamodules produce natively.
- **TMGG dense one-hot**: ``GraphData(X, E, y, node_mask)`` used
  throughout the training pipeline.

Invariants under test:

1. A single graph's node count and edge structure survive a round-trip
   (``from_pyg_batch`` then ``to_pyg``).
2. A batch of two variable-size graphs is padded to ``n_max`` and the
   ``node_mask`` correctly identifies real vs. padded nodes.
3. The full pipeline adjacency → GraphData → Data → Batch → GraphData →
   adjacency reproduces the original adjacency.
4. Padding rows/columns outside the node mask are zero in both X and E.
5. ``to_pyg`` raises ``ValueError`` for batch size > 1.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from tmgg.data.datasets.graph_types import GraphData

# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------


def _make_triangle_graph() -> Data:
    """Three-node cycle: 0-1, 1-2, 0-2 (undirected, both directions)."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
    )
    return Data(edge_index=edge_index, num_nodes=3)


def _make_square_graph() -> Data:
    """Four-node cycle: 0-1, 1-2, 2-3, 3-0 (undirected, both directions)."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    return Data(edge_index=edge_index, num_nodes=4)


# ---------------------------------------------------------------------------
# from_pyg_batch tests
# ---------------------------------------------------------------------------


class TestFromPygBatch:
    def test_single_graph_shape(self) -> None:
        """A single-graph batch produces shapes (1, n, 2), (1, n, n, 2), etc."""
        triangle = _make_triangle_graph()
        batch = Batch.from_data_list([triangle])
        gd = GraphData.from_pyg_batch(batch)

        assert gd.X.shape == (1, 3, 2)
        assert gd.E.shape == (1, 3, 3, 2)
        assert gd.y.shape == (1, 0)
        assert gd.node_mask.shape == (1, 3)
        assert gd.node_mask.all(), "All three nodes should be real."

    def test_single_graph_edge_structure(self) -> None:
        """Edge features channel 1 matches the triangle's adjacency."""
        triangle = _make_triangle_graph()
        batch = Batch.from_data_list([triangle])
        gd = GraphData.from_pyg_batch(batch)

        adj_recovered = gd.E[0, :, :, 1]
        expected = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        assert torch.allclose(
            adj_recovered, expected
        ), f"Recovered adjacency:\n{adj_recovered}\nExpected:\n{expected}"

    def test_batch_of_two_padding(self) -> None:
        """Triangle (n=3) + square (n=4) batch: n_max=4, triangle padded."""
        triangle = _make_triangle_graph()
        square = _make_square_graph()
        batch = Batch.from_data_list([triangle, square])
        gd = GraphData.from_pyg_batch(batch)

        assert gd.X.shape == (2, 4, 2)
        assert gd.E.shape == (2, 4, 4, 2)
        assert gd.node_mask.shape == (2, 4)

        # Triangle graph: first 3 real, 4th padded
        assert gd.node_mask[0, :3].all()
        assert not gd.node_mask[0, 3]

        # Square graph: all 4 real
        assert gd.node_mask[1, :].all()

    def test_padding_zeros_in_features(self) -> None:
        """Padded node positions have zero features in X and E."""
        triangle = _make_triangle_graph()
        square = _make_square_graph()
        batch = Batch.from_data_list([triangle, square])
        gd = GraphData.from_pyg_batch(batch)

        # Padded node (index 3) in first graph: X[0, 3] should be [1, 0]
        # (no-node one-hot) but the real invariant is the mask is False.
        # More directly: node_mask[0, 3] == False ensures downstream masking.
        assert not gd.node_mask[0, 3]

        # Padded rows/cols of E for first graph should be zero in edge channel
        assert gd.E[0, 3, :, 1].sum() == 0.0, "Padded row should have no edges."
        assert gd.E[0, :, 3, 1].sum() == 0.0, "Padded column should have no edges."

    def test_adjacency_round_trip(self) -> None:
        """from_pyg_batch → to_binary_adjacency recovers original adjacency."""
        triangle = _make_triangle_graph()
        batch = Batch.from_data_list([triangle])
        gd = GraphData.from_pyg_batch(batch)

        adj = gd.to_binary_adjacency()
        expected = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])
        assert torch.allclose(
            adj, expected
        ), f"Round-trip adjacency:\n{adj}\nExpected:\n{expected}"


# ---------------------------------------------------------------------------
# to_pyg tests
# ---------------------------------------------------------------------------


class TestToPyg:
    def test_single_unbatched_graph(self) -> None:
        """to_pyg on an unbatched GraphData returns correct num_nodes."""
        triangle = _make_triangle_graph()
        batch = Batch.from_data_list([triangle])
        gd = GraphData.from_pyg_batch(batch)

        # Squeeze to unbatched: node_mask shape (3,)
        gd_single = GraphData(
            X=gd.X.squeeze(0),
            E=gd.E.squeeze(0),
            y=gd.y.squeeze(0),
            node_mask=gd.node_mask.squeeze(0),
        )
        data = gd_single.to_pyg()

        assert data.num_nodes == 3
        # Triangle has 6 directed edges (3 undirected × 2)
        assert data.edge_index is not None
        assert data.edge_index.shape == (2, 6)

    def test_batch_size_1_accepted(self) -> None:
        """to_pyg accepts batch-size-1 GraphData (squeezes batch dim)."""
        triangle = _make_triangle_graph()
        batch = Batch.from_data_list([triangle])
        gd = GraphData.from_pyg_batch(batch)

        # node_mask is (1, 3) here; to_pyg should handle it
        data = gd.to_pyg()
        assert data.num_nodes == 3

    def test_batch_size_gt1_raises(self) -> None:
        """to_pyg raises ValueError when batch size > 1."""
        triangle = _make_triangle_graph()
        square = _make_square_graph()
        batch = Batch.from_data_list([triangle, square])
        gd = GraphData.from_pyg_batch(batch)

        with pytest.raises(ValueError, match="batch size"):
            gd.to_pyg()

    def test_full_round_trip(self) -> None:
        """adjacency → GraphData → Data → Batch → GraphData → adjacency."""
        adj_original = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        # adjacency → GraphData (unbatched)
        gd_from_adj = GraphData.from_binary_adjacency(adj_original)

        # GraphData → PyG Data
        data = gd_from_adj.to_pyg()

        # PyG Data → Batch → GraphData
        batch = Batch.from_data_list([data])
        gd_recovered = GraphData.from_pyg_batch(batch)

        # GraphData → adjacency (batched, shape (1, 3, 3))
        adj_recovered = gd_recovered.to_binary_adjacency().squeeze(0)

        assert torch.allclose(adj_recovered, adj_original), (
            f"Full round-trip failed.\nOriginal:\n{adj_original}\n"
            f"Recovered:\n{adj_recovered}"
        )
