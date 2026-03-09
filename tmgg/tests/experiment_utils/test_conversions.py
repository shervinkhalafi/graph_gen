"""Tests for GraphData conversion classmethods and collation.

Test rationale: Conversion between adjacency and categorical representations
is the bridge between the spectral denoiser pipeline and the discrete
diffusion pipeline. Off-by-one errors in one-hot encoding or incorrect
masking in the collation function produce data that is structurally valid
but semantically wrong, leading to silent training failures. These tests
verify round-trip identity, correct one-hot encoding, padding behavior,
and edge symmetry preservation.

These tests cover ``GraphData.from_adjacency()``, ``GraphData.to_adjacency()``,
and ``GraphData.collate()`` — the methods that replaced the standalone
``conversions.py`` functions.
"""

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData

BS = 3
N = 6


@pytest.fixture()
def binary_adjacency() -> torch.Tensor:
    """A batch of simple binary symmetric adjacency matrices."""
    A = torch.zeros(BS, N, N)
    # Create some edges (symmetric)
    for b in range(BS):
        for i in range(N - 1):
            A[b, i, i + 1] = 1.0
            A[b, i + 1, i] = 1.0
    return A


class TestFromAdjacency:
    """Verify GraphData.from_adjacency() conversion."""

    def test_output_shapes(self, binary_adjacency: torch.Tensor) -> None:
        """All fields have the expected shapes."""
        g = GraphData.from_adjacency(binary_adjacency)
        assert g.X.shape == (BS, N, 2)
        assert g.E.shape == (BS, N, N, 2)
        assert g.y.shape == (BS, 0)
        assert g.node_mask.shape == (BS, N)

    def test_nodes_are_one_hot(self, binary_adjacency: torch.Tensor) -> None:
        """All node features should be [0, 1] (real node)."""
        g = GraphData.from_adjacency(binary_adjacency)
        assert (g.X[:, :, 0] == 0).all()
        assert (g.X[:, :, 1] == 1).all()

    def test_edges_are_one_hot(self, binary_adjacency: torch.Tensor) -> None:
        """Edge features should sum to 1 along the last dim (one-hot)."""
        g = GraphData.from_adjacency(binary_adjacency)
        sums = g.E.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_diagonal_is_no_edge(self, binary_adjacency: torch.Tensor) -> None:
        """Diagonal entries should be [1, 0] (no self-loops)."""
        g = GraphData.from_adjacency(binary_adjacency)
        diag_idx = torch.arange(N)
        assert (g.E[:, diag_idx, diag_idx, 0] == 1).all()
        assert (g.E[:, diag_idx, diag_idx, 1] == 0).all()

    def test_all_nodes_valid(self, binary_adjacency: torch.Tensor) -> None:
        """node_mask should be all True (no padding at this stage)."""
        g = GraphData.from_adjacency(binary_adjacency)
        assert g.node_mask.all()

    def test_single_graph(self) -> None:
        """Unbatched (n, n) input should return unbatched outputs."""
        A = torch.eye(4)
        A[0, 1] = A[1, 0] = 1.0
        g = GraphData.from_adjacency(A)
        assert g.X.dim() == 2
        assert g.X.shape == (4, 2)
        assert g.E.dim() == 3
        assert g.E.shape == (4, 4, 2)
        assert g.node_mask.dim() == 1


class TestToAdjacency:
    """Verify GraphData.to_adjacency() conversion."""

    def test_round_trip(self, binary_adjacency: torch.Tensor) -> None:
        """adjacency -> from_adjacency -> to_adjacency should recover the original."""
        g = GraphData.from_adjacency(binary_adjacency)
        A_recovered = g.to_adjacency()
        # The diagonal is always zeroed (no self-loops)
        A_no_diag = binary_adjacency.clone()
        diag_idx = torch.arange(N)
        A_no_diag[:, diag_idx, diag_idx] = 0
        assert torch.allclose(A_recovered, A_no_diag)

    def test_single_graph_round_trip(self) -> None:
        """Round trip works for a single unbatched graph."""
        A = torch.zeros(5, 5)
        A[0, 1] = A[1, 0] = 1.0
        A[2, 3] = A[3, 2] = 1.0
        g = GraphData.from_adjacency(A)
        A_recovered = g.to_adjacency()
        assert torch.allclose(A_recovered, A)

    def test_masking_zeros_invalid_edges(self) -> None:
        """Masked nodes should have zero adjacency entries."""
        E = torch.zeros(2, 4, 4, 2)
        E[:, :, :, 1] = 1.0  # all edges "present"
        node_mask = torch.ones(2, 4, dtype=torch.bool)
        node_mask[:, 3] = False  # mask last node
        g = GraphData(
            X=torch.zeros(2, 4, 2),
            E=E,
            y=torch.zeros(2, 0),
            node_mask=node_mask,
        )
        A = g.to_adjacency()
        # Last row/col should be zero
        assert (A[:, 3, :] == 0).all()
        assert (A[:, :, 3] == 0).all()

    def test_symmetry_preservation(self, binary_adjacency: torch.Tensor) -> None:
        """Recovered adjacency should be symmetric if input edges were."""
        g = GraphData.from_adjacency(binary_adjacency)
        A = g.to_adjacency()
        assert torch.allclose(A, A.transpose(1, 2))


class TestCollate:
    """Verify GraphData.collate() for variable-size graphs."""

    def _make_graph(self, n: int, dy: int = 0) -> GraphData:
        """Create a simple GraphData with n nodes."""
        return GraphData(
            X=torch.ones(n, 2),
            E=torch.ones(n, n, 2),
            y=torch.zeros(dy),
            node_mask=torch.ones(n, dtype=torch.bool),
        )

    def test_basic_padding(self) -> None:
        """Graphs of different sizes should be padded to the max."""
        g1 = self._make_graph(3)
        g2 = self._make_graph(5)
        batch = GraphData.collate([g1, g2])
        assert batch.X.shape == (2, 5, 2)
        assert batch.E.shape == (2, 5, 5, 2)
        assert batch.node_mask.shape == (2, 5)

    def test_node_mask_correctness(self) -> None:
        """Mask should be True for real nodes, False for padding."""
        g1 = self._make_graph(3)
        g2 = self._make_graph(5)
        batch = GraphData.collate([g1, g2])
        assert batch.node_mask[0, :3].all()
        assert not batch.node_mask[0, 3:].any()
        assert batch.node_mask[1, :5].all()

    def test_padded_nodes_have_no_node_class(self) -> None:
        """Padded node positions should be set to class 0 (no-node)."""
        g1 = self._make_graph(2)
        g2 = self._make_graph(4)
        batch = GraphData.collate([g1, g2])
        # For graph 0, positions 2 and 3 should have class 0 = 1
        assert batch.X[0, 2, 0] == 1.0
        assert batch.X[0, 3, 0] == 1.0

    def test_padded_edges_have_no_edge_class(self) -> None:
        """Padded edge positions should be set to class 0 (no-edge)."""
        g1 = self._make_graph(2)
        g2 = self._make_graph(4)
        batch = GraphData.collate([g1, g2])
        # For graph 0, row 2 (padded) should have E[0, 2, :, 0] = 1
        assert (batch.E[0, 2, :, 0] == 1.0).all()
        assert (batch.E[0, :, 2, 0] == 1.0).all()

    def test_real_data_preserved(self) -> None:
        """Real graph data should be preserved after padding."""
        X1 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        E1 = torch.zeros(2, 2, 2)
        E1[0, 1, 1] = 1.0
        E1[1, 0, 1] = 1.0
        E1[0, 0, 0] = 1.0
        E1[1, 1, 0] = 1.0
        g1 = GraphData(
            X=X1,
            E=E1,
            y=torch.zeros(0),
            node_mask=torch.ones(2, dtype=torch.bool),
        )
        g2 = self._make_graph(3)

        batch = GraphData.collate([g1, g2])
        # Graph 0's real data should be intact
        assert torch.allclose(batch.X[0, :2], X1)
        assert torch.allclose(batch.E[0, :2, :2], E1)

    def test_global_features_preserved(self) -> None:
        """Global features y should be copied correctly."""
        y1 = torch.tensor([1.0, 2.0])
        y2 = torch.tensor([3.0, 4.0])
        g1 = GraphData(
            X=torch.ones(2, 2),
            E=torch.ones(2, 2, 2),
            y=y1,
            node_mask=torch.ones(2, dtype=torch.bool),
        )
        g2 = GraphData(
            X=torch.ones(3, 2),
            E=torch.ones(3, 3, 2),
            y=y2,
            node_mask=torch.ones(3, dtype=torch.bool),
        )
        batch = GraphData.collate([g1, g2])
        assert torch.allclose(batch.y[0], y1)
        assert torch.allclose(batch.y[1], y2)
