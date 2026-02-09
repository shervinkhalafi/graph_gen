"""Tests for categorical conversion utilities.

Test rationale: Conversion between adjacency and categorical representations
is the bridge between the spectral denoiser pipeline and the discrete
diffusion pipeline. Off-by-one errors in one-hot encoding or incorrect
masking in the collation function produce data that is structurally valid
but semantically wrong, leading to silent training failures. These tests
verify round-trip identity, correct one-hot encoding, padding behavior,
and edge symmetry preservation.
"""

import pytest
import torch

from tmgg.experiment_utils.data.conversions import (
    adjacency_to_categorical,
    categorical_to_adjacency,
    collate_categorical,
)

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


class TestAdjacencyToCategorical:
    """Verify adjacency -> categorical conversion."""

    def test_output_shapes(self, binary_adjacency: torch.Tensor) -> None:
        """All outputs have the expected shapes."""
        X, E, y, node_mask = adjacency_to_categorical(binary_adjacency)
        assert X.shape == (BS, N, 2)
        assert E.shape == (BS, N, N, 2)
        assert y.shape == (BS, 0)
        assert node_mask.shape == (BS, N)

    def test_nodes_are_one_hot(self, binary_adjacency: torch.Tensor) -> None:
        """All node features should be [0, 1] (real node)."""
        X, _, _, _ = adjacency_to_categorical(binary_adjacency)
        assert (X[:, :, 0] == 0).all()
        assert (X[:, :, 1] == 1).all()

    def test_edges_are_one_hot(self, binary_adjacency: torch.Tensor) -> None:
        """Edge features should sum to 1 along the last dim (one-hot)."""
        _, E, _, _ = adjacency_to_categorical(binary_adjacency)
        sums = E.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_diagonal_is_no_edge(self, binary_adjacency: torch.Tensor) -> None:
        """Diagonal entries should be [1, 0] (no self-loops)."""
        _, E, _, _ = adjacency_to_categorical(binary_adjacency)
        diag_idx = torch.arange(N)
        assert (E[:, diag_idx, diag_idx, 0] == 1).all()
        assert (E[:, diag_idx, diag_idx, 1] == 0).all()

    def test_all_nodes_valid(self, binary_adjacency: torch.Tensor) -> None:
        """node_mask should be all True (no padding at this stage)."""
        _, _, _, node_mask = adjacency_to_categorical(binary_adjacency)
        assert node_mask.all()

    def test_single_graph(self) -> None:
        """Unbatched (n, n) input should return unbatched outputs."""
        A = torch.eye(4)
        A[0, 1] = A[1, 0] = 1.0
        X, E, y, node_mask = adjacency_to_categorical(A)
        assert X.dim() == 2
        assert X.shape == (4, 2)
        assert E.dim() == 3
        assert E.shape == (4, 4, 2)
        assert node_mask.dim() == 1


class TestCategoricalToAdjacency:
    """Verify categorical -> adjacency conversion."""

    def test_round_trip(self, binary_adjacency: torch.Tensor) -> None:
        """adjacency -> categorical -> adjacency should recover the original."""
        X, E, y, node_mask = adjacency_to_categorical(binary_adjacency)
        A_recovered = categorical_to_adjacency(E, node_mask)
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
        X, E, y, node_mask = adjacency_to_categorical(A)
        A_recovered = categorical_to_adjacency(E, node_mask)
        assert torch.allclose(A_recovered, A)

    def test_masking_zeros_invalid_edges(self) -> None:
        """Masked nodes should have zero adjacency entries."""
        E = torch.zeros(2, 4, 4, 2)
        E[:, :, :, 1] = 1.0  # all edges "present"
        node_mask = torch.ones(2, 4, dtype=torch.bool)
        node_mask[:, 3] = False  # mask last node
        A = categorical_to_adjacency(E, node_mask)
        # Last row/col should be zero
        assert (A[:, 3, :] == 0).all()
        assert (A[:, :, 3] == 0).all()

    def test_symmetry_preservation(self, binary_adjacency: torch.Tensor) -> None:
        """Recovered adjacency should be symmetric if input edges were."""
        _, E, _, node_mask = adjacency_to_categorical(binary_adjacency)
        A = categorical_to_adjacency(E, node_mask)
        assert torch.allclose(A, A.transpose(1, 2))


class TestCollateCategorical:
    """Verify variable-size graph collation."""

    def test_basic_padding(self) -> None:
        """Graphs of different sizes should be padded to the max."""
        g1 = (torch.ones(3, 2), torch.ones(3, 3, 2), torch.zeros(0), 3)
        g2 = (torch.ones(5, 2), torch.ones(5, 5, 2), torch.zeros(0), 5)
        X, E, y, mask = collate_categorical([g1, g2])
        assert X.shape == (2, 5, 2)
        assert E.shape == (2, 5, 5, 2)
        assert mask.shape == (2, 5)

    def test_node_mask_correctness(self) -> None:
        """Mask should be True for real nodes, False for padding."""
        g1 = (torch.ones(3, 2), torch.ones(3, 3, 2), torch.zeros(0), 3)
        g2 = (torch.ones(5, 2), torch.ones(5, 5, 2), torch.zeros(0), 5)
        _, _, _, mask = collate_categorical([g1, g2])
        assert mask[0, :3].all()
        assert not mask[0, 3:].any()
        assert mask[1, :5].all()

    def test_padded_nodes_have_no_node_class(self) -> None:
        """Padded node positions should be set to class 0 (no-node)."""
        g1 = (torch.ones(2, 2), torch.ones(2, 2, 2), torch.zeros(0), 2)
        g2 = (torch.ones(4, 2), torch.ones(4, 4, 2), torch.zeros(0), 4)
        X, _, _, _ = collate_categorical([g1, g2])
        # For graph 0, positions 2 and 3 should have class 0 = 1
        assert X[0, 2, 0] == 1.0
        assert X[0, 3, 0] == 1.0

    def test_padded_edges_have_no_edge_class(self) -> None:
        """Padded edge positions should be set to class 0 (no-edge)."""
        g1 = (torch.ones(2, 2), torch.ones(2, 2, 2), torch.zeros(0), 2)
        g2 = (torch.ones(4, 2), torch.ones(4, 4, 2), torch.zeros(0), 4)
        _, E, _, _ = collate_categorical([g1, g2])
        # For graph 0, row 2 (padded) should have E[0, 2, :, 0] = 1
        assert (E[0, 2, :, 0] == 1.0).all()
        assert (E[0, :, 2, 0] == 1.0).all()

    def test_real_data_preserved(self) -> None:
        """Real graph data should be preserved after padding."""
        X1 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        E1 = torch.zeros(2, 2, 2)
        E1[0, 1, 1] = 1.0
        E1[1, 0, 1] = 1.0
        E1[0, 0, 0] = 1.0
        E1[1, 1, 0] = 1.0
        g1 = (X1, E1, torch.zeros(0), 2)
        g2 = (torch.ones(3, 2), torch.ones(3, 3, 2), torch.zeros(0), 3)

        X, E, _, _ = collate_categorical([g1, g2])
        # Graph 0's real data should be intact
        assert torch.allclose(X[0, :2], X1)
        assert torch.allclose(E[0, :2, :2], E1)

    def test_global_features_preserved(self) -> None:
        """Global features y should be copied correctly."""
        y1 = torch.tensor([1.0, 2.0])
        y2 = torch.tensor([3.0, 4.0])
        g1 = (torch.ones(2, 2), torch.ones(2, 2, 2), y1, 2)
        g2 = (torch.ones(3, 2), torch.ones(3, 3, 2), y2, 3)
        _, _, y, _ = collate_categorical([g1, g2])
        assert torch.allclose(y[0], y1)
        assert torch.allclose(y[1], y2)
