"""Tests for AdjacencyBatch and CategoricalBatch data containers.

Test rationale
--------------
These batch types are the data contract between all pipeline stages: data
loading, noise application, model forward pass, and loss computation. Validation
errors caught here prevent cryptic shape mismatches downstream, so we test both
the happy path (valid construction, factory methods, device transfer) and the
error path (every dimension/shape invariant the containers enforce).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from tmgg.data.batches import AdjacencyBatch, CategoricalBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adj(batch: int = 4, n: int = 5) -> Tensor:
    """Create a simple batched adjacency tensor."""
    return torch.eye(n).unsqueeze(0).expand(batch, -1, -1).clone()


def _make_categorical(
    batch: int = 4,
    n: int = 5,
    dx: int = 2,
    de: int = 2,
    dy: int = 0,
) -> dict[str, Tensor]:
    """Create a valid set of tensors for CategoricalBatch construction."""
    X = torch.randn(batch, n, dx)
    E = torch.randn(batch, n, n, de)
    y = torch.randn(batch, dy)
    node_mask = torch.ones(batch, n, dtype=torch.bool)
    return {"X": X, "E": E, "y": y, "node_mask": node_mask}


# ===========================================================================
# AdjacencyBatch
# ===========================================================================


class TestAdjacencyBatchConstruction:
    """Verify that AdjacencyBatch accepts valid inputs and rejects invalid ones."""

    def test_valid_3d(self) -> None:
        """A 3D square tensor should construct without error."""
        A = _make_adj(batch=4, n=5)
        batch = AdjacencyBatch(A=A)
        assert batch.batch_size == 4
        assert batch.num_nodes == 5

    def test_valid_2d(self) -> None:
        """A 2D square tensor (single graph) should also be accepted."""
        A = torch.eye(5)
        batch = AdjacencyBatch(A=A)
        assert batch.batch_size == 1
        assert batch.num_nodes == 5

    def test_valid_with_timestep(self) -> None:
        """Construction with a matching 1D timestep tensor should work."""
        A = _make_adj(batch=4, n=5)
        t = torch.zeros(4)
        batch = AdjacencyBatch(A=A, t=t)
        assert batch.t is not None

    def test_non_square_2d_raises(self) -> None:
        """A non-square 2D tensor must be rejected."""
        A = torch.randn(3, 5)
        with pytest.raises(ValueError, match="square"):
            AdjacencyBatch(A=A)

    def test_non_square_3d_raises(self) -> None:
        """A 3D tensor that is not square in last two dims must be rejected."""
        A = torch.randn(4, 3, 5)
        with pytest.raises(ValueError, match="square"):
            AdjacencyBatch(A=A)

    def test_1d_raises(self) -> None:
        """A 1D tensor must be rejected (wrong ndim)."""
        A = torch.randn(5)
        with pytest.raises(ValueError, match="2D.*3D"):
            AdjacencyBatch(A=A)

    def test_4d_raises(self) -> None:
        """A 4D tensor must be rejected (wrong ndim)."""
        A = torch.randn(2, 3, 3, 1)
        with pytest.raises(ValueError, match="2D.*3D"):
            AdjacencyBatch(A=A)

    def test_timestep_wrong_ndim_raises(self) -> None:
        """A 2D timestep tensor must be rejected."""
        A = _make_adj()
        t = torch.zeros(4, 1)
        with pytest.raises(ValueError, match="1D"):
            AdjacencyBatch(A=A, t=t)


class TestAdjacencyBatchProperties:
    """Verify that properties return correct values."""

    def test_device(self) -> None:
        """The device property should reflect where the tensor lives."""
        A = _make_adj()
        batch = AdjacencyBatch(A=A)
        assert batch.device == torch.device("cpu")


class TestAdjacencyBatchMethods:
    """Verify to(), ensure_batched(), and from_tensor()."""

    def test_to_cpu(self) -> None:
        """Moving to the same device should produce equal tensors."""
        A = _make_adj()
        batch = AdjacencyBatch(A=A)
        moved = batch.to("cpu")
        assert torch.equal(moved.A, batch.A)

    def test_to_preserves_timestep(self) -> None:
        """to() should also move the timestep tensor."""
        A = _make_adj()
        t = torch.ones(4)
        batch = AdjacencyBatch(A=A, t=t)
        moved = batch.to("cpu")
        assert moved.t is not None
        assert torch.equal(moved.t, t)

    def test_to_none_timestep(self) -> None:
        """to() with t=None should keep t as None."""
        batch = AdjacencyBatch(A=_make_adj())
        moved = batch.to("cpu")
        assert moved.t is None

    def test_ensure_batched_from_2d(self) -> None:
        """A 2D input should become 3D with batch dim of 1."""
        A = torch.eye(5)
        batch = AdjacencyBatch(A=A)
        batched = batch.ensure_batched()
        assert batched.A.ndim == 3
        assert batched.A.shape == (1, 5, 5)
        assert batched.batch_size == 1

    def test_ensure_batched_noop_for_3d(self) -> None:
        """A 3D input should return self, unchanged."""
        A = _make_adj()
        batch = AdjacencyBatch(A=A)
        batched = batch.ensure_batched()
        assert batched is batch

    def test_from_tensor_basic(self) -> None:
        """from_tensor() should produce the same result as direct construction."""
        A = _make_adj()
        batch = AdjacencyBatch.from_tensor(A)
        assert batch.batch_size == 4
        assert batch.num_nodes == 5
        assert batch.t is None

    def test_from_tensor_with_timestep(self) -> None:
        """from_tensor() should forward the timestep argument."""
        A = _make_adj()
        t = torch.arange(4, dtype=torch.float)
        batch = AdjacencyBatch.from_tensor(A, t=t)
        assert batch.t is not None
        assert torch.equal(batch.t, t)


class TestAdjacencyBatchFrozen:
    """Verify immutability of the frozen dataclass."""

    def test_cannot_reassign_A(self) -> None:
        """Direct attribute assignment should raise."""
        batch = AdjacencyBatch(A=_make_adj())
        with pytest.raises(AttributeError):
            batch.A = torch.zeros(4, 5, 5)  # type: ignore[misc]


# ===========================================================================
# CategoricalBatch
# ===========================================================================


class TestCategoricalBatchConstruction:
    """Verify that CategoricalBatch accepts valid inputs and rejects invalid ones."""

    def test_valid_construction(self) -> None:
        """All-valid tensors should construct without error."""
        tensors = _make_categorical()
        batch = CategoricalBatch(**tensors)
        assert batch.batch_size == 4
        assert batch.num_nodes == 5

    def test_valid_with_timestep(self) -> None:
        """Construction with a matching timestep tensor should work."""
        tensors = _make_categorical()
        tensors["t"] = torch.zeros(4)
        batch = CategoricalBatch(**tensors)
        assert batch.t is not None

    def test_X_wrong_ndim_raises(self) -> None:
        """X must be 3D."""
        tensors = _make_categorical()
        tensors["X"] = torch.randn(4, 5)  # 2D instead of 3D
        with pytest.raises(ValueError, match="3D"):
            CategoricalBatch(**tensors)

    def test_E_wrong_ndim_raises(self) -> None:
        """E must be 4D."""
        tensors = _make_categorical()
        tensors["E"] = torch.randn(4, 5, 5)  # 3D instead of 4D
        with pytest.raises(ValueError, match="4D"):
            CategoricalBatch(**tensors)

    def test_E_non_square_raises(self) -> None:
        """E must be square in spatial dims."""
        tensors = _make_categorical()
        tensors["E"] = torch.randn(4, 5, 3, 2)  # 5 != 3
        with pytest.raises(ValueError, match="square"):
            CategoricalBatch(**tensors)

    def test_node_mask_wrong_ndim_raises(self) -> None:
        """node_mask must be 2D."""
        tensors = _make_categorical()
        tensors["node_mask"] = torch.ones(4, 5, 1)  # 3D
        with pytest.raises(ValueError, match="2D"):
            CategoricalBatch(**tensors)

    def test_batch_size_mismatch_raises(self) -> None:
        """Mismatched batch sizes between X and E must be rejected."""
        tensors = _make_categorical()
        tensors["E"] = torch.randn(3, 5, 5, 2)  # batch=3 vs batch=4
        with pytest.raises(ValueError, match="Batch size mismatch"):
            CategoricalBatch(**tensors)

    def test_node_count_mismatch_raises(self) -> None:
        """Mismatched node counts between X and node_mask must be rejected."""
        tensors = _make_categorical(n=5)
        tensors["node_mask"] = torch.ones(4, 7, dtype=torch.bool)  # n=7 vs n=5
        with pytest.raises(ValueError, match="Node count mismatch"):
            CategoricalBatch(**tensors)

    def test_y_wrong_ndim_raises(self) -> None:
        """y must be 2D."""
        tensors = _make_categorical()
        tensors["y"] = torch.randn(4)  # 1D
        with pytest.raises(ValueError, match="y must be 2D"):
            CategoricalBatch(**tensors)

    def test_y_batch_size_mismatch_raises(self) -> None:
        """y batch size must match X batch size."""
        tensors = _make_categorical()
        tensors["y"] = torch.randn(3, 0)  # batch=3 vs batch=4
        with pytest.raises(ValueError, match="y must be 2D"):
            CategoricalBatch(**tensors)

    def test_timestep_wrong_ndim_raises(self) -> None:
        """t must be 1D when present."""
        tensors = _make_categorical()
        tensors["t"] = torch.zeros(4, 1)  # 2D
        with pytest.raises(ValueError, match="1D"):
            CategoricalBatch(**tensors)

    def test_timestep_batch_mismatch_raises(self) -> None:
        """t batch size must match data batch size."""
        tensors = _make_categorical()
        tensors["t"] = torch.zeros(3)  # batch=3 vs batch=4
        with pytest.raises(ValueError, match="Timestep batch size"):
            CategoricalBatch(**tensors)


class TestCategoricalBatchProperties:
    """Verify that properties return correct values."""

    def test_dimensions(self) -> None:
        """dx, de, dy should reflect tensor shapes."""
        tensors = _make_categorical(batch=2, n=7, dx=3, de=4, dy=5)
        batch = CategoricalBatch(**tensors)
        assert batch.batch_size == 2
        assert batch.num_nodes == 7
        assert batch.dx == 3
        assert batch.de == 4
        assert batch.dy == 5

    def test_device(self) -> None:
        """The device property should reflect where the tensors live."""
        batch = CategoricalBatch(**_make_categorical())
        assert batch.device == torch.device("cpu")


class TestCategoricalBatchMethods:
    """Verify to(), mask(), to_adjacency(), and factory methods."""

    def test_to_cpu(self) -> None:
        """Moving to the same device should produce equal tensors."""
        batch = CategoricalBatch(**_make_categorical())
        moved = batch.to("cpu")
        assert torch.equal(moved.X, batch.X)
        assert torch.equal(moved.E, batch.E)

    def test_to_preserves_timestep(self) -> None:
        """to() should also move the timestep tensor."""
        tensors = _make_categorical()
        tensors["t"] = torch.ones(4)
        batch = CategoricalBatch(**tensors)
        moved = batch.to("cpu")
        assert moved.t is not None
        assert torch.equal(moved.t, tensors["t"])

    def test_to_none_timestep(self) -> None:
        """to() with t=None should keep t as None."""
        batch = CategoricalBatch(**_make_categorical())
        moved = batch.to("cpu")
        assert moved.t is None

    def test_mask_zeros_padded_nodes(self) -> None:
        """mask() should zero X features at positions where node_mask is False.

        We construct a batch where node 0 is masked (False) and verify that
        X[0, 0, :] becomes all-zero after masking.
        """
        tensors = _make_categorical(batch=1, n=3, dx=2, de=2)
        tensors["X"] = torch.ones(1, 3, 2)
        tensors["E"] = torch.ones(1, 3, 3, 2)
        tensors["node_mask"] = torch.tensor([[False, True, True]])
        batch = CategoricalBatch(**tensors)

        masked = batch.mask()
        # Node 0 is masked: X should be zero there
        assert torch.all(masked.X[0, 0] == 0)
        # Node 1 and 2 are real: X should be non-zero
        assert torch.all(masked.X[0, 1] == 1)
        assert torch.all(masked.X[0, 2] == 1)

    def test_mask_zeros_edges_at_masked_endpoints(self) -> None:
        """mask() should zero E where either endpoint is masked.

        If node 0 is masked, all edges involving node 0 should be zeroed.
        """
        tensors = _make_categorical(batch=1, n=3, dx=2, de=2)
        tensors["E"] = torch.ones(1, 3, 3, 2)
        tensors["node_mask"] = torch.tensor([[False, True, True]])
        batch = CategoricalBatch(**tensors)

        masked = batch.mask()
        # Edges (0, *) and (*, 0) should be zero
        assert torch.all(masked.E[0, 0, :, :] == 0)
        assert torch.all(masked.E[0, :, 0, :] == 0)
        # Edge (1,2) should be preserved
        assert torch.all(masked.E[0, 1, 2, :] == 1)

    def test_mask_preserves_y_and_node_mask(self) -> None:
        """mask() should not alter y or node_mask."""
        tensors = _make_categorical(batch=1, n=3)
        tensors["node_mask"] = torch.tensor([[False, True, True]])
        batch = CategoricalBatch(**tensors)
        masked = batch.mask()
        assert torch.equal(masked.y, batch.y)
        assert torch.equal(masked.node_mask, batch.node_mask)

    def test_to_adjacency_recovers_binary(self) -> None:
        """to_adjacency() should recover the binary adjacency from edge features.

        We construct E where class 1 encodes edges and class 0 encodes no-edge,
        and verify that to_adjacency() returns the original binary matrix.
        """
        bs, n = 2, 4
        A = torch.zeros(bs, n, n)
        A[0, 0, 1] = 1.0
        A[0, 1, 0] = 1.0
        A[1, 2, 3] = 1.0
        A[1, 3, 2] = 1.0

        cat = CategoricalBatch.from_adjacency(A)
        recovered = cat.to_adjacency()
        assert torch.equal(recovered, A)

    def test_from_adjacency_shapes(self) -> None:
        """from_adjacency() should produce dx=2, de=2 features with correct shapes."""
        bs, n = 3, 6
        A = torch.zeros(bs, n, n)
        cat = CategoricalBatch.from_adjacency(A)

        assert cat.X.shape == (bs, n, 2)
        assert cat.E.shape == (bs, n, n, 2)
        assert cat.y.shape == (bs, 0)
        assert cat.node_mask.shape == (bs, n)
        assert cat.dx == 2
        assert cat.de == 2

    def test_from_adjacency_with_node_mask(self) -> None:
        """from_adjacency() should encode node_mask into X features."""
        bs, n = 1, 4
        A = torch.zeros(bs, n, n)
        node_mask = torch.tensor([[True, True, False, False]])
        cat = CategoricalBatch.from_adjacency(A, node_mask=node_mask)

        # Real nodes: class 1 should be 1, class 0 should be 0
        assert cat.X[0, 0, 1] == 1.0
        assert cat.X[0, 0, 0] == 0.0
        # Padded nodes: class 0 should be 1, class 1 should be 0
        assert cat.X[0, 2, 0] == 1.0
        assert cat.X[0, 2, 1] == 0.0

    def test_from_adjacency_with_timestep(self) -> None:
        """from_adjacency() should pass through the timestep tensor."""
        A = torch.zeros(2, 5, 5)
        t = torch.tensor([0.1, 0.5])
        cat = CategoricalBatch.from_adjacency(A, t=t)
        assert cat.t is not None
        assert torch.equal(cat.t, t)

    def test_from_adjacency_requires_3d(self) -> None:
        """from_adjacency() should reject non-3D input."""
        A = torch.zeros(5, 5)  # 2D, missing batch dim
        with pytest.raises(ValueError, match="3D"):
            CategoricalBatch.from_adjacency(A)

    def test_from_adjacency_default_node_mask(self) -> None:
        """from_adjacency() without node_mask should assume all nodes are real."""
        A = torch.zeros(2, 5, 5)
        cat = CategoricalBatch.from_adjacency(A)
        assert torch.all(cat.node_mask)

    def test_from_placeholder(self) -> None:
        """from_placeholder() should behave identically to direct construction."""
        tensors = _make_categorical()
        batch_direct = CategoricalBatch(**tensors)
        batch_factory = CategoricalBatch.from_placeholder(**tensors)

        assert torch.equal(batch_direct.X, batch_factory.X)
        assert torch.equal(batch_direct.E, batch_factory.E)
        assert torch.equal(batch_direct.y, batch_factory.y)
        assert torch.equal(batch_direct.node_mask, batch_factory.node_mask)

    def test_from_placeholder_with_timestep(self) -> None:
        """from_placeholder() should forward the timestep."""
        tensors = _make_categorical()
        t = torch.zeros(4)
        batch = CategoricalBatch.from_placeholder(**tensors, t=t)
        assert batch.t is not None


class TestCategoricalBatchFrozen:
    """Verify immutability of the frozen dataclass."""

    def test_cannot_reassign_X(self) -> None:
        """Direct attribute assignment should raise."""
        batch = CategoricalBatch(**_make_categorical())
        with pytest.raises(AttributeError):
            batch.X = torch.zeros(4, 5, 2)  # type: ignore[misc]


class TestRoundTrip:
    """Verify that from_adjacency -> to_adjacency round-trips correctly.

    This is a key invariant: converting a binary adjacency matrix to
    categorical format and back should recover the original matrix exactly.
    """

    def test_identity_roundtrip(self) -> None:
        """Empty graph should round-trip to empty graph."""
        A = torch.zeros(2, 5, 5)
        recovered = CategoricalBatch.from_adjacency(A).to_adjacency()
        assert torch.equal(recovered, A)

    def test_complete_graph_roundtrip(self) -> None:
        """Complete graph (excluding diagonal) should round-trip correctly."""
        n = 4
        A = torch.ones(1, n, n) - torch.eye(n).unsqueeze(0)
        recovered = CategoricalBatch.from_adjacency(A).to_adjacency()
        assert torch.equal(recovered, A)

    def test_random_binary_roundtrip(self) -> None:
        """A random binary adjacency matrix should survive the round-trip."""
        A = (torch.rand(3, 6, 6) > 0.5).float()
        # Ensure symmetric
        A = (A + A.transpose(1, 2)).clamp(max=1.0)
        recovered = CategoricalBatch.from_adjacency(A).to_adjacency()
        assert torch.equal(recovered, A)
