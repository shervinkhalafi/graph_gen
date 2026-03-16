"""Tests for the typed graph data containers in graph_types.py.

Testing Strategy
----------------
These tests verify the five frozen dataclasses that replace PlaceHolder
and GraphFeatures, focusing on immutability, masking semantics,
symmetry assertions, and the NoisyData conversion helper.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData, collapse_to_indices
from tmgg.diffusion.diffusion_graph_types import (
    LimitDistribution,
    TransitionMatrices,
)


def _make_symmetric_E(bs: int, n: int, de: int) -> torch.Tensor:
    """Build a symmetric edge tensor for tests."""
    E_upper = torch.randn(bs, n, n, de)
    return (E_upper + E_upper.transpose(1, 2)) / 2


# -- GraphData.mask() -------------------------------------------------------


class TestGraphDataMask:
    """Verify that mask() zeros masked positions and asserts E symmetry.

    Ported from the original PlaceHolder.mask() tests. Key difference:
    frozen dataclasses return new instances rather than mutating in place.
    """

    def test_symmetric_edges_pass(self) -> None:
        """Symmetric E after masking raises no error."""
        bs, n, dx, de = 2, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = _make_symmetric_E(bs, n, de)
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        data = GraphData(X=X, E=E, y=y, node_mask=node_mask)
        result = data.mask()
        assert result is not data  # frozen — returns new instance

    def test_symmetric_edges_with_partial_mask(self) -> None:
        """Masking some nodes preserves symmetry when E starts symmetric."""
        bs, n, dx, de = 2, 5, 3, 2
        X = torch.randn(bs, n, dx)
        E = _make_symmetric_E(bs, n, de)
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)
        node_mask[:, -1] = 0

        data = GraphData(X=X, E=E, y=y, node_mask=node_mask)
        result = data.mask()
        assert result is not data

    def test_asymmetric_edges_raise(self) -> None:
        """Asymmetric E triggers AssertionError after masking."""
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        data = GraphData(X=X, E=E, y=y, node_mask=node_mask)
        with pytest.raises(AssertionError, match="symmetric"):
            data.mask()

    def test_masked_positions_are_zero(self) -> None:
        """Node features and edges involving masked nodes are zeroed."""
        bs, n, dx, de = 1, 3, 2, 2
        X = torch.ones(bs, n, dx)
        E = _make_symmetric_E(bs, n, de).abs() + 0.1
        # Make E symmetric and positive
        E = (E + E.transpose(1, 2)) / 2
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)
        node_mask[0, 2] = 0  # mask out node 2

        result = GraphData(X=X, E=E, y=y, node_mask=node_mask).mask()
        assert result.X[0, 2].abs().sum() == 0
        assert result.E[0, 2, :, :].abs().sum() == 0
        assert result.E[0, :, 2, :].abs().sum() == 0


# -- GraphData.mask_zero_diag() ---------------------------------------------


class TestGraphDataMaskZeroDiag:
    """Verify mask_zero_diag zeros both masked positions and the diagonal."""

    def test_diagonal_zeroed(self) -> None:
        """Edge diagonal E[b, i, i, :] is zero after mask_zero_diag."""
        bs, n, dx, de = 2, 5, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        result = GraphData(X=X, E=E, y=y, node_mask=node_mask).mask_zero_diag()
        for b in range(bs):
            for i in range(n):
                assert result.E[b, i, i].abs().sum() == 0

    def test_masked_positions_zeroed(self) -> None:
        """Masked node features are zeroed."""
        bs, n, dx, de = 1, 4, 2, 2
        X = torch.ones(bs, n, dx)
        E = torch.ones(bs, n, n, de)
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n, dtype=torch.bool)
        node_mask[0, 3] = False

        result = GraphData(X=X, E=E, y=y, node_mask=node_mask).mask_zero_diag()
        assert result.X[0, 3].abs().sum() == 0


# -- GraphData.type_as() ----------------------------------------------------


class TestGraphDataTypeAs:
    """Verify type_as returns a new instance with correct dtype."""

    def test_dtype_changes(self) -> None:
        data = GraphData(
            X=torch.randn(1, 3, 2),
            E=torch.randn(1, 3, 3, 2),
            y=torch.zeros(1, 0),
            node_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        target = torch.zeros(1, dtype=torch.float64)
        result = data.type_as(target)
        assert result is not data
        assert result.X.dtype == torch.float64
        assert result.E.dtype == torch.float64
        assert result.y.dtype == torch.float64


# -- GraphData immutability -------------------------------------------------


class TestGraphDataFrozen:
    """Frozen dataclass prevents field assignment."""

    def test_cannot_assign_X(self) -> None:
        data = GraphData(
            X=torch.randn(1, 3, 2),
            E=torch.randn(1, 3, 3, 2),
            y=torch.zeros(1, 0),
            node_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        with pytest.raises(AttributeError):
            data.X = torch.zeros(1, 3, 2)  # type: ignore[misc]


# -- collapse_to_indices() --------------------------------------------------


class TestCollapseToIndices:
    """Verify argmax collapse with -1 sentinel for masked positions."""

    def test_basic_collapse(self) -> None:
        """One-hot X and E are collapsed to integer class indices."""
        bs, n, dx, de = 1, 3, 4, 2
        X = torch.zeros(bs, n, dx)
        X[0, 0, 2] = 1.0  # node 0 -> class 2
        X[0, 1, 0] = 1.0  # node 1 -> class 0
        X[0, 2, 3] = 1.0  # node 2 -> class 3

        E = torch.zeros(bs, n, n, de)
        E[0, 0, 1, 1] = 1.0  # edge (0,1) -> class 1
        E[0, 1, 0, 1] = 1.0  # symmetric

        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        result = collapse_to_indices(GraphData(X=X, E=E, y=y, node_mask=node_mask))
        assert result.X[0, 0] == 2
        assert result.X[0, 1] == 0
        assert result.X[0, 2] == 3
        assert result.E[0, 0, 1] == 1

    def test_masked_positions_are_negative_one(self) -> None:
        """Masked nodes and edges get sentinel value -1."""
        bs, n, dx, de = 1, 3, 2, 2
        X = torch.zeros(bs, n, dx)
        X[..., 0] = 1.0
        E = torch.zeros(bs, n, n, de)
        E[..., 0] = 1.0
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)
        node_mask[0, 2] = 0

        result = collapse_to_indices(GraphData(X=X, E=E, y=y, node_mask=node_mask))
        assert result.X[0, 2] == -1
        assert result.E[0, 2, 0] == -1
        assert result.E[0, 0, 2] == -1


# -- TransitionMatrices ------------------------------------------------------


class TestTransitionMatrices:
    """Basic construction and immutability."""

    def test_construction(self) -> None:
        tm = TransitionMatrices(
            X=torch.eye(3).unsqueeze(0),
            E=torch.eye(2).unsqueeze(0),
            y=torch.zeros(1, 0, 0),
        )
        assert tm.X.shape == (1, 3, 3)
        assert tm.E.shape == (1, 2, 2)

    def test_frozen(self) -> None:
        tm = TransitionMatrices(
            X=torch.eye(3).unsqueeze(0),
            E=torch.eye(2).unsqueeze(0),
            y=torch.zeros(1, 0, 0),
        )
        with pytest.raises(AttributeError):
            tm.X = torch.zeros(1, 3, 3)  # type: ignore[misc]


# -- LimitDistribution ------------------------------------------------------


class TestLimitDistribution:
    """Basic construction."""

    def test_construction(self) -> None:
        ld = LimitDistribution(
            X=torch.ones(3) / 3,
            E=torch.ones(2) / 2,
            y=torch.zeros(0),
        )
        assert ld.X.shape == (3,)
        assert torch.allclose(ld.X.sum(), torch.tensor(1.0))
