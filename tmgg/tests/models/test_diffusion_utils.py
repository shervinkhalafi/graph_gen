"""Tests for diffusion_utils.py, focusing on GraphData masking invariants.

Testing Strategy
----------------
GraphData.mask() must enforce edge-feature symmetry for undirected
graphs. The original DiGress baseline asserts this; TMGG previously
discarded the check result (P2.7 bug). These tests confirm the restored
assertion catches asymmetric inputs and passes symmetric ones.

See also tests/models/test_graph_types.py for comprehensive GraphData tests.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import GraphData, collapse_to_indices


class TestGraphDataMask:
    """Verify the symmetry assertion in GraphData.mask().

    Starting state: GraphData with batch of (X, E, y) tensors and a
    node_mask. The ``mask()`` method zeros out masked positions, asserts
    E symmetry, and returns a new frozen instance.
    """

    def test_symmetric_edges_pass(self) -> None:
        """Symmetric E after masking raises no error.

        Starting state: E is explicitly symmetric, node_mask keeps all nodes.
        Invariant: mask() returns without assertion failure; result is a
        new instance (frozen dataclass).
        """
        bs, n, dx, de = 2, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E_upper = torch.randn(bs, n, n, de)
        E = (E_upper + E_upper.transpose(1, 2)) / 2
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        data = GraphData(y=y, node_mask=node_mask, X_class=X, E_class=E)
        result = data.mask()
        assert result is not data

    def test_symmetric_edges_with_partial_mask(self) -> None:
        """Masking some nodes preserves symmetry when E starts symmetric.

        Starting state: symmetric E, node_mask zeros out last node per batch.
        Invariant: element-wise masking (E * e_mask1 * e_mask2) preserves
        symmetry because the mask itself is symmetric in (i, j).
        """
        bs, n, dx, de = 2, 5, 3, 2
        X = torch.randn(bs, n, dx)
        E_upper = torch.randn(bs, n, n, de)
        E = (E_upper + E_upper.transpose(1, 2)) / 2
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)
        node_mask[:, -1] = 0

        data = GraphData(y=y, node_mask=node_mask, X_class=X, E_class=E)
        result = data.mask()
        assert result is not data

    def test_asymmetric_edges_raise(self) -> None:
        """Asymmetric E triggers AssertionError after masking.

        Starting state: E is deliberately asymmetric.
        Invariant: the assertion catches the asymmetry and reports
        the magnitude of the discrepancy.
        """
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        data = GraphData(y=y, node_mask=node_mask, X_class=X, E_class=E)
        with pytest.raises(AssertionError, match="symmetric"):
            data.mask()

    def test_collapse_to_indices_skips_symmetry_check(self) -> None:
        """collapse_to_indices converts to class indices without symmetry assertion.

        Starting state: asymmetric E (would fail mask()).
        Invariant: collapse_to_indices produces integer indices via argmax
        without checking E symmetry.
        """
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        data = GraphData(y=y, node_mask=node_mask, X_class=X, E_class=E)
        E_idx, X_idx = collapse_to_indices(data)
        assert X_idx is not None
        assert X_idx.shape == (bs, n)
        assert E_idx.shape == (bs, n, n)
