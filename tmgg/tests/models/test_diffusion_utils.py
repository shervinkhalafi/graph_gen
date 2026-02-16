"""Tests for diffusion_utils.py, focusing on PlaceHolder invariants.

Testing Strategy
----------------
PlaceHolder.mask() must enforce edge-feature symmetry for undirected
graphs. The original DiGress baseline asserts this; TMGG previously
discarded the check result (P2.7 bug). These tests confirm the restored
assertion catches asymmetric inputs and passes symmetric ones.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.models.digress.diffusion_utils import PlaceHolder


class TestPlaceHolderMask:
    """Verify the symmetry assertion in PlaceHolder.mask().

    Starting state: PlaceHolder with batch of (X, E, y) tensors and a
    node_mask. The ``mask()`` method zeros out masked positions and then
    asserts E remains symmetric (as expected for undirected graphs).
    """

    def test_symmetric_edges_pass(self) -> None:
        """Symmetric E after masking raises no error.

        Starting state: E is explicitly symmetric, node_mask keeps all nodes.
        Invariant: mask() returns without assertion failure.
        """
        bs, n, dx, de = 2, 4, 3, 2
        X = torch.randn(bs, n, dx)
        # Build symmetric E
        E_upper = torch.randn(bs, n, n, de)
        E = (E_upper + E_upper.transpose(1, 2)) / 2
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        ph = PlaceHolder(X=X, E=E, y=y)
        result = ph.mask(node_mask)
        # Should succeed and return the PlaceHolder
        assert result is ph

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
        node_mask[:, -1] = 0  # mask out last node

        ph = PlaceHolder(X=X, E=E, y=y)
        result = ph.mask(node_mask)
        assert result is ph

    def test_asymmetric_edges_raise(self) -> None:
        """Asymmetric E triggers AssertionError after masking.

        Starting state: E is deliberately asymmetric.
        Invariant: the restored assertion catches the asymmetry and reports
        the magnitude of the discrepancy.
        """
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)  # random → almost certainly asymmetric
        # Make it definitely asymmetric
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        ph = PlaceHolder(X=X, E=E, y=y)
        with pytest.raises(AssertionError, match="symmetric"):
            ph.mask(node_mask)

    def test_collapse_mode_skips_symmetry_check(self) -> None:
        """In collapse mode (argmax), the symmetry assertion is not run.

        Starting state: asymmetric E, collapse=True.
        Invariant: collapse converts E to class indices via argmax, which
        takes the ``if collapse`` branch and skips the symmetry check.
        """
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0
        y = torch.zeros(bs, 0)
        node_mask = torch.ones(bs, n)

        ph = PlaceHolder(X=X, E=E, y=y)
        # collapse=True should not raise, even with asymmetric E
        result = ph.mask(node_mask, collapse=True)
        assert result is ph
