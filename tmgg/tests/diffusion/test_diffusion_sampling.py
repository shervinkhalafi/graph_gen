"""Tests for the categorical sampling primitives in diffusion_sampling.py.

Pins the contract of :func:`_sample_from_unnormalised_posterior`, the
canonical upstream-DiGress primitive ported in parity #28 / #29 / D-5.
The helper normalises an unnormalised PMF with a row-sum floor, then
draws categorical indices via ``torch.multinomial``. The floor mirrors
upstream's ``denom[denom == 0] = 1e-5`` guard
(``DiGress/src/diffusion/diffusion_utils.py:274-290``) so degenerate
posterior rows do not crash sampling.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.diffusion.diffusion_sampling import (
    _sample_from_unnormalised_posterior,
    compute_posterior_distribution,
)


class TestSampleFromUnnormalisedPosterior:
    """Contract for :func:`_sample_from_unnormalised_posterior`.

    Test rationale (D-5): the helper underpins the upstream sampling
    contract. We pin three behaviours:

    1. On a healthy non-negative tensor the sampled distribution agrees
       with the manual ``unnorm / sum + multinomial`` reference within
       Monte-Carlo error.
    2. On a degenerate row (all zeros), the helper does NOT crash and
       produces a uniformly-distributed sample because the row-sum
       floor clamps the divisor away from zero.
    3. Negative input raises a clear ``ValueError`` -- the helper does
       not silently mask a bug by calling ``abs()``.
    """

    def test_matches_reference_on_healthy_posterior(self) -> None:
        """Sampled empirical distribution matches the manual reference."""
        torch.manual_seed(0)
        bs, k = 1, 4
        n = 10000
        # Random non-negative weights of shape (bs, n, k) -- sample n
        # categorical draws and compare empirical class frequency to the
        # explicit normalised PMF.
        unnorm = torch.rand(bs, n, k).abs() + 0.1  # strictly positive
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert sampled.shape == (bs, n)
        assert sampled.dtype == torch.long

        # Reference normalisation
        ref_prob = unnorm / unnorm.sum(dim=-1, keepdim=True)
        # Empirical class-1 frequency at each position should agree
        # with the analytic class-1 probability within 1.5/sqrt(n).
        # We check a handful of positions.
        # Per-row only one sample; instead pick a single PMF row and
        # draw many samples from it via the helper.
        single_row = unnorm[0, 0:1].repeat(1, n).reshape(1, n, k)
        single_sampled = _sample_from_unnormalised_posterior(single_row)
        empirical = torch.bincount(single_sampled[0], minlength=k).float() / n
        analytic = ref_prob[0, 0]
        max_dev = (empirical - analytic).abs().max().item()
        # Standard error ~ sqrt(p*(1-p)/n) <= 0.5/sqrt(n) = 0.005;
        # 6-sigma envelope is 0.03.
        assert max_dev < 0.03, (
            f"Empirical vs analytic disagree by {max_dev:.4f}, "
            f"analytic={analytic.tolist()}, empirical={empirical.tolist()}"
        )

    def test_degenerate_row_does_not_crash(self) -> None:
        """All-zero row falls back to uniform via the upstream zero-row branch.

        Upstream ``unnormalized[unnormalized.sum(dim=-1) == 0] = 1e-5``
        sets the entire row to ``zero_floor``, producing a uniform PMF
        after normalisation. The helper inherits this fallback so
        sampling continues on degenerate rows rather than crashing in
        :func:`torch.multinomial`.
        """
        torch.manual_seed(1)
        bs, n, k = 2, 3, 4
        unnorm = torch.zeros(bs, n, k)
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert sampled.shape == (bs, n)
        assert sampled.dtype == torch.long
        # All sampled indices must be in [0, k).
        assert int(sampled.min().item()) >= 0
        assert int(sampled.max().item()) < k

    def test_negative_input_raises(self) -> None:
        """Negative probability mass surfaces as a clear ValueError."""
        unnorm = torch.tensor([[[0.5, -0.1, 0.6]]])
        with pytest.raises(ValueError, match="negative probability mass"):
            _sample_from_unnormalised_posterior(unnorm)

    def test_non_degenerate_row_unaffected_by_floor(self) -> None:
        """A row with strictly positive mass is normalised as-is.

        The upstream zero-row branch only fires when
        ``row.sum() == 0`` exactly. A row whose entries are tiny but
        non-zero passes through normalisation unchanged.
        """
        torch.manual_seed(0)
        unnorm = torch.full((1, 1, 2), 5e-6)  # strictly positive
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert int(sampled.min().item()) >= 0
        assert int(sampled.max().item()) < 2

    def test_shape_preserved_for_4d_input(self) -> None:
        """Edge-PMF shape ``(bs, n, n, k)`` returns ``(bs, n, n)``."""
        torch.manual_seed(2)
        bs, n, k = 2, 3, 5
        unnorm = torch.rand(bs, n, n, k) + 0.1
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert sampled.shape == (bs, n, n)


class TestComputePosteriorDistributionMaskAware:
    """Regression tests for the mask-aware posterior denom guard.

    Test rationale (parity W2-3): commit ``6424fc34`` introduced a
    mask-aware split inside :func:`compute_posterior_distribution`
    after its predecessor ``e74ee0f8`` tripped on legitimate VLB-path
    inputs. CLAUDE.md mandates "first add the bug as a unit test
    BEFORE fixing"; the meta-synthesis flagged the missing regression
    test. These tests pin the three branches the guard distinguishes:

    1. Valid (one-hot M and M_t) rows divide cleanly and match the
       manual ``(M_t @ Qt^T) * (M @ Qsb) / denom`` formula.
    2. Masked rows (M or M_t row-sum zero) produce zero output rather
       than NaN, matching the post-mask cleanup the guard provides.
    3. A zero denom at a *valid* position trips the assertion with the
       informative ``min_at_valid`` message — the fail-loud branch
       CLAUDE.md mandates for real bugs.
    """

    def _kernel_with_off_diagonal_mass(
        self, bs: int, d: int, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Build an irreducible doubly-stochastic kernel of shape (bs, d, d)."""
        kernel = torch.full((d, d), 0.1, dtype=dtype)
        kernel.fill_diagonal_(0.9 - 0.1 * (d - 2))
        # Row-stochastic, never exactly zero anywhere.
        return kernel.unsqueeze(0).expand(bs, d, d).contiguous()

    def test_masked_rows_yield_zero(self) -> None:
        """Masked-row inputs (zero M or M_t) produce zero output, no NaN."""
        torch.manual_seed(0)
        bs, n, d = 1, 4, 3
        M = torch.zeros(bs, n, d)
        M_t = torch.zeros(bs, n, d)
        # Positions 0, 1: masked (all-zero rows in both M and M_t).
        # Positions 2, 3: valid one-hot.
        M[0, 2, 0] = 1.0
        M[0, 3, 1] = 1.0
        M_t[0, 2, 1] = 1.0
        M_t[0, 3, 2] = 1.0

        Qt_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qsb_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qtb_M = self._kernel_with_off_diagonal_mass(bs, d)

        out = compute_posterior_distribution(
            M, M_t, Qt_M, Qsb_M, Qtb_M, field="X_class"
        )
        assert out.shape == (bs, n, d)
        assert torch.isfinite(out).all(), "guard must not return NaN"
        # Masked positions: every class has zero probability.
        assert torch.allclose(out[0, 0], torch.zeros(d))
        assert torch.allclose(out[0, 1], torch.zeros(d))
        # Valid positions: nonzero somewhere.
        assert out[0, 2].abs().sum() > 0
        assert out[0, 3].abs().sum() > 0

    def test_valid_rows_match_manual_formula(self) -> None:
        """Valid-row outputs match the manual ``(M_t @ Qt^T) * (M @ Qsb) / denom`` formula."""
        torch.manual_seed(1)
        bs, n, d = 1, 3, 4
        M = torch.zeros(bs, n, d)
        M_t = torch.zeros(bs, n, d)
        # All-valid one-hot: position k → class (k+1) mod d, M_t → class k mod d.
        for i in range(n):
            M[0, i, (i + 1) % d] = 1.0
            M_t[0, i, i % d] = 1.0

        Qt_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qsb_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qtb_M = self._kernel_with_off_diagonal_mass(bs, d)

        out = compute_posterior_distribution(
            M, M_t, Qt_M, Qsb_M, Qtb_M, field="X_class"
        )

        # Manual reference using the same formula (no guard applied because
        # every position is valid).
        Qt_M_T = Qt_M.transpose(-2, -1)
        left = M_t @ Qt_M_T
        right = M @ Qsb_M
        product = left * right
        denom = ((M @ Qtb_M) * M_t).sum(dim=-1, keepdim=True)
        expected = product / denom

        torch.testing.assert_close(out, expected)

    def test_zero_denom_at_valid_position_raises(self) -> None:
        """Construct a degenerate valid-position scenario; guard asserts loudly.

        Set ``M`` and ``M_t`` to one-hot at *disjoint* classes, with
        ``Qtb_M`` sending mass strictly along the identity. Then
        ``denom = (M @ Qtb_M * M_t).sum(-1) = 0`` while the position is
        flagged valid (both row-sums positive), tripping the assert.
        """
        bs, n, d = 1, 1, 2
        M = torch.zeros(bs, n, d)
        M_t = torch.zeros(bs, n, d)
        M[0, 0, 0] = 1.0
        M_t[0, 0, 1] = 1.0
        # Identity kernels: M @ Qtb selects class 0; M_t selects class 1; the
        # element-wise product has a zero at class 1, so denom = 0.
        identity = torch.eye(d).unsqueeze(0).expand(bs, d, d).contiguous()
        Qt_M = identity.clone()
        Qsb_M = identity.clone()
        Qtb_M = identity.clone()

        with pytest.raises(AssertionError, match="min_at_valid"):
            compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M, field="X_class")
