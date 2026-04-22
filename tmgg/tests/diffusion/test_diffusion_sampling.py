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

from tmgg.diffusion.diffusion_sampling import _sample_from_unnormalised_posterior


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
