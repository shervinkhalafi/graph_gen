"""Tests for discrete noise schedules and transition matrices.

Test rationale
--------------
The noise schedule and transition matrices underpin every forward diffusion step
and every reverse sampling step in DiGress-style discrete diffusion.  An error
in betas, alpha-bar, or the transition matrix construction would silently corrupt
training targets and generated samples.  These tests verify the core mathematical
invariants:

* **Monotonicity of betas**: the cosine schedule produces non-decreasing betas,
  so noise increases with timestep.
* **Alpha-bar decay**: alpha-bar starts near 1 (clean data) and decreases toward
  0 (pure noise), ensuring the forward process interpolates correctly.
* **Stochasticity**: all transition matrices are row-stochastic (rows sum to 1),
  a necessary condition for valid probability transitions.
* **Convergence**: at ``alpha_bar ≈ 0`` the cumulative transition matrix should
  collapse to the stationary distribution — uniform for
  ``DiscreteUniformTransition``, marginal for ``MarginalUniformTransition``.
* **Boundary sanity**: t = 0 and t = T produce finite values (no NaN/Inf).
* **Limit distributions**: ``get_limit_dist()`` returns valid probability
  distributions that sum to 1.

See also: ``src/tmgg/models/digress/diffusion_utils.py`` for schedule functions,
and ``src/tmgg/models/digress/noise_schedule.py`` for the classes under test.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.models.digress.noise_schedule import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    PredefinedNoiseScheduleDiscrete,
)

# ---------------------------------------------------------------------------
# PredefinedNoiseScheduleDiscrete
# ---------------------------------------------------------------------------


class TestPredefinedNoiseScheduleDiscrete:
    """Invariants on the precomputed beta / alpha-bar lookup tables."""

    @pytest.fixture()
    def schedule(self) -> PredefinedNoiseScheduleDiscrete:
        return PredefinedNoiseScheduleDiscrete("cosine", 500)

    def test_betas_monotonically_nondecreasing(
        self, schedule: PredefinedNoiseScheduleDiscrete
    ) -> None:
        """Cosine betas should be monotonically non-decreasing."""
        betas = schedule.betas
        diffs = betas[1:] - betas[:-1]
        assert (diffs >= -1e-7).all(), "Betas are not monotonically non-decreasing"

    def test_alphas_bar_starts_near_one_and_decays(
        self, schedule: PredefinedNoiseScheduleDiscrete
    ) -> None:
        """Alpha-bar should start close to 1 and decrease toward 0."""
        ab = schedule.alphas_bar
        assert ab[0] > 0.95, f"alphas_bar[0] = {ab[0]:.4f}, expected > 0.95"
        assert ab[-1] < 0.05, f"alphas_bar[-1] = {ab[-1]:.4f}, expected < 0.05"
        # Monotonically non-increasing
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-7).all(), "alphas_bar is not monotonically non-increasing"

    def test_forward_returns_correct_beta(
        self, schedule: PredefinedNoiseScheduleDiscrete
    ) -> None:
        """forward(t_int=k) should return betas[k]."""
        t = torch.tensor([0, 100, 499])
        result = schedule.forward(t_int=t)
        expected = schedule.betas[t.long()]
        assert torch.allclose(result, expected)

    def test_get_alpha_bar_returns_correct_values(
        self, schedule: PredefinedNoiseScheduleDiscrete
    ) -> None:
        """get_alpha_bar(t_int=k) should return alphas_bar[k]."""
        t = torch.tensor([0, 250, 499])
        result = schedule.get_alpha_bar(t_int=t)
        expected = schedule.alphas_bar[t.long()]
        assert torch.allclose(result, expected)

    def test_exactly_one_argument_required(
        self, schedule: PredefinedNoiseScheduleDiscrete
    ) -> None:
        """Passing both or neither argument should raise."""
        with pytest.raises(ValueError):
            schedule.forward()
        with pytest.raises(ValueError):
            schedule.forward(t_normalized=torch.tensor([0.5]), t_int=torch.tensor([10]))
        with pytest.raises(ValueError):
            schedule.get_alpha_bar()

    def test_boundary_timesteps_are_finite(
        self, schedule: PredefinedNoiseScheduleDiscrete
    ) -> None:
        """t=0 and t=T-1 should produce finite betas and alpha-bar."""
        for t_val in [0, schedule.timesteps - 1]:
            t = torch.tensor([t_val])
            beta = schedule.forward(t_int=t)
            ab = schedule.get_alpha_bar(t_int=t)
            assert torch.isfinite(beta).all(), f"Non-finite beta at t={t_val}"
            assert torch.isfinite(ab).all(), f"Non-finite alpha_bar at t={t_val}"

    def test_unknown_schedule_raises(self) -> None:
        """An unrecognised schedule name should raise immediately."""
        with pytest.raises(NotImplementedError):
            PredefinedNoiseScheduleDiscrete("nonexistent", 100)


# ---------------------------------------------------------------------------
# DiscreteUniformTransition
# ---------------------------------------------------------------------------


class TestDiscreteUniformTransition:
    """Invariants on uniform-stationary transition matrices."""

    @pytest.fixture()
    def transition(self) -> DiscreteUniformTransition:
        return DiscreteUniformTransition(x_classes=3, e_classes=4, y_classes=0)

    def test_get_Qt_rows_sum_to_one(
        self, transition: DiscreteUniformTransition
    ) -> None:
        """Each row of the one-step transition matrices must sum to 1."""
        beta_t = torch.tensor([0.1, 0.5, 0.9])
        ph = transition.get_Qt(beta_t, "cpu")
        for name, q in [("X", ph.X), ("E", ph.E)]:
            row_sums = q.sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-6
            ), f"{name} rows don't sum to 1: {row_sums}"

    def test_get_Qt_bar_rows_sum_to_one(
        self, transition: DiscreteUniformTransition
    ) -> None:
        """Rows of cumulative transition matrices must sum to 1."""
        alpha_bar_t = torch.tensor([0.99, 0.5, 0.01])
        ph = transition.get_Qt_bar(alpha_bar_t, "cpu")
        for name, q in [("X", ph.X), ("E", ph.E)]:
            row_sums = q.sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-6
            ), f"{name} rows don't sum to 1"

    def test_get_Qt_bar_converges_to_uniform(
        self, transition: DiscreteUniformTransition
    ) -> None:
        """At alpha_bar ≈ 0 the cumulative matrix should be nearly uniform."""
        alpha_bar_t = torch.tensor([1e-7])
        ph = transition.get_Qt_bar(alpha_bar_t, "cpu")

        expected_x = torch.ones(1, 3, 3) / 3
        expected_e = torch.ones(1, 4, 4) / 4
        assert torch.allclose(
            ph.X, expected_x, atol=1e-5
        ), f"X not uniform at alpha_bar≈0: {ph.X}"
        assert torch.allclose(
            ph.E, expected_e, atol=1e-5
        ), f"E not uniform at alpha_bar≈0: {ph.E}"

    def test_get_Qt_bar_identity_at_alpha_bar_one(
        self, transition: DiscreteUniformTransition
    ) -> None:
        """At alpha_bar = 1 the cumulative matrix should be the identity."""
        alpha_bar_t = torch.tensor([1.0])
        ph = transition.get_Qt_bar(alpha_bar_t, "cpu")
        assert torch.allclose(ph.X, torch.eye(3).unsqueeze(0), atol=1e-6)
        assert torch.allclose(ph.E, torch.eye(4).unsqueeze(0), atol=1e-6)

    def test_get_limit_dist_sums_to_one(
        self, transition: DiscreteUniformTransition
    ) -> None:
        """Limit distribution for each feature type must sum to 1."""
        limit = transition.get_limit_dist()
        assert torch.allclose(limit.X.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(limit.E.sum(), torch.tensor(1.0), atol=1e-6)

    def test_get_limit_dist_is_uniform(
        self, transition: DiscreteUniformTransition
    ) -> None:
        """Limit distribution should be uniform 1/K."""
        limit = transition.get_limit_dist()
        assert torch.allclose(limit.X, torch.ones(3) / 3, atol=1e-6)
        assert torch.allclose(limit.E, torch.ones(4) / 4, atol=1e-6)


# ---------------------------------------------------------------------------
# MarginalUniformTransition
# ---------------------------------------------------------------------------


class TestMarginalUniformTransition:
    """Invariants on marginal-stationary transition matrices."""

    @pytest.fixture()
    def marginals(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = torch.tensor([0.7, 0.2, 0.1])
        e_m = torch.tensor([0.5, 0.3, 0.15, 0.05])
        return x_m, e_m

    @pytest.fixture()
    def transition(
        self, marginals: tuple[torch.Tensor, torch.Tensor]
    ) -> MarginalUniformTransition:
        x_m, e_m = marginals
        return MarginalUniformTransition(x_m, e_m, y_classes=0)

    def test_get_Qt_rows_sum_to_one(
        self, transition: MarginalUniformTransition
    ) -> None:
        beta_t = torch.tensor([0.1, 0.5, 0.9])
        ph = transition.get_Qt(beta_t, "cpu")
        for name, q in [("X", ph.X), ("E", ph.E)]:
            row_sums = q.sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-6
            ), f"{name} rows don't sum to 1"

    def test_get_Qt_bar_converges_to_marginals(
        self,
        transition: MarginalUniformTransition,
        marginals: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """At alpha_bar ≈ 0, rows of Qt_bar should match the marginal distribution."""
        x_m, e_m = marginals
        alpha_bar_t = torch.tensor([1e-7])
        ph = transition.get_Qt_bar(alpha_bar_t, "cpu")

        # Each row should be approximately the marginal distribution
        for row_idx in range(3):
            assert torch.allclose(
                ph.X[0, row_idx], x_m, atol=1e-5
            ), f"X row {row_idx} doesn't match marginal: {ph.X[0, row_idx]}"
        for row_idx in range(4):
            assert torch.allclose(
                ph.E[0, row_idx], e_m, atol=1e-5
            ), f"E row {row_idx} doesn't match marginal: {ph.E[0, row_idx]}"

    def test_get_Qt_bar_identity_at_alpha_bar_one(
        self, transition: MarginalUniformTransition
    ) -> None:
        alpha_bar_t = torch.tensor([1.0])
        ph = transition.get_Qt_bar(alpha_bar_t, "cpu")
        assert torch.allclose(ph.X, torch.eye(3).unsqueeze(0), atol=1e-6)
        assert torch.allclose(ph.E, torch.eye(4).unsqueeze(0), atol=1e-6)

    def test_get_limit_dist_sums_to_one(
        self, transition: MarginalUniformTransition
    ) -> None:
        limit = transition.get_limit_dist()
        assert torch.allclose(limit.X.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(limit.E.sum(), torch.tensor(1.0), atol=1e-6)

    def test_get_limit_dist_matches_marginals(
        self,
        transition: MarginalUniformTransition,
        marginals: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Limit distribution should equal the input marginals."""
        x_m, e_m = marginals
        limit = transition.get_limit_dist()
        assert torch.allclose(limit.X, x_m, atol=1e-6)
        assert torch.allclose(limit.E, e_m, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases: boundary timesteps produce finite values
# ---------------------------------------------------------------------------


class TestBoundaryFiniteness:
    """Transition matrices at extreme parameter values must remain finite."""

    def test_discrete_uniform_beta_zero(self) -> None:
        """beta_t = 0 should yield identity matrix."""
        tr = DiscreteUniformTransition(2, 2, 0)
        ph = tr.get_Qt(torch.tensor([0.0]), "cpu")
        assert torch.isfinite(ph.X).all()
        assert torch.allclose(ph.X, torch.eye(2).unsqueeze(0), atol=1e-6)

    def test_discrete_uniform_beta_one(self) -> None:
        """beta_t = 1 should yield the stationary (uniform) matrix."""
        tr = DiscreteUniformTransition(2, 2, 0)
        ph = tr.get_Qt(torch.tensor([1.0]), "cpu")
        assert torch.isfinite(ph.X).all()
        expected = torch.ones(1, 2, 2) / 2
        assert torch.allclose(ph.X, expected, atol=1e-6)

    def test_marginal_beta_zero(self) -> None:
        x_m = torch.tensor([0.6, 0.4])
        e_m = torch.tensor([0.8, 0.2])
        tr = MarginalUniformTransition(x_m, e_m, y_classes=0)
        ph = tr.get_Qt(torch.tensor([0.0]), "cpu")
        assert torch.isfinite(ph.X).all()
        assert torch.allclose(ph.X, torch.eye(2).unsqueeze(0), atol=1e-6)
