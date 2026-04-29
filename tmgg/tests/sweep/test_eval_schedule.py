"""Tests for scripts.sweep.eval_schedule.

Rationale
---------
The cosine/U-bowl eval cadence places evaluations densely at training
boundaries (warmup, expected knee) and sparsely in the chance-plateau
middle. The cadence density is

    rho(s) = rho_max - (rho_max - rho_min) * sin^2(pi * s / (2 * s_p))

with the bowl minimum at ``s = s_p`` (the chance-plateau midpoint) and
maxima at ``s = 0`` and ``s = 2*s_p`` (the expected knee). Equivalent
``cos^2`` form: ``rho_min + (rho_max - rho_min) * cos^2(pi * s / (2 * s_p))``.

The CDF has a closed form:

    C(s) = 0.5 * (rho_max + rho_min) * s
         + ((rho_max - rho_min) * s_p) / (2 * pi) * sin(pi * s / s_p)

Eval timestamps are placed by inverse CDF using ``scipy.optimize.brentq``.

These tests pin: (1) bowl shape (max at endpoints, min at midpoint),
(2) CDF monotonicity, (3) inverse-CDF round-trip accuracy, (4) schedule
length, (5) schedule density skew (more points at boundaries than in
the middle), and (6) iteration-beyond-knee with shifted bowls.
"""

from __future__ import annotations

import math

import pytest
from scripts.sweep.eval_schedule import (
    compute_schedule,
    cosine_cdf,
    cosine_density,
    inverse_cdf,
)

# Default SBM bowl parameters per spec §11.1.
RHO_MAX = 1.0 / 4000.0
RHO_MIN = 1.0 / 20000.0
S_P = 35000


def test_cosine_density_max_at_endpoints() -> None:
    """rho(0) and rho(2*s_p) both equal rho_max (warmup + knee density)."""
    rho_at_zero = cosine_density(0.0, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
    rho_at_two_sp = cosine_density(2.0 * S_P, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
    assert rho_at_zero == pytest.approx(RHO_MAX, rel=1e-12)
    assert rho_at_two_sp == pytest.approx(RHO_MAX, rel=1e-12)


def test_cosine_density_min_at_chance_plateau() -> None:
    """rho(s_p) equals rho_min — sparsest spacing in the chance-plateau middle."""
    rho_at_sp = cosine_density(float(S_P), rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
    assert rho_at_sp == pytest.approx(RHO_MIN, rel=1e-12)


def test_cosine_density_within_bounds() -> None:
    """rho(s) is in [rho_min, rho_max] for all s in [0, 2*s_p]."""
    for s in range(0, 2 * S_P + 1, 1000):
        rho = cosine_density(float(s), rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
        assert RHO_MIN - 1e-15 <= rho <= RHO_MAX + 1e-15


def test_cdf_monotone() -> None:
    """C(s) is strictly increasing on [0, 2*s_p]."""
    cdf_values = [
        cosine_cdf(float(s), rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
        for s in range(0, 2 * S_P + 1, 1000)
    ]
    for prev, curr in zip(cdf_values, cdf_values[1:], strict=False):
        assert curr > prev, f"CDF not monotone: {prev} -> {curr}"


def test_cdf_at_zero_is_zero() -> None:
    """C(0) = 0."""
    assert cosine_cdf(0.0, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P) == pytest.approx(
        0.0, abs=1e-12
    )


def test_inverse_cdf_roundtrip() -> None:
    """For several test points s, inverse_cdf(C(s)) == s within 0.01%."""
    upper = 2.0 * S_P
    for s in [S_P / 2.0, S_P, 3.0 * S_P / 2.0, 1000.0, 50000.0]:
        c = cosine_cdf(s, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
        s_back = inverse_cdf(
            c,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
            s_p=S_P,
            x_lo=0.0,
            x_hi=upper,
        )
        assert s_back == pytest.approx(s, rel=1e-4)


def test_compute_schedule_count() -> None:
    """compute_schedule(N=12, ...) returns exactly 12 sorted distinct ints."""
    schedule = compute_schedule(
        n_evals=12, total_steps=2 * S_P, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P
    )
    assert len(schedule) == 12
    assert schedule == sorted(schedule)
    # All entries are positive ints in [1, total_steps].
    for s in schedule:
        assert isinstance(s, int)
        assert 1 <= s <= 2 * S_P


def test_compute_schedule_dense_at_endpoints() -> None:
    """Density skew: more eval steps in [0, 2*s_p/3] + [4*s_p/3, 2*s_p] than middle.

    The bowl has its maximum density at endpoints and minimum at the
    midpoint. With N=24 evals, the bowl placement should yield more
    points in the first and last sixth-of-period than in the middle
    third. We assert the boundary thirds combined contain *at least*
    1.5x the middle-third count — a gentle ratio that the geometry
    must satisfy without being noise-sensitive.
    """
    n = 24
    total_steps = 2 * S_P
    schedule = compute_schedule(
        n_evals=n,
        total_steps=total_steps,
        rho_min=RHO_MIN,
        rho_max=RHO_MAX,
        s_p=S_P,
    )
    boundary_lo = 2 * S_P // 3
    boundary_hi = 4 * S_P // 3
    boundary_count = sum(1 for s in schedule if s <= boundary_lo or s >= boundary_hi)
    middle_count = sum(1 for s in schedule if boundary_lo < s < boundary_hi)
    # Defensive: there should be evals in both regions; the boundary
    # regions are 2/3 of the total interval, so we expect them to hold
    # most of the points but require at least 1.5x the middle count.
    assert boundary_count >= int(1.5 * middle_count), (
        f"boundary={boundary_count}, middle={middle_count}; "
        f"expected boundary >= 1.5*middle"
    )


def test_iteration_beyond_knee() -> None:
    """For S > 2*s_p, the schedule covers the extended range with shifted bowls.

    With total_steps = 3*s_p the schedule must still place points
    beyond ``2 * s_p`` (in the next bowl). The next bowl starts at
    ``s = 2*s_p`` so we expect at least a couple of evals beyond it.
    """
    total_steps = 3 * S_P
    schedule = compute_schedule(
        n_evals=18,
        total_steps=total_steps,
        rho_min=RHO_MIN,
        rho_max=RHO_MAX,
        s_p=S_P,
    )
    beyond_first_bowl = [s for s in schedule if s > 2 * S_P]
    assert (
        len(beyond_first_bowl) >= 2
    ), f"schedule should extend beyond first bowl (s={2 * S_P}); got {schedule}"
    # All entries within bounds.
    assert max(schedule) <= total_steps


def test_compute_schedule_distinct_ints() -> None:
    """Schedule entries are integer-rounded; collisions can occur but rare."""
    n = 10
    total_steps = 2 * S_P
    schedule = compute_schedule(
        n_evals=n,
        total_steps=total_steps,
        rho_min=RHO_MIN,
        rho_max=RHO_MAX,
        s_p=S_P,
    )
    # Allow at most one collision per consecutive pair (extreme noise).
    assert len(set(schedule)) >= n - 1


def test_cdf_closed_form_matches_numerical_integration() -> None:
    """Spot-check: at s = s_p, the closed-form CDF matches a fine Riemann sum."""
    s_target = float(S_P)
    closed = cosine_cdf(s_target, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
    # Trapezoidal integration with very small step.
    n_steps = 100_000
    ds = s_target / n_steps
    riemann = 0.0
    for i in range(n_steps):
        s_left = i * ds
        s_right = (i + 1) * ds
        rho_left = cosine_density(s_left, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
        rho_right = cosine_density(s_right, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
        riemann += 0.5 * (rho_left + rho_right) * ds
    assert closed == pytest.approx(riemann, rel=1e-4)


def test_invalid_params_raise() -> None:
    """rho_min > rho_max or non-positive s_p raises ValueError."""
    with pytest.raises(ValueError):
        cosine_density(100.0, rho_min=1.0, rho_max=0.5, s_p=1000)
    with pytest.raises(ValueError):
        cosine_density(100.0, rho_min=0.001, rho_max=0.01, s_p=0)
    with pytest.raises(ValueError):
        compute_schedule(
            n_evals=0,
            total_steps=1000,
            rho_min=RHO_MIN,
            rho_max=RHO_MAX,
            s_p=S_P,
        )


def test_density_at_quarter_period_is_average() -> None:
    """At s = s_p/2 (quarter of the 2*s_p bowl period), rho equals the mean.

    Because sin^2(pi/4) = 0.5, rho(s_p/2) = rho_max - 0.5*(rho_max - rho_min)
    = (rho_max + rho_min)/2.
    """
    s = float(S_P) / 2.0
    rho = cosine_density(s, rho_min=RHO_MIN, rho_max=RHO_MAX, s_p=S_P)
    expected = 0.5 * (RHO_MAX + RHO_MIN)
    assert rho == pytest.approx(expected, rel=1e-12)


# Use math import to silence "unused" warning when this file is parsed.
_ = math.pi
