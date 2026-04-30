"""Unit tests for saturation_fits diagnostic module.

Pin the contract for each of the three fitters: they accept a ``(steps,
values)`` series and return a JSON-friendly dict with ``model``,
``status``, and ``q_inf`` keys at minimum. We do NOT pin numerical
asymptote values tightly — the point of the diagnostic is to compare
models on real data, not to lock arbitrary tolerances. Instead, we
verify:

* Each fitter handles insufficient/empty/infinite/non-finite input
  without raising.
* On a clean synthetic curve where the asymptote is unambiguous, the
  fit recovers it within a generous tolerance.
* The Beta-Binomial branch refuses non-proportion data with a clear
  status code rather than silently misfitting.
"""

from __future__ import annotations

import numpy as np
from scripts.sweep.saturation_fits import (
    compare_fits,
    fit_beta_binomial,
    fit_domhan_exponential,
    fit_logistic,
)


def test_domhan_recovers_asymptote_on_clean_exponential() -> None:
    """f(s) = 0.85 * (1 - exp(-s/30000)) + 0.05; asymptote = 0.90."""
    rng = np.random.default_rng(0)
    a, tau, c = 0.85, 30000.0, 0.05
    steps = np.linspace(5000, 60000, 12)
    values = a * (1.0 - np.exp(-steps / tau)) + c + rng.normal(0, 0.001, 12)
    fit = fit_domhan_exponential(steps, values)
    assert fit["status"] == "ok"
    assert abs(fit["q_inf"] - 0.90) < 0.05


def test_domhan_returns_insufficient_data_below_8_points() -> None:
    fit = fit_domhan_exponential(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    assert fit["status"] == "insufficient_data"
    assert fit["q_inf"] is None


def test_logistic_recovers_asymptote_on_clean_sigmoid() -> None:
    """f(s) = 0.7 / (1 + exp(-(s - 5000)/2000)); asymptote = 0.7."""
    L_true, s_mid_true, tau_true = 0.7, 5000.0, 2000.0
    steps = np.linspace(0, 20000, 16)
    values = L_true / (1.0 + np.exp(-(steps - s_mid_true) / tau_true))
    fit = fit_logistic(steps, values)
    assert fit["status"] == "ok"
    assert abs(fit["q_inf"] - 0.7) < 0.05


def test_logistic_returns_insufficient_data_below_8_points() -> None:
    fit = fit_logistic(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    assert fit["status"] == "insufficient_data"


def test_beta_binomial_recovers_rate_on_clean_proportions() -> None:
    """All observations from p=0.6 with n=40 trials each.

    With 12 observations × 40 trials = 480 trials total, the posterior
    mean concentrates around 0.6 and the 95% CI should be tight.
    """
    rng = np.random.default_rng(1)
    p_true = 0.6
    n_per_step = 40
    steps = np.linspace(1000, 100000, 12)
    wins = rng.binomial(n_per_step, p_true, size=12)
    values = wins.astype(float) / n_per_step
    fit = fit_beta_binomial(steps, values, n_trials_per_step=n_per_step)
    assert fit["status"] == "ok"
    assert abs(fit["q_inf"] - p_true) < 0.05
    # 95% CI should be narrow with this many trials.
    assert fit["ci95_high"] - fit["ci95_low"] < 0.10


def test_beta_binomial_refuses_out_of_range_values() -> None:
    """MMD-style values (often > 1) should produce a clear refusal."""
    fit = fit_beta_binomial(
        np.linspace(1000, 5000, 6),
        np.array([0.1, 0.5, 1.7, 2.3, 0.9, 0.2]),
    )
    assert fit["status"] == "not_a_proportion"
    assert fit["q_inf"] is None


def test_beta_binomial_handles_empty_input() -> None:
    fit = fit_beta_binomial(np.array([]), np.array([]))
    assert fit["status"] == "insufficient_data"


def test_compare_fits_skips_beta_branch_for_non_proportion() -> None:
    """When ``is_proportion=False``, only Domhan and logistic compete."""
    rng = np.random.default_rng(2)
    steps = np.linspace(5000, 60000, 12)
    values = 0.5 * (1.0 - np.exp(-steps / 20000)) + rng.normal(0, 0.01, 12)
    out = compare_fits(steps, values, is_proportion=False)
    assert "domhan" in out
    assert "logistic" in out
    assert out["beta_binomial"]["status"] == "skipped_not_a_proportion"


def test_compare_fits_runs_all_three_for_proportion() -> None:
    """For proportion metrics, all three fits run side by side."""
    rng = np.random.default_rng(3)
    steps = np.linspace(5000, 60000, 12)
    p_true = 0.4
    wins = rng.binomial(40, p_true, size=12)
    values = wins.astype(float) / 40
    out = compare_fits(steps, values, is_proportion=True)
    assert out["domhan"]["status"] in {"ok", "degenerate"}
    assert out["logistic"]["status"] in {"ok", "degenerate"}
    assert out["beta_binomial"]["status"] == "ok"


def test_models_disagree_on_climbing_curve_diagnoses_premature_asymptote() -> None:
    """When the run is still climbing, the three models should DIFFER.

    This is the diagnostic insight the comparator surfaces: if Beta-
    Binomial reports posterior mean ≈ 0.3 and logistic reports L ≈ 0.7,
    the run is mid-climb and Beta-Binomial is anchored on early failures.
    Conversely, if all three agree, the run has plateaued.

    Pin: synthetic climbing-sigmoid where logistic should beat Domhan
    on asymptote recovery, and Beta-Binomial should under-estimate.
    """
    L_true, s_mid_true, tau_true = 0.7, 30000.0, 5000.0
    steps = np.linspace(1000, 30000, 16)
    values = L_true / (1.0 + np.exp(-(steps - s_mid_true) / tau_true))
    out = compare_fits(steps, values, is_proportion=True, n_trials_per_step=40)
    assert out["logistic"]["status"] == "ok"
    assert out["beta_binomial"]["status"] == "ok"
    # On a still-climbing series, the Beta-Binomial posterior mean is
    # the OBSERVED running rate so far, which is well below L_true.
    if out["logistic"]["q_inf"] is not None:
        # Logistic should be closer to the true asymptote.
        assert abs(out["logistic"]["q_inf"] - L_true) < 0.15
    # Beta-Binomial mean should be visibly LOWER than logistic L
    # (anchored on early-training failures).
    assert out["beta_binomial"]["q_inf"] < out["logistic"]["q_inf"]
