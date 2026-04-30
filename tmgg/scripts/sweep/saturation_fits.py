"""Compare three saturation-curve models on a gen-val time series.

Diagnostic-only: this module does not influence the watcher's flowchart
or any kill/keep decision. Its purpose is to characterise the
asymptote-prediction problem on the running sweep so we can pick a
principled model later. The watcher's existing Domhan exponential
remains the production fit.

Three models, in increasing rigor:

1. **Domhan exponential** — ``f(s) = a (1 - exp(-s/tau)) + c``. The
   classical learning-curve prior. Closed-form via ``scipy.optimize
   .curve_fit``. Asymptote = ``a + c``. Cheap, but assumes a
   monotone-increasing target — silently breaks on the noisy
   non-monotone Bernoulli proportions we see early in training.

2. **Logistic** — ``f(s) = L / (1 + exp(-(s - s_mid)/tau))``. Sigmoid
   shape; better matches the "long warmup, then fast climb, then
   plateau" pattern that gen-val/sbm_accuracy actually shows. The
   asymptote is ``L`` (the saturation height). Still a least-squares
   fit; doesn't model observation noise explicitly.

3. **Beta-Binomial sequential posterior** — for metrics that are
   proportions of N evaluator trials (``sbm_accuracy`` is wins/40 by
   default), each observation is ``Binomial(N, p_t)``. We treat the
   long-run rate as a single Beta-distributed parameter and update
   the posterior across all observed steps. The asymptote becomes
   ``Beta(alpha_post, beta_post).mean()`` with a closed-form 95%
   credible interval. Most rigorous of the three, but only meaningful
   for genuinely-proportional metrics — applying it to MMDs is
   nonsense.

Convention: each fitter returns a JSON-friendly dict with at least
``model``, ``q_inf``, and ``status`` keys. ``status="ok"`` means the
fit converged; ``"degenerate"``, ``"insufficient_data"``, etc. flag
the failure modes that the watcher silently swallows today.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import (  # pyright: ignore[reportMissingImports]
    OptimizeWarning,
    curve_fit,
)
from scipy.stats import beta as beta_dist  # pyright: ignore[reportMissingImports]


def _domhan_func(s: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    """f(s) = a * (1 - exp(-s/tau)) + c. Asymptote = a + c."""
    return a * (1.0 - np.exp(-s / tau)) + c


def _logistic_func(s: np.ndarray, L: float, s_mid: float, tau: float) -> np.ndarray:
    """f(s) = L / (1 + exp(-(s - s_mid)/tau)). Asymptote = L."""
    return L / (1.0 + np.exp(-(s - s_mid) / tau))


def fit_domhan_exponential(steps: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Fit ``a (1 - exp(-s/tau)) + c``; report asymptote = a + c.

    Mirrors the watcher's current production fit. Returns ``status``
    "insufficient_data" when fewer than 8 finite points are available
    (matches ``saturation_fit_partial`` in watch_runs.py), and
    ``"degenerate"`` when curve_fit cannot estimate the covariance
    (the OptimizeWarning we keep seeing in the watcher logs).
    """
    if len(steps) < 8:
        return {
            "model": "domhan",
            "status": "insufficient_data",
            "n_points": int(len(steps)),
            "q_inf": None,
        }
    try:
        # Same initial guess as compute_s_star.py.
        p0 = (float(values[-1] - values[0]), float(steps[-1] / 3), float(values[0]))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(_domhan_func, steps, values, p0=p0, maxfev=5000)
    except (OptimizeWarning, RuntimeError, ValueError) as exc:
        return {
            "model": "domhan",
            "status": "degenerate",
            "reason": str(exc)[:200],
            "n_points": int(len(steps)),
            "q_inf": None,
        }
    a, tau, c = (float(x) for x in popt)
    q_inf = a + c
    # Wald 1-sigma on q_inf via diag of pcov for (a, c).
    var_a = float(pcov[0, 0]) if pcov.shape == (3, 3) else float("nan")
    var_c = float(pcov[2, 2]) if pcov.shape == (3, 3) else float("nan")
    cov_ac = float(pcov[0, 2]) if pcov.shape == (3, 3) else float("nan")
    sigma_q_inf = math.sqrt(max(var_a + var_c + 2.0 * cov_ac, 0.0))
    return {
        "model": "domhan",
        "status": "ok",
        "n_points": int(len(steps)),
        "q_inf": q_inf,
        "sigma_q_inf": sigma_q_inf,
        "a": a,
        "tau": tau,
        "c": c,
    }


def fit_logistic(steps: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Fit ``L / (1 + exp(-(s - s_mid)/tau))``; report asymptote = L.

    Logistic better matches a sigmoid trajectory (slow warmup, fast
    climb, plateau) than the always-concave exponential. Initial
    midpoint guess uses the median step; initial L is the trailing
    max so the fit doesn't get stuck at a low plateau when the true
    asymptote is still climbing.
    """
    if len(steps) < 8:
        return {
            "model": "logistic",
            "status": "insufficient_data",
            "n_points": int(len(steps)),
            "q_inf": None,
        }
    try:
        L0 = float(max(np.max(values), 0.01))
        s_mid0 = float(np.median(steps))
        tau0 = float(steps[-1] / 4.0)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(
                _logistic_func, steps, values, p0=(L0, s_mid0, tau0), maxfev=5000
            )
    except (OptimizeWarning, RuntimeError, ValueError) as exc:
        return {
            "model": "logistic",
            "status": "degenerate",
            "reason": str(exc)[:200],
            "n_points": int(len(steps)),
            "q_inf": None,
        }
    L, s_mid, tau = (float(x) for x in popt)
    sigma_L = (
        math.sqrt(max(float(pcov[0, 0]), 0.0)) if pcov.shape == (3, 3) else float("nan")
    )
    return {
        "model": "logistic",
        "status": "ok",
        "n_points": int(len(steps)),
        "q_inf": L,
        "sigma_q_inf": sigma_L,
        "s_mid": s_mid,
        "tau": tau,
    }


def fit_beta_binomial(
    steps: np.ndarray,
    values: np.ndarray,
    n_trials_per_step: int = 40,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> dict[str, Any]:
    """Sequential Beta posterior on a wins/trials proportion.

    Treats ``values[i]`` as ``wins[i] / n_trials_per_step``. Updates a
    single shared Beta posterior across all observed steps. Reports
    posterior mean + 95% credible interval as the asymptote estimate.

    Caveat: this assumes the per-step proportion is drawn from a
    single underlying rate ``p``. That is FALSE early in training (the
    rate is climbing), so this fit will systematically under-estimate
    the asymptote until training has saturated. Compare against the
    logistic asymptote: when they agree, the run is in a stable
    plateau; when logistic reports much higher L than beta-binomial
    posterior mean, the run is still climbing and the beta estimate
    is anchored on early-training failures.

    Only meaningful for proportion metrics (``sbm_accuracy``,
    ``planarity_accuracy``, ``uniqueness``). Applying to MMDs is
    nonsense; the caller is responsible for filtering.
    """
    if len(steps) < 1:
        return {
            "model": "beta_binomial",
            "status": "insufficient_data",
            "n_points": 0,
            "q_inf": None,
        }
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if len(finite_values) == 0:
        return {
            "model": "beta_binomial",
            "status": "no_finite_observations",
            "n_points": 0,
            "q_inf": None,
        }
    if (finite_values < 0).any() or (finite_values > 1).any():
        return {
            "model": "beta_binomial",
            "status": "not_a_proportion",
            "reason": (
                f"values out of [0,1]: min={float(finite_values.min())}, "
                f"max={float(finite_values.max())}"
            ),
            "n_points": int(len(finite_values)),
            "q_inf": None,
        }
    total_wins = float(np.sum(np.round(finite_values * n_trials_per_step)))
    total_trials = float(len(finite_values) * n_trials_per_step)
    alpha_post = prior_alpha + total_wins
    beta_post = prior_beta + (total_trials - total_wins)
    posterior_mean = alpha_post / (alpha_post + beta_post)
    ci_low, ci_high = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)
    return {
        "model": "beta_binomial",
        "status": "ok",
        "n_points": int(len(finite_values)),
        "q_inf": float(posterior_mean),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "alpha_posterior": float(alpha_post),
        "beta_posterior": float(beta_post),
        "n_trials_per_step": n_trials_per_step,
    }


def compare_fits(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    is_proportion: bool = False,
    n_trials_per_step: int = 40,
) -> dict[str, Any]:
    """Run all three fits and return a structured comparison.

    ``is_proportion`` controls whether the Beta-Binomial fit runs (it's
    only valid for proportion metrics). MMD targets typically have
    ``is_proportion=False``.
    """
    out: dict[str, Any] = {
        "domhan": fit_domhan_exponential(steps, values),
        "logistic": fit_logistic(steps, values),
    }
    if is_proportion:
        out["beta_binomial"] = fit_beta_binomial(
            steps, values, n_trials_per_step=n_trials_per_step
        )
    else:
        out["beta_binomial"] = {
            "model": "beta_binomial",
            "status": "skipped_not_a_proportion",
            "q_inf": None,
        }
    return out
