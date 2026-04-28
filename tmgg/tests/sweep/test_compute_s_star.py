"""Tests for scripts.sweep.compute_s_star.

Rationale: the saturating-exponential fit is a classical Domhan-2015
move. We verify the fitter recovers a known tau and computes the
correct saturation step on a synthetic curve before trusting it on
the v1 long-run.

Schema rationale (Option D, reviewer 2026-04-29). The yaml is *flat*
(top-level ``s_star`` / ``s_star_operational`` / ``s_star_nll_upper_bound``)
because the sweep only ever consumes one dataset's S* at a time, and
flat keys make the dual-key NLL-fallback structure inspect-at-a-glance.
The test assertions below pin that flat shape so the schema cannot
silently regress.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from scripts.sweep.compute_s_star import (
    MIN_POINTS_FOR_FIT,
    NLL_KEY,
    InsufficientSamplesError,
    compute_s_star,
    fit_saturating_exponential,
    saturation_step,
)


def _synthetic_curve(
    *, a: float, tau: float, c: float, n_steps: int, noise_sd: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = np.arange(1, n_steps + 1, dtype=float)
    truth = a * (1.0 - np.exp(-steps / tau)) + c
    return steps, truth + rng.normal(0.0, noise_sd, size=steps.shape)


def test_fitter_recovers_known_tau() -> None:
    steps, values = _synthetic_curve(
        a=1.0, tau=1000.0, c=0.0, n_steps=10_000, noise_sd=0.005, seed=0
    )
    a_hat, tau_hat, c_hat = fit_saturating_exponential(steps, values)
    assert tau_hat == pytest.approx(1000.0, rel=0.05)
    assert a_hat == pytest.approx(1.0, rel=0.05)
    assert abs(c_hat) < 0.05


def test_saturation_step_at_95pct() -> None:
    """ln(1 / 0.05) ≈ 2.9957, so 95% saturation ≈ 3 * tau."""
    s95 = saturation_step(a=1.0, tau=1000.0, _c=0.0, fraction=0.95)
    assert s95 == pytest.approx(3000.0, rel=0.01)


def test_compute_s_star_picks_latest_metric(tmp_path: Path) -> None:
    """Two metrics; one saturates fast, one slow; S* is the slow one."""
    steps_fast, values_fast = _synthetic_curve(
        a=1.0, tau=500.0, c=0.0, n_steps=5_000, noise_sd=0.001, seed=1
    )
    steps_slow, values_slow = _synthetic_curve(
        a=1.0, tau=2000.0, c=0.0, n_steps=10_000, noise_sd=0.001, seed=2
    )
    # Lightning-style metrics.csv: long format with 'step' column.
    n = max(len(steps_fast), len(steps_slow))
    df = pl.DataFrame(
        {
            "step": np.arange(1, n + 1),
            "epoch": np.zeros(n),
            "metric_a": np.concatenate(
                [values_fast, np.full(n - len(values_fast), values_fast[-1])]
            ),
            "metric_b": np.concatenate(
                [values_slow, np.full(n - len(values_slow), values_slow[-1])]
            ),
        }
    )
    csv_path = tmp_path / "metrics.csv"
    df.write_csv(csv_path)

    out_path = tmp_path / "s_star.yaml"
    compute_s_star(
        metrics_csv=csv_path,
        metric_keys=["metric_a", "metric_b"],
        out_path=out_path,
        dataset="synthetic",
        fractions=(0.95,),
    )

    result = yaml.safe_load(out_path.read_text())
    # Flat schema: per_metric is at the top level, keyed by metric name.
    s_star_a = result["per_metric"]["metric_a"]["s_at_0.95"]
    s_star_b = result["per_metric"]["metric_b"]["s_at_0.95"]
    assert s_star_a == pytest.approx(1500.0, rel=0.10)
    assert s_star_b == pytest.approx(6000.0, rel=0.10)
    # S* across metrics is the maximum, written as a top-level int.
    assert result["s_star"] == pytest.approx(6000, rel=0.10)
    assert result["s_star_status"] == "fitted"
    assert result["dataset"] == "synthetic"


def test_insufficient_gen_val_falls_through_to_nll_upper_bound(tmp_path: Path) -> None:
    """Option D fallback: 3 gen-val/* points + ~50 NLL points.

    Without ``--nll-fallback`` the script must raise
    ``InsufficientSamplesError`` because no gen-val/* metric meets
    ``MIN_POINTS_FOR_FIT``. With ``--nll-fallback`` and an
    ``--operational-cap``, the script must produce ``s_star: null``
    (we refuse to confuse an NLL upper bound with a true structural
    S*) plus a populated ``s_star_nll_upper_bound`` and
    ``s_star_operational``.
    """
    # Three gen-val/* points spaced at the v1 cadence (~75k apart).
    # That mirrors the actual v1 long-run: validation-time generation
    # was so expensive it only fired three times across 232k steps.
    gen_val_steps = np.array([74_999, 149_999, 224_999], dtype=float)
    gen_val_values = np.array([0.30, 0.45, 0.55], dtype=float)

    # ~50 NLL points across the same horizon, sampled densely enough
    # to identify the saturating curve. Sign-flipped here on the
    # *generation* side so the flipped value rises and saturates;
    # the script under test will sign-flip again internally.
    nll_steps = np.arange(4_999, 230_000, 5_000, dtype=float)
    assert nll_steps.size >= MIN_POINTS_FOR_FIT, "regression sentinel"
    rng = np.random.default_rng(42)
    a_true, tau_true, c_true = 2.0, 30_000.0, 1.0
    nll_signal = a_true * (1.0 - np.exp(-nll_steps / tau_true)) + c_true
    nll_signal += rng.normal(0.0, 0.05, size=nll_signal.shape)
    # Stored sign-flipped → val/epoch_NLL itself decreases over time.
    nll_values_stored = -nll_signal

    # Stitch into a single Lightning-style metrics.csv. Use an
    # all-rows step column; gen-val/* and NLL each appear only on
    # the steps where they were actually logged (else null).
    all_steps = np.union1d(gen_val_steps, nll_steps).astype(int)
    n = all_steps.size
    gen_col: list[float | None] = [None] * n
    nll_col: list[float | None] = [None] * n
    step_to_idx = {int(s): i for i, s in enumerate(all_steps)}
    for s, v in zip(gen_val_steps, gen_val_values, strict=False):
        gen_col[step_to_idx[int(s)]] = float(v)
    for s, v in zip(nll_steps, nll_values_stored, strict=False):
        nll_col[step_to_idx[int(s)]] = float(v)

    df = pl.DataFrame(
        {
            "step": all_steps,
            "val/gen/sbm_accuracy": gen_col,
            NLL_KEY: nll_col,
        }
    )
    csv_path = tmp_path / "metrics.csv"
    df.write_csv(csv_path)

    out_path = tmp_path / "s_star.yaml"

    # (a) Without --nll-fallback, must raise InsufficientSamplesError.
    with pytest.raises(InsufficientSamplesError):
        compute_s_star(
            metrics_csv=csv_path,
            metric_keys=["val/gen/sbm_accuracy"],
            out_path=out_path,
            dataset="spectre_sbm",
            fractions=(0.95,),
            nll_fallback=False,
        )

    # (b) With --nll-fallback --operational-cap 100000, succeeds and
    # writes the dual-key Option-D schema.
    result = compute_s_star(
        metrics_csv=csv_path,
        metric_keys=["val/gen/sbm_accuracy"],
        out_path=out_path,
        dataset="spectre_sbm",
        fractions=(0.95,),
        nll_fallback=True,
        operational_cap=100_000,
    )

    on_disk = yaml.safe_load(out_path.read_text())
    assert on_disk == result, "in-memory return must match yaml on disk"

    assert on_disk["s_star"] is None
    assert on_disk["s_star_status"] == "nll-fallback-only"
    assert on_disk["s_star_operational"] == 100_000
    assert on_disk["nll_fit_status"] == "fitted"
    assert isinstance(on_disk["s_star_nll_upper_bound"], int)
    assert on_disk["s_star_nll_upper_bound"] > 0
    # Recovered tau ≈ 30000, so 95% saturation ≈ 3 * tau ≈ 90000.
    # Allow ±25% to absorb noise — this test pins the order of
    # magnitude, not the third decimal.
    assert on_disk["s_star_nll_upper_bound"] == pytest.approx(90_000, rel=0.25)
    # Gen-val sample counts must be reported even when no fit
    # succeeded — the operator needs to see why.
    assert on_disk["gen_val_sample_counts"]["val/gen/sbm_accuracy"] == 3
    assert on_disk["nll_sample_count"] >= MIN_POINTS_FOR_FIT
    # Caveat and sanity-check strings must be present when the NLL
    # path fired, so a reader of the yaml cannot miss the warning.
    assert "upper bound" in on_disk["s_star_nll_caveat"].lower()
    assert "escalate" in on_disk["sanity_check"].lower()
