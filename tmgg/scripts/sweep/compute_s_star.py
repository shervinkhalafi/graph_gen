"""Compute the saturation step S* from a Lightning metrics.csv.

Per spec §4.1 / §5.6: fit a saturating exponential to each metric of
interest, compute the step at which the metric reaches a target
fraction of its predicted terminal value, take the latest such step
across metrics as S*.

Reads legacy v1 namespace keys (``val/gen/*``) by default; the new
namespace (``gen-val/*``) is auto-mapped via ``--metric-keys`` if the
caller passes them in explicitly.

Option-D fallback (reviewer 2026-04-29). When the v1 long-run was
sampled too coarsely to fit the gen-val/* curves (the v1 cadence
gives only three points across 232k steps), the script can fall back
to fitting sign-flipped ``val/epoch_NLL`` and reporting the resulting
saturation step under a *separate* top-level key
``s_star_nll_upper_bound``. This is an upper bound only — per
README §"Pitfall to design around", NLL plateaus do not always
correspond to sample-quality plateaus, so an NLL-derived S* can
overshoot the structural-quality saturation. The operational sweep
cap is set independently via ``--operational-cap``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.optimize import curve_fit

#: Minimum number of finite samples required to identify the
#: three-parameter Domhan saturating exponential. Raising this from
#: the previous "≥ 4" floor reflects empirical instability of the fit
#: at five-to-seven points: tau is poorly identified there and the
#: 95% saturation step swings by an order of magnitude with noise.
MIN_POINTS_FOR_FIT = 8

#: NLL key used by the ``--nll-fallback`` path. Hard-coded; if a future
#: dataset logs it under a different name, update both this constant
#: and the README pitfall section.
NLL_KEY = "val/epoch_NLL"

#: Caveat string written verbatim into ``s_star.yaml`` whenever the
#: NLL fallback fires. Kept here so the test suite can pin the exact
#: phrasing, and so reviewers see the same caveat regardless of CLI
#: invocation.
NLL_CAVEAT = (
    "NLL-derived S* is an upper bound only.\n"
    "Per README docs/reports/2026-04-28-hp-tuning-and-knee-identification\n"
    "section 'Pitfall to design around': NLL plateaus do not always\n"
    "correspond to sample-quality plateaus in this regime — the model\n"
    "can match marginal degree/spectrum statistics without recovering\n"
    "block structure. Treat s_star_nll_upper_bound as a soft ceiling\n"
    "on training horizon; the structural-quality S* (gen-val/*) is\n"
    "what actually governs the sweep, and is generally smaller.\n"
    "Replace with a full gen-val/* fit once a v1-equivalent run with\n"
    "eval_every_n_steps <= 10000 is available."
)

#: Sanity-check string: if NLL upper bound is more than 1.5x the
#: operational cap, escalate. The 1.5x factor matches the default
#: tolerance_x used elsewhere in the sweep (anchors.yaml MMD entries).
SANITY_CHECK = (
    "If s_star_nll_upper_bound > 1.5 * s_star_operational, escalate.\n"
    "Either the operational cap is too aggressive, or NLL is\n"
    "saturating much later than the structural metrics — both\n"
    "warrant a re-think before launching the sweep."
)


class InsufficientSamplesError(KeyError):
    """Raised when a requested metric has fewer than ``MIN_POINTS_FOR_FIT``
    finite samples in the supplied metrics.csv.

    Subclasses ``KeyError`` because the symptom is the same for the
    caller — a metric they asked for cannot be produced — and
    ``check_threshold.py`` already special-cases ``KeyError`` for the
    "missing required metric" failure mode (spec §7).
    """


def _model(s: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    return a * (1.0 - np.exp(-s / tau)) + c


def fit_saturating_exponential(
    steps: np.ndarray, values: np.ndarray
) -> tuple[float, float, float]:
    """Return (a, tau, c) for ``f(s) = a * (1 - exp(-s/tau)) + c``."""
    finite = np.isfinite(values)
    s = np.asarray(steps, dtype=float)[finite]
    v = np.asarray(values, dtype=float)[finite]
    if s.size < MIN_POINTS_FOR_FIT:
        raise InsufficientSamplesError(
            f"need at least {MIN_POINTS_FOR_FIT} finite points; got {s.size}"
        )
    a0 = float(v[-1] - v[0])
    tau0 = float((s[-1] - s[0]) / 3.0)
    c0 = float(v[0])
    popt, _ = curve_fit(
        _model,
        s,
        v,
        p0=(a0, tau0, c0),
        maxfev=10_000,
    )
    a, tau, _c = (float(x) for x in popt)
    return a, tau, _c


def saturation_step(*, a: float, tau: float, _c: float, fraction: float) -> float:
    """Step at which ``a*(1-exp(-s/tau)) >= fraction * a``.

    Solving: ``s = -tau * ln(1 - fraction)``. Genuinely independent
    of the offset ``_c`` (hence the leading underscore in the
    parameter name) — kept in the signature for symmetry with the
    fit return tuple, never read.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}")
    if a == 0.0:
        return 0.0
    return float(-tau * np.log(1.0 - fraction))


def _count_finite(df: pl.DataFrame, key: str) -> int:
    """Number of finite samples for ``key`` in a Lightning metrics.csv."""
    if key not in df.columns:
        return 0
    sub = df.select(["step", key]).drop_nulls()
    if sub.is_empty():
        return 0
    values = sub[key].to_numpy().astype(float)
    return int(np.isfinite(values).sum())


def _fit_one_metric(
    df: pl.DataFrame, key: str, fractions: tuple[float, ...]
) -> dict[str, float]:
    """Fit a single metric column and return per-fraction saturation steps.

    Raises ``InsufficientSamplesError`` when the metric has fewer than
    ``MIN_POINTS_FOR_FIT`` finite samples. Caller decides whether to
    propagate or accumulate.
    """
    if key not in df.columns:
        raise KeyError(f"metric {key!r} not in metrics.csv; columns={df.columns}")
    sub = df.select(["step", key]).drop_nulls()
    if sub.is_empty():
        raise InsufficientSamplesError(f"metric {key!r} has zero non-null rows")
    steps = sub["step"].to_numpy().astype(float)
    values = sub[key].to_numpy().astype(float)
    a, tau, c = fit_saturating_exponential(steps, values)
    entry: dict[str, float] = {"a": a, "tau": tau, "c": c}
    for frac in fractions:
        entry[f"s_at_{frac:.2f}"] = saturation_step(a=a, tau=tau, _c=c, fraction=frac)
    return entry


def compute_s_star(
    *,
    metrics_csv: Path,
    metric_keys: Iterable[str],
    out_path: Path,
    dataset: str,
    fractions: tuple[float, ...] = (0.90, 0.95, 0.99),
    nll_fallback: bool = False,
    operational_cap: int | None = None,
) -> dict[str, object]:
    """Fit each metric, write a flat S* yaml to ``out_path``.

    The output schema is **flat**: top-level keys ``s_star``,
    ``s_star_status``, ``s_star_operational``, ``s_star_nll_upper_bound``,
    plus provenance and per-metric blocks. This deliberately differs
    from the previous nested-by-dataset shape — the sweep only ever
    reads one dataset's S* file at a time, and flat keys make the
    "Option D dual key" structure (operational cap + NLL ceiling)
    inspect-at-a-glance.

    Parameters
    ----------
    metrics_csv : Path
        Lightning ``metrics.csv`` from the v1 long-run.
    metric_keys : Iterable[str]
        gen-val/* (or legacy val/gen/*) column names to fit.
    out_path : Path
        Where to write the resulting yaml.
    dataset : str
        Recorded in the yaml as provenance; not used in keying.
    fractions : tuple of float
        Fractions of terminal value at which to compute saturation
        steps. ``s_star`` itself is taken at the median fraction
        (defaults to 0.95).
    nll_fallback : bool
        If True, also fit sign-flipped ``val/epoch_NLL`` and write the
        resulting saturation step under ``s_star_nll_upper_bound``.
        When zero gen-val/* metrics meet ``MIN_POINTS_FOR_FIT``, the
        NLL fit is the only thing that produces a number, but
        ``s_star`` itself is still written as ``null`` to make it
        impossible to silently confuse the upper bound with the
        structural-quality S*.
    operational_cap : int or None
        If given, recorded as ``s_star_operational`` — the actual
        step cap the sweep will use. Independent of any fit.
    """
    df = pl.read_csv(metrics_csv)
    if "step" not in df.columns:
        raise KeyError(f"'step' column missing from {metrics_csv}")

    metric_keys_list = list(metric_keys)
    gen_val_sample_counts = {k: _count_finite(df, k) for k in metric_keys_list}

    # Fit each gen-val/* metric, accumulating insufficient-sample
    # errors rather than raising on the first one — we need the full
    # picture before deciding whether to fall through to NLL.
    per_metric: dict[str, dict[str, float]] = {}
    insufficient: list[tuple[str, InsufficientSamplesError]] = []
    s_star_at: dict[float, float] = {f: 0.0 for f in fractions}

    for key in metric_keys_list:
        try:
            entry = _fit_one_metric(df, key, fractions)
        except InsufficientSamplesError as e:
            insufficient.append((key, e))
            continue
        per_metric[key] = entry
        for frac in fractions:
            s_star_at[frac] = max(s_star_at[frac], entry[f"s_at_{frac:.2f}"])

    n_fitted = len(per_metric)
    n_insufficient = len(insufficient)

    if n_fitted == 0 and not nll_fallback:
        # Zero gen-val/* metrics could be fit and the caller did not
        # opt into the NLL fallback — fail loud (spec §7, "fail loud").
        msg_lines = [
            f"No gen-val/* metric in {metrics_csv} has >= {MIN_POINTS_FOR_FIT}",
            "finite samples. Either re-run with denser eval cadence,",
            "or pass --nll-fallback to fit sign-flipped val/epoch_NLL",
            "as an upper-bound proxy.",
            "Per-metric finite-sample counts:",
        ]
        for k, n in gen_val_sample_counts.items():
            msg_lines.append(f"  {k}: {n}")
        raise InsufficientSamplesError("\n".join(msg_lines))

    canonical_fraction = 0.95 if 0.95 in s_star_at else max(s_star_at)

    # NLL fallback — fit sign-flipped val/epoch_NLL so the curve
    # rises and saturates rather than falls. The fitter is
    # direction-agnostic but the saturation_step formula assumes a
    # rising curve.
    #
    # Degenerate-fit handling: if the NLL signal is too noisy or
    # too flat (e.g. v1 long-run NLL fluctuates around a near-
    # constant mean — std/mean ≈ 0.19, slope ≈ 0), curve_fit
    # converges to negative tau or near-zero a. Both produce
    # meaningless saturation steps. We detect both and record
    # ``s_star_nll_upper_bound: null`` plus a fit-status note,
    # rather than writing a misleading number.
    nll_block: dict[str, float] | None = None
    nll_sample_count: int | None = None
    nll_fit_status: str | None = None
    nll_upper_bound: int | None = None
    if nll_fallback:
        nll_sample_count = _count_finite(df, NLL_KEY)
        if NLL_KEY not in df.columns:
            raise KeyError(
                f"--nll-fallback requested but {NLL_KEY!r} missing from {metrics_csv}"
            )
        nll_sub = df.select(["step", NLL_KEY]).drop_nulls()
        nll_steps = nll_sub["step"].to_numpy().astype(float)
        nll_values = -nll_sub[NLL_KEY].cast(pl.Float64).to_numpy()
        a, tau, c = fit_saturating_exponential(nll_steps, nll_values)
        nll_block = {"a": a, "tau": tau, "c": c}
        for frac in fractions:
            nll_block[f"s_at_{frac:.2f}"] = saturation_step(
                a=a, tau=tau, _c=c, fraction=frac
            )
        # Decide whether the fit is usable.
        sat = nll_block[f"s_at_{canonical_fraction:.2f}"]
        if tau <= 0.0 or sat <= 0.0 or not np.isfinite(sat):
            nll_fit_status = "degenerate-fit-nll-too-noisy"
        else:
            nll_fit_status = "fitted"
            nll_upper_bound = int(round(sat))

    # Status assignment.
    if n_fitted > 0 and n_insufficient == 0:
        status = "fitted"
    elif n_fitted == 0 and nll_block is not None:
        status = "nll-fallback-only"
    else:
        status = "insufficient-gen-val-samples"

    s_star_value: int | None = (
        int(round(s_star_at[canonical_fraction])) if n_fitted > 0 else None
    )

    result: dict[str, object] = {
        "dataset": dataset,
        "source_csv": str(metrics_csv),
        "s_star": s_star_value,
        "s_star_status": status,
        "s_star_fraction": canonical_fraction,
        "fractions": list(fractions),
        "gen_val_sample_counts": gen_val_sample_counts,
        "per_metric": per_metric,
    }
    if operational_cap is not None:
        result["s_star_operational"] = int(operational_cap)
    if nll_block is not None:
        result["s_star_nll_upper_bound"] = nll_upper_bound
        result["nll_fit_status"] = nll_fit_status
        result["nll_per_metric"] = {NLL_KEY: nll_block}
        result["nll_sample_count"] = nll_sample_count
        result["s_star_nll_caveat"] = NLL_CAVEAT
        result["sanity_check"] = SANITY_CHECK
    if n_insufficient > 0:
        result["insufficient_metrics"] = {k: str(e) for k, e in insufficient}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(result, sort_keys=False))
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--dataset", required=True, help="Provenance label in the output yaml"
    )
    p.add_argument(
        "--metric-keys",
        nargs="+",
        required=True,
        help="CSV column names to fit (legacy v1: val/gen/*; new: gen-val/*)",
    )
    p.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
    )
    p.add_argument(
        "--nll-fallback",
        action="store_true",
        help=(
            "When set, also fit sign-flipped val/epoch_NLL and write the "
            "result under s_star_nll_upper_bound (upper bound only — see "
            "README pitfall section)."
        ),
    )
    p.add_argument(
        "--operational-cap",
        type=int,
        default=None,
        help=(
            "Operational sweep cap (training step budget). Recorded as "
            "s_star_operational in the output yaml. Independent of any "
            "fit; chosen by the human/reviewer."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compute_s_star(
        metrics_csv=args.metrics_csv,
        metric_keys=args.metric_keys,
        out_path=args.out,
        dataset=args.dataset,
        fractions=tuple(args.fractions),
        nll_fallback=args.nll_fallback,
        operational_cap=args.operational_cap,
    )


if __name__ == "__main__":
    main()
