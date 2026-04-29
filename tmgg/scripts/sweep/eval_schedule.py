"""Cosine/U-bowl eval cadence — schedule generator for the smallest-config sweep.

Per spec §11.1: the eval cadence places generation evaluations densely
near the boundaries of expected behaviour (warmup near ``s = 0``, the
expected knee near ``s = 2*s_p``) and sparsely in the chance-plateau
middle (``s = s_p``). This is a simple D-optimal-design-flavoured
heuristic — Atkinson & Donev (1992) motivate concentrating
observations where curvature is highest, and the empirical learning
curves we observe (e.g. v1 long-run ``val/gen/sbm_accuracy``) are
near-flat for tens of thousands of steps before phase-transitioning.

Density formula (single-bowl, period ``2 * s_p``):

    rho(s) = rho_max - (rho_max - rho_min) * sin^2(pi * s / (2 * s_p))

Equivalent ``cos^2`` form:

    rho(s) = rho_min + (rho_max - rho_min) * cos^2(pi * s / (2 * s_p))

Properties:

- ``rho(0) = rho_max`` (densest; warmup).
- ``rho(s_p) = rho_min`` (sparsest; chance-plateau midpoint).
- ``rho(2 * s_p) = rho_max`` (densest; expected knee at ``s_k = 2 * s_p``).

Closed-form CDF (the integral of rho from 0 to s):

    C(s) = ((rho_max + rho_min) / 2) * s
         + ((rho_max - rho_min) * s_p) / (2 * pi) * sin(pi * s / s_p)

Eval timestamps are placed by inverse CDF — root-find ``s_i`` such that
``C(s_i) = (i / N) * C(S)`` for ``i = 1, ..., N``. The CDF is strictly
increasing (rho > 0 everywhere) so ``brentq`` always converges.

Iteration beyond the knee
-------------------------
For ``S > 2 * s_p``, the bowl repeats: the next bowl spans ``[2*s_p,
4*s_p]`` shifted by ``Delta = 2 * s_p``, then ``[4*s_p, 6*s_p]`` and so
on. The density is therefore periodic with period ``2 * s_p``; the CDF
extends linearly across bowls because the integral over each full
period equals ``(rho_max + rho_min) * s_p`` (the sin term vanishes at
multiples of ``s_p``).

Integration consumer
--------------------
This module computes the schedule as a sorted list of integer training
steps. Wiring the schedule into the Lightning trainer (so val_check
fires at exactly those steps) is a separate training-side patch, flagged
in spec §10 as Phase 0.4.

CLI
---

::

    uv run python -m scripts.sweep.eval_schedule \
        --dataset spectre_sbm \
        --total-steps 100000 \
        --n-evals 24

Writes ``eval_schedule_<dataset>.yaml`` next to ``s_star.yaml`` and
prints a tabular schedule preview.

References
----------
Atkinson, A. C., & Donev, A. N. (1992). *Optimum Experimental Designs*.
Oxford University Press.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from scipy.optimize import brentq

# Default parameters per spec §11.1 — SBM bowl.
DEFAULT_RHO_MIN = 1.0 / 20000.0
DEFAULT_RHO_MAX = 1.0 / 4000.0
DEFAULT_S_P = 35000

# Module-level path constant: where to load defaults from per dataset.
DEFAULT_ANCHORS_PATH = Path(
    "docs/experiments/sweep/smallest-config-2026-04-29/anchors.yaml"
)


def _validate_params(*, rho_min: float, rho_max: float, s_p: float) -> None:
    """Common validation for density/CDF inputs.

    Raises
    ------
    ValueError
        If ``rho_min >= rho_max`` (no contrast) or ``s_p <= 0`` (no period).
    """
    if rho_min <= 0.0 or rho_max <= 0.0:
        raise ValueError(f"rho must be positive: rho_min={rho_min}, rho_max={rho_max}")
    if rho_min >= rho_max:
        raise ValueError(
            f"rho_min must be < rho_max: rho_min={rho_min}, rho_max={rho_max}"
        )
    if s_p <= 0.0:
        raise ValueError(f"s_p must be positive: s_p={s_p}")


def cosine_density(s: float, *, rho_min: float, rho_max: float, s_p: float) -> float:
    """Bowl density at training step ``s``.

    Parameters
    ----------
    s
        Training step (non-negative).
    rho_min
        Minimum density (sparsest spacing's reciprocal).
    rho_max
        Maximum density (densest spacing's reciprocal).
    s_p
        Half-period; bowl minimum at ``s = s_p``, bowl maxima at
        ``s = 0`` and ``s = 2 * s_p``.

    Returns
    -------
    float
        Density value in ``[rho_min, rho_max]``.

    Notes
    -----
    Periodic with period ``2 * s_p`` so callers can use this for
    iterate-beyond-knee schedules without modular arithmetic.
    """
    _validate_params(rho_min=rho_min, rho_max=rho_max, s_p=s_p)
    sin_term = math.sin(math.pi * s / (2.0 * s_p))
    return rho_max - (rho_max - rho_min) * sin_term * sin_term


def cosine_cdf(s: float, *, rho_min: float, rho_max: float, s_p: float) -> float:
    """Closed-form cumulative density: integral of ``cosine_density`` from 0 to s.

    The closed form follows from ``sin^2(x) = (1 - cos(2x)) / 2``:

        C(s) = 0.5 * (rho_max + rho_min) * s
             + ((rho_max - rho_min) * s_p) / (2 * pi) * sin(pi * s / s_p)

    Strictly increasing on ``[0, infinity)`` because the density is
    everywhere positive.
    """
    _validate_params(rho_min=rho_min, rho_max=rho_max, s_p=s_p)
    linear = 0.5 * (rho_max + rho_min) * s
    sinusoidal = ((rho_max - rho_min) * s_p / (2.0 * math.pi)) * math.sin(
        math.pi * s / s_p
    )
    return linear + sinusoidal


def inverse_cdf(
    target: float,
    *,
    rho_min: float,
    rho_max: float,
    s_p: float,
    x_lo: float = 0.0,
    x_hi: float | None = None,
) -> float:
    """Find ``s`` such that ``cosine_cdf(s) == target`` via Brent root-finding.

    Parameters
    ----------
    target
        Target CDF value.
    rho_min, rho_max, s_p
        Density parameters.
    x_lo
        Lower bound for the search; default 0.
    x_hi
        Upper bound for the search; if None, default ``2 * s_p`` (one
        full bowl). Caller must pass a larger value for iterate-beyond-
        knee schedules.

    Raises
    ------
    ValueError
        If ``target`` is outside the CDF range on ``[x_lo, x_hi]``.
    """
    _validate_params(rho_min=rho_min, rho_max=rho_max, s_p=s_p)
    if x_hi is None:
        x_hi = 2.0 * s_p
    cdf_lo = cosine_cdf(x_lo, rho_min=rho_min, rho_max=rho_max, s_p=s_p)
    cdf_hi = cosine_cdf(x_hi, rho_min=rho_min, rho_max=rho_max, s_p=s_p)
    if not (cdf_lo - 1e-12 <= target <= cdf_hi + 1e-12):
        raise ValueError(
            f"target={target} outside CDF range [{cdf_lo}, {cdf_hi}] "
            f"on [{x_lo}, {x_hi}]"
        )

    def residual(x: float) -> float:
        return cosine_cdf(x, rho_min=rho_min, rho_max=rho_max, s_p=s_p) - target

    # ``brentq`` returns either the root or ``(root, results)`` depending on
    # ``full_output``; we never set ``full_output=True`` so the return is the
    # root scalar. basedpyright cannot narrow the union without ``cast``.
    root = brentq(  # pyright: ignore[reportUnknownVariableType]
        residual, x_lo, x_hi, xtol=1e-6
    )
    return float(root)  # pyright: ignore[reportArgumentType]


def compute_schedule(
    *,
    n_evals: int,
    total_steps: int,
    rho_min: float,
    rho_max: float,
    s_p: float,
) -> list[int]:
    """Compute the integer eval-step list for a sweep run.

    Places ``n_evals`` evaluations on ``(0, total_steps]`` with spacing
    that follows the cosine bowl. For ``total_steps > 2 * s_p`` the
    bowl repeats periodically; the placement uses the closed-form CDF
    with ``x_hi = total_steps`` (no need for explicit shifting because
    the density is periodic and the CDF accumulates linearly across
    bowls).

    Parameters
    ----------
    n_evals
        Number of evaluation points to place (must be >= 1).
    total_steps
        Upper bound on training steps (must be >= 1).
    rho_min, rho_max, s_p
        Density parameters.

    Returns
    -------
    list[int]
        Sorted list of integer step indices in ``[1, total_steps]``.
    """
    if n_evals < 1:
        raise ValueError(f"n_evals must be >= 1; got {n_evals}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1; got {total_steps}")
    _validate_params(rho_min=rho_min, rho_max=rho_max, s_p=s_p)

    cdf_total = cosine_cdf(
        float(total_steps), rho_min=rho_min, rho_max=rho_max, s_p=s_p
    )
    schedule: list[int] = []
    for i in range(1, n_evals + 1):
        target = (i / n_evals) * cdf_total
        s_real = inverse_cdf(
            target,
            rho_min=rho_min,
            rho_max=rho_max,
            s_p=s_p,
            x_lo=0.0,
            x_hi=float(total_steps),
        )
        schedule.append(max(1, int(round(s_real))))
    return sorted(schedule)


def load_dataset_defaults(
    dataset: str, anchors_path: Path | None = None
) -> tuple[float, float, int]:
    """Resolve per-dataset cosine-bowl defaults.

    Currently every dataset in scope (``spectre_sbm``, ``pyg_enzymes``)
    uses the spec §11.1 SBM defaults. The function is structured so a
    future ``anchors.yaml`` extension that carries per-dataset cosine
    parameters under an ``eval_cadence`` key can be plugged in without
    changing the CLI surface.

    Returns
    -------
    (rho_min, rho_max, s_p)
    """
    _ = dataset  # currently unused; placeholder for future per-dataset overrides
    _ = anchors_path
    return DEFAULT_RHO_MIN, DEFAULT_RHO_MAX, DEFAULT_S_P


def write_schedule_yaml(
    *,
    out_path: Path,
    dataset: str,
    n_evals: int,
    total_steps: int,
    rho_min: float,
    rho_max: float,
    s_p: float,
    schedule: list[int],
) -> None:
    """Write the schedule + parameters to ``out_path`` as YAML."""
    payload = {
        "dataset": dataset,
        "n_evals": n_evals,
        "total_steps": total_steps,
        "params": {
            "rho_min": rho_min,
            "rho_max": rho_max,
            "s_p": s_p,
            "expected_knee_s_k": 2 * s_p,
        },
        "schedule": schedule,
        "doc": (
            "Cosine/U-bowl eval cadence per spec §11.1. Density is "
            "rho(s) = rho_max - (rho_max - rho_min) * sin^2(pi * s / (2 * s_p)); "
            "min at s_p (chance-plateau midpoint), max at 0 and 2*s_p."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _format_table(schedule: list[int]) -> str:
    lines = [f"{'idx':>4}  {'step':>10}"]
    for i, s in enumerate(schedule, start=1):
        lines.append(f"{i:>4}  {s:>10}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset key (e.g. spectre_sbm, pyg_enzymes). Currently used "
            "only for output-yaml provenance and per-dataset default "
            "lookup; the bowl parameters are spec §11.1 defaults."
        ),
    )
    _ = p.add_argument(
        "--total-steps",
        type=int,
        required=True,
        help="Total training steps (max_steps).",
    )
    _ = p.add_argument(
        "--n-evals",
        type=int,
        required=True,
        help="Number of generation-eval points to place.",
    )
    _ = p.add_argument(
        "--rho-min",
        type=float,
        default=None,
        help=(
            f"Override rho_min (default per dataset; spec §11.1 SBM "
            f"default = {DEFAULT_RHO_MIN!s})."
        ),
    )
    _ = p.add_argument(
        "--rho-max",
        type=float,
        default=None,
        help=(
            f"Override rho_max (default per dataset; spec §11.1 SBM "
            f"default = {DEFAULT_RHO_MAX!s})."
        ),
    )
    _ = p.add_argument(
        "--s-p",
        type=int,
        default=None,
        help=(
            f"Override s_p (chance-plateau midpoint, half the expected "
            f"knee). Default per dataset; SBM default = {DEFAULT_S_P}."
        ),
    )
    _ = p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output yaml path; defaults to "
            "docs/experiments/sweep/smallest-config-2026-04-29/"
            "eval_schedule_<dataset>.yaml"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rho_min_default, rho_max_default, s_p_default = load_dataset_defaults(args.dataset)
    rho_min = args.rho_min if args.rho_min is not None else rho_min_default
    rho_max = args.rho_max if args.rho_max is not None else rho_max_default
    s_p = args.s_p if args.s_p is not None else s_p_default

    schedule = compute_schedule(
        n_evals=args.n_evals,
        total_steps=args.total_steps,
        rho_min=rho_min,
        rho_max=rho_max,
        s_p=s_p,
    )

    if args.out is None:
        out_path = (
            Path("docs/experiments/sweep/smallest-config-2026-04-29")
            / f"eval_schedule_{args.dataset}.yaml"
        )
    else:
        out_path = args.out

    write_schedule_yaml(
        out_path=out_path,
        dataset=args.dataset,
        n_evals=args.n_evals,
        total_steps=args.total_steps,
        rho_min=rho_min,
        rho_max=rho_max,
        s_p=s_p,
        schedule=schedule,
    )

    print(f"# eval schedule for dataset={args.dataset}")
    print(
        f"# total_steps={args.total_steps}  n_evals={args.n_evals}  "
        f"rho_min={rho_min}  rho_max={rho_max}  s_p={s_p}"
    )
    print(_format_table(schedule))
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
