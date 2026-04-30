"""Run all three saturation-fit models on the active sweep and log them.

Diagnostic-only CLI. Pulls the live W&B history for every running
launch, runs ``compare_fits`` on each gen-val metric, prints a
side-by-side comparison, and appends a structured row to
``saturation_fits.jsonl``. Does NOT touch the watcher's flowchart or
any kill/keep decision.

Use for "what does the saturation surface actually look like for this
sweep" — we want to know whether Domhan, logistic, or Beta-Binomial
gives the most stable + meaningful asymptote prediction across the
metrics we actually threshold against. Once we have a few dozen rows,
we'll know which model to ship into the watcher (a separate decision).

Invocation::

    doppler run -- uv run python -m scripts.sweep.diag_saturation_fits

Per-metric proportionality classification (``is_proportion`` arg to
Beta-Binomial):

* Proportions (in [0,1], drawn from N=eval_num_samples trials):
  ``sbm_accuracy``, ``planarity_accuracy``, ``uniqueness``,
  ``modularity_q`` (modularity is bounded but technically continuous,
  not a proportion — exclude unless we wire a special prior).
* Non-proportions (continuous, unbounded above): ``degree_mmd``,
  ``clustering_mmd``, ``orbit_mmd``, ``spectral_mmd``,
  ``full_chain_vlb``, ``mean_step_kl``, ``empirical_p_in``,
  ``empirical_p_out``.

For non-proportion MMDs we pre-transform the value so the fit targets
a *quality* curve (monotone non-decreasing as model learns):
``q(s) = max(0, 1 - mmd(s) / mmd_anchor)``. The asymptote is then a
fraction of "perfect" relative to the anchor target. This keeps the
three models comparable across all gen-val keys.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from scripts.sweep.fetch_outcomes import find_pending_launches, read_rounds
from scripts.sweep.saturation_fits import compare_fits

# Gen-val keys we fit, with per-key flags for the Beta-Binomial branch.
PROPORTION_KEYS: dict[str, bool] = {
    "gen-val/sbm_accuracy": True,
    "gen-val/planarity_accuracy": True,
    "gen-val/uniqueness": True,
    "gen-val/modularity_q": False,  # bounded but continuous; not a Bernoulli
    "gen-val/degree_mmd": False,
    "gen-val/clustering_mmd": False,
    "gen-val/orbit_mmd": False,
    "gen-val/spectral_mmd": False,
}

# For MMD-style metrics we transform raw value -> quality fraction
# q = max(0, 1 - mmd / anchor) so all three fits target a monotone-up
# quality curve. Anchors below mirror the conservative ceilings used
# in fetch_outcomes.check_run; missing entries fall back to fitting
# the raw MMD (asymptote then has units of MMD, lower is better).
MMD_ANCHOR_BY_DATASET_AND_METRIC: dict[str, dict[str, float]] = {
    "spectre_sbm": {
        "gen-val/degree_mmd": 0.0013,
        "gen-val/clustering_mmd": 0.0498,
        "gen-val/orbit_mmd": 0.0433,
        "gen-val/spectral_mmd": 0.01,
    },
    "pyg_enzymes": {
        "gen-val/degree_mmd": 0.004,
        "gen-val/clustering_mmd": 0.083,
        "gen-val/orbit_mmd": 0.002,
        # spectral_mmd anchor is path-D in-house; not pinned yet.
    },
}


def _quality_from_mmd(mmd: np.ndarray, anchor: float) -> np.ndarray:
    """Convert a raw MMD series to a quality fraction in [0, 1]."""
    return np.clip(1.0 - mmd / max(anchor, 1e-12), 0.0, 1.0)


def fetch_history_for_run(
    *, entity: str, project: str, run_uid: str
) -> list[dict[str, Any]]:
    """Pull all gen-val history rows from W&B; reuse the watcher's split-query trick.

    Returns rows ordered by ``trainer/global_step`` ascending. Each row
    carries whatever gen-val/* keys the eval worker logged at that
    step plus ``trainer/global_step``.
    """
    import wandb  # local import; only needed at runtime

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_uid})
    run_list = list(runs)
    if not run_list:
        return []
    run_list.sort(key=lambda r: getattr(r, "created_at", "") or "", reverse=True)
    run = run_list[0]
    rows = list(
        run.scan_history(
            keys=[
                "trainer/global_step",
                "_step",
                *PROPORTION_KEYS.keys(),
            ],
            page_size=200,
        )
    )
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        if r is None:
            continue
        step_raw = r.get("trainer/global_step")
        if step_raw is None:
            continue
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        merged = by_step.setdefault(step, {})
        for k, v in r.items():
            if v is not None:
                merged[k] = v
    return [merged for _step, merged in sorted(by_step.items())]


def fits_for_run(
    *,
    history: list[dict[str, Any]],
    dataset: str,
    n_trials_per_step: int = 40,
) -> dict[str, dict[str, Any]]:
    """Run compare_fits on every gen-val metric we care about."""
    if not history:
        return {}
    steps = np.asarray(
        [int(r.get("trainer/global_step", 0)) for r in history], dtype=float
    )
    out: dict[str, dict[str, Any]] = {}
    anchors_for_dataset = MMD_ANCHOR_BY_DATASET_AND_METRIC.get(dataset, {})
    for key, is_prop in PROPORTION_KEYS.items():
        raw = np.asarray([float(r.get(key, np.nan)) for r in history], dtype=float)
        finite = np.isfinite(raw)
        if finite.sum() < 1:
            out[key] = {
                "raw_or_quality": "raw",
                "n_finite": 0,
                "fits": {
                    "domhan": {"status": "no_finite_observations"},
                    "logistic": {"status": "no_finite_observations"},
                    "beta_binomial": {"status": "no_finite_observations"},
                },
            }
            continue
        finite_steps = steps[finite]
        finite_values = raw[finite]
        anchor = anchors_for_dataset.get(key)
        if anchor is not None and not is_prop:
            # MMD path: transform to quality so all three fits are
            # comparable on a [0,1] scale.
            q = _quality_from_mmd(finite_values, anchor)
            fits = compare_fits(
                finite_steps,
                q,
                is_proportion=True,
                n_trials_per_step=n_trials_per_step,
            )
            out[key] = {
                "raw_or_quality": "quality",
                "anchor": anchor,
                "n_finite": int(finite.sum()),
                "min_step": float(finite_steps.min()),
                "max_step": float(finite_steps.max()),
                "fits": fits,
            }
        else:
            fits = compare_fits(
                finite_steps,
                finite_values,
                is_proportion=is_prop,
                n_trials_per_step=n_trials_per_step,
            )
            out[key] = {
                "raw_or_quality": "raw",
                "n_finite": int(finite.sum()),
                "min_step": float(finite_steps.min()),
                "max_step": float(finite_steps.max()),
                "fits": fits,
            }
    return out


def write_log_row(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def pretty_print_fits(
    run_uid: str, dataset: str, by_metric: dict[str, dict[str, Any]]
) -> None:
    sep = "-" * 72
    print(sep)
    print(f"SAT-FITS  run={run_uid}  dataset={dataset}")
    for key, entry in by_metric.items():
        kind = entry.get("raw_or_quality", "raw")
        n = entry.get("n_finite", 0)
        smin = entry.get("min_step")
        smax = entry.get("max_step")
        print(
            f"  {key} [{kind}, n_finite={n}, "
            f"steps={int(smin) if smin is not None else '?'}..{int(smax) if smax is not None else '?'}]"
        )
        for model_name in ("domhan", "logistic", "beta_binomial"):
            fit = entry["fits"].get(model_name, {})
            status = fit.get("status", "?")
            q_inf = fit.get("q_inf")
            sigma = fit.get("sigma_q_inf")
            ci_lo = fit.get("ci95_low")
            ci_hi = fit.get("ci95_high")
            extras: list[str] = []
            if sigma is not None:
                extras.append(f"sigma={sigma:.3f}")
            if ci_lo is not None and ci_hi is not None:
                extras.append(f"95%CI=[{ci_lo:.3f},{ci_hi:.3f}]")
            extras_str = " ".join(extras)
            q_inf_str = (
                f"q_inf={q_inf:.4f}" if isinstance(q_inf, float) else f"q_inf={q_inf}"
            )
            print(f"    {model_name:>14}: status={status:<22} {q_inf_str} {extras_str}")
    print(sep)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    _ = p.add_argument(
        "--log-jsonl",
        type=Path,
        default=Path(
            "docs/experiments/sweep/smallest-config-2026-04-29/saturation_fits.jsonl"
        ),
    )
    _ = p.add_argument("--entity", default="graph_denoise_team")
    _ = p.add_argument("--project", default="tmgg-smallest-config-sweep")
    _ = p.add_argument("--n-trials-per-step", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = read_rounds(args.rounds_jsonl)
    pending = find_pending_launches(rows)
    print(f"# {len(pending)} running launches found in {args.rounds_jsonl}")
    ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for launched in pending:
        run_uid = launched["run_uid"]
        dataset = launched.get("dataset", "?")
        try:
            history = fetch_history_for_run(
                entity=args.entity, project=args.project, run_uid=run_uid
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic CLI, broad-catch is OK
            print(f"# SKIP {run_uid}: {type(exc).__name__}: {exc}")
            continue
        by_metric = fits_for_run(
            history=history,
            dataset=dataset,
            n_trials_per_step=args.n_trials_per_step,
        )
        pretty_print_fits(run_uid, dataset, by_metric)
        log_row = {
            "kind": "saturation_fit_diagnostic",
            "ts_utc": ts,
            "run_uid": run_uid,
            "dataset": dataset,
            "n_history_rows": len(history),
            "by_metric": by_metric,
        }
        write_log_row(args.log_jsonl, log_row)


if __name__ == "__main__":  # pragma: no cover
    main()
