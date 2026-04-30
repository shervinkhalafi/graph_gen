"""Option-C in-flight watcher — snapshot fetcher + decision-flowchart hint.

Per the spec extension (§11 In-flight watch loop): Claude periodically
polls W&B for in-flight runs, applies the literature-informed decision
flowchart, and decides keep / kill / extend-watch in prose. This script
is the *snapshot fetcher* — it pulls partial W&B history, applies the
flowchart, prints a recommendation. Claude reviews the recommendation,
writes the actual decision to ``progress.md`` (prose) and appends a
row to ``watches.jsonl`` (structured). The script does NOT modify
``watches.jsonl`` — that's intentional, to keep the audit trail clean.

There is no auto-kill. The watcher is informational.

Cadence and freshness
---------------------
The watcher runs every 15 minutes (Claude scheduling via
``ScheduleWakeup``). The freshness gate skips runs whose
``current_step`` has not advanced since their prior watch row, so a
re-run within the cadence window after no new gen-val cycle landed
produces zero new rows.

Decision flowchart
------------------
Cheapest / highest-confidence kills first, dynamics-side kills next,
phase-transition guard before saturation, relative-bottom-of-peers
last. See spec §11.B.4 for the mermaid + per-criterion citations.

#. ``diverged`` — NLL inf/NaN or jump > 50%; or NLL non-decreasing for
   5 consecutive validation cycles AND above the round's anchor NLL.
#. ``cost_cap`` — wall-time > 1.5x budget.
#. ``samples_degenerate`` — sample images degenerate for 3+ watches.
#. ``opt_health_confluence`` — every block at ``grad_cosine < 0.1``
   AND ``update_to_weight < 1e-4`` AND quality flat for 3 watches.
#. ``extend_watch`` — modularity rising or bits/edge descending; max
   one extension per run.
#. ``saturation_heuristic`` — saturation fit (≥ 8 points) predicts
   terminal within 3% of current value.
#. ``hyperband_relative`` — at ≥ 33% step_cap, in bottom ``1/eta``
   (eta = 3) of peers, stable for 3 watches.
#. otherwise ``keep``.

References for thresholds
-------------------------
- Saturation heuristic — Domhan-inspired; the 3% threshold is calibrated
  to the v1 noise floor (``std/mean ≈ 0.19`` per ``s_star.yaml``).
- Hyperband — Li et al. 2018, ``eta=3``; the rule here is gentler than
  canonical Hyperband (which kills bottom ``1 - 1/eta = 2/3``) — it
  kills only bottom ``1/eta`` and only after stability for 3 watches.
- Phase-transition guard — empirical observation in the v1 long-run
  that ``gen-val/sbm_accuracy`` is flat at chance for tens of thousands
  of steps before phase-transitioning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scripts.sweep._eval_manifest import (
    evals_lag as _evals_lag_from_rows,
)
from scripts.sweep._eval_manifest import (
    latest_status_per_step as _latest_status_per_step,
)
from scripts.sweep._eval_manifest import (
    read_manifest as _read_manifest,
)
from scripts.sweep.compute_s_star import (
    MIN_POINTS_FOR_FIT,
    fit_saturating_exponential,
)

# Watcher cadence (seconds). Claude calls ``ScheduleWakeup`` with
# ``delaySeconds`` near this value when in the in-flight watch loop.
WATCH_CADENCE_SECONDS = 900  # 15 minutes

# Hyperband-style ``eta`` parameter.
HYPERBAND_ETA = 3
# Saturation-heuristic threshold (fraction of current value).
SATURATION_TOLERANCE = 0.03
# Optimizer-health confluence thresholds.
GRAD_COSINE_DEAD = 0.1
UPDATE_TO_WEIGHT_DEAD = 1e-4
# Cost-cap wall-time multiplier.
COST_CAP_MULTIPLIER = 1.5

# Async-eval starvation thresholds (plan: compressed-tumbling-whale §7).
# When ``evals_lag`` (scheduled-and-passed minus completed evals) exceeds
# ``EVALS_LAG_BLOCKS_SATURATION``, the saturation-kill path is suppressed
# in favour of ``extend_watch`` — the saturation fit on starved data has
# too few completed-eval points to be defensible.
EVALS_LAG_BLOCKS_SATURATION = 3
# When the eval system is so starved we cannot judge quality, the
# watcher gives up: a high lag, at-most-one completed eval, AND the
# watcher has already extended several times. The thresholds are
# deliberately conservative so a single slow batch of evals does not
# trigger the kill.
EVAL_STARVATION_LAG_THRESHOLD = 5
EVAL_STARVATION_MAX_COMPLETED = 1
EVAL_STARVATION_MIN_WATCH_COUNT = 3


def read_watches(path: Path) -> list[dict[str, Any]]:
    """Read ``watches.jsonl``, skipping schema rows and blank lines.

    Missing file returns an empty list (not raised) — the watcher is
    expected to run before any rows have been written.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "schema":
            continue
        rows.append(obj)
    return rows


def find_running_launches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Launched rows from ``rounds.jsonl`` without a paired outcome.

    Single source of truth lives in ``fetch_outcomes.find_pending_launches``
    so the watcher and ``fetch_outcomes`` agree on what counts as still-
    running. The watcher historically had its own copy with the older
    uid-only pairing rule, which silently disagreed with fetch_outcomes
    after the relaunch-after-kill flow landed (uid-only marks the
    relaunch as already-paired by the cancel-outcome). Re-export the
    canonical implementation here so callers and tests do not have to
    track which module owns it.
    """
    from scripts.sweep.fetch_outcomes import find_pending_launches

    return find_pending_launches(rows)


def find_prior_watches(
    watches: list[dict[str, Any]], *, run_uid: str
) -> list[dict[str, Any]]:
    """Append-order subset of ``watches`` for a given run_uid."""
    return [w for w in watches if w.get("run_uid") == run_uid]


def freshness_gate_passes(
    *, current_step: int, prior_watches: list[dict[str, Any]]
) -> bool:
    """True if no prior watch or current_step strictly exceeds the latest prior."""
    if not prior_watches:
        return True
    latest_prior = prior_watches[-1].get("current_step", 0)
    return current_step > int(latest_prior)


def saturation_fit_partial(
    *, steps: np.ndarray, values: np.ndarray
) -> dict[str, float] | None:
    """Fit the partial gen-val/<quality> curve.

    Returns a dict with ``a, tau, c, q_infinity_hat, residual_std,
    n_points_used`` if the fit is well-defined; ``None`` if there are
    fewer than ``MIN_POINTS_FOR_FIT`` finite samples.

    The ``q_infinity_hat`` is the predicted asymptotic quality
    (``a + c``) — terminal value the metric will reach if the curve
    keeps following the saturating-exponential model.
    """
    s = np.asarray(steps, dtype=float)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    s = s[finite]
    v = v[finite]
    if s.size < MIN_POINTS_FOR_FIT:
        return None
    try:
        a, tau, c = fit_saturating_exponential(s, v)
    except Exception:
        # curve_fit failed (degenerate); refuse to invent a value.
        return None
    pred = a * (1.0 - np.exp(-s / tau)) + c
    residual_std = float(np.std(v - pred))
    return {
        "a": float(a),
        "tau": float(tau),
        "c": float(c),
        "q_infinity_hat": float(a + c),
        "residual_std": residual_std,
        "n_points_used": int(s.size),
    }


@dataclass
class FlowchartInput:
    """Inputs to ``apply_flowchart``.

    Each field is a clean numeric/categorical signal; the snapshot-
    fetching layer (``build_flowchart_input``) translates raw W&B
    history into this shape.
    """

    run_uid: str
    current_step: int
    step_cap: int
    anchor_nll: float
    wall_time_ratio: float
    nll_history: list[float]
    samples_degenerate_count: int
    grad_cosine_blocks: dict[str, float]
    update_to_weight_blocks: dict[str, float]
    quality_history: tuple[list[float], list[float]] | None
    quality_flat_count: int
    modularity_rising: bool
    bits_per_edge_descending: bool
    extension_count: int
    peer_quality_quantile: float | None
    peer_count: int
    fit: dict[str, float] | None = field(default=None)
    # Async-eval signals (None for legacy in-band runs without a manifest).
    evals_lag: int | None = field(default=None)
    completed_eval_count: int = field(default=0)
    watch_count: int = field(default=0)


def _is_diverged(inp: FlowchartInput) -> bool:
    nll = inp.nll_history
    if not nll:
        return False
    last = nll[-1]
    if math.isinf(last) or math.isnan(last):
        return True
    if len(nll) >= 2:
        prev = nll[-2]
        if (
            prev > 0
            and math.isfinite(prev)
            and math.isfinite(last)
            and last > 1.5 * prev
        ):
            return True
    # Stuck-at-init: NLL non-decreasing for 5 cycles AND above anchor.
    if len(nll) >= 5:
        last5 = nll[-5:]
        if (
            all(math.isfinite(v) for v in last5)
            and all(last5[i + 1] >= last5[i] - 1e-9 for i in range(4))
            and last > inp.anchor_nll
        ):
            return True
    return False


def _all_blocks_dead(inp: FlowchartInput) -> bool:
    if not inp.grad_cosine_blocks or not inp.update_to_weight_blocks:
        return False
    cosine_dead = all(v < GRAD_COSINE_DEAD for v in inp.grad_cosine_blocks.values())
    update_dead = all(
        v < UPDATE_TO_WEIGHT_DEAD for v in inp.update_to_weight_blocks.values()
    )
    return cosine_dead and update_dead


def apply_flowchart(inp: FlowchartInput) -> dict[str, Any]:
    """Apply the decision flowchart, return ``{"recommendation": ..., "reason": ...}``.

    The recommendation is one of ``keep``, ``kill``, ``extend_watch``.
    For ``kill``, ``reason`` is one of ``diverged``, ``cost_cap``,
    ``samples_degenerate``, ``opt_health_confluence``,
    ``saturation_heuristic``, ``hyperband_relative``,
    ``eval_starvation``. For other recommendations, ``reason`` is
    ``None``.

    Order is fixed: cheapest / highest-confidence kills first, then
    dynamics-side kills, then phase-transition guard, then the async-
    eval guards (saturation broadening + starvation kill), then
    saturation, then relative.

    Async-eval interaction (plan: compressed-tumbling-whale §7)
    -----------------------------------------------------------
    When ``evals_lag`` is known (i.e. the run has a manifest), two
    branches engage. First, ``evals_lag > EVALS_LAG_BLOCKS_SATURATION``
    suppresses the saturation-kill path: a saturation fit on starved
    data has too few completed-eval points to defensibly mark
    saturation. Second, the dedicated ``kill: eval_starvation`` branch
    fires when the eval system is broken AND we cannot tell quality —
    the run has been watched several cycles, has at most one completed
    eval, and the lag has grown past the threshold. Both branches
    require a non-``None`` ``evals_lag``: legacy in-band runs without a
    manifest skip them and follow the original flowchart unchanged.
    """
    if _is_diverged(inp):
        return {"recommendation": "kill", "reason": "diverged"}
    if inp.wall_time_ratio > COST_CAP_MULTIPLIER:
        return {"recommendation": "kill", "reason": "cost_cap"}
    if inp.samples_degenerate_count >= 3:
        return {"recommendation": "kill", "reason": "samples_degenerate"}
    if _all_blocks_dead(inp) and inp.quality_flat_count >= 3:
        return {"recommendation": "kill", "reason": "opt_health_confluence"}
    # Eval-starvation kill: the eval system is broken AND we cannot
    # tell quality. Conservative thresholds so a single slow batch of
    # evals does not trigger.
    if (
        inp.evals_lag is not None
        and inp.evals_lag >= EVAL_STARVATION_LAG_THRESHOLD
        and inp.completed_eval_count <= EVAL_STARVATION_MAX_COMPLETED
        and inp.watch_count >= EVAL_STARVATION_MIN_WATCH_COUNT
    ):
        return {"recommendation": "kill", "reason": "eval_starvation"}
    if (
        inp.modularity_rising or inp.bits_per_edge_descending
    ) and inp.extension_count < 1:
        return {"recommendation": "extend_watch", "reason": None}
    # Saturation heuristic. If async-eval lag exceeds the threshold,
    # broaden to ``extend_watch`` rather than kill — the fit cannot be
    # trusted with that many missing eval points.
    fit = inp.fit
    if fit is None and inp.quality_history is not None:
        steps_list, values_list = inp.quality_history
        fit = saturation_fit_partial(
            steps=np.asarray(steps_list, dtype=float),
            values=np.asarray(values_list, dtype=float),
        )
    if fit is not None and inp.quality_history is not None:
        _, values_list = inp.quality_history
        if values_list:
            current_value = float(values_list[-1])
            if current_value > 0.0:
                gap = abs(fit["q_infinity_hat"] - current_value) / current_value
                if gap < SATURATION_TOLERANCE:
                    if (
                        inp.evals_lag is not None
                        and inp.evals_lag > EVALS_LAG_BLOCKS_SATURATION
                    ):
                        # Starved evals block saturation kill — broaden
                        # to extend_watch so we accumulate more eval
                        # points before deciding.
                        return {"recommendation": "extend_watch", "reason": None}
                    return {
                        "recommendation": "kill",
                        "reason": "saturation_heuristic",
                    }
    # Hyperband-style relative kill.
    at_late_stage = inp.current_step >= inp.step_cap // 3
    bottom_quantile = (
        inp.peer_quality_quantile is not None
        and inp.peer_count > 0
        and inp.peer_quality_quantile <= 1.0 / HYPERBAND_ETA
    )
    if at_late_stage and bottom_quantile and inp.quality_flat_count >= 3:
        return {"recommendation": "kill", "reason": "hyperband_relative"}
    return {"recommendation": "keep", "reason": None}


def compute_evals_lag_from_manifest(
    *,
    manifest_path: Path | None,
    schedule: list[int],
    current_step: int,
) -> int | None:
    """Compute ``evals_lag`` from a per-run async-eval manifest.

    Returns ``None`` for legacy in-band runs that have no manifest
    (either ``manifest_path`` is ``None`` or the file does not exist).
    A ``None`` lag tells the flowchart to skip the async-eval branches
    so existing behaviour is preserved.

    Parameters
    ----------
    manifest_path
        Path to ``eval_manifest.jsonl`` for the run. ``None`` for runs
        that never opted into async-eval.
    schedule
        Integer training-step schedule the trainer was given. Empty
        schedule yields ``0`` lag (nothing to be late on).
    current_step
        The trainer's current ``global_step``. Schedule entries past
        this step are not yet expected.
    """
    if manifest_path is None or not manifest_path.exists():
        return None
    rows = _read_manifest(manifest_path)
    return _evals_lag_from_rows(rows, schedule, current_step)


def count_completed_evals_from_manifest(
    *,
    manifest_path: Path | None,
) -> int:
    """Count distinct ``scheduled_step``s with a terminal ``completed`` row.

    Returns ``0`` when the manifest does not exist (legacy in-band).
    Used by the eval-starvation branch alongside ``evals_lag`` and the
    watcher's ``watch_count``.
    """
    if manifest_path is None or not manifest_path.exists():
        return 0
    rows = _read_manifest(manifest_path)
    latest = _latest_status_per_step(rows)
    return sum(1 for r in latest.values() if r.get("status") == "completed")


def build_recommendation(
    *,
    run_uid: str,
    flowchart_input: FlowchartInput,
    observed_metrics: dict[str, float],
    observed_diagnostics: dict[str, dict[str, float]],
    snapshot_sha256: str,
    previous_decision_ref: str | None,
) -> dict[str, Any]:
    """Compose the recommendation row that gets printed for Claude to review.

    The schema mirrors ``watches.jsonl`` so the operator can copy the
    relevant fields verbatim into the watch entry. The ``kind`` here is
    ``watch_recommendation`` (advisory) so it cannot be confused with a
    ``watch`` row that Claude actually committed.

    The ``evals_lag`` field is propagated from the flowchart input so
    downstream readers (and Claude's review) can see the async-eval
    state at-a-glance. ``None`` means a legacy in-band run with no
    manifest — the saturation broadening/starvation-kill branches did
    not engage.
    """
    decision = apply_flowchart(flowchart_input)
    return {
        "kind": "watch_recommendation",
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "run_uid": run_uid,
        "current_step": flowchart_input.current_step,
        "step_cap": flowchart_input.step_cap,
        "previous_decision_ref": previous_decision_ref,
        "snapshot_sha256": snapshot_sha256,
        "observed_metrics": observed_metrics,
        "observed_diagnostics": observed_diagnostics,
        "recommendation": decision["recommendation"],
        "reason": decision["reason"],
        "fit_diagnostics": flowchart_input.fit,
        "evals_lag": flowchart_input.evals_lag,
    }


def hash_snapshot(payload: dict[str, Any]) -> str:
    """Deterministic SHA-256 of the JSON-serialised snapshot."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def fetch_snapshot_from_wandb(  # pragma: no cover — wraps live W&B API
    *,
    entity: str,
    project: str,
    run_name: str,
    last_k_validations: int = 30,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Pull a partial run snapshot from W&B.

    Returns ``(summary, history_subset, sample_image_records)``. Kept
    out of the unit-tested surface — exercised by the first real
    in-flight watch loop. Prefer ``build_flowchart_input`` for unit
    testing.
    """
    import wandb  # local import; only needed at runtime

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    run_list = list(runs)
    if not run_list:
        raise LookupError(f"no W&B run named {run_name!r} in {entity}/{project}")
    if len(run_list) > 1:
        raise LookupError(f"ambiguous: {len(run_list)} W&B runs named {run_name!r}")
    run = run_list[0]
    summary = dict(run.summary)
    history_keys = [
        "val/epoch_NLL",
        "gen-val/sbm_accuracy",
        "gen-val/modularity_q",
        "gen-val/degree_mmd",
        "gen-val/clustering_mmd",
        "diagnostics-val/progress/bits_per_edge",
    ]
    raw_history = list(run.scan_history(keys=["_step", *history_keys], page_size=200))
    # Truncate to last k validation cycles (those with any gen-val/* key).
    val_records = [r for r in raw_history if any(k.startswith("gen-val/") for k in r)]
    history_subset = val_records[-last_k_validations:]
    return summary, dict(run.summary), history_subset


def write_recommendation_stdout(rec: dict[str, Any]) -> None:
    """Pretty-print a recommendation for Claude (or human) review."""
    sep = "-" * 72
    print(sep)
    print(f"WATCH SNAPSHOT  run={rec['run_uid']}  step={rec['current_step']}")
    print(f"  recommendation: {rec['recommendation']}")
    print(f"  reason:         {rec.get('reason')}")
    print(f"  snapshot_sha:   {rec['snapshot_sha256'][:16]}…")
    if rec.get("fit_diagnostics"):
        fit = rec["fit_diagnostics"]
        print(
            f"  fit:            tau={fit.get('tau'):.0f} "
            f"q_inf={fit.get('q_infinity_hat'):.4f} "
            f"n_points={fit.get('n_points_used')}"
        )
    print(f"  observed:       {rec['observed_metrics']}")
    print(sep)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    _ = p.add_argument(
        "--watches-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/watches.jsonl"),
    )
    _ = p.add_argument(
        "--entity",
        default="graph_denoise_team",
    )
    _ = p.add_argument(
        "--project",
        default="tmgg-smallest-config-sweep",
    )
    _ = p.add_argument(
        "--last-k-validations",
        type=int,
        default=30,
        help="How many trailing validation cycles to fetch per run.",
    )
    _ = p.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help=(
            "Root of the per-run outputs tree (mirrors the Modal "
            "tmgg-outputs volume layout: ``<root>/<run_id>/eval_manifest.jsonl``). "
            "When set, the watcher reads each run's manifest to compute "
            "``evals_lag``. Legacy in-band runs without a manifest are "
            "treated as ``evals_lag=None``."
        ),
    )
    return p.parse_args()


def _manifest_path_for_run(*, outputs_root: Path | None, run_uid: str) -> Path | None:
    """Derive ``<outputs_root>/<run_uid>/eval_manifest.jsonl``.

    Returns ``None`` when ``outputs_root`` is unset; callers then treat
    the run as a legacy in-band run (no manifest).
    """
    if outputs_root is None:
        return None
    return outputs_root / run_uid / "eval_manifest.jsonl"


def _read_rounds(rounds_jsonl: Path) -> list[dict[str, Any]]:
    if not rounds_jsonl.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in rounds_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "schema":
            continue
        rows.append(obj)
    return rows


def main() -> None:  # pragma: no cover — CLI driver, exercised by integration
    args = _parse_args()
    rounds = _read_rounds(args.rounds_jsonl)
    watches = read_watches(args.watches_jsonl)
    running = find_running_launches(rounds)
    print(f"# {len(running)} running launches found in {args.rounds_jsonl}")
    for launched in running:
        run_uid = launched["run_uid"]
        prior = find_prior_watches(watches, run_uid=run_uid)
        try:
            summary, _, history = fetch_snapshot_from_wandb(
                entity=args.entity,
                project=args.project,
                run_name=run_uid,
                last_k_validations=args.last_k_validations,
            )
        except LookupError as exc:
            print(f"# SKIP {run_uid}: {exc}")
            continue
        current_step = int(summary.get("_step", 0) or 0)
        if not freshness_gate_passes(current_step=current_step, prior_watches=prior):
            print(f"# SKIP {run_uid}: freshness gate (step={current_step} unchanged)")
            continue
        # Build the flowchart input from the snapshot. The full
        # translation is left to the operator-side review for the
        # first round; the CLI here prints the raw snapshot and the
        # flowchart recommendation against a minimal input.
        nll_history: list[float] = [
            float(r["val/epoch_NLL"]) for r in history if "val/epoch_NLL" in r
        ]
        quality_steps: list[float] = [
            float(r["_step"]) for r in history if "gen-val/sbm_accuracy" in r
        ]
        quality_values: list[float] = [
            float(r["gen-val/sbm_accuracy"])
            for r in history
            if "gen-val/sbm_accuracy" in r
        ]
        quality_history = (quality_steps, quality_values) if quality_steps else None
        fit = (
            saturation_fit_partial(
                steps=np.asarray(quality_steps, dtype=float),
                values=np.asarray(quality_values, dtype=float),
            )
            if quality_history is not None
            else None
        )
        # Read the per-run async-eval manifest. ``manifest_path`` is
        # ``None`` when ``--outputs-root`` is unset or when the run
        # never opted into async-eval; both paths yield ``evals_lag =
        # None`` and skip the async-eval branches of the flowchart.
        manifest_path = _manifest_path_for_run(
            outputs_root=args.outputs_root, run_uid=run_uid
        )
        eval_schedule_raw = launched.get("eval_schedule") or []
        eval_schedule: list[int] = [int(s) for s in eval_schedule_raw]
        evals_lag_value = compute_evals_lag_from_manifest(
            manifest_path=manifest_path,
            schedule=eval_schedule,
            current_step=current_step,
        )
        completed_eval_count = count_completed_evals_from_manifest(
            manifest_path=manifest_path
        )
        flowchart_input = FlowchartInput(
            run_uid=run_uid,
            current_step=current_step,
            step_cap=int(launched.get("step_cap", 100000)),
            anchor_nll=float(summary.get("anchor_nll", 1000.0)),  # placeholder
            wall_time_ratio=0.5,  # operator updates from Modal billing
            nll_history=nll_history,
            samples_degenerate_count=0,  # operator updates from sample images
            grad_cosine_blocks={},  # populated from summary in operator review
            update_to_weight_blocks={},
            quality_history=quality_history,
            quality_flat_count=0,
            modularity_rising=False,
            bits_per_edge_descending=False,
            extension_count=len(prior),
            peer_quality_quantile=None,
            peer_count=0,
            fit=fit,
            evals_lag=evals_lag_value,
            completed_eval_count=completed_eval_count,
            watch_count=len(prior),
        )
        snapshot_payload = {"summary": summary, "history": history}
        snapshot_sha = hash_snapshot(snapshot_payload)
        rec = build_recommendation(
            run_uid=run_uid,
            flowchart_input=flowchart_input,
            observed_metrics={
                k: float(v)
                for k, v in summary.items()
                if isinstance(v, int | float) and k.startswith("gen-val/")
            },
            observed_diagnostics={},
            snapshot_sha256=snapshot_sha,
            previous_decision_ref=prior[-1]["run_uid"] if prior else None,
        )
        write_recommendation_stdout(rec)


if __name__ == "__main__":  # pragma: no cover
    main()
