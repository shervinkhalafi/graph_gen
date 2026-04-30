"""Fetch outcomes for launched runs and append outcome rows to rounds.jsonl.

Per spec §4.3: outcome rows reference the launched row by ``run_uid``.
This script:
  1. Reads rounds.jsonl, finds launched rows lacking a paired outcome.
  2. For each, queries W&B by run name (``run_uid``) within the
     pinned project (tmgg-smallest-config-sweep).
  3. Pulls terminal summary metrics + opt-health diagnostics scalars.
  4. Applies check_threshold.check_run.
  5. Appends an outcome row.

Per spec §7 invariant 2: refuses to write threshold_pass=true if any
required metric is missing — raises and aborts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import wandb
from scripts.sweep._eval_manifest import evals_completeness, read_manifest
from scripts.sweep.check_threshold import (
    check_run,  # MetricMissingError / AnchorMissingError propagate from check_run per §7 invariant 2
)
from wandb.apis.public.runs import Run as WandbRun

REQUIRED_DIAGNOSTIC_KEYS = (
    "diagnostics-train/opt-health/grad_snr",
    "diagnostics-train/opt-health/grad_cosine",
    "diagnostics-train/opt-health/update_to_weight",
)

# History keys pulled when a manifest is present. The trainer registers
# ``trainer/global_step`` as the custom step axis via ``define_metric``,
# so we pull it explicitly to key the per-step dict. The four anchored
# gen-val metrics span every dataset's threshold rule (sbm_accuracy is
# only checked for ``spectre_sbm`` but pulling it costs nothing).
HISTORY_KEYS = (
    "trainer/global_step",
    "gen-val/sbm_accuracy",
    "gen-val/degree_mmd",
    "gen-val/clustering_mmd",
    "gen-val/orbit_mmd",
    "gen-val/spectral_mmd",
    "gen-val/modularity_q",
)

# Metrics required to consider a step "fully anchored" — i.e. usable as
# the terminal-step source for the threshold check. We deliberately do
# not require ``sbm_accuracy`` here: non-SBM datasets do not log it, so
# requiring it would forbid the terminal-step source on every other
# dataset. ``check_run`` separately enforces dataset-specific anchors.
ANCHORED_GENVAL_KEYS = (
    "gen-val/degree_mmd",
    "gen-val/clustering_mmd",
    "gen-val/orbit_mmd",
)


def read_rounds(rounds_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in rounds_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "schema":
            continue
        rows.append(obj)
    return rows


def find_pending_launches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Launched rows that have no matching outcome row.

    Pairing is two-layered:

    1. **Direct ID match.** If a launched row has a
       ``modal_function_call_id`` and an outcome row exists with the same
       value, the launched row is paired regardless of ts. This is the
       authoritative case — the outcome explicitly identifies which Modal
       call it refers to.
    2. **Timestamp fallback per uid.** For launched rows without a
       ts-after-everything outcome match by call_id, fall back to the
       per-uid latest-outcome timestamp: the launched row is pending iff
       no outcome with the same ``run_uid`` has ``ts_utc >= launched ts``.

    Layer 1 covers the operator-cancel-then-relaunch sequence where the
    cancel-outcome may be appended after the relaunch (e.g. round-2
    relaunch on 2026-04-30: cancel-outcomes for the killed pods landed
    in rounds.jsonl after the relaunched-launched rows, with a wall-time
    gap of seconds between the launch and the bookkeeping write — so
    layer 2 alone would mark the relaunches as paired). Layer 2 covers
    the abandoned-smoke / legacy-launch cases where no call_id was
    captured.
    """
    outcome_call_ids: set[str] = {
        cid
        for cid in (
            r.get("modal_function_call_id") for r in rows if r.get("kind") == "outcome"
        )
        if cid
    }
    # Layer 2 considers only call_id-less outcomes — call_id-bearing
    # outcomes are interpreted strictly via layer 1, so a cancel-outcome
    # written *after* a relaunch (with the killed pod's call_id) does
    # not shadow the relaunch via ts ordering.
    latest_outcome_ts: dict[str, str] = {}
    for r in rows:
        if r.get("kind") != "outcome":
            continue
        if r.get("modal_function_call_id"):
            continue
        ts = r.get("ts_utc", "")
        prev = latest_outcome_ts.get(r["run_uid"], "")
        if ts > prev:
            latest_outcome_ts[r["run_uid"]] = ts
    pending: list[dict[str, Any]] = []
    for r in rows:
        if r.get("kind") != "launched":
            continue
        # Layer 1: direct call_id match (authoritative).
        call_id = r.get("modal_function_call_id")
        if call_id and call_id in outcome_call_ids:
            continue
        # Layer 2: uid + latest-outcome-ts fallback (call_id-less outcomes only).
        outcome_ts = latest_outcome_ts.get(r["run_uid"])
        if outcome_ts is None or outcome_ts < r.get("ts_utc", ""):
            pending.append(r)
    return pending


def resolve_wandb_run(*, entity: str, project: str, run_name: str) -> WandbRun:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    run_list = list(runs)
    if len(run_list) == 0:
        raise LookupError(f"no W&B run named {run_name!r} in {entity}/{project}")
    if len(run_list) > 1:
        raise LookupError(
            f"ambiguous: {len(run_list)} W&B runs named {run_name!r} in {entity}/{project}"
        )
    return run_list[0]


def fetch_block_keyed_diagnostic(run: WandbRun, key_prefix: str) -> dict[str, float]:
    """Pull terminal block-keyed scalars matching ``<key_prefix>/<block>``.

    W&B summary stores logged scalars under their full key (with ``/``
    separators). We pick out keys starting with ``<key_prefix>/`` and
    return ``{<block>: <value>}``.
    """
    out: dict[str, float] = {}
    for k, v in run.summary.items():
        if not isinstance(k, str):
            continue
        if not k.startswith(key_prefix + "/"):
            continue
        block = k[len(key_prefix) + 1 :]
        try:
            out[block] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _scan_history_per_step(
    run: WandbRun,
) -> dict[int, dict[str, float]]:
    """Pull gen-val metrics from W&B history, keyed by ``trainer/global_step``.

    Async-eval logs gen-val/* without an explicit ``step=`` kwarg; the
    custom step axis ``trainer/global_step`` (registered by the trainer
    via ``define_metric``) routes them into history rows. We pull those
    rows here and re-key them by integer step so callers can pick the
    terminal step or build a saturation-fit series.
    """
    metrics_per_step: dict[int, dict[str, float]] = {}
    for hrow in run.scan_history(keys=list(HISTORY_KEYS)):
        if hrow is None:
            continue
        raw_step = hrow.get("trainer/global_step")
        if raw_step is None:
            continue
        step = int(raw_step)
        record: dict[str, float] = {}
        for key in HISTORY_KEYS:
            if key == "trainer/global_step":
                continue
            value = hrow.get(key)
            if isinstance(value, int | float):
                record[key] = float(value)
        if record:
            metrics_per_step[step] = record
    return metrics_per_step


def _terminal_step_metrics(
    metrics_per_step: dict[int, dict[str, float]],
) -> tuple[int | None, dict[str, float]]:
    """Pick the highest step where every anchored gen-val key is present.

    Returns ``(step, metrics)`` or ``(None, {})`` if no step is fully
    anchored. The threshold check runs against the returned ``metrics``;
    ``check_run`` will raise ``MetricMissingError`` if a dataset-specific
    anchor (e.g. ``sbm_accuracy`` for SBM) is still absent.
    """
    fully_anchored = [
        step
        for step, record in metrics_per_step.items()
        if all(k in record for k in ANCHORED_GENVAL_KEYS)
    ]
    if not fully_anchored:
        return None, {}
    terminal = max(fully_anchored)
    return terminal, dict(metrics_per_step[terminal])


def build_outcome_row(
    *,
    launched: dict[str, Any],
    run: WandbRun,
    anchors_path: Path,
    manifest_path: Path | None = None,
    expected_schedule: list[int] | None = None,
) -> dict[str, Any]:
    """Build an outcome row for ``launched`` from a finished W&B ``run``.

    Two branches:

    1. **Legacy (manifest_path is None).** Read terminal metrics from
       ``run.summary`` and apply ``check_run`` directly. This is the
       in-band-eval path; ``metrics_per_step``, ``evals_completeness``,
       and ``gate_reason`` are all ``None``.
    2. **Async-eval (manifest_path provided).** Read the JSONL manifest
       to get the eval-worker view of completeness. Pull W&B history
       keyed by ``trainer/global_step`` so the per-step metrics are
       available for the saturation fit. Apply ``check_run`` to the
       **terminal-step** history values (not the summary, which can hold
       stale or unrelated values). Refuse ``threshold_pass=true`` if any
       expected scheduled step is missing both from the manifest's
       terminal status and from W&B history.
    """
    summary = dict(run.summary)
    metrics_subset = {
        k: v
        for k, v in summary.items()
        if k.startswith("gen-val/")
        or k == "val/epoch_NLL"
        or k == "diagnostics-val/progress/bits_per_edge"
    }

    if run.state in ("crashed", "killed", "failed"):
        return {
            "kind": "outcome",
            "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "run_uid": launched["run_uid"],
            "wandb_run_id": run.id,
            "status": "failed",
            "failure_kind": _classify_failure(run),
            "last_logged_step": int(summary.get("_step", 0) or 0),
        }

    use_manifest = manifest_path is not None

    metrics_per_step: dict[int, dict[str, float]] | None = None
    completeness: dict[str, int | list[int]] | None = None
    gate_reason: str | None = None
    terminal_step_from_history: int | None = None

    if use_manifest:
        assert manifest_path is not None  # narrow for type-checker
        manifest_rows = read_manifest(manifest_path)
        schedule = list(expected_schedule or [])
        completeness = evals_completeness(manifest_rows, schedule)
        metrics_per_step = _scan_history_per_step(run)
        terminal_step_from_history, terminal_metrics = _terminal_step_metrics(
            metrics_per_step
        )
        # Async-eval: replace summary-derived metrics with the
        # terminal-step history values. The summary may still hold a
        # stale gen-val key from an earlier in-band path or from a
        # manual log; history is authoritative for per-step gen-val.
        threshold_metrics_source: dict[str, float] = dict(terminal_metrics)
    else:
        threshold_metrics_source = {
            k: float(v) for k, v in metrics_subset.items() if isinstance(v, int | float)
        }

    # Decide whether expected evals are missing AND not in history.
    expected_evals_missing = False
    if use_manifest and completeness is not None:
        # ``evals_completeness`` returns a union-typed dict; the
        # "missing" entry is always a list[int] by construction. Narrow
        # explicitly so basedpyright accepts the iteration.
        missing_value = completeness["missing"]
        assert isinstance(missing_value, list)
        manifest_missing: list[int] = list(missing_value)
        history_steps = (
            set(metrics_per_step.keys()) if metrics_per_step is not None else set()
        )
        truly_missing = [s for s in manifest_missing if s not in history_steps]
        if truly_missing:
            expected_evals_missing = True

    breakdown = check_run(
        metrics=threshold_metrics_source,
        dataset=launched["dataset"],
        anchors_path=anchors_path,
    )

    threshold_pass = bool(breakdown["pass"])
    if expected_evals_missing:
        threshold_pass = False
        gate_reason = "expected_evals_missing"

    diag_terminal = {
        prefix.rsplit("/", 1)[-1]: fetch_block_keyed_diagnostic(run, prefix)
        for prefix in REQUIRED_DIAGNOSTIC_KEYS
    }

    if use_manifest and terminal_step_from_history is not None:
        terminal_step = terminal_step_from_history
    else:
        terminal_step = int(summary.get("_step", launched["step_cap"]))

    if use_manifest:
        # Emit the terminal-step metrics as the canonical "metrics"
        # field — saturation fits and downstream readers want these,
        # not the summary view.
        out_metrics: dict[str, float] = dict(threshold_metrics_source)
    else:
        out_metrics = {
            k: float(v) for k, v in metrics_subset.items() if isinstance(v, int | float)
        }

    row: dict[str, Any] = {
        "kind": "outcome",
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "run_uid": launched["run_uid"],
        "wandb_run_id": run.id,
        "status": "finished",
        "terminal_step": terminal_step,
        "metrics": out_metrics,
        "diagnostics_terminal": diag_terminal,
        "threshold_pass": threshold_pass,
        "threshold_breakdown": breakdown["per_metric"],
        "metrics_per_step": metrics_per_step,
        "evals_completeness": completeness,
        "gate_reason": gate_reason,
    }
    return row


def _classify_failure(run: WandbRun) -> str:
    state = run.state
    if state == "crashed":
        # Heuristic: SIGILL leaves exit_code=-4; OOM leaves a CUDA OOM in the
        # error log. The W&B public API doesn't expose exit_code reliably, so
        # we leave classification to the operator's vibe note when uncertain.
        return "sigill_or_oom_or_diverged"
    if state == "killed":
        return "killed"
    return "failed"


def append_outcome_row(*, rounds_jsonl: Path, row: dict[str, Any]) -> None:
    with rounds_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    p.add_argument(
        "--anchors",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/anchors.yaml"),
    )
    p.add_argument(
        "--entity",
        default="graph_denoise_team",
    )
    p.add_argument(
        "--project",
        default="tmgg-smallest-config-sweep",
    )
    args = p.parse_args()

    rows = read_rounds(args.rounds_jsonl)
    pending = find_pending_launches(rows)
    print(f"found {len(pending)} launched rows without paired outcome")

    for launched in pending:
        try:
            run = resolve_wandb_run(
                entity=args.entity, project=args.project, run_name=launched["run_uid"]
            )
        except LookupError as exc:
            print(f"SKIP {launched['run_uid']}: {exc}")
            continue
        if run.state == "running":
            print(f"SKIP {launched['run_uid']}: still running")
            continue
        outcome = build_outcome_row(
            launched=launched, run=run, anchors_path=args.anchors
        )
        append_outcome_row(rounds_jsonl=args.rounds_jsonl, row=outcome)
        print(
            f"OUTCOME {launched['run_uid']} status={outcome['status']} "
            f"pass={outcome.get('threshold_pass', '-')}"
        )


if __name__ == "__main__":
    main()
