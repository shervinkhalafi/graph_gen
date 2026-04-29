"""Reader helpers for the async-eval JSONL manifest.

The async-eval architecture (plan: ``compressed-tumbling-whale``) has
two writers append rows to ``/data/outputs/{run_id}/eval_manifest.jsonl``:
the trainer-side ``AsyncEvalSpawnCallback`` writes ``status: spawned``
when it fires a Modal eval call, and the eval worker writes
``status: completed`` or ``status: failed`` when it finishes. The first
line of the file is a ``{"kind": "schema", "version": 1, "doc": ...}``
header, mirroring the convention already used by ``rounds.jsonl``.

This module exposes pure-function readers — the only I/O happens in
``read_manifest``. Three downstream callers consume them:

- ``fetch_outcomes.py`` cross-references the manifest with W&B history
  to decide whether all expected eval steps are accounted for before
  emitting ``threshold_pass``.
- ``watch_runs.py`` reads ``evals_lag`` to flag eval-starvation, which
  the recommendation flowchart uses to broaden ``extend_watch`` over
  ``kill: saturation``.
- ``reconcile_evals.py`` walks ``status: spawned`` rows whose Modal
  call has since terminated and writes the missing terminal row.

Manifest row schema (per the plan)::

    {"kind": "eval_event",
     "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
     "wandb_run_id": "abc1234",
     "scheduled_step": 5237,
     "global_step": 5240,
     "ts_utc": "2026-04-29T...",
     "status": "spawned" | "completed" | "failed",
     "modal_call_id": "fc-...",
     "checkpoint_path": "/data/outputs/{run_id}/checkpoints/step_5240.ckpt",
     "metrics": {"gen-val/sbm_accuracy": 0.62, ...} | null,
     "error_tail": null | "..."}

``scheduled_step`` and ``global_step`` differ when
``accumulate_grad_batches > 1`` skips the exact scheduled step; the
eval fires at the first step >= scheduled. ``scheduled_step`` is the
key used for completeness — it matches the schedule the trainer was
given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read all eval-event rows from the JSONL manifest.

    Skips the leading ``{"kind": "schema", ...}`` header row and any
    blank or whitespace-only lines. Every other line must parse as JSON
    and have ``kind == "eval_event"``; rows with any other ``kind`` are
    ignored so future row-types stay forward-compatible without
    breaking older readers.

    Parameters
    ----------
    path
        Filesystem path to ``eval_manifest.jsonl``.

    Returns
    -------
    list[dict[str, Any]]
        Parsed eval-event rows in file order. Empty list if the file
        contains only a schema header (or is empty).
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("kind") == "eval_event":
                rows.append(row)
    return rows


def latest_status_per_step(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group rows by ``scheduled_step``, take the latest by ``ts_utc``.

    The manifest is append-only, so a single ``scheduled_step`` may
    have a ``spawned`` row and a later terminal (``completed`` or
    ``failed``) row. Downstream completeness checks need the terminal
    state; we pick by ISO-8601 ``ts_utc`` which sorts lexicographically.

    Keys by ``scheduled_step`` rather than ``global_step`` because
    completeness is judged against the schedule the trainer was given,
    not against the actual step the eval landed on.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["scheduled_step"])
        ts = str(row["ts_utc"])
        prev = by_step.get(step)
        if prev is None or str(prev["ts_utc"]) < ts:
            by_step[step] = row
    return by_step


def expected_steps(schedule: list[int]) -> set[int]:
    """Return the schedule as a set.

    Convenience wrapper so callers can phrase completeness questions as
    set algebra without re-importing builtins.
    """
    return set(schedule)


def evals_completeness(
    rows: list[dict[str, Any]],
    schedule: list[int],
) -> dict[str, int | list[int]]:
    """Summarise terminal-state coverage of ``schedule``.

    A scheduled step is ``completed`` if its latest row is
    ``status: completed``, ``failed`` if its latest row is
    ``status: failed``, and ``missing`` otherwise (no row, or only a
    ``spawned`` row). ``missing`` is sorted ascending so callers can
    print it without further work.

    Returns
    -------
    dict
        ``{"expected": int, "completed": int, "failed": int,
        "missing": list[int]}``.
    """
    latest = latest_status_per_step(rows)
    completed = 0
    failed = 0
    missing: list[int] = []
    for step in schedule:
        entry = latest.get(step)
        if entry is None:
            missing.append(step)
            continue
        status = entry.get("status")
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        else:
            # "spawned" or anything non-terminal counts as missing —
            # the eval has not produced a result yet.
            missing.append(step)
    return {
        "expected": len(schedule),
        "completed": completed,
        "failed": failed,
        "missing": sorted(missing),
    }


def evals_lag(
    rows: list[dict[str, Any]],
    schedule: list[int],
    current_step: int,
) -> int:
    """Return scheduled-and-passed steps minus completed evals.

    A positive lag means the trainer has passed eval points whose
    results have not yet been logged. ``watch_runs.py`` uses this to
    flag eval-starvation: a saturation-fit on starved data has too few
    points to be defensible, so the recommendation broadens to
    ``extend_watch`` rather than ``kill: saturation``.

    Only ``status: completed`` counts as resolved here. ``status: failed``
    is intentionally not counted: a failed eval has produced no metric
    for the saturation fit, so it still represents an information gap.
    """
    latest = latest_status_per_step(rows)
    passed = sum(1 for s in schedule if s <= current_step)
    completed = sum(
        1
        for s in schedule
        if s <= current_step and (latest.get(s) or {}).get("status") == "completed"
    )
    return passed - completed
