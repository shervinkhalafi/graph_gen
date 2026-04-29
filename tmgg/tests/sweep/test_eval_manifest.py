"""Tests for scripts.sweep._eval_manifest.

Rationale
---------
The async-eval architecture (plan: compressed-tumbling-whale) splits
sampling+MMD evaluation off the trainer onto a separate Modal worker.
Both writers (the trainer-side ``AsyncEvalSpawnCallback`` and the
worker-side ``_evaluate_mmd_async_impl``) append rows to a JSONL
manifest at ``/data/outputs/{run_id}/eval_manifest.jsonl``. The first
row is always a ``{"kind": "schema", ...}`` header that downstream
readers must skip — same convention the sweep already uses for
``rounds.jsonl``.

This module pins the small reader helpers used by ``fetch_outcomes``,
``watch_runs``, and ``reconcile_evals``:

- ``read_manifest`` parses JSONL, skipping schema rows and blank lines.
- ``latest_status_per_step`` groups by ``scheduled_step`` (NOT
  ``global_step`` — completeness is judged against the schedule the
  trainer was given) and keeps the row with the latest ``ts_utc``.
- ``evals_completeness`` and ``evals_lag`` give ``fetch_outcomes`` and
  ``watch_runs`` cheap predicates for "is this run done evaluating?"
  and "is this run lagging on its eval schedule?".

Invariants exercised:

1. Schema row at file head is always skipped.
2. Blank lines are tolerated (some writers append a trailing newline).
3. When the same scheduled_step has multiple rows (``spawned`` then
   ``completed``), the later ``ts_utc`` wins.
4. Completeness keys (``expected``, ``completed``, ``failed``,
   ``missing``) match the documented contract from the plan.
5. Lag is computed against the trainer's current step, not wall-clock.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.sweep._eval_manifest import (
    evals_completeness,
    evals_lag,
    expected_steps,
    latest_status_per_step,
    read_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as JSONL — one JSON object per line, terminating newline."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            _ = fh.write(json.dumps(row))
            _ = fh.write("\n")


def test_read_manifest_skips_schema_row(tmp_path: Path) -> None:
    """Schema row at file head must not appear in the parsed event list."""
    path = tmp_path / "eval_manifest.jsonl"
    _write_jsonl(
        path,
        [
            {"kind": "schema", "version": 1, "doc": "manifest header"},
            {
                "kind": "eval_event",
                "run_uid": "rA",
                "wandb_run_id": "abc",
                "scheduled_step": 100,
                "global_step": 100,
                "ts_utc": "2026-04-29T00:00:00Z",
                "status": "spawned",
                "modal_call_id": "fc-1",
                "checkpoint_path": "/data/x.ckpt",
                "metrics": None,
                "error_tail": None,
            },
            {
                "kind": "eval_event",
                "run_uid": "rA",
                "wandb_run_id": "abc",
                "scheduled_step": 100,
                "global_step": 100,
                "ts_utc": "2026-04-29T00:01:00Z",
                "status": "completed",
                "modal_call_id": "fc-1",
                "checkpoint_path": "/data/x.ckpt",
                "metrics": {"gen-val/sbm_accuracy": 0.5},
                "error_tail": None,
            },
        ],
    )
    rows = read_manifest(path)
    assert len(rows) == 2
    assert all(row["kind"] == "eval_event" for row in rows)


def test_read_manifest_skips_blanks(tmp_path: Path) -> None:
    """Blank/whitespace-only lines must not raise a JSON decode error."""
    path = tmp_path / "eval_manifest.jsonl"
    schema_line = json.dumps({"kind": "schema", "version": 1})
    event_line = json.dumps(
        {
            "kind": "eval_event",
            "run_uid": "rA",
            "wandb_run_id": "abc",
            "scheduled_step": 100,
            "global_step": 100,
            "ts_utc": "2026-04-29T00:00:00Z",
            "status": "completed",
            "modal_call_id": "fc-1",
            "checkpoint_path": "/data/x.ckpt",
            "metrics": {"gen-val/sbm_accuracy": 0.5},
            "error_tail": None,
        }
    )
    # Blank line in the middle, whitespace-only line at the end —
    # both must be tolerated by read_manifest.
    payload = f"{schema_line}\n\n{event_line}\n   \n"
    _ = path.write_text(payload, encoding="utf-8")
    rows = read_manifest(path)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_latest_status_per_step_picks_newer_ts() -> None:
    """When two rows share scheduled_step, the later ts_utc wins."""
    rows = [
        {
            "kind": "eval_event",
            "scheduled_step": 200,
            "global_step": 200,
            "ts_utc": "2026-04-29T00:00:01Z",
            "status": "spawned",
            "modal_call_id": "fc-2",
        },
        {
            "kind": "eval_event",
            "scheduled_step": 200,
            "global_step": 200,
            "ts_utc": "2026-04-29T00:00:02Z",
            "status": "completed",
            "modal_call_id": "fc-2",
        },
    ]
    latest = latest_status_per_step(rows)
    assert set(latest.keys()) == {200}
    assert latest[200]["status"] == "completed"


def test_evals_completeness_counts_correctly() -> None:
    """Two completed, one failed, one missing → keys reflect that exactly."""
    schedule = [100, 200, 300, 400]
    rows = [
        {
            "kind": "eval_event",
            "scheduled_step": 100,
            "ts_utc": "2026-04-29T00:00:01Z",
            "status": "spawned",
        },
        {
            "kind": "eval_event",
            "scheduled_step": 100,
            "ts_utc": "2026-04-29T00:00:02Z",
            "status": "completed",
        },
        {
            "kind": "eval_event",
            "scheduled_step": 200,
            "ts_utc": "2026-04-29T00:00:03Z",
            "status": "completed",
        },
        {
            "kind": "eval_event",
            "scheduled_step": 300,
            "ts_utc": "2026-04-29T00:00:04Z",
            "status": "spawned",
        },
        {
            "kind": "eval_event",
            "scheduled_step": 300,
            "ts_utc": "2026-04-29T00:00:05Z",
            "status": "failed",
        },
        # No row at 400 at all.
    ]
    result = evals_completeness(rows, schedule)
    assert result == {
        "expected": 4,
        "completed": 2,
        "failed": 1,
        "missing": [400],
    }


def test_evals_lag_at_current_step() -> None:
    """Two scheduled steps have passed, only one completed → lag = 1.

    Schedule is ``[100, 200, 300]``; trainer is at step 250 so steps
    100 and 200 should have completed. Only step 100 has a completed
    row, so the trainer is one eval behind.
    """
    schedule = [100, 200, 300]
    rows = [
        {
            "kind": "eval_event",
            "scheduled_step": 100,
            "ts_utc": "2026-04-29T00:00:01Z",
            "status": "completed",
        },
    ]
    assert evals_lag(rows, schedule, current_step=250) == 1


def test_evals_lag_zero_when_all_completed() -> None:
    """All scheduled-and-passed evals completed → lag = 0."""
    schedule = [100, 200]
    rows = [
        {
            "kind": "eval_event",
            "scheduled_step": 100,
            "ts_utc": "2026-04-29T00:00:01Z",
            "status": "completed",
        },
        {
            "kind": "eval_event",
            "scheduled_step": 200,
            "ts_utc": "2026-04-29T00:00:02Z",
            "status": "completed",
        },
    ]
    assert evals_lag(rows, schedule, current_step=300) == 0


def test_expected_steps_matches_set_of_schedule() -> None:
    """Trivial wrapper, but pin the contract so callers can rely on it."""
    assert expected_steps([100, 200, 300]) == {100, 200, 300}
    assert expected_steps([]) == set[int]()
