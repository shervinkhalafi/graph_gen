"""Tests for ``scripts.sweep.reconcile_evals``.

Rationale
---------
The async-eval architecture (plan: compressed-tumbling-whale) lets the
trainer's ``on_fit_end`` drain loop time out after 15 minutes of idle
progress. Any in-flight Modal eval still running at that point leaves
its manifest row in ``status="spawned"`` indefinitely. The reconciler
runs later (manually or from ``fetch_outcomes.py``) to query Modal for
the call's terminal state via ``modal.FunctionCall.from_id(call_id)``
and append the appropriate ``completed`` or ``failed`` row.

Invariants exercised
--------------------
1. Terminal success → manifest gains a ``completed`` row carrying the
   call's return-value dict as ``metrics``.
2. Terminal failure → manifest gains a ``failed`` row whose
   ``error_tail`` contains the exception's stringified message
   (truncated to 1000 chars).
3. Non-terminal call → no new row, ``still_pending`` counter
   incremented. The reconciler never blocks on a live call.
4. ``--respawn`` is forward-compat scaffolding; passing
   ``respawn_failed=True`` raises ``NotImplementedError``.
5. Returned counts ``{reconciled, still_pending, respawned}`` reflect
   the mixed-scenario truth.

The reconciler appends rows; it never deletes or mutates existing
rows. The audit trail is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from modal.exception import OutputExpiredError
from scripts.sweep.reconcile_evals import reconcile_manifest


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to ``path``, one object per line."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def _read_manifest_lines(path: Path) -> list[dict[str, Any]]:
    """Read all JSON rows from ``path``, skipping blanks."""
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _spawned_row(scheduled_step: int, modal_call_id: str) -> dict[str, Any]:
    """Construct a canonical ``status="spawned"`` row."""
    return {
        "kind": "eval_event",
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "wandb_run_id": "abc1234",
        "scheduled_step": scheduled_step,
        "global_step": scheduled_step,
        "ts_utc": "2026-04-29T00:00:00Z",
        "status": "spawned",
        "modal_call_id": modal_call_id,
        "checkpoint_path": f"/data/outputs/run/checkpoints/step_{scheduled_step}.ckpt",
        "metrics": None,
        "error_tail": None,
    }


def test_reconciler_writes_completed_row_for_terminal_success(
    tmp_path: Path,
) -> None:
    """A spawned row whose Modal call returned a metrics dict gets a
    matching ``completed`` row appended."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=100, modal_call_id="fc-success"),
        ],
    )

    fake_call = MagicMock()
    fake_call.get.return_value = {
        "metrics": {"gen-val/sbm_accuracy": 0.85, "gen-val/degree_mmd": 0.0014},
        "status": "completed",
    }
    resolver = MagicMock(return_value=fake_call)

    counts = reconcile_manifest(manifest, modal_call_resolver=resolver)

    assert counts == {"reconciled": 1, "still_pending": 0, "respawned": 0}
    rows = _read_manifest_lines(manifest)
    completed_rows = [
        r
        for r in rows
        if r.get("status") == "completed" and r.get("scheduled_step") == 100
    ]
    assert len(completed_rows) == 1
    assert completed_rows[0]["metrics"] == {
        "gen-val/sbm_accuracy": 0.85,
        "gen-val/degree_mmd": 0.0014,
    }
    # The original spawned row is preserved.
    spawned_rows = [
        r
        for r in rows
        if r.get("status") == "spawned" and r.get("scheduled_step") == 100
    ]
    assert len(spawned_rows) == 1
    fake_call.get.assert_called_once_with(timeout=0)
    resolver.assert_called_once_with("fc-success")


def test_reconciler_writes_failed_row_for_terminal_failure(tmp_path: Path) -> None:
    """A spawned row whose Modal call raised on ``.get(timeout=0)`` gets
    a ``failed`` row with ``error_tail`` carrying the exception text."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=200, modal_call_id="fc-fail"),
        ],
    )

    fake_call = MagicMock()
    fake_call.get.side_effect = RuntimeError("OOM in worker")
    resolver = MagicMock(return_value=fake_call)

    counts = reconcile_manifest(manifest, modal_call_resolver=resolver)

    assert counts == {"reconciled": 1, "still_pending": 0, "respawned": 0}
    rows = _read_manifest_lines(manifest)
    failed_rows = [r for r in rows if r.get("status") == "failed"]
    assert len(failed_rows) == 1
    assert "OOM" in failed_rows[0]["error_tail"]
    assert failed_rows[0]["scheduled_step"] == 200


def test_reconciler_skips_still_pending(tmp_path: Path) -> None:
    """An ``OutputExpiredError`` (or the modal TimeoutError parent
    class) means the call is still in flight — no new row is written
    and the ``still_pending`` counter advances."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=300, modal_call_id="fc-pending"),
        ],
    )
    rows_before = _read_manifest_lines(manifest)

    fake_call = MagicMock()
    fake_call.get.side_effect = OutputExpiredError("not yet ready")
    resolver = MagicMock(return_value=fake_call)

    counts = reconcile_manifest(manifest, modal_call_resolver=resolver)

    assert counts == {"reconciled": 0, "still_pending": 1, "respawned": 0}
    rows_after = _read_manifest_lines(manifest)
    assert rows_after == rows_before


def test_reconciler_respawn_flag_raises_notimplemented(tmp_path: Path) -> None:
    """The ``--respawn`` flag exists for forward-compat; passing
    ``respawn_failed=True`` must raise rather than silently no-op."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=400, modal_call_id="fc-respawn"),
        ],
    )
    resolver = MagicMock()

    with pytest.raises(NotImplementedError):
        reconcile_manifest(manifest, respawn_failed=True, modal_call_resolver=resolver)


def test_reconciler_returns_counts_for_mixed_scenario(tmp_path: Path) -> None:
    """Mixed scenario — 2 completed, 1 failed, 1 still pending — the
    counts dict reflects all four rows accurately."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=100, modal_call_id="fc-ok-1"),
            _spawned_row(scheduled_step=200, modal_call_id="fc-ok-2"),
            _spawned_row(scheduled_step=300, modal_call_id="fc-fail"),
            _spawned_row(scheduled_step=400, modal_call_id="fc-pending"),
        ],
    )

    def make_call(return_value: Any = None, side_effect: Any = None) -> MagicMock:
        c = MagicMock()
        if side_effect is not None:
            c.get.side_effect = side_effect
        else:
            c.get.return_value = return_value
        return c

    call_map = {
        "fc-ok-1": make_call(return_value={"metrics": {"gen-val/sbm_accuracy": 0.8}}),
        "fc-ok-2": make_call(return_value={"metrics": {"gen-val/sbm_accuracy": 0.82}}),
        "fc-fail": make_call(side_effect=RuntimeError("CUDA OOM")),
        "fc-pending": make_call(side_effect=OutputExpiredError("running")),
    }

    def resolver(call_id: str) -> MagicMock:
        return call_map[call_id]

    counts = reconcile_manifest(manifest, modal_call_resolver=resolver)
    assert counts == {"reconciled": 3, "still_pending": 1, "respawned": 0}

    rows = _read_manifest_lines(manifest)
    completed_steps = sorted(
        r["scheduled_step"] for r in rows if r.get("status") == "completed"
    )
    failed_steps = sorted(
        r["scheduled_step"] for r in rows if r.get("status") == "failed"
    )
    assert completed_steps == [100, 200]
    assert failed_steps == [300]


def test_reconciler_ignores_already_completed_rows(tmp_path: Path) -> None:
    """If a row is already terminal (``completed`` or ``failed``), the
    reconciler must not re-query Modal for it — it only acts on rows
    whose latest status is ``spawned``."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=100, modal_call_id="fc-1"),
            {
                **_spawned_row(scheduled_step=100, modal_call_id="fc-1"),
                "ts_utc": "2026-04-29T00:01:00Z",
                "status": "completed",
                "metrics": {"gen-val/sbm_accuracy": 0.7},
            },
        ],
    )
    resolver = MagicMock()

    counts = reconcile_manifest(manifest, modal_call_resolver=resolver)
    assert counts == {"reconciled": 0, "still_pending": 0, "respawned": 0}
    resolver.assert_not_called()


def test_reconciler_truncates_long_error_tail(tmp_path: Path) -> None:
    """Error messages longer than 1000 characters are truncated."""
    manifest = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"kind": "schema", "version": 1},
            _spawned_row(scheduled_step=500, modal_call_id="fc-bigerror"),
        ],
    )

    long_msg = "X" * 5000
    fake_call = MagicMock()
    fake_call.get.side_effect = RuntimeError(long_msg)
    resolver = MagicMock(return_value=fake_call)

    counts = reconcile_manifest(manifest, modal_call_resolver=resolver)
    assert counts == {"reconciled": 1, "still_pending": 0, "respawned": 0}

    rows = _read_manifest_lines(manifest)
    failed_rows = [r for r in rows if r.get("status") == "failed"]
    assert len(failed_rows) == 1
    assert len(failed_rows[0]["error_tail"]) <= 1000
