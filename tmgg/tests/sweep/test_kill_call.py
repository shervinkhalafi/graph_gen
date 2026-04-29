"""Unit tests for ``scripts.sweep.kill_call``.

Rationale
---------
``kill_call`` is the manual gate the operator (Claude) uses to terminate a
specific Modal call ID when the watcher's flowchart recommends a kill. It is
intentionally thin around ``modal.FunctionCall.from_id(...).cancel()`` and
must:

- Translate a successful ``cancel()`` into ``action="cancelled"``.
- Translate ``modal.exception.NotFoundError`` (raised when the ID is unknown
  or the call already terminated) into ``action="not_found"`` — never crash.
- Translate any other ``Exception`` into ``action="failed"`` with a short
  ``error`` tail so the operator can see why; we still return rather than
  re-raise so the CLI can report mixed-batch outcomes per call_id.
- ``--from-manifest`` lists ``spawned``-status rows whose paired ``completed``
  / ``failed`` row is absent, defaulting to dry-run; ``--yes`` actually
  triggers cancellation for each.

These tests pin those branches without taking a real Modal dependency by
patching ``modal.FunctionCall.from_id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_cancel_call_success_returns_cancelled() -> None:
    """A successful cancel returns ``action="cancelled"`` and no error."""
    from scripts.sweep import kill_call as mod

    handle = MagicMock()
    handle.cancel = MagicMock(return_value=None)

    with patch.object(mod, "_function_call_from_id", return_value=handle):
        result = mod.cancel_call("fc-01TESTSUCCESS")

    handle.cancel.assert_called_once()
    assert result == {
        "call_id": "fc-01TESTSUCCESS",
        "action": "cancelled",
        "error": None,
    }


def test_cancel_call_not_found_returns_not_found() -> None:
    """``NotFoundError`` on ``from_id`` collapses to ``action="not_found"``."""
    from scripts.sweep import kill_call as mod

    not_found_exc = mod._not_found_exception_class()  # the class kill_call catches

    def raise_not_found(_call_id: str) -> Any:
        raise not_found_exc("unknown id")

    with patch.object(mod, "_function_call_from_id", side_effect=raise_not_found):
        result = mod.cancel_call("fc-01MISSING")

    assert result["call_id"] == "fc-01MISSING"
    assert result["action"] == "not_found"
    # ``error`` may be the original message or None — but never a stack tail.
    assert result["error"] is None or "unknown id" in str(result["error"])


def test_cancel_call_unexpected_exception_returns_failed() -> None:
    """Any other exception collapses to ``action="failed"`` with an error tail."""
    from scripts.sweep import kill_call as mod

    def raise_runtime(_call_id: str) -> Any:
        raise RuntimeError("transient grpc deadline exceeded" + "X" * 500)

    with patch.object(mod, "_function_call_from_id", side_effect=raise_runtime):
        result = mod.cancel_call("fc-01BOOM")

    assert result["call_id"] == "fc-01BOOM"
    assert result["action"] == "failed"
    assert result["error"] is not None
    assert "transient grpc deadline" in result["error"]
    # Error is bounded — we cap to 300 chars to keep stdout one-line.
    assert len(result["error"]) <= 300


def test_cancel_call_failure_during_cancel_returns_failed() -> None:
    """A handle that explodes inside ``.cancel()`` also lands in ``failed``."""
    from scripts.sweep import kill_call as mod

    handle = MagicMock()
    handle.cancel = MagicMock(side_effect=ValueError("rpc dropped"))

    with patch.object(mod, "_function_call_from_id", return_value=handle):
        result = mod.cancel_call("fc-01CANCELFAIL")

    assert result["action"] == "failed"
    assert result["error"] is not None
    assert "rpc dropped" in result["error"]


# -----------------------------------------------------------------------------
# CLI / manifest-mode tests.
# -----------------------------------------------------------------------------


def _write_manifest_row(
    manifest_dir: Path,
    *,
    scheduled_step: int,
    status: str,
    discriminator: str,
    modal_call_id: str | None = None,
) -> None:
    """Write a single eval-event row in the directory layout."""
    fname = f"{scheduled_step:07d}-{status}-{discriminator}.json"
    row: dict[str, Any] = {
        "kind": "eval_event",
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "wandb_run_id": "abc1234",
        "scheduled_step": scheduled_step,
        "global_step": scheduled_step,
        "ts_utc": f"2026-04-29T00:00:0{scheduled_step % 10}",
        "status": status,
    }
    if modal_call_id is not None:
        row["modal_call_id"] = modal_call_id
    (manifest_dir / fname).write_text(json.dumps(row))


def test_collect_kill_candidates_lists_only_pending_spawns(tmp_path: Path) -> None:
    """``--from-manifest`` mode: only steps without a terminal row are listed.

    Setup: one ``spawned`` row at step 200 with no terminal row, one ``spawned``
    + ``completed`` pair at step 800. The kill-candidate list must contain
    only the dangling 200 row's call ID.
    """
    from scripts.sweep import kill_call as mod

    manifest_dir = tmp_path / "eval_manifest.d"
    manifest_dir.mkdir()
    _write_manifest_row(
        manifest_dir,
        scheduled_step=200,
        status="spawned",
        discriminator="fc01PENDING",
        modal_call_id="fc-01PENDING",
    )
    _write_manifest_row(
        manifest_dir,
        scheduled_step=800,
        status="spawned",
        discriminator="fc01DONE",
        modal_call_id="fc-01DONE",
    )
    _write_manifest_row(
        manifest_dir,
        scheduled_step=800,
        status="completed",
        discriminator="evalworker-uuid",
    )

    candidates = mod.collect_kill_candidates(tmp_path)
    call_ids = [c["modal_call_id"] for c in candidates]
    assert call_ids == ["fc-01PENDING"]


def test_kill_from_manifest_dry_run_does_not_invoke_cancel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--from-manifest`` without ``--yes`` lists candidates and exits 0.

    Default behaviour is dry-run; we must never trigger a Modal cancel
    just by listing the dangling spawns.
    """
    from scripts.sweep import kill_call as mod

    manifest_dir = tmp_path / "eval_manifest.d"
    manifest_dir.mkdir()
    _write_manifest_row(
        manifest_dir,
        scheduled_step=200,
        status="spawned",
        discriminator="fc01PENDING",
        modal_call_id="fc-01PENDING",
    )

    with patch.object(mod, "cancel_call") as mock_cancel:
        rc = mod.run_cli(
            ["--from-manifest", str(tmp_path)],
        )

    assert rc == 0  # listing succeeded
    mock_cancel.assert_not_called()
    captured = capsys.readouterr()
    assert "fc-01PENDING" in captured.out


def test_kill_from_manifest_with_yes_triggers_cancel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--from-manifest --yes`` cancels every dangling spawn."""
    from scripts.sweep import kill_call as mod

    manifest_dir = tmp_path / "eval_manifest.d"
    manifest_dir.mkdir()
    _write_manifest_row(
        manifest_dir,
        scheduled_step=200,
        status="spawned",
        discriminator="fc01PENDING",
        modal_call_id="fc-01PENDING",
    )

    fake_result = {
        "call_id": "fc-01PENDING",
        "action": "cancelled",
        "error": None,
    }
    with patch.object(mod, "cancel_call", return_value=fake_result) as mock_cancel:
        rc = mod.run_cli(
            ["--from-manifest", str(tmp_path), "--yes"],
        )

    assert rc == 0  # at least one cancellation succeeded
    mock_cancel.assert_called_once_with("fc-01PENDING")
    captured = capsys.readouterr()
    # JSON line per call printed to stdout.
    assert '"action": "cancelled"' in captured.out


def test_kill_explicit_call_ids_invokes_cancel_for_each(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--call-id`` (repeatable) triggers a cancel per id and prints JSON."""
    from scripts.sweep import kill_call as mod

    results = [
        {"call_id": "fc-01A", "action": "cancelled", "error": None},
        {"call_id": "fc-01B", "action": "not_found", "error": None},
    ]
    with patch.object(mod, "cancel_call", side_effect=results) as mock_cancel:
        rc = mod.run_cli(["--call-id", "fc-01A", "--call-id", "fc-01B"])

    assert rc == 0  # >=1 cancelled
    assert mock_cancel.call_count == 2
    captured = capsys.readouterr()
    assert '"call_id": "fc-01A"' in captured.out
    assert '"call_id": "fc-01B"' in captured.out


def test_kill_exits_nonzero_when_nothing_cancelled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If every call returned ``not_found``/``failed``, exit code is 1.

    The operator may want to chain this in a script; non-zero exit
    signals "no actual termination occurred" so they can take a fallback.
    """
    from scripts.sweep import kill_call as mod

    results = [
        {"call_id": "fc-01A", "action": "not_found", "error": None},
        {"call_id": "fc-01B", "action": "failed", "error": "boom"},
    ]
    with patch.object(mod, "cancel_call", side_effect=results):
        rc = mod.run_cli(["--call-id", "fc-01A", "--call-id", "fc-01B"])

    assert rc == 1
