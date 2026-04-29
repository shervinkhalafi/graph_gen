"""Post-hoc reconciler for the async-eval manifest.

Per plan ``compressed-tumbling-whale.md`` Step 9: when the trainer's
15-minute progress-reset drain timeout fires (``on_fit_end`` in
``AsyncEvalSpawnCallback``), some Modal eval calls may still be
in-flight. Their manifest rows remain ``status="spawned"`` indefinitely.
This reconciler walks those rows, queries Modal for each call's
terminal status via ``modal.FunctionCall.from_id(call_id).get(timeout=0)``,
and appends the appropriate ``completed`` / ``failed`` row.

Design notes
------------
- Append-only. Existing rows are never modified or deleted; the audit
  trail is preserved. Latest-row-per-step semantics (per
  ``_eval_manifest.latest_status_per_step``) ensure downstream readers
  see the resolved state.
- Non-blocking. We call ``.get(timeout=0)`` so a still-running eval
  returns control immediately; Modal raises ``OutputExpiredError`` (or
  the parent ``modal.exception.TimeoutError``) in that case and we
  count the row as ``still_pending`` without writing.
- Forward-compat ``--respawn``. The flag exists so callers can wire
  the future re-spawn loop without changing the CLI shape; today it
  raises ``NotImplementedError`` because re-spawn requires storing the
  full task dict in the spawned-row payload, which is out of scope.

CLI usage
---------
.. code-block:: bash

    uv run python -m scripts.sweep.reconcile_evals \\
        --manifest /data/outputs/<run_id>/eval_manifest.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal
from modal.exception import TimeoutError as ModalTimeoutError

# TODO: replace with scripts.sweep._eval_manifest.read_manifest /
# latest_status_per_step once Step 5 lands. The fallbacks below
# duplicate the documented behaviour (skip schema rows + blanks; group
# by scheduled_step and keep latest ts_utc) so this module is usable
# in isolation while Step 5 is in flight.
try:
    from scripts.sweep._eval_manifest import (
        latest_status_per_step as _ext_latest_status_per_step,
    )
    from scripts.sweep._eval_manifest import (
        read_manifest as _ext_read_manifest,
    )

    _have_eval_manifest = True
except ImportError:
    _have_eval_manifest = False
    _ext_read_manifest = None  # type: ignore[assignment]
    _ext_latest_status_per_step = None  # type: ignore[assignment]


_ERROR_TAIL_MAX_CHARS = 1000


def _fallback_read_manifest(path: Path) -> list[dict[str, Any]]:
    """Inline JSONL parser; mirrors ``_eval_manifest.read_manifest``.

    Skips ``kind == "schema"`` rows and blank/whitespace-only lines.
    Used only when ``_eval_manifest`` is not yet importable (Step 5
    landing in parallel).
    """
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") == "schema":
            continue
        rows.append(obj)
    return rows


def _fallback_latest_status_per_step(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Inline grouper; mirrors ``_eval_manifest.latest_status_per_step``.

    Group rows by ``scheduled_step`` and keep the row with the largest
    ``ts_utc`` per group. ISO-8601 strings sort lexically in time order
    when the timezone suffix is consistent, which it is in our writers.
    """
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("scheduled_step")
        if step is None:
            continue
        ts = row.get("ts_utc", "")
        prev = latest.get(step)
        if prev is None or ts > prev.get("ts_utc", ""):
            latest[step] = row
    return latest


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if _have_eval_manifest and _ext_read_manifest is not None:
        return _ext_read_manifest(path)
    return _fallback_read_manifest(path)


def _latest_per_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    if _have_eval_manifest and _ext_latest_status_per_step is not None:
        return _ext_latest_status_per_step(rows)
    return _fallback_latest_status_per_step(rows)


def _default_modal_call_resolver(call_id: str) -> Any:
    """Default resolver: look up a live FunctionCall by id."""
    return modal.FunctionCall.from_id(call_id)


def _append_row(path: Path, row: dict[str, Any]) -> None:
    """Append a single JSON object as a JSONL row to ``path``."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False))
        fh.write("\n")


def _build_event_row(
    *,
    base: dict[str, Any],
    status: str,
    metrics: dict[str, Any] | None,
    error_tail: str | None,
) -> dict[str, Any]:
    """Construct an ``eval_event`` row inheriting identifiers from ``base``.

    The reconciler keeps the row's identifying triple (``run_uid``,
    ``wandb_run_id``, ``scheduled_step``) and ``modal_call_id`` so the
    row can be matched back to its spawn event. Mutable fields
    (``status``, ``metrics``, ``error_tail``, ``ts_utc``) reflect the
    reconciliation outcome.
    """
    return {
        "kind": "eval_event",
        "run_uid": base.get("run_uid"),
        "wandb_run_id": base.get("wandb_run_id"),
        "scheduled_step": base.get("scheduled_step"),
        "global_step": base.get("global_step"),
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "status": status,
        "modal_call_id": base.get("modal_call_id"),
        "checkpoint_path": base.get("checkpoint_path"),
        "metrics": metrics,
        "error_tail": error_tail,
        "reconciled": True,
    }


def _coerce_metrics(return_value: Any) -> dict[str, Any] | None:
    """Extract a metrics dict from a Modal call's return value.

    ``_evaluate_mmd_async_impl`` (Step 1 of the plan) returns a dict
    with a ``"metrics"`` key. Older or hand-rolled callers may return
    a bare metrics dict, which we accept too. Anything else becomes
    ``None`` — a known limitation worth surfacing when reconciling
    non-standard callers.
    """
    if not isinstance(return_value, dict):
        return None
    metrics = return_value.get("metrics", return_value)
    if not isinstance(metrics, dict):
        return None
    return metrics


def reconcile_manifest(
    manifest_path: Path,
    *,
    respawn_failed: bool = False,
    modal_call_resolver: Callable[[str], Any] | None = None,
) -> dict[str, int]:
    """Reconcile ``status="spawned"`` rows against Modal's call status.

    Parameters
    ----------
    manifest_path : Path
        JSONL manifest at ``/data/outputs/<run_id>/eval_manifest.jsonl``.
    respawn_failed : bool, optional
        Forward-compat flag. Currently raises ``NotImplementedError``
        because re-spawn requires storing the full task dict in the
        spawned-row payload first (out of scope for Step 9).
    modal_call_resolver : Callable[[str], Any], optional
        Function mapping a ``modal_call_id`` to a Modal FunctionCall
        handle. Defaults to ``modal.FunctionCall.from_id``. Override
        in tests to inject mocks.

    Returns
    -------
    dict[str, int]
        ``{"reconciled": int, "still_pending": int, "respawned": int}``.
        ``reconciled`` counts rows that gained a terminal
        (``completed`` / ``failed``) row this invocation. ``respawned``
        is always 0 today.

    Raises
    ------
    NotImplementedError
        If ``respawn_failed=True``.
    """
    if respawn_failed:
        # TODO: respawn loop — requires storing task dict in the
        # spawned-row payload first so we can re-invoke
        # ``modal.Function.from_name(...).spawn(task)`` with the same
        # task. Until then, fail loudly so callers can't silently
        # depend on a no-op.
        raise NotImplementedError(
            "respawn_failed=True is not yet implemented. The spawned-row "
            "payload would need to carry the full task dict before we can "
            "re-invoke modal.Function.from_name(...).spawn(task)."
        )

    resolver = modal_call_resolver or _default_modal_call_resolver

    rows = _read_rows(manifest_path)
    latest = _latest_per_step(rows)

    counts = {"reconciled": 0, "still_pending": 0, "respawned": 0}

    for _step, row in sorted(latest.items()):
        if row.get("status") != "spawned":
            continue

        call_id = row.get("modal_call_id")
        if not call_id:
            # A spawned row without a call id is malformed; skip.
            # Loud failure here would block the whole reconcile pass,
            # which is worse than logging-and-continuing for one row.
            continue

        call = resolver(call_id)
        try:
            return_value = call.get(timeout=0)
        except ModalTimeoutError:
            # Non-terminal: the call is still running. Modal raises
            # ``OutputExpiredError`` (a subclass of
            # ``modal.exception.TimeoutError``) for "not yet ready".
            counts["still_pending"] += 1
            continue
        except Exception as exc:  # noqa: BLE001 — call-side failure
            error_tail = str(exc)[:_ERROR_TAIL_MAX_CHARS]
            failed_row = _build_event_row(
                base=row,
                status="failed",
                metrics=None,
                error_tail=error_tail,
            )
            _append_row(manifest_path, failed_row)
            counts["reconciled"] += 1
            continue

        metrics = _coerce_metrics(return_value)
        completed_row = _build_event_row(
            base=row,
            status="completed",
            metrics=metrics,
            error_tail=None,
        )
        _append_row(manifest_path, completed_row)
        counts["reconciled"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to eval_manifest.jsonl on the local filesystem.",
    )
    parser.add_argument(
        "--respawn",
        action="store_true",
        help=(
            "Forward-compat flag: re-spawn evals that the manifest marks "
            "failed. Currently raises NotImplementedError; the spawned-row "
            "payload needs to carry the task dict first."
        ),
    )
    args = parser.parse_args()

    counts = reconcile_manifest(args.manifest, respawn_failed=args.respawn)
    print(
        f"reconciled={counts['reconciled']} "
        f"still_pending={counts['still_pending']} "
        f"respawned={counts['respawned']}"
    )


if __name__ == "__main__":
    main()
