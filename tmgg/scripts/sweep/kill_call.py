"""Manual kill mechanism for in-flight Modal calls.

This is the operator-side gate behind the watcher's flowchart: when
``watch_runs.py`` recommends ``kill``, Claude invokes this script with
the call ID(s) to cancel. There is no auto-kill from the watcher — the
flowchart is informational, this CLI is the action.

Two ways to feed it call IDs:

1. ``--call-id fc-...`` (repeatable) — direct cancellation per ID.
2. ``--from-manifest <run_dir>`` — read the run's
   ``eval_manifest.d/`` for ``status: spawned`` rows that lack a
   paired ``completed`` / ``failed`` row, then list (default) or
   cancel (``--yes``) the corresponding eval-worker calls.

The trainer's own ``FunctionCall`` ID is captured at spawn time and
recorded on the matching ``rounds.jsonl`` ``launched`` row under
``modal_function_call_id``; pass it through ``--call-id`` to cancel
the trainer.

Each call's outcome is printed as a JSON line on stdout:

::

    {"call_id": "fc-...", "action": "cancelled" | "not_found" | "failed",
     "error": null | "..."}

Exit code is 0 when at least one call was cancelled, else 1, so the
script can be chained in shell pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# We keep the ``modal`` import lazy at function-call time so unit tests
# can patch ``_function_call_from_id`` and ``_not_found_exception_class``
# without forcing the heavy SDK to load. Production code paths still see
# the real Modal classes; tests substitute stubs.
_ERROR_TAIL_LIMIT = 300


def _function_call_from_id(call_id: str) -> Any:
    """Resolve a ``FunctionCall`` handle from its object ID.

    Lazy ``import modal`` here keeps test patching cheap and avoids
    pulling Modal into module-import-time of every consumer of this
    file (including pytest collection on machines without Modal
    credentials).
    """
    import modal  # noqa: PLC0415 — intentional lazy import

    return modal.FunctionCall.from_id(call_id)


def _not_found_exception_class() -> type[BaseException]:
    """Return the exception class used for "unknown / terminal call ID".

    Modal raises ``modal.exception.NotFoundError`` when a function-call
    ID doesn't resolve. Importing it lazily mirrors
    :func:`_function_call_from_id` so tests can stub the ID lookup
    without touching the SDK.
    """
    import modal  # noqa: PLC0415 — intentional lazy import

    return modal.exception.NotFoundError


def cancel_call(call_id: str) -> dict[str, Any]:
    """Cancel a Modal function call by its FunctionCall ID (``fc-...``).

    Parameters
    ----------
    call_id
        The ``fc-...`` object ID to cancel.

    Returns
    -------
    dict
        ``{"call_id": str, "action": "cancelled" | "not_found" | "failed",
        "error": str | None}``. ``not_found`` is returned for unknown
        or already-terminal IDs (Modal's ``NotFoundError``); ``failed``
        captures any other exception with the first 300 chars of
        ``str(exc)`` as the diagnostic tail.
    """
    not_found_cls = _not_found_exception_class()
    try:
        handle = _function_call_from_id(call_id)
    except not_found_cls as exc:
        # The cancel target is gone (already terminal or never existed);
        # surface a structured marker rather than a stacktrace so
        # batched-cancel callers can keep going.
        return {
            "call_id": call_id,
            "action": "not_found",
            "error": str(exc)[:_ERROR_TAIL_LIMIT] if str(exc) else None,
        }
    except Exception as exc:  # noqa: BLE001 — we report-and-continue here
        return {
            "call_id": call_id,
            "action": "failed",
            "error": str(exc)[:_ERROR_TAIL_LIMIT],
        }

    try:
        handle.cancel()
    except not_found_cls as exc:
        return {
            "call_id": call_id,
            "action": "not_found",
            "error": str(exc)[:_ERROR_TAIL_LIMIT] if str(exc) else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "call_id": call_id,
            "action": "failed",
            "error": str(exc)[:_ERROR_TAIL_LIMIT],
        }

    return {"call_id": call_id, "action": "cancelled", "error": None}


def collect_kill_candidates(run_dir: Path) -> list[dict[str, Any]]:
    """Inspect a run's manifest dir for dangling ``spawned`` rows.

    Returns the list of eval-event rows whose ``scheduled_step`` has
    no terminal companion (``completed`` / ``failed``). Each entry
    carries the full JSON row so the caller can show context (step,
    run_uid, ts) alongside the call ID. The list is sorted ascending
    by ``scheduled_step``.

    A "dangling" row is the watcher's input for the manifest-mode
    kill: those eval workers are the ones currently in-flight, and
    cancelling them is what the operator typically wants when a run
    has gone off the rails.
    """
    from tmgg.sweep._eval_manifest import (  # noqa: PLC0415 — runtime dep
        latest_status_per_step,
        read_manifest,
    )

    # ``run_dir`` may be the run output dir (containing
    # ``eval_manifest.d/``) or the manifest dir / legacy JSONL itself.
    # Try the inner ``eval_manifest.d`` first so callers can pass the
    # outer run dir (matches the smallest-config-search workflow).
    candidate_inner = run_dir / "eval_manifest.d"
    if candidate_inner.is_dir():
        rows = read_manifest(candidate_inner)
    else:
        rows = read_manifest(run_dir)
    latest = latest_status_per_step(rows)
    candidates: list[dict[str, Any]] = []
    for step, row in sorted(latest.items()):
        if row.get("status") != "spawned":
            continue
        if not row.get("modal_call_id"):
            # No call ID means we have nothing to cancel; skip rather
            # than emit a half-broken candidate.
            continue
        _ = step  # value already keyed by latest_status_per_step
        candidates.append(row)
    return candidates


def _print_json_line(payload: dict[str, Any]) -> None:
    """Emit one stable-keyed JSON line on stdout."""
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def run_cli(argv: list[str] | None = None) -> int:
    """Parse args, run cancellation(s), print one JSON line per call.

    Returns
    -------
    int
        ``0`` when at least one call was cancelled, else ``1``.
        Manifest-mode dry-run (no ``--yes``) returns ``0`` if any
        candidate exists, else ``1`` — listing nothing means the user's
        target run has no dangling spawns to act on.
    """
    parser = argparse.ArgumentParser(
        description="Cancel Modal FunctionCalls by ID, or list dangling "
        "spawns from a run's eval_manifest directory."
    )
    parser.add_argument(
        "--call-id",
        action="append",
        default=[],
        metavar="fc-...",
        help="A FunctionCall ID to cancel. Repeatable.",
    )
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Path to a run output dir containing ``eval_manifest.d/``. "
        "List dangling ``spawned`` rows; dry-run unless ``--yes``.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="With ``--from-manifest``: actually cancel the listed calls.",
    )
    args = parser.parse_args(argv)

    call_ids: list[str] = list(args.call_id)
    candidates: list[dict[str, Any]] = []

    if args.from_manifest is not None:
        candidates = collect_kill_candidates(args.from_manifest)
        if not args.yes:
            # Dry-run: list candidates as JSON lines and return.
            if not candidates:
                sys.stderr.write(
                    f"No dangling 'spawned' rows under {args.from_manifest}\n"
                )
                return 1
            for row in candidates:
                _print_json_line(
                    {
                        "call_id": row.get("modal_call_id"),
                        "action": "dry_run_listed",
                        "scheduled_step": row.get("scheduled_step"),
                        "run_uid": row.get("run_uid"),
                        "wandb_run_id": row.get("wandb_run_id"),
                    }
                )
            return 0
        # ``--yes``: feed candidate call IDs into the cancel queue.
        call_ids.extend(
            str(c["modal_call_id"]) for c in candidates if c.get("modal_call_id")
        )

    if not call_ids:
        parser.error("no --call-id given and --from-manifest produced no candidates")

    cancelled_count = 0
    for cid in call_ids:
        result = cancel_call(cid)
        _print_json_line(result)
        if result["action"] == "cancelled":
            cancelled_count += 1

    return 0 if cancelled_count >= 1 else 1


def main() -> None:
    """Entry point for ``python -m scripts.sweep.kill_call``."""
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
