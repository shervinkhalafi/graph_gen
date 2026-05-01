"""Idempotent W&B history cache for the active sweep.

For every ``run_uid`` in ``rounds.jsonl``, fetches gen-val/* + NLL +
opt-health history from W&B and writes it under
``wandb_export/sweep_cache/<safe_uid>/``::

    history.jsonl   one JSON row per training step (sorted)
    manifest.json   {wandb_run_id, run_state, last_fetched_step,
                      terminal_step, history_complete, fetched_at_utc}

Idempotent rules:
  * If ``manifest.history_complete`` is true and the run's outcome row
    is ``status=finished``: SKIP (no W&B call).
  * Otherwise resolve the W&B run by display_name, fetch only rows
    with ``trainer/global_step`` strictly greater than
    ``manifest.last_fetched_step``, append, update the manifest.
  * If the W&B run state is ``finished`` or ``failed``, mark
    ``history_complete = true`` so the next invocation skips it.

Designed to be safe to call from cron / CI: no W&B API calls beyond
what is needed to extend each run's cache.

Invocation::

    doppler run -- uv run python -m scripts.sweep.cache_history
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from scripts.sweep.fetch_outcomes import read_rounds

DEFAULT_KEYS = [
    "trainer/global_step",
    "_step",
    "val/epoch_NLL",
    "gen-val/sbm_accuracy",
    "gen-val/modularity_q",
    "gen-val/degree_mmd",
    "gen-val/clustering_mmd",
    "gen-val/orbit_mmd",
    "gen-val/spectral_mmd",
    "gen-val/full_chain_vlb",
    "gen-val/uniqueness",
    "gen-val/empirical_p_in",
    "gen-val/empirical_p_out",
    "diagnostics-val/progress/bits_per_edge",
]


def safe_uid(run_uid: str) -> str:
    return run_uid.replace("/", "__")


def manifest_path(cache_root: Path, run_uid: str) -> Path:
    return cache_root / safe_uid(run_uid) / "manifest.json"


def history_path(cache_root: Path, run_uid: str) -> Path:
    return cache_root / safe_uid(run_uid) / "history.jsonl"


def load_manifest(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {
            "wandb_run_id": None,
            "run_state": None,
            "last_fetched_step": -1,
            "terminal_step": None,
            "history_complete": False,
            "fetched_at_utc": None,
        }
    return json.loads(p.read_text())


def write_manifest(p: Path, m: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True))


def append_history(p: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def find_outcome_for(
    rounds: list[dict[str, Any]], run_uid: str
) -> dict[str, Any] | None:
    matches = [
        r for r in rounds if r.get("kind") == "outcome" and r.get("run_uid") == run_uid
    ]
    if not matches:
        return None
    return matches[-1]


def find_launches_for(
    rounds: list[dict[str, Any]], run_uid: str
) -> list[dict[str, Any]]:
    return [
        r for r in rounds if r.get("kind") == "launched" and r.get("run_uid") == run_uid
    ]


def fetch_run_history(
    *,
    entity: str,
    project: str,
    run_uid: str,
    min_step_exclusive: int,
    keys: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (new_rows_sorted_by_step, run_metadata).

    ``run_metadata`` is None if no W&B run was found. New rows have
    ``trainer/global_step > min_step_exclusive``. Rows are merged so
    that gen-val metrics (logged with custom step axis) align with
    NLL keys at the same trainer step.
    """
    import wandb  # local import; only needed at runtime

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_uid}))
    if not runs:
        return [], None
    runs.sort(key=lambda r: getattr(r, "created_at", "") or "", reverse=True)
    run = runs[0]

    by_step: dict[int, dict[str, Any]] = {}
    rows = list(run.scan_history(keys=keys, page_size=200))
    for r in rows:
        if r is None:
            continue
        s_raw = r.get("trainer/global_step")
        if s_raw is None:
            continue
        try:
            s = int(s_raw)
        except (TypeError, ValueError):
            continue
        if s <= min_step_exclusive:
            continue
        merged = by_step.setdefault(s, {"trainer/global_step": s})
        for k, v in r.items():
            if v is not None:
                merged[k] = v

    sorted_rows = [by_step[s] for s in sorted(by_step.keys())]
    metadata = {
        "wandb_run_id": run.id,
        "run_state": run.state,
        "summary_terminal_step": run.summary.get("trainer/global_step"),
    }
    return sorted_rows, metadata


def cache_one_run(
    *,
    cache_root: Path,
    run_uid: str,
    rounds: list[dict[str, Any]],
    entity: str,
    project: str,
    keys: list[str],
    force: bool,
) -> dict[str, Any]:
    """Returns a status dict describing what happened for this uid."""
    mp = manifest_path(cache_root, run_uid)
    hp = history_path(cache_root, run_uid)
    manifest = load_manifest(mp)

    outcome = find_outcome_for(rounds, run_uid)
    outcome_status = outcome.get("status") if outcome else None
    is_outcome_finished = outcome_status in {"finished", "finished_early", "failed"}

    if not force and manifest.get("history_complete") and is_outcome_finished:
        return {"run_uid": run_uid, "action": "skip_complete", "rows_added": 0}

    last_step = int(manifest.get("last_fetched_step") or -1)
    new_rows, meta = fetch_run_history(
        entity=entity,
        project=project,
        run_uid=run_uid,
        min_step_exclusive=last_step,
        keys=keys,
    )
    if meta is None:
        return {"run_uid": run_uid, "action": "skip_no_wandb_run", "rows_added": 0}

    append_history(hp, new_rows)

    new_last_step = int(new_rows[-1]["trainer/global_step"]) if new_rows else last_step
    run_state = meta["run_state"]
    is_history_complete = run_state in {"finished", "failed", "crashed"}
    new_manifest = {
        "wandb_run_id": meta["wandb_run_id"],
        "run_state": run_state,
        "last_fetched_step": new_last_step,
        "terminal_step": (
            int(meta["summary_terminal_step"])
            if meta["summary_terminal_step"] is not None
            else None
        ),
        "history_complete": is_history_complete,
        "fetched_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    write_manifest(mp, new_manifest)

    return {
        "run_uid": run_uid,
        "action": "fetched",
        "rows_added": len(new_rows),
        "last_step": new_last_step,
        "run_state": run_state,
        "history_complete": is_history_complete,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    _ = p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("wandb_export/sweep_cache"),
    )
    _ = p.add_argument("--entity", default="graph_denoise_team")
    _ = p.add_argument("--project", default="tmgg-smallest-config-sweep")
    _ = p.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    _ = p.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if manifest says history_complete.",
    )
    _ = p.add_argument(
        "--only",
        default=None,
        help="Comma-separated run_uid substrings; restrict to those that match.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rounds = read_rounds(args.rounds_jsonl)
    uids = sorted({r["run_uid"] for r in rounds if r.get("kind") == "launched"})
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        uids = [u for u in uids if any(w in u for w in wanted)]

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    args.cache_root.mkdir(parents=True, exist_ok=True)

    print(f"# {len(uids)} run_uids to consider; cache_root={args.cache_root}")
    n_skip = n_fetch = n_norun = 0
    rows_added_total = 0
    for uid in uids:
        try:
            status = cache_one_run(
                cache_root=args.cache_root,
                run_uid=uid,
                rounds=rounds,
                entity=args.entity,
                project=args.project,
                keys=keys,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic CLI
            print(f"# ERROR {uid}: {type(exc).__name__}: {exc}")
            continue
        if status["action"] == "skip_complete":
            n_skip += 1
            print(f"  SKIP    {uid}")
        elif status["action"] == "skip_no_wandb_run":
            n_norun += 1
            print(f"  NO-RUN  {uid}")
        else:
            n_fetch += 1
            rows_added_total += status["rows_added"]
            print(
                f"  FETCH   {uid}  +{status['rows_added']:3d} rows "
                f"last_step={status['last_step']}  state={status['run_state']}"
            )
    print(
        f"# done. fetched={n_fetch} (rows+={rows_added_total}) "
        f"skipped={n_skip} no_wandb_run={n_norun}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
