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

from scripts.sweep.check_threshold import (
    check_run,  # MetricMissingError / AnchorMissingError propagate from check_run per §7 invariant 2
)

REQUIRED_DIAGNOSTIC_KEYS = (
    "diagnostics-train/opt-health/grad_snr",
    "diagnostics-train/opt-health/grad_cosine",
    "diagnostics-train/opt-health/update_to_weight",
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
    """Launched rows that have no matching outcome row."""
    outcome_uids = {r["run_uid"] for r in rows if r.get("kind") == "outcome"}
    return [
        r
        for r in rows
        if r.get("kind") == "launched" and r["run_uid"] not in outcome_uids
    ]


def resolve_wandb_run(
    *, entity: str, project: str, run_name: str
) -> wandb.apis.public.Run:
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


def fetch_block_keyed_diagnostic(
    run: wandb.apis.public.Run, key_prefix: str
) -> dict[str, float]:
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


def build_outcome_row(
    *,
    launched: dict[str, Any],
    run: wandb.apis.public.Run,
    anchors_path: Path,
) -> dict[str, Any]:
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

    breakdown = check_run(
        metrics={
            k: float(v) for k, v in metrics_subset.items() if isinstance(v, int | float)
        },
        dataset=launched["dataset"],
        anchors_path=anchors_path,
    )

    diag_terminal = {
        prefix.rsplit("/", 1)[-1]: fetch_block_keyed_diagnostic(run, prefix)
        for prefix in REQUIRED_DIAGNOSTIC_KEYS
    }

    row = {
        "kind": "outcome",
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "run_uid": launched["run_uid"],
        "wandb_run_id": run.id,
        "status": "finished",
        "terminal_step": int(summary.get("_step", launched["step_cap"])),
        "metrics": {
            k: float(v) for k, v in metrics_subset.items() if isinstance(v, int | float)
        },
        "diagnostics_terminal": diag_terminal,
        "threshold_pass": breakdown["pass"],
        "threshold_breakdown": breakdown["per_metric"],
    }
    return row


def _classify_failure(run: wandb.apis.public.Run) -> str:
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
