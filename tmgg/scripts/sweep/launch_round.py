"""Launch a single sweep round on Modal.

Reads a ``round.yaml`` (the per-round launch table from progress.md),
invokes the appropriate Modal wrapper for each row, parses Modal
output for the app ID, and appends a ``launched`` row to
``rounds.jsonl`` per the spec §4.3 schema.

Per spec §7 invariant 1: refuses to append a launched row if the
wrapper exited non-zero.

The W&B run ID is *not* known at launch time when Modal detaches.
Launched rows therefore have ``wandb_run_id: null``; fetch_outcomes.py
resolves it from the W&B API by matching ``run_uid == run.name``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

WRAPPER_BY_DATASET = {
    "spectre_sbm": "run-upstream-digress-sbm-modal-a100.zsh",
    "pyg_enzymes": "run-discrete-pyg-enzymes-modal-a100.zsh",
}

MODAL_APP_ID_RE = re.compile(r"\b(ap-[A-Za-z0-9]{10,})\b")


def config_hash(overrides: dict[str, Any]) -> str:
    blob = json.dumps(overrides, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def make_run_uid(
    *, dataset: str, round_no: int, axis_changed: str, cfg_hash: str
) -> str:
    return f"smallest-cfg/{dataset}/r{round_no}/{axis_changed}/{cfg_hash}"


def build_wrapper_invocation(
    *, dataset: str, run_uid: str, seed: int, overrides: dict[str, Any]
) -> list[str]:
    """Compose the wrapper command line.

    `wandb_name` is the Hydra key that maps to the W&B run-name; the
    existing tmgg-modal CLI resolves it via the wandb-logger config.
    If Task 10's first launch shows the W&B run name is auto-generated
    (i.e. ``wandb_name=`` was ignored), the operator must:
      1. Inspect the actual wandb-logger config interpolation
         (e.g. ``${run_name}`` or ``${experiment_name}``).
      2. Update ``WRAPPER_BY_DATASET`` or this helper to set the
         correct Hydra key.
      3. Re-run round 1 (the misnamed runs are non-fatal — fetch_outcomes
         simply can't resolve them by display_name).
    """
    wrapper = WRAPPER_BY_DATASET[dataset]
    cmd: list[str] = [f"./{wrapper}"]
    cmd += [f"seed={seed}", f"wandb_name={run_uid}"]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    return cmd


def parse_modal_app_id(stdout: str) -> str | None:
    """Find the first ``ap-...`` token in wrapper output, or None."""
    m = MODAL_APP_ID_RE.search(stdout)
    return m.group(1) if m else None


def append_launched_row(
    *,
    rounds_jsonl: Path,
    row: dict[str, Any],
) -> None:
    with rounds_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False) + "\n")


def launch_one(
    *,
    dataset: str,
    round_no: int,
    axis_changed: str,
    axis_value: Any,
    seed: int,
    step_cap: int,
    overrides: dict[str, Any],
    rounds_jsonl: Path,
    session_tag: str,
    dry_run: bool,
) -> dict[str, Any]:
    cfg_hash = config_hash(overrides)
    run_uid = make_run_uid(
        dataset=dataset, round_no=round_no, axis_changed=axis_changed, cfg_hash=cfg_hash
    )
    cmd = build_wrapper_invocation(
        dataset=dataset, run_uid=run_uid, seed=seed, overrides=overrides
    )
    print("LAUNCH:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return {"run_uid": run_uid, "dry_run": True}
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"wrapper exited {proc.returncode} for run_uid={run_uid}; "
            f"refusing to append launched row per spec §7 invariant 1"
        )
    app_id = parse_modal_app_id(proc.stdout) or parse_modal_app_id(proc.stderr)
    row = {
        "kind": "launched",
        "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "run_uid": run_uid,
        "round": round_no,
        "dataset": dataset,
        "model_arch": "digress_official",
        "axis_changed": axis_changed,
        "axis_value": axis_value,
        "seed": seed,
        "step_cap": step_cap,
        "config_hash": cfg_hash,
        "config_overrides": overrides,
        "wandb_entity": "graph_denoise_team",
        "wandb_project": "tmgg-smallest-config-sweep",
        "wandb_run_id": None,
        "wandb_group": f"round-{round_no}-{dataset}-{axis_changed}",
        "modal_app_id": app_id,
        "launched_by_session": session_tag,
    }
    append_launched_row(rounds_jsonl=rounds_jsonl, row=row)
    return row


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-yaml", type=Path, required=True)
    p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--session-tag", default=dt.datetime.now().strftime("claude-%Y-%m-%d-%H")
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    spec = yaml.safe_load(args.round_yaml.read_text())
    round_no = int(spec["round"])
    for entry in spec["launches"]:
        launch_one(
            dataset=entry["dataset"],
            round_no=round_no,
            axis_changed=entry["axis_changed"],
            axis_value=entry["axis_value"],
            seed=int(entry["seed"]),
            step_cap=int(entry["step_cap"]),
            overrides=dict(entry["overrides"]),
            rounds_jsonl=args.rounds_jsonl,
            session_tag=args.session_tag,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
