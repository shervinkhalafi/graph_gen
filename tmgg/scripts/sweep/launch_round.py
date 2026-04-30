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

# The wrapper subprocess (``tmgg-modal run --detach``) prints a stable
# ``MODAL_FUNCTION_CALL_ID=fc-...`` line so the launcher can record the
# trainer's call ID for the manual-kill workflow (see
# ``scripts/sweep/kill_call.py``). Anchored on the prefix to avoid
# false-positives from log lines that happen to mention an ``fc-...`` ID.
MODAL_FUNCTION_CALL_ID_RE = re.compile(r"MODAL_FUNCTION_CALL_ID=(fc-[A-Za-z0-9]+)")

# All sweep runs land in this single W&B project. ``fetch_outcomes`` and
# ``watch_runs`` query it by default and ignore the ``wandb_project`` field
# in rounds.jsonl rows, so a wrapper that silently lands in a different
# project is invisible to the rest of the pipeline.
CANONICAL_WANDB_PROJECT = "tmgg-smallest-config-sweep"

# The wrappers echo the rendered ``tmgg-modal run ...`` command (which
# carries the wrapper's own ``wandb_project=<value>`` token) before
# spawning. We scan that echo to assert the effective project matches the
# canonical one — guarding against the round-1 incident where the SBM
# wrapper defaulted ``WANDB_PROJECT=discrete-diffusion`` and the runs
# vanished from the sweep namespace.
WRAPPER_WANDB_PROJECT_RE = re.compile(r"\bwandb_project=([A-Za-z0-9_\-]+)")


def config_hash(overrides: dict[str, Any]) -> str:
    blob = json.dumps(overrides, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def make_run_uid(
    *, dataset: str, round_no: int, axis_changed: str, cfg_hash: str
) -> str:
    return f"smallest-cfg/{dataset}/r{round_no}/{axis_changed}/{cfg_hash}"


def load_async_eval_schedule(schedule_path: Path) -> list[int]:
    """Read the integer ``schedule`` list from an ``eval_schedule_*.yaml`` file.

    The YAML schema is produced by ``scripts/sweep/eval_schedule.py``:

    .. code-block:: yaml

        dataset: spectre_sbm
        n_evals: 24
        total_steps: 100000
        params: {rho_min: ..., rho_max: ..., s_p: ..., expected_knee_s_k: ...}
        schedule: [2590, 5237, 8011, ...]
        doc: "..."

    Only the ``schedule`` key is consumed here; we cast to ``int`` so we
    inline an unambiguous Hydra list literal (no quotes, no decimals).
    """
    payload = yaml.safe_load(schedule_path.read_text())
    if "schedule" not in payload:
        raise KeyError(
            f"{schedule_path}: missing required 'schedule' key "
            f"(produced by scripts/sweep/eval_schedule.py)"
        )
    return [int(s) for s in payload["schedule"]]


def build_wrapper_invocation(
    *,
    dataset: str,
    run_uid: str,
    seed: int,
    overrides: dict[str, Any],
    async_eval_schedule_path: Path | None = None,
    async_eval_gpu_tier: str = "standard",
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

    Async-eval mode (``async_eval_schedule_path`` is not None) appends
    the Hydra overrides that bind the ``default_with_async_eval``
    callback group, enable the ``async_eval_spawn`` callback, inline the
    integer schedule list, and forward ``run_uid`` + ``gpu_tier`` to the
    spawned worker. The override syntax uses ``base/callbacks=...`` (no
    leading ``+``) because ``_base_infra.yaml`` already binds
    ``base/callbacks: default``; ``+`` would attempt an append, which
    Hydra rejects.
    """
    wrapper = WRAPPER_BY_DATASET[dataset]
    cmd: list[str] = [f"./{wrapper}"]
    cmd += [f"seed={seed}", f"+wandb_name={run_uid}"]
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    if async_eval_schedule_path is not None:
        schedule = load_async_eval_schedule(async_eval_schedule_path)
        # Hydra accepts list literals via ``key=[a,b,c]`` on the CLI; no
        # spaces inside the brackets and no quoting around individual
        # integers.
        schedule_literal = "[" + ",".join(str(s) for s in schedule) + "]"
        cmd += [
            "base/callbacks=default_with_async_eval",
            "callbacks.async_eval_spawn.enabled=true",
            f"callbacks.async_eval_spawn.schedule={schedule_literal}",
            f"callbacks.async_eval_spawn.run_uid={run_uid}",
            f"callbacks.async_eval_spawn.gpu_tier={async_eval_gpu_tier}",
        ]
    return cmd


def parse_modal_app_id(stdout: str) -> str | None:
    """Find the first ``ap-...`` token in wrapper output, or None."""
    m = MODAL_APP_ID_RE.search(stdout)
    return m.group(1) if m else None


def parse_modal_function_call_id(stdout: str) -> str | None:
    """Extract the trainer's ``fc-...`` FunctionCall ID from wrapper output.

    The wrapper prints ``MODAL_FUNCTION_CALL_ID=fc-...`` after a
    detached spawn. Returning ``None`` is non-fatal: legacy runs and
    dry-runs do not emit the marker, and the manual-kill workflow then
    falls back to ``modal app stop <app_id>``.
    """
    m = MODAL_FUNCTION_CALL_ID_RE.search(stdout)
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
    async_eval_schedule_path: Path | None = None,
    async_eval_gpu_tier: str = "standard",
) -> dict[str, Any]:
    cfg_hash = config_hash(overrides)
    run_uid = make_run_uid(
        dataset=dataset, round_no=round_no, axis_changed=axis_changed, cfg_hash=cfg_hash
    )
    cmd = build_wrapper_invocation(
        dataset=dataset,
        run_uid=run_uid,
        seed=seed,
        overrides=overrides,
        async_eval_schedule_path=async_eval_schedule_path,
        async_eval_gpu_tier=async_eval_gpu_tier,
    )
    print("LAUNCH:", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        # Mirror the production-row schema's async-eval keys so dry-run
        # consumers can audit the full plan; downstream tests rely on
        # ``async_eval_enabled`` being absent or false when the flag is off.
        dry: dict[str, Any] = {"run_uid": run_uid, "dry_run": True}
        if async_eval_schedule_path is not None:
            dry["async_eval_enabled"] = True
            dry["async_eval_schedule_path"] = str(async_eval_schedule_path)
            dry["async_eval_gpu_tier"] = async_eval_gpu_tier
        return dry
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"wrapper exited {proc.returncode} for run_uid={run_uid}; "
            f"refusing to append launched row per spec §7 invariant 1"
        )
    project_match = WRAPPER_WANDB_PROJECT_RE.search(
        proc.stdout
    ) or WRAPPER_WANDB_PROJECT_RE.search(proc.stderr)
    if project_match is None:
        raise RuntimeError(
            f"wrapper output for run_uid={run_uid} did not contain a "
            f"'wandb_project=...' token; cannot verify the run landed in "
            f"the canonical sweep project ({CANONICAL_WANDB_PROJECT}). "
            f"Inspect the wrapper {WRAPPER_BY_DATASET[dataset]} or "
            f"the WRAPPER_WANDB_PROJECT_RE regex."
        )
    actual_project = project_match.group(1)
    if actual_project != CANONICAL_WANDB_PROJECT:
        raise RuntimeError(
            f"wrapper for run_uid={run_uid} rendered "
            f"wandb_project={actual_project!r} but the canonical sweep "
            f"project is {CANONICAL_WANDB_PROJECT!r}. The trainer pod will "
            f"land in the wrong W&B namespace and fetch_outcomes/watch_runs "
            f"will silently miss it. Fix the wrapper default "
            f"({WRAPPER_BY_DATASET[dataset]}) or set "
            f"WANDB_PROJECT={CANONICAL_WANDB_PROJECT} in the environment, "
            f"then relaunch."
        )
    app_id = parse_modal_app_id(proc.stdout) or parse_modal_app_id(proc.stderr)
    fc_id = parse_modal_function_call_id(proc.stdout) or parse_modal_function_call_id(
        proc.stderr
    )
    row: dict[str, Any] = {
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
        "wandb_project": CANONICAL_WANDB_PROJECT,
        "wandb_run_id": None,
        "wandb_group": f"round-{round_no}-{dataset}-{axis_changed}",
        "modal_app_id": app_id,
        # The trainer's FunctionCall ID for direct ``modal.FunctionCall.cancel``
        # via ``scripts.sweep.kill_call``. ``None`` when the wrapper did not
        # emit the marker (legacy output / blocking mode); operators then
        # fall back to ``modal app stop <modal_app_id>``.
        "modal_function_call_id": fc_id,
        "launched_by_session": session_tag,
    }
    if async_eval_schedule_path is not None:
        row["async_eval_enabled"] = True
        row["async_eval_schedule_path"] = str(async_eval_schedule_path)
        row["async_eval_gpu_tier"] = async_eval_gpu_tier
    append_launched_row(rounds_jsonl=rounds_jsonl, row=row)
    return row


# Sentinel used by argparse when ``--async-eval`` is passed bare (no value).
# We prefer this over ``nargs='?'`` + ``const=True`` to keep mypy happy with a
# concrete sentinel type and to make the "no value supplied" case unambiguous.
_ASYNC_EVAL_DEFAULT_SENTINEL = "__use_default_path__"


def _default_schedule_path(dataset: str) -> Path:
    """Resolve the default ``eval_schedule_<dataset>.yaml`` path.

    Matches ``scripts/sweep/eval_schedule.py``'s ``--out`` default.
    """
    return (
        Path("docs/experiments/sweep/smallest-config-2026-04-29")
        / f"eval_schedule_{dataset}.yaml"
    )


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
    p.add_argument(
        "--async-eval",
        nargs="?",
        const=_ASYNC_EVAL_DEFAULT_SENTINEL,
        default=None,
        help=(
            "Enable async-eval mode for every launch in the round. "
            "Optionally takes a path to an ``eval_schedule_<dataset>.yaml`` "
            "file; if omitted, defaults to "
            "``docs/experiments/sweep/smallest-config-2026-04-29/"
            "eval_schedule_<dataset>.yaml`` (per-launch dataset). Per-launch "
            "``async_eval: true`` entries in round.yaml override this CLI default "
            "when the CLI flag is unset."
        ),
    )
    p.add_argument(
        "--async-eval-gpu-tier",
        default="standard",
        choices=["debug", "standard", "fast"],
        help="GPU tier for the spawned async-eval workers (default: standard).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    spec = yaml.safe_load(args.round_yaml.read_text())
    round_no = int(spec["round"])
    for entry in spec["launches"]:
        # Async-eval activation: CLI flag wins, then per-launch entry, else off.
        # Per-launch ``async_eval_gpu_tier`` overrides the CLI tier.
        per_launch_async = bool(entry.get("async_eval", False))
        per_launch_tier = entry.get("async_eval_gpu_tier")
        if args.async_eval is not None:
            if args.async_eval == _ASYNC_EVAL_DEFAULT_SENTINEL:
                schedule_path: Path | None = _default_schedule_path(entry["dataset"])
            else:
                schedule_path = Path(args.async_eval)
            tier = per_launch_tier or args.async_eval_gpu_tier
        elif per_launch_async:
            schedule_path = _default_schedule_path(entry["dataset"])
            tier = per_launch_tier or args.async_eval_gpu_tier
        else:
            schedule_path = None
            tier = args.async_eval_gpu_tier
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
            async_eval_schedule_path=schedule_path,
            async_eval_gpu_tier=tier,
        )


if __name__ == "__main__":
    main()
