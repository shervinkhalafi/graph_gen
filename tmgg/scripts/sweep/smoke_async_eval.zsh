#!/usr/bin/env zsh

set -euo pipefail

# Smoke launcher for the async-eval architecture.
#
# A self-contained two-step gate that exercises the entire async-eval
# wiring on the cheapest configuration that still proves it works:
# 1k training steps on ``spectre_sbm`` with async-eval calls fired at
# steps 200 and 800. Pass criteria live in ``Agent2-comms.md`` under
# "Async-eval ready for smoke" — three holdable conditions on W&B
# history rows, the eval-manifest JSONL, and trainer wall-clock.
#
# What this script does, in order:
#   1. (optional) ``uv run tmgg-modal deploy`` so the async-eval Modal
#      wrappers are registered.
#   2. Materialises a fresh ``round.yaml`` under ``/tmp`` with the smoke
#      launch table baked in.
#   3. Invokes ``scripts.sweep.launch_round`` against that round.yaml,
#      pointing it at ``smoke_eval_schedule.yaml`` (a 2-step schedule
#      committed alongside this script).
#   4. Reads the most recent ``kind=launched`` row from ``rounds.jsonl``
#      and prints copy-paste verification commands for the three pass
#      criteria.
#
# Style notes: matches ``run-upstream-digress-sbm-modal.zsh`` — env-var
# defaults via ``: "${VAR:=default}"``, optional doppler prefix, no hard
# fails on the doppler path so the script also works without a Doppler
# token configured locally.

usage() {
  cat <<'EOF'
Usage: scripts/sweep/smoke_async_eval.zsh [--help]

Smoke launcher for the async-eval architecture: deploy, write a fresh
/tmp round.yaml, invoke scripts.sweep.launch_round with the committed
2-step smoke schedule, then print verification commands.

Environment variables:
  WANDB_PROJECT  W&B project the smoke run writes to.
                 Default: tmgg-smallest-config-sweep.
  WANDB_ENTITY   W&B entity. Default: graph_denoise_team.
  USE_DOPPLER    1 = wrap calls in ``doppler run --``. Default: 1.
  DEPLOY         1 = run ``uv run tmgg-modal deploy`` first. Default: 1.
  DRY_RUN        1 = pass ``--dry-run`` to the launcher (no Modal
                 spend, no rounds.jsonl row). Default: 0.

Pass criteria (all three required for smoke green):
  1. W&B history has ``gen-val/sbm_accuracy`` rows at steps 200, 800.
  2. ``eval_manifest.jsonl`` contains 2 ``spawned`` + 2 ``completed``
     rows, scheduled-step paired.
  3. Trainer wall-clock ~ training time alone (no eval blocking).
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

# --- defaults --------------------------------------------------------
: "${WANDB_PROJECT:=tmgg-smallest-config-sweep}"
: "${WANDB_ENTITY:=graph_denoise_team}"
: "${USE_DOPPLER:=1}"
: "${DEPLOY:=1}"
: "${DRY_RUN:=0}"
export WANDB_PROJECT WANDB_ENTITY

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
schedule_yaml="${repo_root}/docs/experiments/sweep/smallest-config-2026-04-29/smoke_eval_schedule.yaml"
rounds_jsonl="${repo_root}/docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"

if [[ ! -f "${schedule_yaml}" ]]; then
  print -ru2 -- "ERROR: smoke schedule not found: ${schedule_yaml}"
  exit 1
fi

run_prefixed() {
  if [[ "${USE_DOPPLER}" == "1" ]]; then
    doppler run -- "$@"
  else
    "$@"
  fi
}

# --- 1. deploy -------------------------------------------------------
if [[ "${DEPLOY}" == "1" ]]; then
  print -r -- "[smoke] Deploying tmgg-modal app..."
  run_prefixed uv run tmgg-modal deploy
else
  print -r -- "[smoke] DEPLOY=0 — skipping ``uv run tmgg-modal deploy``."
fi

# --- 2. write fresh /tmp round.yaml ----------------------------------
ts=$(date -u +%Y%m%dT%H%M%SZ)
round_yaml="/tmp/smoke-round-${ts}.yaml"
cat >"${round_yaml}" <<'EOF'
session_tag: smoke-async-eval
round: 0
launches:
  - dataset: spectre_sbm
    axis_changed: smoke
    axis_value: smoke
    seed: 0
    step_cap: 1000
    async_eval: true
    overrides:
      trainer.max_steps: 1000
      trainer.val_check_interval: 1000  # disable in-band gen-val; async does it
      force_fresh: true                 # bypass skip-completed check (key exists in default config; no + prefix)
EOF
print -r -- "[smoke] Wrote round.yaml: ${round_yaml}"

# --- 3. invoke launcher ---------------------------------------------
typeset -a launcher_cmd
launcher_cmd=(
  uv run python -m scripts.sweep.launch_round
  --round-yaml "${round_yaml}"
  --async-eval "${schedule_yaml}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  launcher_cmd+=(--dry-run)
fi

print -r -- "[smoke] Launching:"
printf '  %q' "${launcher_cmd[@]}"
print

run_prefixed "${launcher_cmd[@]}"

# --- 4. print verification commands ---------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
  print -r -- "[smoke] DRY_RUN=1 — no rounds.jsonl row appended; skipping verification block."
  exit 0
fi

if [[ ! -f "${rounds_jsonl}" ]]; then
  print -ru2 -- "[smoke] WARN: rounds.jsonl missing: ${rounds_jsonl}"
  exit 0
fi

# Pull the most recent kind=launched row.
last_launched=$(
  uv run python - <<PY
import json, sys
from pathlib import Path
rows = [
    json.loads(line)
    for line in Path("${rounds_jsonl}").read_text().splitlines()
    if line.strip()
]
launched = [r for r in rows if r.get("kind") == "launched"]
if not launched:
    sys.exit(0)
print(json.dumps(launched[-1]))
PY
)

if [[ -z "${last_launched}" ]]; then
  print -ru2 -- "[smoke] WARN: no kind=launched rows in ${rounds_jsonl}; cannot print verification commands."
  exit 0
fi

run_uid=$(print -r -- "${last_launched}" | uv run python -c 'import json,sys; print(json.loads(sys.stdin.read())["run_uid"])')
wandb_run_id=$(print -r -- "${last_launched}" | uv run python -c 'import json,sys; v=json.loads(sys.stdin.read()).get("wandb_run_id"); print(v if v is not None else "")')

print
print -r -- "============================================================"
print -r -- "[smoke] Launched. run_uid=${run_uid}"
print -r -- "[smoke] wandb_run_id=${wandb_run_id:-<unresolved-yet>}"
print -r -- "============================================================"
print -r -- "Next-step verification commands:"
print
print -r -- "1. W&B history scan (criterion 1: gen-val rows at steps 200, 800)"
if [[ -n "${wandb_run_id}" ]]; then
  print -r -- "   uv run python -c 'import wandb; api = wandb.Api(); run = api.run(\"${WANDB_ENTITY}/${WANDB_PROJECT}/${wandb_run_id}\"); print(list(run.scan_history(keys=[\"gen-val/sbm_accuracy\", \"trainer/global_step\"])))'"
else
  print -r -- "   # wandb_run_id not yet resolved — fetch it first:"
  print -r -- "   uv run python -m scripts.sweep.fetch_outcomes  # resolves run_uid -> wandb_run_id"
  print -r -- "   # Then run the scan_history command above with the resolved id."
fi
print
print -r -- "2. Modal volume manifest fetch (criterion 2: 2 spawned + 2 completed rows)"
print -r -- "   modal volume get tmgg-outputs ${run_uid}/eval_manifest.jsonl /tmp/eval_manifest.jsonl"
print -r -- "   cat /tmp/eval_manifest.jsonl"
print
print -r -- "3. Trainer wall-clock (criterion 3: ~training time alone, ~7 min on A100)"
print -r -- "   # Inspect Modal app logs or W&B run wall-clock; trainer should not block on evals."
print
print -r -- "All three criteria must hold for the smoke to be green."
print -r -- "Pass-criteria reference: Agent2-comms.md, 'Async-eval ready for smoke'."
