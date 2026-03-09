# Simplify Modal Execution to CLI Transport Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `TaskInput`/`TaskOutput`/`_dispatch_task` machinery with a generic Modal function that runs experiment CLIs as subprocesses, writing an append-only `confirmation.jsonl` on the volume. Remove Tigris storage entirely.

**Architecture:** The Modal function is a pure subprocess transport: it receives a CLI command name, a config name, and Hydra overrides, then runs the CLI. For sweep mode (pre-resolved configs), the caller uploads YAML files to the Modal volume beforehand and the function runs against those. Experiments handle their own W&B logging (they already do). W&B is the source of truth; the Modal volume is the backup. A single `confirmation.jsonl` file provides a lightweight execution log.

**Tech Stack:** Python, Modal, Hydra/OmegaConf, subprocess, JSONL

---

## Design Decisions (confirmed)

1. **Pure subprocess invocation.** `modal_run_cli(cmd, config_name, overrides)` runs `subprocess.run([cmd, "--config-name="+config_name] + overrides)`. No Hydra Compose API, no importlib.

2. **Sweep configs via volume.** For sweeps, `config_builder.py` produces fully-resolved config dicts. These are serialized to YAML and uploaded to the Modal volume (e.g. `/data/outputs/sweep_configs/{sweep_id}/config_001.yaml`). The Modal function runs `cmd --config-path=/data/outputs/sweep_configs/{sweep_id} --config-name=config_001`. Since configs are already fully resolved (no `defaults:` chain), they work standalone.

3. **Tigris removed entirely.** W&B is the source of truth for metrics. The Modal volume stores outputs and the confirmation log. `storage.py` is deleted; `tigris-credentials` is removed from all functions.

---

## Context: What Exists Today

### Current data flow (training)

```
ModalRunner._create_task_input()
  -> prepare_config_for_remote(config)  # strips logger/paths, resolves, extracts W&B
  -> TaskInput(config=dict, run_id, gpu_tier, timeout, tags, task_type)
  -> asdict(task_input)  # serialize to dict
  -> modal_fn.remote(task_dict)  # send to Modal

Inside Modal container:
  modal_execute_task(task_dict)
    -> TaskInput(**task_dict)
    -> execute_task(task, get_storage=get_storage_from_env)
      -> _resolve_execution_paths(run_id)  # sets output_dir from env
      -> OmegaConf.create(task.config)     # reconstruct config
      -> merge additional_tags into _wandb_config
      -> _write_volume_confirmation("started")
      -> _dispatch_task(task_type, config, run_experiment_fn)
        -> run_experiment(config)  OR  execute_eigenstructure(config)  etc.
      -> extract best_val_loss, upload checkpoint to Tigris
      -> upload TaskOutput metrics to Tigris
      -> _write_volume_confirmation("completed")
      -> return TaskOutput as dict
```

### Target data flow

```
ModalRunner.run_experiment(config, cmd="tmgg-discrete-gen")
  -> config_yaml = OmegaConf.to_yaml(config)   # or overrides list
  -> modal_fn.remote(cmd, config_name, overrides, run_id)

Inside Modal container:
  modal_run_cli(cmd, config_name, overrides, run_id)
    -> append_confirmation(confirmation.jsonl, run_id, "started", cmd)
    -> subprocess.run([cmd, "--config-name="+config_name] + overrides)
    -> append_confirmation(confirmation.jsonl, run_id, "completed"|"failed")
    -> return {"status": ..., "run_id": ..., "exit_code": ...}
```

For sweep mode:

```
config_builder.py produces list of resolved config dicts
ModalRunner.upload_sweep_configs(configs, sweep_id)
  -> write each to YAML on Modal volume: /data/outputs/sweep_configs/{sweep_id}/config_NNN.yaml
for each config:
  modal_fn.remote(cmd, config_path="/data/outputs/sweep_configs/{sweep_id}", config_name="config_NNN", run_id=...)
```

### What this eliminates

| Component | Reason |
|-----------|--------|
| `TaskInput.task_type` | Dispatch replaced by CLI command name |
| `TaskOutput` | W&B + volume confirmation replaces structured return |
| `execute_task()` | Each experiment runs its own CLI |
| `_dispatch_task()` | No more type-based routing |
| `_resolve_execution_paths()` | Hydra resolves paths itself |
| `_write_volume_confirmation()` | Replaced by `append_confirmation()` JSONL |
| `prepare_config_for_remote()` | Config serialized as YAML or passed as overrides |
| `TigrisStorage` / `storage.py` | W&B is source of truth; volume is backup |
| `StorageProtocol` | No more storage injection |

### What survives

| Component | Why |
|-----------|-----|
| `_extract_wandb_config()` in task.py | Used by `config_builder.py` for sweep config generation |
| `CloudRunner` / `ExperimentResult` | Clean abstraction, ModalRunner still implements it |
| Evaluation functions in `_functions.py` | Self-contained with own `EvaluationInput`/`EvaluationOutput`. Separate migration. |
| `modal_list_checkpoints` | Reads from volume, unrelated to execution |

---

## Files Overview

### Files to create
- `src/tmgg/modal/confirmation.py` — append-only JSONL writer/reader
- `tests/modal/test_confirmation.py`

### Files to heavily modify
- `src/tmgg/modal/_functions.py` — replace `modal_execute_task*` with `modal_run_cli*`; remove `tigris_secret` from CLI functions
- `src/tmgg/modal/runner.py` — simplify `ModalRunner` to build CLI args; remove Tigris dependency
- `src/tmgg/experiment_utils/task.py` — remove `TaskInput`, `TaskOutput`, `execute_task`, `_dispatch_task`, keep `_extract_wandb_config`

### Files to delete
- `src/tmgg/modal/storage.py`

### Files with minor edits
- `src/tmgg/experiments/eigenstructure_study/execute.py` — remove `best_val_loss`/`best_model_path` shims
- `src/tmgg/experiments/embedding_study/execute.py` — same
- `src/tmgg/exp_configs/base_config_eigenstructure.yaml` — remove `task_type`
- `src/tmgg/exp_configs/base_config_embedding_study.yaml` — remove `task_type`
- `src/tmgg/exp_configs/*.yaml` — add `_cli_cmd` field to each experiment config
- `src/tmgg/experiments/stages/runner.py` — update ModalRunner usage
- `src/tmgg/experiment_utils/cloud/base.py` — simplify `ExperimentResult`
- `src/tmgg/modal/volumes.py` — no changes needed (already cleaned up)

### Test files to update
- `tests/experiment_utils/test_task.py` — remove tests for deleted functions
- `tests/modal/test_runner.py` — update for new ModalRunner interface

---

## Task 1: Create confirmation.jsonl writer

**Files:**
- Create: `src/tmgg/modal/confirmation.py`
- Create: `tests/modal/test_confirmation.py`

**Step 1: Write the failing test**

```python
# tests/modal/test_confirmation.py
"""Tests for append-only experiment confirmation log.

Test rationale:
    The confirmation log provides a backup record of experiment execution
    on the Modal volume, independent of W&B. It must be append-only (safe
    for concurrent writers) and parseable for status queries.

Invariants:
    - Each append produces exactly one newline-terminated JSON line
    - Reading back parses all lines correctly
    - Multiple appends accumulate (no overwriting)
    - run_id is present in every entry
"""

import json
from pathlib import Path

from tmgg.modal.confirmation import append_confirmation, read_confirmations


class TestAppendConfirmation:
    def test_creates_file_if_missing(self, tmp_path: Path):
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="abc123", status="started")
        assert log_path.exists()

    def test_appends_valid_jsonl(self, tmp_path: Path):
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="abc123", status="started")
        append_confirmation(log_path, run_id="abc123", status="completed", exit_code=0)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry_0 = json.loads(lines[0])
        assert entry_0["run_id"] == "abc123"
        assert entry_0["status"] == "started"
        assert "timestamp" in entry_0

        entry_1 = json.loads(lines[1])
        assert entry_1["status"] == "completed"
        assert entry_1["exit_code"] == 0

    def test_includes_wandb_run_id_when_provided(self, tmp_path: Path):
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(
            log_path, run_id="abc123", status="completed", wandb_run_id="wandb-xyz"
        )
        entry = json.loads(log_path.read_text().strip())
        assert entry["wandb_run_id"] == "wandb-xyz"

    def test_extra_fields_preserved(self, tmp_path: Path):
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(
            log_path, run_id="abc123", status="completed", cmd="tmgg-discrete-gen"
        )
        entry = json.loads(log_path.read_text().strip())
        assert entry["cmd"] == "tmgg-discrete-gen"


class TestReadConfirmations:
    def test_reads_all_entries(self, tmp_path: Path):
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="r1", status="started")
        append_confirmation(log_path, run_id="r1", status="completed")
        append_confirmation(log_path, run_id="r2", status="started")

        entries = read_confirmations(log_path)
        assert len(entries) == 3

    def test_filter_by_run_id(self, tmp_path: Path):
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="r1", status="started")
        append_confirmation(log_path, run_id="r2", status="started")
        append_confirmation(log_path, run_id="r1", status="completed")

        entries = read_confirmations(log_path, run_id="r1")
        assert len(entries) == 2
        assert all(e["run_id"] == "r1" for e in entries)

    def test_returns_empty_for_missing_file(self, tmp_path: Path):
        log_path = tmp_path / "nonexistent.jsonl"
        assert read_confirmations(log_path) == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/modal/test_confirmation.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/tmgg/modal/confirmation.py
"""Append-only experiment confirmation log.

Writes one JSON line per status event to a JSONL file on the Modal volume.
Provides a lightweight backup record of experiment execution independent
of W&B. Each write appends a single newline-terminated JSON object.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONFIRMATION_PATH = Path("/data/outputs/confirmation.jsonl")


def append_confirmation(
    path: Path,
    *,
    run_id: str,
    status: str,
    wandb_run_id: str | None = None,
    **extra: Any,
) -> None:
    """Append a status entry to the confirmation log.

    Parameters
    ----------
    path
        Path to the JSONL file.
    run_id
        Experiment run identifier.
    status
        One of "started", "completed", "failed".
    wandb_run_id
        W&B run ID if available (typically known only after completion).
    **extra
        Additional fields (cmd, exit_code, error, etc.).
    """
    entry: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("HOSTNAME", "unknown"),
    }
    if wandb_run_id is not None:
        entry["wandb_run_id"] = wandb_run_id
    entry.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_confirmations(
    path: Path,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read entries from the confirmation log.

    Parameters
    ----------
    path
        Path to the JSONL file.
    run_id
        If provided, return only entries matching this run_id.

    Returns
    -------
    list[dict]
        Parsed entries, optionally filtered.
    """
    if not path.exists():
        return []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if run_id is None or entry.get("run_id") == run_id:
                entries.append(entry)
    return entries
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/modal/test_confirmation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tmgg/modal/confirmation.py tests/modal/test_confirmation.py
git commit -m "feat(modal): add append-only confirmation.jsonl writer"
```

---

## Task 2: Replace `modal_execute_task` with generic `modal_run_cli`

**Files:**
- Modify: `src/tmgg/modal/_functions.py`

**Step 1: Rewrite the execution functions**

Replace the three `modal_execute_task` variants with three `modal_run_cli` variants. The new function:

```python
@app.function(
    name="modal_run_cli",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=DEFAULT_TIMEOUTS["standard"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli(
    cmd: str,
    config_name: str,
    run_id: str,
    overrides: list[str] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI inside the Modal container.

    Parameters
    ----------
    cmd
        CLI entry point name (e.g. "tmgg-discrete-gen").
    config_name
        Hydra config name (e.g. "base_config_discrete_diffusion_generative").
    run_id
        Experiment identifier for the confirmation log.
    overrides
        Hydra CLI overrides (e.g. ["seed=42", "data.num_nodes=20"]).
    config_path
        Override Hydra's config search path. Used for sweep configs
        uploaded to the volume. If None, uses the experiment's default.
    """
    import subprocess

    from tmgg.modal.confirmation import DEFAULT_CONFIRMATION_PATH, append_confirmation

    append_confirmation(
        DEFAULT_CONFIRMATION_PATH, run_id=run_id, status="started", cmd=cmd
    )

    cli_args = [cmd]
    if config_path:
        cli_args.append(f"--config-path={config_path}")
    cli_args.append(f"--config-name={config_name}")
    if overrides:
        cli_args.extend(overrides)

    result = subprocess.run(cli_args, capture_output=True, text=True)

    wandb_run_id = _extract_wandb_run_id(result.stdout)

    if result.returncode == 0:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="completed",
            exit_code=0,
            wandb_run_id=wandb_run_id,
            cmd=cmd,
        )
        return {"status": "completed", "run_id": run_id, "wandb_run_id": wandb_run_id}
    else:
        error_tail = result.stderr[-500:] if result.stderr else ""
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            exit_code=result.returncode,
            error=error_tail,
            cmd=cmd,
        )
        return {
            "status": "failed",
            "run_id": run_id,
            "exit_code": result.returncode,
            "error": error_tail,
        }
```

Also create `modal_run_cli_fast` (A100) and `modal_run_cli_debug` (T4) variants that delegate via `.local()`.

Remove `tigris_secret` from CLI function secrets. Keep it on eval functions.

Remove the old `modal_execute_task`, `modal_execute_task_fast`, `modal_execute_task_debug`.

Keep the `_extract_wandb_run_id` helper (regex on stdout) and the local entrypoint (update it to call new function names).

**Step 2: Run existing tests to check nothing else broke**

Run: `uv run pytest tests/modal/ -v --ignore=tests/modal/test_runner.py`
Expected: Most pass; runner tests will be updated in Task 3.

**Step 3: Commit**

```bash
git add src/tmgg/modal/_functions.py
git commit -m "feat(modal): replace modal_execute_task with generic modal_run_cli"
```

---

## Task 3: Simplify `ModalRunner` to build CLI args

**Files:**
- Modify: `src/tmgg/modal/runner.py`
- Modify: `tests/modal/test_runner.py`

**Step 1: Rewrite ModalRunner**

The runner no longer needs `TaskInput`, `prepare_config_for_remote`, or `TigrisStorage`.

New `run_experiment()` flow:
1. Read `cmd` from `config.get("_cli_cmd")` or `cmd` parameter
2. Read `config_name` from Hydra metadata or from a `_config_name` field
3. Build overrides list from config differences vs base
4. Call `modal_fn.remote(cmd, config_name, run_id, overrides, config_path)`

For simple runs where we have the original Hydra config:
```python
def run_experiment(self, config, cmd=None, ...):
    resolved_cmd = cmd or config.get("_cli_cmd")
    config_name = config.get("_config_name", "base_config_training")
    # For simple invocations, serialize the full resolved config to the volume
    # and point config_path at it (sweep-style)
    config_yaml = OmegaConf.to_yaml(config, resolve=True)
    run_id = config.get("run_id", str(uuid.uuid4())[:8])
    # Upload config via volume
    # ... or pass as YAML string and have the Modal function write it
```

Actually, the simplest approach: always use the "upload resolved YAML to volume" path. This avoids having to decompose a resolved config back into `(config_name, overrides)`:

```python
def run_experiment(self, config, cmd=None, ...):
    resolved_cmd = cmd or config.get("_cli_cmd")
    run_id = config.get("run_id", str(uuid.uuid4())[:8])
    config_yaml = OmegaConf.to_yaml(config, resolve=True)

    modal_fn = self._select_modal_function(gpu)
    result = modal_fn.remote(
        cmd=resolved_cmd,
        config_yaml=config_yaml,  # Modal function writes to temp file
        run_id=run_id,
    )
```

Wait — this brings back the pre-resolved YAML issue. For training configs that use `defaults:`, we need the defaults chain. But if the config is already fully resolved (all interpolations computed), a bare YAML with `defaults: [_self_]` should work.

**Revised approach:** The Modal function gets `config_yaml` (string), writes it to a temp dir, and runs `cmd --config-path=/tmp/xxx --config-name=run_config`. Inject `defaults: [_self_]` into the YAML so Hydra doesn't look for external defaults. This works because the config is already fully resolved — all the values from `base/trainer/default.yaml` etc. are already inline.

This means the `modal_run_cli` function signature becomes:
```python
def modal_run_cli(cmd, config_yaml, run_id, overrides=None)
```
Not `(cmd, config_name, overrides)`. The `config_yaml` is the serialized resolved config.

**Step 2: Update check_modal_deployment to look for new function name**

```python
fn = modal.Function.from_name(MODAL_APP_NAME, "modal_run_cli")
```

**Step 3: Remove imports of TaskInput, prepare_config_for_remote, TigrisStorage**

**Step 4: Update `get_status()` to read confirmation.jsonl**

```python
def get_status(self, run_id: str) -> str:
    if run_id in self._active_runs:
        return "running"
    # No remote check — user should check W&B or volume directly
    return "unknown"
```

(Remote confirmation.jsonl reading would require a Modal function call just to check status, which is heavyweight. Keep it simple.)

**Step 5: Update tests**

Update `tests/modal/test_runner.py` to mock new function signatures.

**Step 6: Commit**

```bash
git add src/tmgg/modal/runner.py tests/modal/test_runner.py
git commit -m "refactor(modal): simplify ModalRunner to CLI transport"
```

---

## Task 4: Delete Tigris storage and gut task.py

**Files:**
- Delete: `src/tmgg/modal/storage.py`
- Modify: `src/tmgg/experiment_utils/task.py`
- Modify: `tests/experiment_utils/test_task.py`

**Step 1: Delete storage.py**

```bash
git rm src/tmgg/modal/storage.py
```

**Step 2: Gut task.py**

Keep only `_extract_wandb_config` (used by `config_builder.py`). Remove everything else:
- `TaskInput`
- `TaskOutput`
- `execute_task`
- `_dispatch_task`
- `_resolve_execution_paths`
- `_write_volume_confirmation`
- `prepare_config_for_remote`
- `StorageProtocol`
- `run_experiment` (the lazy import wrapper)

**Step 3: Update test_task.py**

Remove all tests for deleted functions. Keep tests for `_extract_wandb_config`.

**Step 4: Check for stale imports**

```bash
rg "from tmgg.experiment_utils.task import" src/ tests/ --type py
rg "from tmgg.modal.storage import" src/ tests/ --type py
```

Fix any remaining references.

**Step 5: Commit**

```bash
git add -u
git commit -m "refactor: remove TaskInput/TaskOutput, execute_task, Tigris storage"
```

---

## Task 5: Clean up execute.py shims and config fields

**Files:**
- Modify: `src/tmgg/experiments/eigenstructure_study/execute.py` lines 60-61
- Modify: `src/tmgg/experiments/embedding_study/execute.py` (same pattern)
- Modify: `src/tmgg/exp_configs/base_config_eigenstructure.yaml`
- Modify: `src/tmgg/exp_configs/base_config_embedding_study.yaml`

**Step 1: Remove `best_val_loss`/`best_model_path` shims**

In `eigenstructure_study/execute.py`, delete:
```python
    result.setdefault("best_val_loss", None)
    result.setdefault("best_model_path", None)
```

Same in `embedding_study/execute.py`.

**Step 2: Remove `task_type` from configs**

Delete `task_type: eigenstructure` from `base_config_eigenstructure.yaml`.
Delete `task_type: embedding_study` from `base_config_embedding_study.yaml`.

**Step 3: Add `_cli_cmd` to experiment configs**

Add to each experiment config that has a CLI entry point:
- `base_config_eigenstructure.yaml`: `_cli_cmd: tmgg-eigenstructure-exp`
- `base_config_embedding_study.yaml`: `_cli_cmd: tmgg-embedding-study-exp`
- `base_config_discrete_diffusion_generative.yaml`: `_cli_cmd: tmgg-discrete-gen`
- And so on for all other experiment configs with `[project.scripts]` entries

**Step 4: Commit**

```bash
git add -u
git commit -m "refactor: remove task_type shims, add _cli_cmd to configs"
```

---

## Task 6: Update stages runner and Hydra launcher

**Files:**
- Modify: `src/tmgg/experiments/stages/runner.py`
- Modify: `src/tmgg/hydra_plugins/tmgg_launcher/launcher.py` (if it imports TaskInput)
- Modify: `src/tmgg/experiment_utils/cloud/base.py` (if ExperimentResult needs updating)

**Step 1: Update stages/runner.py**

It currently does `from tmgg.modal.runner import ModalRunner` and calls `runner.run_experiment(config)`. The new signature takes `cmd` — ensure the composed config has `_cli_cmd` set (from Task 5 config additions), or pass `cmd` explicitly.

**Step 2: Check Hydra launcher plugin**

```bash
rg "TaskInput|execute_task|prepare_config" src/tmgg/hydra_plugins/
```

If it uses any of the removed functions, update.

**Step 3: Commit**

```bash
git add -u
git commit -m "refactor: update stages runner and Hydra launcher for CLI transport"
```

---

## Task 7: Final verification

**Step 1: Check for stale imports**

```bash
rg "TaskOutput|_dispatch_task|TigrisStorage|get_storage_from_env|execute_task" src/ tests/ --type py
```

Should return zero matches outside archived tests.

**Step 2: Run all tests**

```bash
uv run pytest tests/ -x -v --ignore=tests/archived
```

**Step 3: Verify Modal deployment (dry run)**

```bash
uv run modal deploy -m tmgg.modal._functions --dry-run 2>&1 | head -20
```

**Step 4: Commit and tag**

```bash
git commit --allow-empty -m "refactor: complete Modal CLI transport simplification"
git tag v0.X.0-modal-cli-transport
```
