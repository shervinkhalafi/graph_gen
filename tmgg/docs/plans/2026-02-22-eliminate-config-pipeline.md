# Eliminate Config Generation Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the superfluous config generation pipeline (`config_compose`, `config_builder`, `generate_configs`, `stage_definitions`, `launch_sweep`, `spawn_single`) and make `--multirun` with `TmggLauncher` the canonical sweep mechanism. Move the W&B skip-existing check into `run_experiment()` as a preflight.

**Architecture:** The config generation pipeline exists to work around a perceived Hydra limitation (programmatic composition of config groups across architectures). Hydra's `--multirun` handles the same cartesian product natively, and the `TmggLauncher` + `ModalRunner` already dispatch jobs to Modal. The pipeline generates intermediate JSON files that `launch_sweep.py` reads — this intermediate step is now unnecessary. The W&B dedup check (skip runs that already exist) moves from the deleted `launch_sweep.py` into `run_experiment.py`, where it runs for every execution mode.

**Tech Stack:** Hydra `--multirun`, OmegaConf, W&B Python API, pytest

---

## Background: What run_id_template Needed

The `run_id_template` (e.g. `"stage1_{arch}_{lr}_{wd}_{k}_s{seed}"`) in `stage_definitions/*.yaml` produced human-readable run names from the sweep grid. `ExperimentConfigBuilder.generate_run_id()` populated it by:

1. Extracting the last segment of the architecture path (e.g. `models/spectral/linear_pe` → `linear_pe`)
2. Formatting HP values with short prefixes: `lr1e-4`, `wd1e-2`, `k8` (scientific notation for values < 0.01)
3. Appending `s{seed}`

With `--multirun`, Hydra resolves each config fully before dispatching. The `run_id` field can be auto-generated from the resolved config values in `run_experiment()`. This function replaces both `ExperimentConfigBuilder.generate_run_id()` and the manual run_id construction in `ExperimentCoordinator.generate_configs()`.

## Files to Delete

| File | Reason |
|---|---|
| `src/tmgg/modal/config_compose.py` | Thin Hydra compose wrapper, only used by config_builder |
| `src/tmgg/modal/config_builder.py` | Two-phase pipeline, replaced by native `--multirun` |
| `src/tmgg/modal/cli/generate_configs.py` | Generates JSON files from stage definitions |
| `src/tmgg/modal/cli/launch_sweep.py` | Reads JSON files and spawns Modal functions |
| `src/tmgg/modal/cli/spawn_single.py` | Spawns single JSON config on Modal |
| `src/tmgg/modal/run_single.py` | Helper for CLI commands, used by deleted CLIs |
| `src/tmgg/modal/stage_definitions/` | Entire directory (13 YAML stage definitions + `__init__.py`) |
| `src/tmgg/modal/stages/` | Entire directory (stage1.py, stage2.py, `__init__.py` — all deprecated) |
| `src/tmgg/experiment_utils/task.py` | Only contains `_extract_wandb_config`, sole consumer was config_builder |
| `tests/modal/test_config_compose.py` | Tests for deleted config_compose |
| `tests/modal/test_config_builder.py` | Tests for deleted config_builder |
| `tests/modal/test_stage_definitions.py` | Tests for deleted stage_definitions |
| `tests/experiment_utils/test_task.py` | Tests for deleted `_extract_wandb_config` |

## Files to Modify

| File | Change |
|---|---|
| `src/tmgg/experiment_utils/run_experiment.py` | Add W&B preflight check + auto-generate run_id |
| `src/tmgg/modal/paths.py` | Remove `get_exp_configs_path`, `require_exp_configs_path` (dead after config_compose deletion) |
| `src/tmgg/modal/cli/__init__.py` | Rewrite docstring (deleted CLI tools) |
| `tests/modal/test_runner.py` | Remove `config_builder` import from integration test, use direct Hydra compose + OmegaConf merge |

---

### Task 1: W&B Preflight Skip Check + Run ID Auto-Generation

**Files:**
- Modify: `src/tmgg/experiment_utils/run_experiment.py:1-73`
- Test: `tests/experiment_utils/test_run_experiment_preflight.py`

This task adds two utilities to `run_experiment.py`:

1. `generate_run_id(config)` — produces a human-readable name from config values, replacing the deleted `run_id_template` machinery
2. `check_wandb_run_exists(entity, project, run_name)` — queries W&B API for a run with that display name

Both are called at the top of `run_experiment()` before any expensive setup.

**Step 1: Write failing tests for run_id generation**

Create `tests/experiment_utils/test_run_experiment_preflight.py`:

```python
"""Tests for run_experiment preflight utilities.

These test the run_id auto-generation and W&B duplicate detection
that replaced the config_builder pipeline. The run_id generator
produces human-readable names from resolved Hydra config values,
matching the format previously handled by ExperimentConfigBuilder.generate_run_id().
The W&B check queries the API for an existing run with the same display name.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from tmgg.experiment_utils.run_experiment import (
    check_wandb_run_exists,
    generate_run_id,
)


class TestGenerateRunId:
    """Tests for generate_run_id().

    The function produces a compact, human-readable identifier from
    the resolved config, using short prefixes and scientific notation
    for small values (matching the old run_id_template convention).
    """

    def test_spectral_arch_config(self):
        """Full spectral arch config produces expected format."""
        config = OmegaConf.create({
            "experiment_name": "stage1",
            "model": {"_target_": "tmgg.models.SpectralArch", "k": 8},
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "seed": 1,
        })
        result = generate_run_id(config)
        assert result == "stage1_SpectralArch_lr1e-4_wd1e-2_k8_s1"

    def test_scientific_notation_small_values(self):
        """Values < 0.01 use scientific notation without zero-padded exponent."""
        config = OmegaConf.create({
            "experiment_name": "exp",
            "model": {"_target_": "tmgg.models.Foo"},
            "learning_rate": 5e-5,
            "weight_decay": 1e-3,
            "seed": 2,
        })
        result = generate_run_id(config)
        assert "lr5e-5" in result
        assert "wd1e-3" in result
        assert result.endswith("_s2")

    def test_large_values_no_scientific(self):
        """Values >= 0.01 use plain string representation."""
        config = OmegaConf.create({
            "experiment_name": "exp",
            "model": {"_target_": "tmgg.models.Foo"},
            "learning_rate": 0.01,
            "weight_decay": 0.1,
            "seed": 1,
        })
        result = generate_run_id(config)
        assert "lr0.01" in result
        assert "wd0.1" in result

    def test_missing_optional_fields(self):
        """Works with minimal config (only seed required)."""
        config = OmegaConf.create({"seed": 42})
        result = generate_run_id(config)
        assert result == "s42"

    def test_diffusion_steps_included(self):
        """model.diffusion_steps appears as T prefix."""
        config = OmegaConf.create({
            "model": {"_target_": "tmgg.models.Foo", "diffusion_steps": 500},
            "seed": 1,
        })
        result = generate_run_id(config)
        assert "T500" in result

    def test_existing_run_id_not_overwritten(self):
        """If config already has run_id, generate_run_id returns it unchanged."""
        config = OmegaConf.create({
            "run_id": "my_custom_id",
            "seed": 1,
        })
        result = generate_run_id(config)
        assert result == "my_custom_id"


class TestCheckWandbRunExists:
    """Tests for check_wandb_run_exists().

    Uses mocked W&B API to avoid real network calls. The function
    queries by displayName filter to check for a single run efficiently.
    """

    @patch("tmgg.experiment_utils.run_experiment.wandb")
    def test_returns_true_when_run_found(self, mock_wandb):
        """Returns True when W&B API returns a matching run."""
        mock_run = MagicMock()
        mock_run.name = "stage1_linear_pe_lr1e-4_s1"
        mock_wandb.Api.return_value.runs.return_value = [mock_run]

        assert check_wandb_run_exists("team", "project", "stage1_linear_pe_lr1e-4_s1")
        mock_wandb.Api.return_value.runs.assert_called_once_with(
            "team/project",
            filters={"displayName": "stage1_linear_pe_lr1e-4_s1"},
        )

    @patch("tmgg.experiment_utils.run_experiment.wandb")
    def test_returns_false_when_no_match(self, mock_wandb):
        """Returns False when W&B API returns empty list."""
        mock_wandb.Api.return_value.runs.return_value = []

        assert not check_wandb_run_exists("team", "project", "nonexistent")

    @patch("tmgg.experiment_utils.run_experiment.wandb")
    def test_returns_false_on_api_error(self, mock_wandb):
        """Returns False (not crash) on W&B API errors — network issues
        should not block experiment execution."""
        mock_wandb.Api.return_value.runs.side_effect = Exception("network error")

        assert not check_wandb_run_exists("team", "project", "run_name")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiment_utils/test_run_experiment_preflight.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_run_id'`

**Step 3: Implement generate_run_id and check_wandb_run_exists**

Add the following to `src/tmgg/experiment_utils/run_experiment.py`, right after the existing imports (before `_LAST_CHECKPOINT`):

```python
import wandb as _wandb_module  # lazy; only used by check_wandb_run_exists

# Alias for patching in tests
wandb = _wandb_module


def _format_hp(value: float) -> str:
    """Format a hyperparameter value: scientific notation for < 0.01, plain otherwise."""
    if isinstance(value, float) and value < 0.01:
        return f"{value:.0e}".replace("e-0", "e-")
    return str(value)


def generate_run_id(config: DictConfig) -> str:
    """Generate a human-readable run ID from resolved config values.

    If the config already contains a ``run_id`` key, returns it unchanged.
    Otherwise, builds a compact identifier from experiment name, model class,
    key hyperparameters, and seed. Small float values use scientific notation
    (e.g. ``lr1e-4``), matching the convention of the former ``run_id_template``.

    Parameters
    ----------
    config
        Resolved Hydra configuration.

    Returns
    -------
    str
        Run identifier like ``stage1_SpectralArch_lr1e-4_wd1e-2_k8_s1``.
    """
    existing = config.get("run_id")
    if existing is not None:
        return str(existing)

    parts: list[str] = []

    # Experiment name (e.g. "stage1", "discrete_gen")
    exp_name = config.get("experiment_name")
    if exp_name:
        parts.append(str(exp_name))

    # Model class short name (last segment of _target_)
    model_cfg = config.get("model", {})
    target = model_cfg.get("_target_", "") if model_cfg else ""
    if target:
        parts.append(str(target).split(".")[-1])

    # Key hyperparameters
    lr = config.get("learning_rate")
    if lr is not None:
        parts.append(f"lr{_format_hp(lr)}")

    wd = config.get("weight_decay")
    if wd is not None:
        parts.append(f"wd{_format_hp(wd)}")

    k = model_cfg.get("k") if model_cfg else None
    if k is not None:
        parts.append(f"k{k}")

    diff_steps = model_cfg.get("diffusion_steps") if model_cfg else None
    if diff_steps is not None:
        parts.append(f"T{diff_steps}")

    # Seed (always present)
    parts.append(f"s{config.seed}")

    return "_".join(parts)


def check_wandb_run_exists(entity: str, project: str, run_name: str) -> bool:
    """Check whether a W&B run with this display name already exists.

    Uses the ``displayName`` filter for an efficient single-run lookup
    rather than fetching all runs. Returns False on any API error so
    that network issues never block experiment execution.

    Parameters
    ----------
    entity
        W&B entity (team or username).
    project
        W&B project name.
    run_name
        Display name to search for.

    Returns
    -------
    bool
        True if a run with that display name exists.
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"displayName": run_name})
        return len(list(runs)) > 0
    except Exception:
        return False
```

Note: the `wandb` import should be done carefully. wandb is a heavy import. Use a lazy import pattern — import inside `check_wandb_run_exists` rather than at module level:

```python
def check_wandb_run_exists(entity: str, project: str, run_name: str) -> bool:
    # ... docstring ...
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"displayName": run_name})
        return len(list(runs)) > 0
    except Exception:
        return False
```

Update the test mock path to patch `wandb` inside the function. Since we use a lazy import, the mock in the test needs `@patch("wandb.Api")` or use `monkeypatch` on `sys.modules`. Simpler approach: make the tests use `monkeypatch`:

```python
# In tests, use monkeypatch instead of @patch:
def test_returns_true_when_run_found(self, monkeypatch):
    mock_api = MagicMock()
    mock_run = MagicMock()
    mock_run.name = "stage1_linear_pe_lr1e-4_s1"
    mock_api.return_value.runs.return_value = [mock_run]
    monkeypatch.setattr("wandb.Api", mock_api)

    assert check_wandb_run_exists("team", "project", "stage1_linear_pe_lr1e-4_s1")
```

**Step 4: Wire into run_experiment()**

In `run_experiment()`, add this block right after `set_seed(config.seed)` and `configure_matmul_precision()`, before the directory creation:

```python
    # Auto-generate run_id if not set
    if config.get("run_id") is None:
        with open_dict(config):
            config.run_id = generate_run_id(config)

    # W&B preflight: skip if a run with this name already exists
    if config.get("skip_if_wandb_exists", False):
        entity = config.get("wandb_entity")
        project = config.get("wandb_project")
        run_name = config.get("run_id")
        if entity and project and run_name:
            if check_wandb_run_exists(str(entity), str(project), str(run_name)):
                loguru.info(
                    f"Skipping: W&B run '{run_name}' already exists "
                    f"in {entity}/{project}"
                )
                return {"skipped": True, "reason": "wandb_run_exists"}
```

Add `from omegaconf import open_dict` to the imports (it's used to mutate the struct config).

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/experiment_utils/test_run_experiment_preflight.py -v`
Expected: all 9 tests PASS

**Step 6: Commit**

```bash
git add src/tmgg/experiment_utils/run_experiment.py tests/experiment_utils/test_run_experiment_preflight.py
git commit -m "feat: add W&B preflight skip check and run_id auto-generation

Replaces the run_id_template machinery from config_builder and the
skip-existing logic from launch_sweep.py. Both now live in
run_experiment() where they apply to all execution modes."
```

---

### Task 2: Delete Config Generation Pipeline

**Files to delete:**
- `src/tmgg/modal/config_compose.py`
- `src/tmgg/modal/config_builder.py`
- `src/tmgg/modal/cli/generate_configs.py`
- `src/tmgg/modal/cli/launch_sweep.py`
- `src/tmgg/modal/cli/spawn_single.py`
- `src/tmgg/modal/run_single.py`
- `src/tmgg/modal/stage_definitions/` (entire directory)
- `src/tmgg/modal/stages/` (entire directory)
- `src/tmgg/experiment_utils/task.py`
- `tests/modal/test_config_compose.py`
- `tests/modal/test_config_builder.py`
- `tests/modal/test_stage_definitions.py`
- `tests/experiment_utils/test_task.py`

**Step 1: Delete source files**

```bash
git rm src/tmgg/modal/config_compose.py
git rm src/tmgg/modal/config_builder.py
git rm src/tmgg/modal/cli/generate_configs.py
git rm src/tmgg/modal/cli/launch_sweep.py
git rm src/tmgg/modal/cli/spawn_single.py
git rm src/tmgg/modal/run_single.py
git rm -r src/tmgg/modal/stage_definitions/
git rm -r src/tmgg/modal/stages/
git rm src/tmgg/experiment_utils/task.py
```

**Step 2: Delete test files**

```bash
git rm tests/modal/test_config_compose.py
git rm tests/modal/test_config_builder.py
git rm tests/modal/test_stage_definitions.py
git rm tests/experiment_utils/test_task.py
```

**Step 3: Run fast tests to confirm nothing breaks**

Run: `uv run pytest tests/ -x -q -m "not slow" --ignore=tests/modal/test_runner.py`

The `test_runner.py` ignore is temporary — it has a `config_builder` import that Task 3 will fix. All other tests should pass because the deleted modules had no importers outside the deleted set (except test_runner.py and paths.py, handled in the next tasks).

Expected: All pass (no test outside the deleted set imports from deleted modules).

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: delete config generation pipeline

Removes config_compose, config_builder, generate_configs, launch_sweep,
spawn_single, run_single, stage_definitions/, stages/, and task.py.
The --multirun + TmggLauncher path replaces this entire pipeline.
Net deletion: ~2500 lines of source + tests."
```

---

### Task 3: Clean Up Remaining References

**Files to modify:**
- `src/tmgg/modal/paths.py:61-103` — remove dead functions
- `src/tmgg/modal/cli/__init__.py` — rewrite docstring
- `tests/modal/test_runner.py:350-406` — fix config_builder import

**Step 1: Clean up paths.py**

Remove `get_exp_configs_path()` (line 61-70) and `require_exp_configs_path()` (line 84-103). Keep `discover_tmgg_path`, `_is_valid_tmgg_path`, and `require_tmgg_path`.

The file should look like:

```python
"""Path discovery utilities for Modal experiments.

Since modal is now part of the tmgg package, path discovery is straightforward.
The module lives at src/tmgg/modal/, so paths are resolved relative to that.
"""

from __future__ import annotations

import os
from pathlib import Path


def discover_tmgg_path() -> Path:
    """Get the tmgg package root directory.

    Since modal is now inside tmgg, this is straightforward.
    The package structure is:
        tmgg/           <- returned path
          src/
            tmgg/
              modal/
                paths.py  <- this file

    Returns
    -------
    Path
        Path to tmgg package root (the directory containing src/).

    Notes
    -----
    TMGG_PATH environment variable can override for special cases
    (e.g., development with editable installs in non-standard locations).
    """
    env_path = os.environ.get("TMGG_PATH")
    if env_path:
        path = Path(env_path)
        if _is_valid_tmgg_path(path):
            return path
        raise ValueError(
            f"TMGG_PATH={env_path} does not contain valid tmgg package "
            "(missing src/tmgg directory)"
        )

    # modal/ is at src/tmgg/modal/, so go up 4 levels to get to tmgg root
    # paths.py -> modal/ -> tmgg/ -> src/ -> tmgg_root/
    return Path(__file__).parent.parent.parent.parent


def _is_valid_tmgg_path(path: Path) -> bool:
    """Check if path contains a valid tmgg package."""
    if not path.exists():
        return False

    src_tmgg = path / "src" / "tmgg"
    if not src_tmgg.exists():
        return False

    return bool(list(src_tmgg.glob("*.py")))


def require_tmgg_path() -> Path:
    """Get tmgg path (always succeeds since we're inside the package).

    Returns
    -------
    Path
        Path to tmgg package root.
    """
    return discover_tmgg_path()
```

**Step 2: Rewrite modal/cli/__init__.py**

```python
"""CLI tools for Modal experiment orchestration.

The primary experiment workflow uses Hydra's --multirun with the TmggLauncher::

    tmgg-experiment --multirun \\
        model=models/spectral/linear_pe,models/spectral/filter_bank \\
        learning_rate=1e-4,5e-4,1e-3 \\
        seed=1,2,3 \\
        hydra/launcher=tmgg_modal

For interactive debugging, use the @app.local_entrypoint() in _functions.py::

    doppler run -- uv run modal run -m tmgg.modal._functions \\
        --config ./config.json --gpu debug
"""
```

**Step 3: Fix test_runner.py integration test**

The test at line 350 (`test_modal_single_step`) imports `ExperimentConfigBuilder` and `deep_merge` from the deleted `config_builder`. Replace with direct Hydra compose + OmegaConf merge:

Replace lines 360-390 (the config_builder usage) with:

```python
        # 1. Compose base config
        exp_configs = (
            Path(__file__).parent.parent.parent / "src" / "tmgg" / "exp_configs"
        )
        overrides = [
            f"paths.output_dir={tmp_path}",
            f"paths.results_dir={tmp_path}/results",
            "trainer.max_steps=2",
            "trainer.accelerator=auto",
            "~logger",
            "data.batch_size=2",
            "data.num_workers=0",
            "seed=42",
            f"hydra.run.dir={tmp_path}",
        ]

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(exp_configs)):
            cfg = compose(config_name=base_config, overrides=overrides)

        # 2. Load and merge architecture via OmegaConf (replaces config_builder)
        import yaml
        arch_yaml = exp_configs / f"{arch}.yaml"
        with open(arch_yaml) as f:
            arch_config = yaml.safe_load(f)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        base_model = cfg_dict.get("model", {})
        # Deep merge: arch values override base
        merged = {**base_model, **arch_config}
        cfg_dict["model"] = merged
```

Remove the `from tmgg.modal.config_builder import ExperimentConfigBuilder, deep_merge` import.

**Step 4: Run the full fast test suite**

Run: `uv run pytest tests/ -x -q -m "not slow"`
Expected: All pass

**Step 5: Commit**

```bash
git add src/tmgg/modal/paths.py src/tmgg/modal/cli/__init__.py tests/modal/test_runner.py
git commit -m "refactor: clean up references to deleted config pipeline

Remove dead functions from paths.py, update cli/__init__.py docstring,
fix test_runner.py integration test to use direct Hydra compose."
```

---

### Task 4: Grep for Stale References and Final Verification

**Step 1: Grep for stale references**

```bash
rg 'config_compose|config_builder|generate_configs|stage_definitions|launch_sweep|spawn_single|run_single|_extract_wandb_config' src/ tests/ --glob '!*.md'
```

Expected: zero hits in `src/` and `tests/` (docs may still reference old pipeline, that's fine).

If any hits remain, fix them.

**Step 2: Check for stale imports of task.py**

```bash
rg 'from tmgg\.experiment_utils\.task' src/ tests/
```

Expected: zero hits.

**Step 3: Check for stale imports of modal.stages**

```bash
rg 'from tmgg\.modal\.stages' src/ tests/
```

Expected: zero hits.

**Step 4: Run the full fast test suite**

```bash
uv run pytest tests/ -x -q -m "not slow"
```

Expected: All tests pass, no import errors.

**Step 5: Run basedpyright**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/experiment_utils/run_experiment.py src/tmgg/modal/paths.py
```

Expected: no new errors.

**Step 6: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: resolve stale references to deleted pipeline"
```

---

## Post-Cleanup Notes

### What Still Exists (intentionally)

- **`ExperimentCoordinator`** (`experiment_utils/cloud/coordinator.py`): Still used by `stages/runner.py` for the `sweep=true` path. This is a separate cleanup — it duplicates some of what `--multirun` does, but it also provides result aggregation and checkpointing. Keep for now.
- **`_wandb_config` auto-inject in `create_loggers()`** (line 194-224 of `logging.py`): Fallback path that fires when no logger section exists but `WANDB_API_KEY` is set. Still useful for ad-hoc remote execution where the logger section gets stripped. The `_extract_wandb_config` function that produced the `_wandb_config` dict is deleted, but the auto-inject code reads it from the config if present — harmless no-op when absent.
- **`configs/` directories** (`configs/discrete_gen/`, `configs/sanity/`): Untracked pre-generated JSON files. Now obsolete. Delete manually if desired: `rm -rf configs/`.

### Canonical Sweep Usage After This Change

```bash
# Local sweep
tmgg-spectral-arch --multirun \
    model=models/spectral/linear_pe,models/spectral/filter_bank,models/spectral/self_attention \
    learning_rate=1e-4,5e-4,1e-3 \
    weight_decay=1e-2,1e-3 \
    +model.k=8,16 \
    seed=1,2,3

# Modal sweep
tmgg-spectral-arch --multirun \
    hydra/launcher=tmgg_modal \
    model=models/spectral/linear_pe,models/spectral/filter_bank \
    learning_rate=1e-4,5e-4 \
    seed=1,2,3

# With W&B dedup (skip already-completed runs)
tmgg-spectral-arch --multirun \
    skip_if_wandb_exists=true \
    hydra/launcher=tmgg_modal \
    model=models/spectral/linear_pe,models/spectral/filter_bank \
    learning_rate=1e-4,5e-4 \
    seed=1,2,3
```
