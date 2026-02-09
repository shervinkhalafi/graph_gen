# W&B Routing: Steps 1.2 and 1.4

Steps 1.2 and 1.4 route all W&B project/entity values through the top-level `wandb_project` and `wandb_entity` keys added by Layer 0 in `base_config_training.yaml`. No logger config or consumer code hardcodes these values anymore.

## Files Modified

### Step 1.2 -- W&B Config: Update Logger and Consumer Code

**`src/tmgg/exp_configs/base/logger/default.yaml`**
Replaced hardcoded `project: tmgg-${oc.select:stage,default}` with `project: ${wandb_project}`, and `entity: graph_denoise_team` with `entity: ${wandb_entity}`. The wandb logger section now derives both values from the top-level config keys set in `base_config_training.yaml`.

**`src/tmgg/experiment_utils/task.py`** (`_extract_wandb_config`)
Added a primary extraction path at the top of `_extract_wandb_config` that reads `config.get("wandb_project")` and `config.get("wandb_entity")` directly. When `wandb_project` is present, it returns immediately with entity/project from these keys (plus tags/log_model from the logger config if available). The existing logger-config parsing is preserved as a fallback for legacy configs that lack the top-level keys.

**`src/tmgg/experiment_utils/logging.py`** (`create_loggers`)
In the auto-create W&B logger block (for remote/Modal execution where logger config is stripped), replaced the fallback `config.get("experiment_name", "tmgg")` with `config.get("wandb_project", "sandbox")` for project, and `entity = None` with `entity = config.get("wandb_entity")`. The `_wandb_config` path also falls through to these top-level keys via `config.get("wandb_project", "sandbox")` and `config.get("wandb_entity")`.

**`src/tmgg/exp_configs/base_config_spectral.yaml`**
Added `wandb_project: architecture-study` after `experiment_name`. This overrides the base `sandbox` default for spectral architecture experiments.

**`src/tmgg/exp_configs/base_config_gnn.yaml`**
Added `wandb_project: architecture-study` after `experiment_name`.

**`src/tmgg/exp_configs/base_config_hybrid.yaml`**
Added `wandb_project: architecture-study` after `experiment_name`.

**`src/tmgg/exp_configs/base_config_digress.yaml`**
Added `wandb_project: architecture-study` after `experiment_name`.

### Step 1.4 -- Logger Config Updates

**`src/tmgg/exp_configs/base/logger/wandb.yaml`**
Replaced `project: tmgg-attention-denoising` with `project: ${wandb_project}`, and `entity: null` with `entity: ${wandb_entity}`.

**`src/tmgg/exp_configs/base/logger/grid_wandb.yaml`**
Replaced `project: "tmgg-grid-search-4k"` with `project: ${wandb_project}`, and `entity: null` with `entity: ${wandb_entity}`.

## Discrepancies from Spec

1. **`logging.py` -- no `config.get("experiment_name", "tmgg")` for project name.** The spec said to replace this pattern in `create_loggers`. In the actual code, this pattern only appeared in the auto-create W&B logger block (the fallback for remote execution with no `_wandb_config`). The main logger creation loop reads project from `logger_params.get("project")` directly (already templated via YAML interpolation). The auto-create fallback was updated as described above.

2. **`task.py` -- no function signature change.** The spec referenced `_extract_wandb_config`; it exists exactly as described. No discrepancy in structure.

3. **`default.yaml` -- project used `tmgg-${oc.select:stage,default}` not a plain hardcoded string.** The spec said "replace any hardcoded `project:`". This was a Hydra interpolation expression, not a literal string, but the effect is the same: it now uses `${wandb_project}` and the stage-based naming is no longer baked into the logger config.

4. **`wandb.yaml` had `entity: null`, not a hardcoded entity string.** Replaced with `${wandb_entity}` as specified. Same for `grid_wandb.yaml`.

## Verification Results

### Tests: 26/26 passed

```
tests/test_config_composition.py::TestConfigComposition::test_config_composes_successfully[base_config_spectral] PASSED
tests/test_config_composition.py::TestConfigComposition::test_config_composes_successfully[base_config_gnn] PASSED
tests/test_config_composition.py::TestConfigComposition::test_config_composes_successfully[base_config_hybrid] PASSED
tests/test_config_composition.py::TestConfigComposition::test_config_composes_successfully[base_config_digress] PASSED
tests/test_config_composition.py::TestConfigComposition::test_model_instantiation[base_config_spectral] PASSED
tests/test_config_composition.py::TestConfigComposition::test_model_instantiation[base_config_gnn] PASSED
tests/test_config_composition.py::TestConfigComposition::test_model_instantiation[base_config_hybrid] PASSED
tests/test_config_composition.py::TestConfigComposition::test_model_instantiation[base_config_digress] PASSED
tests/test_config_composition.py::TestConfigComposition::test_data_module_instantiation[base_config_spectral] PASSED
tests/test_config_composition.py::TestConfigComposition::test_data_module_instantiation[base_config_gnn] PASSED
tests/test_config_composition.py::TestConfigComposition::test_data_module_instantiation[base_config_hybrid] PASSED
tests/test_config_composition.py::TestConfigComposition::test_data_module_instantiation[base_config_digress] PASSED
(+ 14 more forward pass, trainer, and stage composition tests -- all passed)
```

### basedpyright: 0 errors, 141 warnings

All warnings are pre-existing `reportAny` / `reportUnknownMemberType` issues inherent to OmegaConf's dynamic typing (`DictConfig.get()` returns `Any`). The new code in `task.py` adds a few of these (lines 118-132) because it calls `config.get("wandb_project")` etc., which is the same pattern used throughout both files. No new warning categories were introduced.
