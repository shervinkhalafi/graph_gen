# Change Manifest: Config Inheritance + Modal Stage Deprecation

Steps 1.3 and 1.7 of the codebase transition plan.

## Files Modified

### 1. `src/tmgg/exp_configs/base_config_generative.yaml` (Step 1.3)

**What changed:** Refactored to inherit from `base_config_training.yaml` via Hydra `defaults: [base_config_training, _self_]`, matching the pattern already used by `base_config_spectral.yaml`.

**Removed (duplicated from base):**
- `seed: 42` (inherited)
- `learning_rate`, `weight_decay`, `optimizer_type` top-level keys (inherited)
- `force_fresh: false`, `sanity_check: false` (inherited)
- `paths:` block (inherited, identical)
- `base/trainer/default@trainer`, `base/logger/default`, `base/callbacks/default`, `base/progress_bar/default` from defaults list (all pulled in by `base_config_training`)

**Added:**
- `wandb_project: diffusion-study`

**Kept (generative-specific):**
- `experiment_name: "generative_graph"`
- Full `model:` block (generative-specific `_target_`, diffusion settings, MMD settings)
- Full `data:` block (generative-specific `_target_`, dataset config)
- `noise_levels: [0.1, 0.3, 0.5]` (overrides base's `[0.01, 0.05, 0.1, 0.2, 0.3]`)
- `hydra:` output directory layout (generative-specific directory structure)

### 2. `src/tmgg/exp_configs/grid_search_base.yaml` (Step 1.3)

**What changed:** Refactored to inherit from `base_config_training.yaml`. Converted epoch-based training settings to step-based.

**Removed (duplicated from base):**
- `seed: 42`, `sanity_check: false` (inherited)
- `learning_rate`, `weight_decay`, `optimizer_type` (inherited)
- `scheduler_config:` block (inherited, identical)
- `loss_type: BCEWithLogits` (inherited)
- `noise_type`, `noise_levels`, `eval_noise_levels`, `fixed_noise_seed` (inherited)
- `data:` noise sync block (inherited)
- `paths:` block (inherited)

**Epoch-to-step conversions:** Assumed ~31 steps/epoch based on 1000 graphs at batch_size 32 (ceil(1000/32) = 31.25).

| Original (epoch-based)          | New (step-based)              | Rationale                                              |
|---------------------------------|-------------------------------|--------------------------------------------------------|
| `max_epochs: 200`              | `max_steps: 6000`            | 200 * 31 = 6200, rounded to 6000                      |
| `check_val_every_n_epoch: 5`   | `val_check_interval: 150`    | 5 * 31 = 155, rounded to 150                           |
| `visualization_epochs: 50`     | `visualization_interval: 1500`| 50 * 31 = 1550, rounded to 1500                       |
| `patience: 20` (epochs)        | `patience: 4` (val checks)   | 20 epochs / 5 epoch val interval = 4 validation checks |

**Also:** Changed `max_epochs: -1` (from base trainer) is now inherited. The `trainer:` override block sets `max_steps: 6000` explicitly.

**Kept (grid-search-specific):**
- Model/data/logger default overrides in defaults list
- `experiment_name`
- Trainer override block with grid-search-specific callbacks
- `model:` sync block
- `evaluation:` block
- Grid search sweep parameters (`noise_level`, `model_name`)
- Hydra output directory layout (version-based)

### 3. `src/tmgg/modal/stages/stage1.py` (Step 1.7)

**What changed:**
- Added `import warnings` and module-level `warnings.warn(...)` with `DeprecationWarning`
- Added deprecation note to module docstring
- Changed `"models/spectral/filter_bank_nonlinear"` to `"models/spectral/filter_bank"` in `STAGE1_ARCHITECTURES` (the nonlinear variant config file does not exist; only `filter_bank.yaml` exists)

### 4. `src/tmgg/modal/stages/stage2.py` (Step 1.7)

**What changed:**
- Added `import warnings` and module-level `warnings.warn(...)` with `DeprecationWarning`
- Added deprecation note to module docstring
- Changed `"models/spectral/filter_bank_nonlinear"` to `"models/spectral/filter_bank"` in `STAGE2_ARCHITECTURES`

## Discrepancies from Spec

- **`filter_bank_nonlinear` config never existed.** Only `filter_bank.yaml`, `filter_bank_asymmetric.yaml`, and `filter_bank_pearl.yaml` exist under `models/spectral/`. The stage files referenced a nonexistent config, so the rename to `filter_bank` is both spec-compliant and a bug fix.
- **`base_config_generative.yaml` did not use `loss_type: BCEWithLogits`** -- it used `loss_type: MSE` in the model block. This is generative-specific and was kept as-is (the base's `loss_type: BCEWithLogits` is not relevant to the generative model which receives loss_type through its own model config).
- **`grid_search_base.yaml` already had `base/logger/grid_wandb@logger`** which overrides the default logger from base_config_training. This is intentional and preserved.
- **Checkpoint filename in grid_search_base.yaml** was changed from `epoch={epoch:02d}` to `step={step:06d}` to match the step-based training paradigm.

## Decisions Made

1. **Epoch-to-step conversion factor:** 31 steps/epoch (1000 graphs / 32 batch_size). All step values rounded to clean multiples of 150 for consistency.
2. **Early stopping patience conversion:** The original 20-epoch patience with 5-epoch validation interval means 4 validation checks worth of patience. This is preserved exactly.
3. **`wandb_project` not added to grid_search_base.yaml** since it inherits from base_config_training (`sandbox`) and uses a grid-specific WandB logger (`grid_wandb`) that already defines its own project name (`tmgg-grid-search-4k`). Adding a top-level `wandb_project` would be unused.

### 5. `src/tmgg/analysis/figures.py` (pre-commit unblock)

**What changed:** Fixed basedpyright `reportReturnType` error where `plt.subplots()` returns `Figure | SubFigure` but the return type declared `Figure`. Added `isinstance` assertion for the `subplots()` branch and a `TypeError` raise for the `get_figure()` branch (SubFigure axes are not supported). This is consistent with the project's "fail loudly" principle.

### 6. `src/tmgg/analysis/statistics.py` (pre-commit unblock)

**What changed:** Removed redundant type annotation on `df` reassignment at line 182 that caused basedpyright `reportRedeclaration` (parameter `df` was redeclared with `df: pd.DataFrame = ...`). Changed to bare `df = ...` which preserves the same runtime behavior.

## Verification Results

```
$ uv run pytest tests/test_config_composition.py -v
26 passed, 6 warnings in 8.21s

$ uv run python -W error::DeprecationWarning -c "..."
stage1 deprecation OK
stage2 deprecation OK

$ git commit (pre-commit hooks)
ruff ................ Passed
ruff-format ......... Passed
basedpyright ........ Passed
tach check .......... Passed
```
