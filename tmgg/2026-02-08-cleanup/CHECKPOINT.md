# Cleanup Checkpoint — 2026-02-13 (post polish)

**Branch:** `cleanup`
**Head commit:** `6436a22`
**Tests:** 1122 passed, 2 skipped, 0 failures (non-slow) | basedpyright: 0 errors, ~6090 warnings (PyTorch stubs)
**Commits since last checkpoint:** `40ea85a` checkpoint update, `6436a22` polish

## Status by Priority and Step

### P1: Discrete Diffusion (Critical Path)

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1.1 | Noise schedule + transition models | DONE | `models/digress/noise_schedule.py`, 21 tests |
| 1.2 | Categorical DataModule | DONE | `experiments/discrete_diffusion/datamodule.py`, 20 tests |
| 1.3 | Discrete model wrapper | DONE | `models/digress/discrete_transformer.py`, 7 tests |
| 1.4 | Extra features | DONE | Full DiGress port: `ExtraFeatures` with "cycles", "eigenvalues", "all" modes. Cycle counts (k=3-6 via matrix powers), Laplacian eigendecomposition, LCC indicator, eigenvectors. `extra_features_dims()` helper for input_dims computation. 14 tests. |
| 1.5 | Discrete Diffusion LightningModule | DONE | `experiments/discrete_diffusion/lightning_module.py`, 9 tests. Full training (CE loss), validation (VLB), sampling loop |
| 1.6 | Evaluation pipeline | DONE | `evaluate.py` (conversion utils, MMD) + `evaluate_cli.py` (checkpoint eval, CLI `tmgg-discrete-eval`). Wired into `on_validation_epoch_end`. 7 tests. |
| 1.7 | Hydra configs | DONE | `base_config_discrete_diffusion.yaml` + `runner.py` with custom `run_discrete_experiment()` (cannot use `run_experiment` — requires `noise_generator`). CLI: `tmgg-discrete-run`. Callbacks monitor `val/epoch_NLL`. 10 tests. |
| 1.8 | Validation against baseline | DONE | `scripts/validate_discrete_diffusion.py` with `--quick` (CPU, ~10s) and `--full` (GPU, ~1h) modes. DiGress SBM reference thresholds (degree < 0.05, clustering < 0.10, spectral < 0.15). Quick smoke test passes. 2 tests. |

### P2: Correctness Bugs

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 2.1 | ShrinkageWrapper threshold bug | DONE (b1) | Overrode `logits_to_graph()` to threshold at 0.5. 4 regression tests. |
| 2.2 | SBM diagonal not zeroed | DONE | `generate_sbm_adjacency` in `sbm.py` calls `np.fill_diagonal(adj_matrix, 0)` |
| 2.3 | Batch noise averaging | DONE (b2) | Per-element noise via loop over `add_noise`. 1 regression test. |
| 2.4 | `rng` parameter in `add_digress_noise` | DONE | Removed unused `rng` parameter entirely. 1 regression test verifies signature. |
| 2.5 | Operator precedence in MMD | DONE (b1) | Fixed isinstance precedence, widened type sig. 4 regression tests. |
| 2.6 | `final_eval.py` model type conditioning | DONE (b2) | Injected `NoiseGenerator` parameter. 3 regression tests. |
| 2.7 | PlaceHolder symmetry assertion | DONE | Restored with diagnostic message in `diffusion_utils.py:41` |
| 2.8 | masked_softmax NaN guard | DONE | Returns zeros for all-zero mask, replaces NaN with zeros for per-row masking |

### P3: Dead Code Removal

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 3.1 | Delete orphaned files | DONE (b1) | Deleted 5 files, cleaned `__init__.py` exports. |
| 3.2 | Archive stale scripts | DONE (b1→deleted) | Originally archived to `scripts/archived/`; folder deleted in polish pass (recoverable from git). |
| 3.3 | Remove dead plotting functions | DONE (b2) | Removed 11 dead functions + wandb import. 6 kept. |
| 3.4 | Remove dead metrics | DONE (b3) | Deleted `evaluate_noise_robustness`, `compute_adjacency_spectral_histogram`, `compute_laplacian_histogram`. |
| 3.5 | Archive Ray/SLURM backends | DONE (b2) | ray_runner, slurm_runner archived. Launcher/config cleaned. |
| 3.6 | Clean up `base_config_generative.yaml` | DONE (b3) | Already inherited; removed redundant `subdir`, annotated overrides. |

### P4: Documentation and Config Fixes

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 4.1 | Fix `tmgg-attention` references | DONE (b3) | All docs updated to `tmgg-spectral`. Deleted `attention_denoising.md`. |
| 4.2 | Fix missing data configs | DONE (b3) | Configs existed; uncommented stale reference in deprecated stage2.py. |
| 4.3 | Optimizer drift in stages 3-5 | CLOSED | Already harmonized: all stages use `adamw`, `weight_decay: 1e-12`, `amsgrad: true`. |
| 4.4 | Fix dependencies in pyproject.toml | DONE (b3) | Removed `zensical`. Moved docs deps to optional group. |
| 4.5 | Add generative experiment docs | DONE (b4) | Added ~100-line section to `docs/experiments.md` covering diffusion pipeline, architectures, config, MMD metrics. Entry added to `docs/index.md`. |
| 4.6 | Annotate 2025-12-22 review | CLOSED | No review artifact found in repo. |

### P5: Model/Architecture Cleanup

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 5.1 | Disambiguate embedding abstractions | CLOSED | Not a problem. Classes live in separate subsystems, naming is contextually clear. |
| 5.2 | Normalize forward() signatures | CLOSED | Already normalized per family, enforced by two `@runtime_checkable` protocols. |
| 5.3 | Extract visualization from base module | CLOSED | Not worth extracting. ~50 lines of Lightning orchestration glue that needs module state. |
| 5.4 | Remove redundant GNN forward() override | CLOSED | Not redundant — override necessary because GNN models don't accept timestep `t`. |
| 5.5 | Deduplicate model factory logic | DONE (b4) | Created `models/factory.py` (238 lines, 16 model types). All 5 lightning modules delegate to shared factory. Net -200 lines. |

### Post-cleanup polish (6436a22)

| Item | Status | Notes |
|------|--------|-------|
| Fix import cycle in `discrete_diffusion/` | DONE | Split `evaluate.py` into utilities + `evaluate_cli.py`. Cycle resolved. |
| Fix `generate_sbm_batch` pyright error | DONE | Direct submodule import with inline suppression (basedpyright resolution quirk). |
| Delete redundant `log_hyperparams` TODO | DONE | Removed 10-line block from `run_experiment.py`. Lightning handles this via `save_hyperparameters`. |
| Delete speculative multi-GPU TODOs | DONE | Replaced with notes on `.type_as()` device-placement in `diffusion_utils.py`. |
| Delete `scripts/archived/` | DONE | 6 files (-831 lines). All superseded by CLI tools and pytest. |
| Add experiment READMEs | DONE | 8 READMEs across all experiment directories documenting paradigm, models, CLI. |

## Summary

| Priority | Total | Done/Closed |
|----------|-------|-------------|
| P1 Discrete Diffusion | 8 | 8 |
| P2 Correctness Bugs | 8 | 8 |
| P3 Dead Code | 6 | 6 |
| P4 Docs/Config | 6 | 6 |
| P5 Architecture | 5 | 5 |
| Polish | 6 | 6 |
| **Total** | **39** | **39** |

All items complete. basedpyright: 0 errors. Tests: 1122 passed, 2 skipped.

## Remaining TODOs in code (kept intentionally)

- `modal/stages/stage1.py:157`, `stage2.py:163` — "Integrate with TaskInput when refactored" (tied to deprecation of stage files)
- `base_lightningmodule.py:1` — "add criterion, training setup as inheritable" (design aspiration)

## Files in This Checkpoint

Layers 0–3 reports: `agent-diffs/00-layer0-completion-report.md` through `agent-diffs/02-layer3-completion-report.md`
Action plan: `08-action-plan.md`
Audit reports: `01-diffusion-correctness.md` through `07-diffusion-gap-analysis.md`
