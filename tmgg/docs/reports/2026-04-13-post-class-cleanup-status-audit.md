# Post-Class Cleanup Status Audit

Date: 2026-04-13
Branch: `igork/noise-proocess-interface`
Audited tree: `e1de87a7`
Source tracker: `docs/reports/2026-03-06-post-class-cleanup/SUMMARY.md`

This audit re-checks every row in the March summary that was still not marked
`RESOLVED` or `INVALID`. It does not assume the March tracker is current.
It also records two follow-up major rows added after current-tree verification
exposed gaps in the March tracker: `M-53` and `M-54`.

Status labels used here:

- `Open`: the concern is still materially true in the current tree.
- `Resolved in current tree`: the March row is now stale; the code no longer
  matches the problem description.
- `Needs re-review`: the original concern is plausible, but static inspection
  cannot prove that the config or helper is truly dead rather than intentionally
  selectable.
- `No-fix`: the concern is real, but the cleanup was consciously declined
  because the churn would be noisier than the value of the refactor.

## Summary

- `Open`: 0 rows
- `Resolved in current tree`: 38 rows
- `Needs re-review`: 0 rows
- `No-fix`: 1 row

## Data and Shared Utilities

| Row | Updated status | Current evidence | Notes |
| --- | --- | --- | --- |
| `M-1` | Resolved in current tree | `src/tmgg/data/data_modules/_split.py:12`; `src/tmgg/data/data_modules/graph_generation.py:26`; `tests/test_synthetic_graphs.py:230` | The helper now lives inside `data_modules`, the production import is local, and the direct helper regression test uses the new path. |
| `M-2` | Resolved in current tree | `src/tmgg/data/datasets/synthetic_graphs.py:601`; `src/tmgg/data/datasets/pyg_datasets.py:15`; `tests/test_synthetic_graphs.py:166`; `tests/test_synthetic_graphs.py:267` | Both wrappers now subclass `torch.utils.data.Dataset`, and the regression tests lock that protocol surface explicitly. |
| `M-3` | Resolved in current tree | Fixed-size dataset metadata now comes from `src/tmgg/data/data_modules/base_data_module.py:147` via `get_size_distribution()` | The old `get_dataset_info` path no longer exists in `src/tmgg/data/data_modules`. |
| `M-53` | Resolved in current tree | `src/tmgg/experiments/exp_configs/task/denoising.yaml:19-23`; `tests/test_config_composition.py:157`; `tests/test_config_composition.py:178` | Denoising noise settings now stay at top level and are merged only into model config. `cfg.data` no longer receives `noise_type` or `noise_levels`, and the config composition regression tests now pass. |
| `M-24` | Resolved in current tree | `src/tmgg/training/orchestration/sanity_check.py` exposes only one public helper: `maybe_run_sanity_check` at `:464`; the other helpers are private (`_...`) | The March row said six public functions. That is no longer true. |
| `M-25` | Resolved in current tree | No `plotting.py` remains under `src/tmgg` | The file named in the row has already been deleted. |
| `M-26` | No-fix | `src/tmgg/training/orchestration/run_experiment.py:244` | The cohesion concern is real, but we are intentionally leaving it in place. Splitting the function further would add more plumbing and names than value right now. |
| `M-27` | Resolved in current tree | `src/tmgg/training/lightning_modules/diffusion_module.py:200` | `_train_loss_discrete` now has an explicit annotation: `TrainLossDiscrete | None`. |
| `M-28` | Resolved in current tree | `matplotlib.use("Agg")` now appears only in `src/tmgg/training/orchestration/run_experiment.py:10` | The repeated side effect across three files is gone. |

## Models

| Row | Updated status | Current evidence | Notes |
| --- | --- | --- | --- |
| `M-54` | Resolved in current tree | `src/tmgg/experiments/exp_configs/models/digress/digress_base.yaml:13-20`; `src/tmgg/models/digress/extra_features.py:234-283`; `tests/test_config_composition.py:234`; `tests/experiments/test_digress_denoising_module.py:35` | DiGress denoising now uses `E=1` to match the generic scalar edge-state bridge, and `EigenvectorAugmentation` accepts both scalar and categorical edge encodings. The composed forward-path regression now passes. |

## Experiments

| Row | Updated status | Current evidence | Notes |
| --- | --- | --- | --- |
| `M-31` | Resolved in current tree | `src/tmgg/experiments/eigenstructure_study/noised_collector.py:252-266`; `src/tmgg/experiments/eigenstructure_study/cli.py:511-513`; `src/tmgg/experiments/eigenstructure_study/execute.py:169-170` | Covariance-evolution serialization now lives on the typed result dataclass via `CovarianceEvolutionResult.to_json_dict()`, and both entry points delegate to it. |
| `M-32` | Resolved in current tree | `src/tmgg/experiments/discrete_diffusion_generative/evaluate_cli.py:24`; `:104`; `tests/experiments/test_discrete_evaluate_cli.py:48-102` | `compute_mmd_metrics` now lives in the module import block, and the regression test patches the module-level symbol to keep the call path explicit. |
| `M-35` | Resolved in current tree | Denoising runners now return `run_experiment(cfg)`, for example `src/tmgg/experiments/spectral_arch_denoising/runner.py:31`, `src/tmgg/experiments/gaussian_diffusion_generative/runner.py:31`, `src/tmgg/experiments/discrete_diffusion_generative/runner.py:30` | The Hydra sweep result-loss issue described in March appears fixed for the denoising runners. |
| `M-36` | Resolved in current tree | `src/tmgg/experiments/spectral_arch_denoising/runner.py:4`; `src/tmgg/experiments/gaussian_diffusion_generative/runner.py:4`; `src/tmgg/experiments/spectral_arch_denoising/__init__.py:9` | The stale `python -m ...` guidance is gone; the runner and package docstrings now point at the canonical CLI commands. |
| `M-37` | Resolved in current tree | `src/tmgg/experiments/embedding_study/execute.py:136`; `tests/experiments/test_embedding_study_execute.py:17-30` | `_generate_graphs()` now returns `list[torch.Tensor]`, and the helper test checks that the generated values are square float adjacency tensors. |
| `M-38` | Resolved in current tree | `src/tmgg/experiments/eigenstructure_study/noised_collector.py:182-266`; `:449-918`; `tests/experiment_utils/test_eigenstructure_study.py:1039-1169` | The noised comparison layer now returns typed dataclasses (`DriftComparisonResult`, `GapDeltaComparisonResult`, `NoiseLevelComparisonResult`, `CovarianceEvolutionResult`, etc.) and the regression tests lock both the typed API and the serialized JSON boundary. |
| `M-40` | Resolved in current tree | `src/tmgg/experiments/exp_configs/base_config_embedding_study.yaml:9`; `tests/modal/test_config_resolution.py:13-17` | The embedding-study base config now declares `_cli_cmd`, and the Modal config-resolution test keeps the command discoverable. |
| `M-41` | Resolved in current tree | `src/tmgg/experiments/exp_configs/grid_search_base.yaml:10-16` | `grid_search_base.yaml` now selects `override /base/callbacks: grid_search` through Hydra defaults and no longer carries `trainer.callbacks`. |
| `M-42` | Resolved in current tree | `src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml:12-17` | The discrete base config now selects `discrete_wandb` through `base/logger` instead of defining an inline logger block. |

## Experiment Configs

| Row | Updated status | Current evidence | Notes |
| --- | --- | --- | --- |
| `M-43` | Resolved in current tree | `src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml:12-17`; no TensorBoard logger configs remain under `src/tmgg/experiments/exp_configs/base/logger` | The March interpolation-error row no longer matches the current config surface. The discrete base config no longer selects a TensorBoard S3 logger at all. |
| `M-46` | Resolved in current tree | `src/tmgg/experiments/exp_configs/data/sbm_n100.yaml:4-8`; `src/tmgg/experiments/exp_configs/data/sbm_n200.yaml:4-9`; `tests/test_config_composition.py:407-443` | The sized SBM configs now inherit `sbm_default` and override only the graph-size-specific fields. |
| `M-47` | Resolved in current tree | `src/tmgg/experiments/exp_configs/stage/_shared_runtime.yaml:1-15`; `src/tmgg/experiments/exp_configs/stage/stage3_diversity.yaml:9-11`; `stage4_benchmarks.yaml:9-11`; `stage5_full.yaml:9-11`; `tests/test_config_composition.py:368-405` | Stage 3-5 now inherit one shared runtime config for optimizer, scheduler, and noise-level defaults, and the composition tests lock the inherited values. |
| `M-48` | Resolved in current tree | `src/tmgg/experiments/exp_configs/grid_search_base.yaml:10-16`; `src/tmgg/experiments/exp_configs/base/logger/grid_wandb.yaml:1` | `grid_wandb.yaml` is now live: the grid-search config selects it through Hydra defaults. |
| `M-49` | Resolved in current tree | No `tensorboard_s3.yaml` remains under `src/tmgg/experiments/exp_configs/base/logger` | The dangerous placeholder config was deleted. |
| `M-50` | Resolved in current tree | The four files no longer exist under `src/tmgg/experiments/exp_configs/data`, and `docs/how-to-run-experiments.md:225` no longer advertises `er_spectral`; `tests/test_config_consistency.py:181-207` | The deeper review established that these presets were not part of a supported config surface. They are now deleted, and a consistency test keeps both the files and the public docs reference from reappearing. |
| `M-51` | Resolved in current tree | `src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml:12-17` | The discrete base config now selects `discrete_nll` and `discrete_wandb` through `base/callbacks/` and `base/logger/` Hydra overrides. |

## Nitpicks

| Row | Updated status | Current evidence | Notes |
| --- | --- | --- | --- |
| `N-1` | Resolved in current tree | `src/tmgg/models/gnn/nvgnn.py:13` | The `NodeVarGNN` docstring is now NumPy-style; the old Google-style `Args:` / `Returns:` markers are gone. |
| `N-2` | Resolved in current tree | No `factory.py` remains under `src/tmgg/models` | The empty `Baselines` section comment cited in March no longer has a target file. |
| `N-3` | Resolved in current tree | No `sample_gaussian` helper remains under `src/tmgg` | The trivial wrapper identified in March has already been removed. |
| `N-4` | Resolved in current tree | `src/tmgg/models/base.py:12`; `:69`; `src/tmgg/training/logging.py:26`; `:332`; `tests/models/test_graph_model_base.py:166-171` | `parameter_count()` now returns a recursive `ParameterCountTree`, and both the logger formatter and the base-model test consume the typed tree without changing the emitted runtime structure. |
| `N-5` | Resolved in current tree | `src/tmgg/experiments/eigenstructure_study/analyzer.py:443`; `src/tmgg/experiments/eigenstructure_study/execute.py:93`; `src/tmgg/experiments/eigenstructure_study/cli.py:194`; `tests/experiment_utils/test_eigenstructure_study.py:469-570` | The CLI and Hydra analyze paths now route their primary `analysis.json` write through `SpectralAnalyzer.save_results()`, and the regression tests lock both call sites onto that helper. |
| `N-6` | Resolved in current tree | `src/tmgg/experiments/grid_search_runner.py:9` | The config path is now the standard `"../exp_configs"` form. |
| `N-7` | Resolved in current tree | `src/tmgg/experiments/stages/runner.py:15`; `:39` | `stages/runner.py` now imports `run_experiment` at module scope, matching the other runner modules. |
| `N-8` | Resolved in current tree | `src/tmgg/experiments/embedding_study/execute.py:91` | The old `# type: ignore[assignment]` is gone; the code now uses `cast(...)` around `OmegaConf.to_container(...)`. |
| `N-9` | Resolved in current tree | `src/tmgg/experiments/exp_configs/_base_infra.yaml:11-16`; `src/tmgg/experiments/exp_configs/base/logger/wandb.yaml:1`; no `multi.yaml` remains under `src/tmgg/experiments/exp_configs/base/logger` | The March row no longer matches the current tree. `wandb.yaml` is the canonical base logger, and the dead `multi.yaml` file has been deleted. |
| `N-10` | Resolved in current tree | `src/tmgg/experiments/exp_configs/grid_search_base.yaml:40-41`; `tests/test_config_consistency.py:101-113` | `evaluation.final_eval_noise_levels` is gone from `grid_search_base.yaml`, and the consistency test keeps that dead key from reappearing. |
| `N-11` | Resolved in current tree | `src/tmgg/experiments/exp_configs/stage/stage3_diversity.yaml:1-17`; `stage4_benchmarks.yaml:1-17`; `stage5_full.yaml:1-17`; `tests/test_config_consistency.py:116-141` | The future-stage configs now keep only executable settings, and the consistency test asserts the documentation-only metadata keys stay removed. |
| `N-12` | Resolved in current tree | `src/tmgg/experiments/exp_configs/data/pyg_enzymes.yaml:1`; `ring_of_cliques.yaml:1`; `sbm_default.yaml:1`; `grid_gaussian.yaml:1`; `tests/test_config_consistency.py:144-177` | Every remaining data config now declares its Hydra package explicitly, using `data` for ordinary data-group files and `_global_` only for the grid overlays that intentionally set top-level keys. |
| `N-13` | Resolved in current tree | `src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml:3` | The header now points to the CLI command `tmgg-gaussian-gen`. |

## Practical Burn-Down

If this tracker is to become current again, the next useful move is to split the
remaining rows into three buckets and update `SUMMARY.md` accordingly:

- rows that are already fixed and should be marked `RESOLVED`
- rows that are still open and can be scheduled for implementation
- rows that need a short human decision because they may be intentionally
  retained rather than dead
- rows that are consciously accepted as `No-fix` because cleanup churn would
  outweigh the benefit
