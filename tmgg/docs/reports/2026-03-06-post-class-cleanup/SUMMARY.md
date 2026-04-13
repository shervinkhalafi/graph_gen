# Post-Cleanup Code Review: Combined Summary

Reviewed: 2026-03-06, `cleanup` branch. Five reviews covering `src/tmgg/data/`, `src/tmgg/models/`, `src/tmgg/experiments/_shared_utils/`, `src/tmgg/experiments/` (individual experiments), and `src/tmgg/experiments/exp_configs/` (YAML configs). Total: 101 findings across ~15k lines of Python and 90 YAML files (some overlap between the experiments and exp_configs reviews is noted via cross-references).

Findings below are ordered by priority, grouped by source within each tier. Where a finding appears in multiple reports (e.g., config drift visible from both Python and YAML sides), it is listed once under the most specific source with a cross-reference note.

---

## Critical — will crash or produce silently wrong results

### exp_configs

**[TM-1] `regular_spectral.yaml`: `graph_type: d_regular` does not exist in `VALID_TYPES`**
`SyntheticGraphDataset` raises `ValueError` at construction. Fix: change to `graph_type: regular`.

**[TM-2] `base_config_digress.yaml` interpolates `${model.digress_arch}` and `${model.digress_mode}`, which are undefined in every DiGress model config**
OmegaConf throws `InterpolationKeyError` at composition time. Fix: use `${model.model_type}` (present in all DiGress configs) or a static string.

**[TM-3] `base_config_gaussian_diffusion.yaml`: `_target_: tmgg.data.noise.DigressNoiseGenerator` is not a valid Python path**
No `tmgg.data.noise` module exists. Hydra `instantiate()` raises `ImportError`. Fix: change to `tmgg.data.DigressNoiseGenerator`.

**[TM-4] `base_config_discrete_diffusion_generative.yaml` uses `p_in`/`p_out` but Python reads `p_intra`/`p_inter`**
`MultiGraphDataModule._generate_and_split()` reads `graph_config.get("p_intra", 0.7)`. The keys `p_in`/`p_out` are silently ignored; defaults (0.7/0.1) happen to coincide with intended values, masking the bug. Fix: rename to `p_intra`/`p_inter`.

### experiments

**[EXP-C-1] `grid_search_runner.py` uses `version_base="1.3"` while every other runner uses `version_base=None`**
Activates different Hydra output-directory semantics, causing silent path-resolution divergence. Fix: change to `version_base=None`.

**[EXP-C-2] `stages/runner.py` hard-codes `base_config_spectral_arch` as default config**
The "unified entry point for all model families" silently defaults to a spectral denoising configuration. Fix: use `base_config_training` or require `model=` explicitly.

---

## Important — silent behavioral divergence, significant redundancy, or architectural debt

### data

~~**[SOC-1] `train_val_test_split` duplicated in both dataset classes, belongs in data modules**~~
~~RESOLVED (`adf95f72`): Removed from both datasets; data module calls `split_indices` directly.~~

~~**[SOC-2] Lazy import of `_split` inside dataset methods is a workaround for SOC-1**~~
~~RESOLVED (`adf95f72`): Lazy imports removed with SOC-1.~~

~~**[INC-1] `SyntheticGraphDataset` and `PyGDatasetWrapper` are structurally asymmetric**~~
~~RESOLVED (`bdb96170`): `get_masks()` → `np.ndarray` in both; `_num_graphs` → `num_graphs`; removed unused `to_torch()`.~~

~~**[RED-1] Duplicate PyG loading logic across `PyGDatasetWrapper` and `SingleGraphDataModule`**~~
~~RESOLVED (`c37f105d`): `SingleGraphDataModule._load_pyg_graph` delegates to `PyGDatasetWrapper`; symmetrize+binarize moved into wrapper.~~

### models

~~**[M/R-1] Eigenvector truncation/padding block copy-pasted three times**~~
~~RESOLVED (`5184d83c`): `TruncatedEigenEmbedding` wrapper replaces 3x copy-pasted truncation blocks.~~

~~**[M/R-2] Spectral polynomial loop duplicated in `GraphFilterBank` and `SpectralProjectionLayer`**~~
~~RESOLVED (`4b49e114`): `spectral_polynomial()` extracted to `layers/graph_ops.py`; both call sites delegate.~~

~~**[M/I-5] `EigenEmbedding` and `TopKEigenLayer` are two parallel eigenvector implementations**~~
~~RESOLVED (`a0d493f8`): `EigenEmbedding` wraps `TopKEigenLayer`; diagnostics + sign-fix + reg consolidated.~~

~~**[M/S-1] `diffusion_utils.py` mixes mathematical utilities and training-loop concerns**~~
~~RESOLVED (`9d2d831e`): Split into `diffusion_math.py` and `diffusion_sampling.py`; original deleted.~~

~~**[M/S-2] `train_loss.py` lives inside `models/digress/` rather than training package**~~
~~RESOLVED (`6e491161`): Moved to `experiments/_shared_utils/train_loss_discrete.py`.~~

~~**[M/T-1] `extra_features: Any` in `GraphTransformer`**~~
~~RESOLVED (`beebf036`): `ExtraFeaturesProvider` Protocol replaces `Any`.~~

### shared_utils

~~**[SU/RD-1] `graph_evaluator.py` duplicates all structural metric functions from `graph_structure_metrics.py`**~~
~~RESOLVED (`0349a6bc`): Dead `graph_structure_metrics.py` deleted; tests redirected to evaluator functions.~~

~~**[SU/RD-2] Spectral metrics triplication across `metrics.py`, `spectral_deltas.py`, `analyzer.py`**~~
~~RESOLVED (`1b4e640c`): Four primitive functions (`eigenvalue_drift`, `subspace_distance`, `eigengap_delta`, `alg_connectivity_delta`) consolidated into `spectral_deltas.py` as the single source of truth. `metrics.py` migrated from scipy to torch `eigh`/`eigvalsh` and delegates to `eigenvalue_drift`. `analyzer.py` delta functions now call the shared primitives. Scipy dependency removed from metrics.~~

~~**[SU/DC-1] `graph_structure_metrics.py` is functionally dead**~~
~~RESOLVED (`0349a6bc`): Deleted together with SU/RD-1.~~

~~**[SU/CO-1] `sanity_check.py` couples via unsafe `cast` to `SingleStepDenoisingModule` internals**~~
~~RESOLVED (`c2250dc5`): `SanityCheckableModel` Protocol replaces double-cast.~~

### experiments

**[EXP-R-1] `stages/runner.py` and `spectral_arch_denoising/runner.py` are byte-for-byte identical**
Both 48 lines. Fix: make `stages/runner.py` genuinely unified (different default config).

~~**[EXP-R-2] `evaluate_checkpoint.py` redundantly symmetrizes/zeros SBM adjacency matrices**~~
~~RESOLVED (`65c2a14a`): Replaced manual SBM loop with `generate_sbm_batch`; removed redundant post-processing and unused imports.~~

~~**[EXP-R-3] Graph-type alias maps duplicated across `evaluate_checkpoint.py` and `collector.py`**~~
~~RESOLVED (`9ec3f9e6`): Fixed alias resolution order in `SyntheticGraphDataset`; removed caller-side alias maps.~~

~~**[EXP-R-4] `NoisedAnalysisComparator` repeats batch-loading boilerplate in six methods**~~
~~RESOLVED: Extracted `_iter_paired_batches(eps)` generator with `strict=True`; all 6 methods delegate to it. `compute_full_comparison` rewritten as single-pass (5x→1x disk reads per noise level).~~

~~**[EXP-R-6] `embedding_study/cli.py` reimplements SBM generation with O(n^2) nested loops**~~
~~RESOLVED (`9a3fd61a`): Deleted `create_sbm_graphs()`; both `cli.py` and `execute.py` use `generate_sbm_batch`.~~

~~**[EXP-S-1] `evaluate_cli.py` accesses private `_generate_graphs` from `DiffusionModule`**~~
~~RESOLVED: Renamed to public `generate_graphs`; all call sites updated.~~

~~**[EXP-S-3] `generate_reference_graphs` lives in `gaussian_diffusion_generative/` but is imported by `discrete_diffusion_generative/`**~~
~~RESOLVED: Moved to `_shared_utils/reference_graphs.py`; `evaluate_checkpoint.py` deleted; all importers updated.~~

~~**[EXP-K-1] `NoisedAnalysisComparator` uses `strict=False` in six `zip()` calls**~~
~~RESOLVED: Fixed together with EXP-R-4; `_iter_paired_batches` uses `strict=True`.~~

~~**[EXP-CD-3] `base_config_embedding_study.yaml` and `base_config_eigenstructure.yaml` do not inherit from `_base_infra`, replicate infrastructure keys**~~
~~RESOLVED: Extracted `_base_structural_study.yaml` with shared `wandb_entity`, `seed`, `paths.results_dir`; both study configs inherit from it.~~

### exp_configs

~~**[IE-1] Seven data configs set `noise_levels: ${noise_levels}` without also setting `noise_type`**~~
~~RESOLVED: Added `noise_type: ${noise_type}` to all 7 configs.~~

~~**[IE-2] All seven report configs reference `/paths: default`, which does not exist**~~
~~RESOLVED: Created `exp_configs/paths/default.yaml` with `unified_parquet`, `reports`, `eigenstructure_results`, `embedding_results`.~~

~~**[IE-3] `grid_search_base.yaml` model noise uses `${noise_type}` but data config uses hardcoded `"gaussian"`**~~
~~RESOLVED: Changed to `${data.noise_type}` and `${data.noise_levels}` in model section.~~

~~**[RED-1/cfg] Grid data configs (`grid_digress`, `grid_gaussian`, `grid_rotation`) are near-identical**~~
~~RESOLVED (`595f1999`): Extracted `grid_base.yaml`; three configs now inherit and override only `noise_type`.~~

**[RED-2/cfg] DiGress model configs share extensive boilerplate across 14 files**
Same `_target_`, `model_type`, `output_dims`, noise settings, loss type, seed repeated in all. Meaningful variation is `hidden_dims`, `n_layers`, GNN/spectral flags, optimizer settings. Fix: extract `digress_base.yaml` and use Hydra defaults-list inheritance.

**[SD-1] GNN model configs split into two incompatible schema families**
Explicit-interpolation family vs `lightning_base` inheritance family. The latter silently drops `log_spectral_deltas`, `spectral_k`, `seed`. *(Cross-ref: overlaps with EXP-CD-1, EXP-CD-2.)* Fix: deprecate `lightning_base` approach; use explicit interpolations.

**[SD-2] DiGress model configs split into two optimizer-configuration styles**
`digress_sbm_*` hard-codes optimizer settings; `digress_transformer_*` omits `amsgrad` entirely, silently defaulting to `false`. Fix: add `amsgrad: ${amsgrad}` and a top-level default in `_base_infra.yaml`.

---

## Minor — inconsistency, minor dead code, or minor coupling

### data

| ID | Description |
|----|-------------|
| SOC-3 | `_split.py` position ambiguous; should be in `data_modules/` |
| INC-2 | Neither dataset class inherits `torch.utils.data.Dataset`; `__getitem__` vestigial |
| INC-3 | `get_dataset_info` re-reads config instead of `self.num_nodes` |
| RED-2 | `_split_indices` static method is a pure passthrough |
| RED-3 | `import random` deferred inside methods without reason |
| DC-1 | `__getitem__`/`__len__` never used via Dataset protocol |
| DC-2 | `noise_levels`/`noise_type` stored, never used, undocumented |

### models

| ID | Description |
|----|-------------|
| M/R-3 | `get_features` duplicates `_spectral_forward` transformer pass in `bilinear.py` |
| M/R-4 | `GNNSymmetric.__init__` near-verbatim copy of `GNN.__init__` |
| M/R-5 | `_TransformerBlock` reimplements MHA independently |
| M/I-2 | `get_config` includes `model_class` key only in some model families |
| M/I-3 | Unbatched handling in spectral family only; never exercised in production |
| M/I-4 | `SequentialDenoisingModel` uses `hasattr` instead of typed Protocol |
| M/S-3 | `NodeEdgeBlock.forward` falls back to wrong adjacency channel (potential correctness bug) |
| M/D-1 | `digress/graph_types.py` is a pure re-export module |
| M/D-3 | `MultiLayerAttention` not registered in `ModelRegistry` |
| M/T-2 | `ShrinkageWrapper._get_inner_feature_dim` gives wrong answer for `MultiLayerBilinearDenoiser` (latent shape bug) |
| M/T-3 | Behavioral flags smuggled into `dict[str, int]` dimension dict |

### shared_utils

| ID | Description |
|----|-------------|
| RD-3 | `create_graph_denoising_figure` is a pure wrapper with no added behavior |
| DC-2 | `DebugCallback` has no callers |
| DC-3 | `compute_mmd_from_adjacencies` uncalled |
| DC-4 | `GraphEvaluator.set_num_nodes()` uncalled, attribute unused |
| DC-5 | `visualization_interval` stored, never used |
| DC-6 | `sanity_check.py` five public functions only reachable indirectly |
| DC-7 | `plotting.py` (~840 lines) unreferenced from production path |
| CO-2 | `run_experiment.py` mixes too many concerns (seed, W&B, checkpoints, training, S3 sync) |
| CO-3 | `DiffusionModule._train_loss_discrete` conditionally created, no type annotation |
| CO-4 | `matplotlib.use("Agg")` called as module-level side effect in three files |
| CFG-1 | Cosine-warmup scheduler falls back to epoch-based estimation (violates step-based convention) |
| CFG-2 | `run_experiment.py` accesses `config.trainer.max_steps` without guarding |

### experiments

| ID | Description |
|----|-------------|
| EXP-R-5 | Covariance evolution serialization duplicated between CLI and Hydra execute |
| EXP-S-2 | Inline import of `compute_mmd_metrics` despite top-level imports from same module |
| EXP-K-2 | Hydra execute imports from CLI module (inverted dependency) |
| EXP-D-1 | `MolecularDataModule` placeholder (never implemented per scope) |
| EXP-C-3 | `main() -> None` in most runners discards `run_experiment()` return dict |
| EXP-C-4 | Docstrings show `python -m` instead of `tmgg-*` CLI name |
| EXP-T-1 | `_generate_graphs` returns bare `list` instead of `list[torch.Tensor]` |
| EXP-T-3 | `dict[str, Any]` return types in comparator methods lose schema information |
| EXP-CD-4 | `digress_sbm_small_vanilla.yaml` `dy: 32` vs 128 in other "small DiGress" variants |
| EXP-CD-5 | `base_config_embedding_study.yaml` missing `_cli_cmd` key |
| EXP-CS-1 | `grid_search_base.yaml` inline callbacks bypasses shared infrastructure |
| EXP-CS-2 | `base_config_discrete_diffusion_generative.yaml` inline logger duplicates S3 path pattern |

### exp_configs

| ID | Description |
|----|-------------|
| IE-4 | TensorBoard S3 path in discrete diffusion config lacks local fallback |
| SD-4 | `nx.yaml`, `nx_square.yaml`, `nx_star.yaml` use capitalized `"Digress"` |
| SD-5 | Mixed `_target_` import paths for same `GraphDataModule` class across data configs |
| RED-3/cfg | `sbm_n100.yaml`/`sbm_n200.yaml` near-copies of `sbm_default.yaml` |
| RED-4/cfg | Stage configs 3-5 repeat identical optimizer blocks |
| DC-1 | `base/logger/grid_wandb.yaml` never referenced; duplicated inline |
| DC-2 | `base/logger/tensorboard_s3.yaml` hardcoded `your-bucket` placeholder |
| DC-4 | Four `_spectral` data configs appear unused by any pipeline |
| HBP-2 | `base_config_discrete_diffusion_generative.yaml` inline callback/logger overrides bypass config groups |
| DOC-1 | `single_graph_base.yaml` hardcodes noise values, breaks CLI overrides silently |

---

## Nitpick

| Source | ID | Description |
|--------|----|-------------|
| models | M/I-1 | Google-style docstring in NumPy-style codebase (`nvgnn.py`) |
| models | M/I-6 | Empty `Baselines` section comment in `factory.py` |
| models | M/D-2 | `diffusion_utils.sample_gaussian` is trivial `torch.randn` wrapper |
| models | M/T-4 | `BaseModel.parameter_count` returns `dict[str, Any]` |
| experiments | EXP-D-2 | `SpectralAnalyzer.save_results` never called |
| experiments | EXP-C-5 | `grid_search_runner.py` uses `Path(__file__)` vs relative string |
| experiments | EXP-C-6 | `stages/runner.py` lazy-imports `run_experiment` without explanation |
| experiments | EXP-T-2 | `type: ignore[assignment]` on `OmegaConf.to_container` without narrowing |
| exp_configs | DC-3 | `base/logger/{multi,csv,wandb}.yaml`, `lightning_base.yaml` unreferenced from defaults lists |
| exp_configs | DC-5 | `grid_search_base.yaml` `final_eval_noise_levels` read by no code |
| exp_configs | DC-6 | Stage configs 3-5 `trigger_conditions`, `ablations`, `statistical_tests` read by no code |
| exp_configs | HBP-1 | `@package data` header inconsistent across data configs |
| exp_configs | DOC-2 | `base_config_gaussian_diffusion.yaml` comment references `python -m` instead of CLI entry point |

---

## Statistics

| Priority | Count |
|----------|-------|
| Critical | 6 |
| Important | 30 |
| Minor | 52 |
| Nitpick | 13 |
| **Total** | **101** |

| Source | Critical | Important | Minor | Nitpick | Total |
|--------|----------|-----------|-------|---------|-------|
| `data/` | 0 | 4 | 7 | 0 | 11 |
| `models/` | 0 | 6 | 11 | 4 | 21 |
| `_shared_utils/` | 0 | 4 | 12 | 0 | 16 |
| experiments (Python) | 2 | 9 | 12 | 4 | 27 |
| exp_configs (YAML) | 4 | 7 | 10 | 5 | 26 |

Two exp_configs findings (SD-1, SD-2) overlap with experiments findings (EXP-CD-1, EXP-CD-2) and are listed once each under exp_configs with cross-references.

The six critical findings are all straightforward one-line fixes. The important tier clusters around three themes: (1) structural metric duplication and drift (SU/RD-1, SU/DC-1, SU/RD-2), (2) dataset/data-module separation of concerns (SOC-1, SOC-2, RED-1), and (3) config schema drift between model families (SD-1, SD-2, IE-1, IE-2, RED-1/cfg, RED-2/cfg). Addressing these three clusters would resolve roughly half the important findings.

---

## Consolidated Reference Table

Every finding in a single table with unified IDs. The `#` column assigns a sequential number; `Original` maps back to the per-report ID for traceability. `File(s)` gives the primary file(s) affected. The table is self-contained: each row carries enough context to understand the issue without reading the prose sections above.

### Critical

| # | Source | Original | Category | File(s) | Description | Fix |
|---|--------|----------|----------|---------|-------------|-----|
| C-1 | exp_configs | TM-1 | Target-Mismatch | `data/regular_spectral.yaml` | ~~RESOLVED (`831db677`)~~ `graph_type: d_regular` → `regular` |
| C-2 | exp_configs | TM-2 | Interpolation-Error | `base_config_digress.yaml:11` | ~~RESOLVED (`831db677`)~~ `${model.digress_arch}_${model.digress_mode}` → `${model.model_type}` |
| C-3 | exp_configs | TM-3 | Target-Mismatch | `base_config_gaussian_diffusion.yaml:28,34` | ~~RESOLVED (`831db677`)~~ `tmgg.data.noise.` → `tmgg.data.` |
| C-4 | exp_configs | TM-4 | Key-Mismatch | `base_config_discrete_diffusion_generative.yaml:47-48` | ~~RESOLVED (`831db677`)~~ `p_in`/`p_out` → `p_intra`/`p_inter` |
| C-5 | experiments | EXP-C-1 | Inconsistency | `grid_search_runner.py:15` | ~~RESOLVED (`efcee542`)~~ `version_base="1.3"` → `None` |
| C-6 | experiments | EXP-C-2 | Inconsistency | `stages/runner.py:22-26` | ~~RESOLVED (`c23107a5`)~~ Config hierarchy flattened; stages runner uses `base_config_denoising` |

### Important

| # | Source | Original | Category | File(s) | Description | Fix |
|---|--------|----------|----------|---------|-------------|-----|
| I-1 | data | SOC-1 | Separation | `synthetic_graphs.py:798`, `pyg_datasets.py:169` | ~~RESOLVED (`adf95f72`)~~ `train_val_test_split` removed from both datasets; data module calls `split_indices` directly |
| I-2 | data | SOC-2 | Separation | `synthetic_graphs.py:820`, `pyg_datasets.py:195` | ~~RESOLVED (`adf95f72`)~~ Lazy imports removed with I-1 |
| I-3 | data | INC-1 | Inconsistency | `synthetic_graphs.py:776`, `pyg_datasets.py:155` | ~~RESOLVED (`bdb96170`)~~ `get_masks()` → `np.ndarray` in both; `_num_graphs` → `num_graphs`; removed unused `to_torch()` and `import torch` |
| I-4 | data | RED-1 | Redundancy | `pyg_datasets.py:61-116`, `single_graph_data_module.py:217-284` | ~~RESOLVED (`c37f105d`)~~ `SingleGraphDataModule._load_pyg_graph` delegates to `PyGDatasetWrapper`; symmetrize+binarize moved into wrapper |
| I-5 | models | M/R-1 | Redundancy | `gnn/gnn_sym.py:80-91`, `gnn/nvgnn.py:69-80` | ~~RESOLVED (`5184d83c`)~~ `TruncatedEigenEmbedding` wrapper (composition, not subclass — LSP) replaces 3x copy-pasted truncation blocks |
| I-6 | models | M/R-2 | Redundancy | `filter_bank.py:154-173`, `spectral_projection.py:93-100` | ~~RESOLVED (`4b49e114`)~~ `spectral_polynomial()` extracted to `layers/graph_ops.py`; both call sites delegate |
| I-7 | models | M/I-5 | Inconsistency | `layers/eigen_embedding.py`, `spectral_denoisers/topk_eigen.py` | ~~RESOLVED (`a0d493f8`)~~ `EigenEmbedding` wraps `TopKEigenLayer`; diagnostics + sign-fix + reg consolidated in `TopKEigenLayer`; `EigenDecompositionError` canonical in `topk_eigen.py` |
| I-8 | models | M/S-1 | Separation | `digress/diffusion_utils.py` | ~~RESOLVED (`9d2d831e`)~~ Split into `diffusion_math.py` (14 pure-math functions) and `diffusion_sampling.py` (13 sampling/posterior/masking functions); original deleted; all 5 consumer imports updated |
| I-9 | models | M/S-2 | Separation | `digress/train_loss.py` | ~~RESOLVED (`6e491161`)~~ `TrainLossDiscrete` moved to `experiments/_shared_utils/train_loss_discrete.py`; consumer imports updated |
| I-10 | models | M/T-1 | Type-Safety | `digress/transformer_model.py:803` | ~~RESOLVED (`beebf036`)~~ `ExtraFeaturesProvider` Protocol added to `extra_features.py`; `Any` replaced with `ExtraFeaturesProvider \| None` in transformer + factory |
| I-11 | shared_utils | SU/RD-1 | Redundancy | `graph_evaluator.py:133-434`, `graph_structure_metrics.py` | ~~RESOLVED (`0349a6bc`)~~ Dead `graph_structure_metrics.py` deleted; tests redirected to `graph_evaluator.py` functions |
| I-12 | shared_utils | SU/RD-2 | Redundancy | `metrics.py:12-136`, `spectral_deltas.py:23-116`, `analyzer.py` | ~~RESOLVED (`1b4e640c`)~~ Four primitives extracted into `spectral_deltas.py`; `metrics.py` migrated scipy→torch, delegates to `eigenvalue_drift`; `analyzer.py` delegates to shared primitives | Consolidate primitives in `spectral_deltas.py` |
| I-13 | shared_utils | SU/DC-1 | Dead-Code | `graph_structure_metrics.py` | ~~RESOLVED (`0349a6bc`)~~ Deleted together with I-11; 442 lines removed |
| I-14 | shared_utils | SU/CO-1 | Coupling | `sanity_check.py:442-476` | ~~RESOLVED (`c2250dc5`)~~ `SanityCheckableModel` Protocol replaces double-cast; direct attribute access with runtime `isinstance` assertion for `nn.Module` |
| I-15 | experiments | EXP-R-1 | Redundancy | `stages/runner.py`, `spectral_arch_denoising/runner.py` | ~~RESOLVED (`c23107a5`)~~ `stages/runner.py` now uses `base_config_denoising`; no longer identical to spectral runner |
| I-16 | experiments | EXP-R-2 | Redundancy | `evaluate_checkpoint.py:67-71` | ~~RESOLVED (`65c2a14a`)~~ Replaced manual SBM loop (block_size computation + `generate_sbm_adjacency` + redundant post-processing) with `generate_sbm_batch`; removed unused `numpy` and `generate_sbm_adjacency` imports | Remove the three post-processing lines |
| I-17 | experiments | EXP-R-3 | Redundancy | `evaluate_checkpoint.py:76-83`, `collector.py:176-182` | ~~RESOLVED (`9ec3f9e6`)~~ Fixed alias resolution order in `SyntheticGraphDataset` (resolve before validate); removed caller-side alias maps in both files; cleaned aliases from `VALID_TYPES` | Extract `resolve_graph_type()` into shared utility |
| I-18 | experiments | EXP-R-4 | Redundancy | `noised_collector.py:180-670` | ~~RESOLVED (`367a3e96`)~~ Extracted `_iter_paired_batches(eps)` generator with `strict=True`; all 6 comparison methods use it; `compute_full_comparison` rewritten as single-pass (5x→1x disk reads per noise level) | Extract `_iter_paired_batches(eps)` generator; single-pass metrics |
| I-19 | experiments | EXP-R-6 | Redundancy | `embedding_study/cli.py:28-73` | ~~RESOLVED (`9a3fd61a`)~~ Deleted `create_sbm_graphs()` (46 lines); both `cli.py` and `execute.py` now use `generate_sbm_batch`; seed propagation fixed in `execute.py` | Replace with wrapper around canonical utility |
| I-20 | experiments | EXP-S-1 | Coupling | `evaluate_cli.py:97` | ~~RESOLVED~~ Renamed `_generate_graphs` → `generate_graphs`; all call sites updated |
| I-21 | experiments | EXP-S-3 | Separation | `evaluate_cli.py:24-26` | ~~RESOLVED~~ Moved to `_shared_utils/reference_graphs.py`; `evaluate_checkpoint.py` deleted; all importers updated |
| I-22 | experiments | EXP-K-1 | Coupling | `noised_collector.py` (6 locations) | ~~RESOLVED (`367a3e96`)~~ Fixed together with I-18: `_iter_paired_batches` uses `strict=True`; all 6 `strict=False` zip calls eliminated | Change to `strict=True` |
| I-23 | experiments | EXP-CD-3 | Inconsistency | `base_config_embedding_study.yaml`, `base_config_eigenstructure.yaml` | ~~RESOLVED~~ Extracted `_base_structural_study.yaml`; both study configs inherit shared `wandb_entity`, `seed`, `paths.results_dir` |
| I-24 | exp_configs | IE-1 | Interpolation-Error | 7 data configs (`ring_of_cliques`, `er_spectral`, etc.) | ~~RESOLVED~~ Added `noise_type: ${noise_type}` to all 7 configs |
| I-25 | exp_configs | IE-2 | Missing-Config | 7 report configs | ~~RESOLVED~~ Created `exp_configs/paths/default.yaml` |
| I-26 | exp_configs | IE-3 | Interpolation-Error | `grid_search_base.yaml:64-65` | ~~RESOLVED~~ Changed to `${data.noise_type}` / `${data.noise_levels}` in model section |
| I-27 | exp_configs | RED-1/cfg | Redundancy | `grid_digress.yaml`, `grid_gaussian.yaml`, `grid_rotation.yaml` | ~~RESOLVED (`595f1999`)~~ Extracted `grid_base.yaml` (35 lines shared); each variant now 7 lines inheriting via Hydra defaults, overriding only `noise_type` | Create `data/grid_base.yaml`; three minimal children |
| I-28 | exp_configs | RED-2/cfg | Redundancy | 14 `models/digress/*.yaml` | ~~RESOLVED (`66146cf9`)~~ Extracted `digress_base.yaml` with shared keys; all 12 children inherit via `defaults: [digress_base, _self_]` |
| I-29 | exp_configs | SD-1 | Schema-Drift | `symmetric_gnn.yaml`, `nodevar_gnn.yaml`, `hybrid_gnn_only.yaml` | ~~RESOLVED (`7149fdf0`)~~ Deleted `lightning_base.yaml`; 3 configs converted to explicit interpolations matching `standard_gnn.yaml` |
| I-30 | exp_configs | SD-2 | Schema-Drift | 6 `digress_transformer_*.yaml` | ~~RESOLVED (`66146cf9`)~~ Added `amsgrad: false` to `_base_infra.yaml`; `digress_base.yaml` uses `${amsgrad}`; SBM configs override to `true` |

### Minor

| # | Source | Original | Category | File(s) | Description |
|---|--------|----------|----------|---------|-------------|
| M-1 | data | SOC-3 | Separation | `_split.py` | ~~RESOLVED (`007fe4ef`)~~ Moved to `data_modules/_split.py`; production caller imports locally from the data-modules layer |
| M-2 | data | INC-2 | Inconsistency | `synthetic_graphs.py`, `pyg_datasets.py` | ~~RESOLVED (`f9a76062`)~~ `SyntheticGraphDataset` and `PyGDatasetWrapper` now subclass `torch.utils.data.Dataset`; protocol regression tests cover both wrappers |
| M-3 | data | INC-3 | Inconsistency | `data_module.py:250` | `get_dataset_info` re-reads `graph_config.get("num_nodes")` instead of using `self.num_nodes` |
| M-4 | data | RED-2 | Redundancy | `multigraph_data_module.py:162-173` | ~~RESOLVED (`adf95f72`)~~ `_split_indices` wrapper removed; direct `split_indices` call |
| M-5 | data | RED-3 | Redundancy | `synthetic_graphs.py:89,373`, `multigraph_data_module.py:268` | ~~RESOLVED (`53dcc3bc`)~~ Hoisted `import random` to module level in `synthetic_graphs.py`; `multigraph_data_module.py` deferral appropriate (only used in one method) |
| M-6 | data | DC-1 | Dead-Code | `synthetic_graphs.py:750-754` | ~~INVALID~~ `__getitem__`/`__len__` called by `reference_graphs.py:68` |
| M-7 | data | DC-2 | Dead-Code | `data_module.py:124-126` | ~~RESOLVED (`53dcc3bc`)~~ Removed dead `noise_levels`/`noise_type` attrs from `GraphDataModule` and `SingleGraphDataModule`; the separate Hydra-constructor compatibility drift is now tracked as `M-53` |
| M-8 | models | M/R-3 | Redundancy | `spectral_denoisers/bilinear.py:527-562` | ~~RESOLVED (`6d593c32`)~~ Extracted `_compute_hidden()` in `MultiLayerBilinearDenoiser`; both `_spectral_forward` and `get_features` delegate |
| M-9 | models | M/R-4 | Redundancy | `gnn/gnn.py`, `gnn/gnn_sym.py` | ~~RESOLVED (`969261d7`)~~ `GNNSymmetric` inherits from `GNN`; deletes `out_y`; overrides `forward()` for symmetric reconstruction |
| M-10 | models | M/R-5 | Redundancy | `spectral_denoisers/bilinear.py:297-377`, `layers/mha_layer.py` | ~~RESOLVED (`9ad24a4d`)~~ `_TransformerBlock` composes `MultiHeadAttention` instead of reimplementing MHA |
| M-11 | models | M/I-2 | Inconsistency | various `get_config` | ~~RESOLVED (`9ad24a4d`)~~ Removed unused `model_class` key from 3 `get_config()` implementations |
| M-12 | models | M/I-3 | ~~INVALID~~ | `spectral_denoisers/` | ~~Unbatched (2D) tensor handling is live code: `GraphData.to_adjacency()` can return 2D for unbatched graphs~~ |
| M-13 | models | M/I-4 | Inconsistency | `hybrid/hybrid.py:64-69` | ~~RESOLVED (`783aa762`)~~ Added `EmbeddingProvider` Protocol; typed `embedding_model` properly; fixed broken `_make_hybrid` factory |
| M-14 | models | M/S-3 | Correctness | `digress/transformer_model.py:392-393` | ~~RESOLVED (`8aa10757`)~~ `E[..., 0]` fallback replaced with `ValueError`; `GraphStructure` bundles adjacency/eigenvectors/eigenvalues; `_GraphTransformer.forward` accepts `GraphData` |
| M-15 | models | M/D-1 | Dead-Code | `digress/graph_types.py` | ~~RESOLVED (`9ad24a4d`)~~ Removed `GraphData`/`collapse_to_indices` re-exports; importers redirect to canonical source |
| M-16 | models | M/D-3 | Dead-Code | `attention/attention.py` | ~~RESOLVED (`e8e7528d`)~~ Registered `MultiLayerAttention` as `"attention"` in `ModelRegistry` |
| M-17 | models | M/T-2 | Type-Safety | `spectral_denoisers/shrinkage_wrapper.py:186-197` | ~~RESOLVED (`2cd3c95b`)~~ Added `feature_dim` property to `SpectralDenoiser` hierarchy; `ShrinkageWrapper` reads it directly |
| M-18 | models | M/T-3 | Type-Safety | `digress/transformer_model.py:589-604` | ~~RESOLVED (`e8e7528d`)~~ Projection flags split into `projection_config: dict[str, bool \| int]`; 7 YAML configs updated |
| M-19 | shared_utils | RD-3 | Redundancy | `plotting.py:319-358` | ~~RESOLVED (`d93c5912`)~~ Deleted dead `create_graph_denoising_figure` wrapper |
| M-20 | shared_utils | DC-2 | Dead-Code | `debug_callback.py` | ~~RESOLVED (`d93c5912`)~~ Deleted dead `DebugCallback` module |
| M-21 | shared_utils | DC-3 | Dead-Code | `mmd_metrics.py:486-534` | ~~RESOLVED (`d93c5912`)~~ Deleted dead `compute_mmd_from_adjacencies` and tests |
| M-22 | shared_utils | DC-4 | Dead-Code | `graph_evaluator.py:525-533` | ~~RESOLVED (`d93c5912`)~~ Deleted dead `set_num_nodes()` and `self.num_nodes` |
| M-23 | shared_utils | DC-5 | Dead-Code | `denoising_module.py:84,109,147` | ~~RESOLVED (`53dcc3bc`)~~ Removed dead `visualization_interval` param, docstring, and attr |
| M-24 | shared_utils | DC-6 | Dead-Code | `sanity_check.py` | Six public functions; only `maybe_run_sanity_check` reachable externally; five should be private |
| M-25 | shared_utils | DC-7 | Dead-Code | `plotting.py` (~840 lines) | No import from any runner, module, or test file; may be notebook-only |
| M-26 | shared_utils | CO-2 | Cohesion | `run_experiment.py:156-269` | ~~INVALID (`2026-04-13`, no-fix)~~ Cohesion concern acknowledged, but further splitting is intentionally declined because the churn would be noisier than the value of the refactor |
| M-27 | shared_utils | CO-3 | Type-Safety | `diffusion_module.py:137-144` | `_train_loss_discrete` conditionally created with no class-level annotation; presence inferred via `isinstance` |
| M-28 | shared_utils | CO-4 | Cohesion | `logging.py:10`, `sanity_check.py:10`, `plotting.py:10` | `matplotlib.use("Agg")` module-level side effect repeated in three files |
| M-29 | shared_utils | CFG-1 | Config | `optimizer_config.py:193-220` | ~~RESOLVED (`7cef584f`)~~ Removed epoch-based fallback; `RuntimeError` if `trainer.max_steps` unset; step-based only |
| M-30 | shared_utils | CFG-2 | Config | `run_experiment.py:193` | ~~RESOLVED (`53dcc3bc`)~~ Guarded with `OmegaConf.select(config, "trainer.max_steps", default=None)` |
| M-31 | experiments | EXP-R-5 | Redundancy | `cli.py:528-544`, `execute.py:169-184` | ~~RESOLVED (`007fe4ef`)~~ Covariance-evolution serialization now lives on the typed result dataclass, and both entry points delegate to `to_json_dict()` |
| M-32 | experiments | EXP-S-2 | Inconsistency | `evaluate_cli.py:100-101` | ~~RESOLVED (`f9a76062`)~~ Hoisted `compute_mmd_metrics` into the top-level import block in `evaluate_cli.py`; regression test patches the module-level symbol directly |
| M-33 | experiments | EXP-K-2 | Coupling | `embedding_study/execute.py:164` | ~~RESOLVED (`9a3fd61a`)~~ Fixed together with I-19: `execute.py` now imports `generate_sbm_batch` directly from `tmgg.data.datasets.sbm` |
| M-34 | experiments | EXP-D-1 | Dead-Code | `discrete_diffusion_generative/datamodule.py:327-344` | ~~RESOLVED (`53dcc3bc`)~~ Deleted `MolecularDataModule` placeholder |
| M-35 | experiments | EXP-C-3 | Inconsistency | all denoising runners | `main() -> None` discards `dict` return from `run_experiment()`; breaks Hydra sweep result collection |
| M-36 | experiments | EXP-C-4 | Inconsistency | `spectral_arch_denoising/runner.py:3-7`, `gaussian_diffusion_generative/runner.py:3-7` | ~~RESOLVED (`f9a76062`)~~ Runner docstrings now point at the canonical CLI entry points `uv run tmgg-spectral-arch` and `uv run tmgg-gaussian-gen` |
| M-37 | experiments | EXP-T-1 | Type-Safety | `embedding_study/execute.py:140` | ~~RESOLVED (`f9a76062`)~~ `_generate_graphs()` now returns `list[torch.Tensor]`, and the embedding-study helper test locks the adjacency-tensor output contract |
| M-38 | experiments | EXP-T-3 | Type-Safety | `noised_collector.py`, `analyzer.py` | ~~RESOLVED (`007fe4ef`)~~ Noised comparison methods now return typed dataclasses, with JSON serialization handled at the result-object boundary |
| M-39 | experiments | EXP-CD-4 | Config-Drift | `digress_sbm_small_vanilla.yaml` | ~~RESOLVED (`53dcc3bc`)~~ Fixed `dy: 128` → `32` in `digress_sbm_small.yaml` and `_highlr.yaml` |
| M-40 | experiments | EXP-CD-5 | Config-Drift | `base_config_embedding_study.yaml` | ~~RESOLVED (`f9a76062`)~~ Added `_cli_cmd: tmgg-embedding-study-exp`; Modal config-resolution coverage now keeps the entry point discoverable |
| M-41 | experiments | EXP-CS-1 | Config-Structure | `grid_search_base.yaml:49-61` | ~~RESOLVED (`ce5ac068`)~~ Grid search now selects `grid_search` through `base/callbacks` defaults; no inline `trainer.callbacks` remain |
| M-42 | experiments | EXP-CS-2 | Config-Structure | `base_config_discrete_diffusion_generative.yaml:69-81` | ~~RESOLVED (`ce5ac068`)~~ Discrete diffusion now selects `discrete_wandb` through `base/logger` defaults instead of defining an inline logger block |
| M-43 | exp_configs | IE-4 | Interpolation-Error | `base_config_discrete_diffusion_generative.yaml:71` | ~~RESOLVED (`ce5ac068`)~~ TensorBoard logger configs were deleted; the discrete base config no longer selects a TensorBoard S3 logger at all |
| M-44 | exp_configs | SD-4 | Consistency | `data/nx.yaml`, `data/nx_square.yaml`, `data/nx_star.yaml` | ~~RESOLVED (`53dcc3bc`)~~ Normalized to `"digress"`, removed `.lower()` fallback in `noise.py` |
| M-45 | exp_configs | SD-5 | Consistency | 6 data configs | ~~RESOLVED (`53dcc3bc`)~~ Normalized all `_target_` paths to canonical `tmgg.data.GraphDataModule` |
| M-46 | exp_configs | RED-3/cfg | Redundancy | `data/sbm_n100.yaml`, `data/sbm_n200.yaml` | ~~RESOLVED (`007fe4ef`)~~ `sbm_n100` and `sbm_n200` now inherit `sbm_default` and override only the size-specific fields |
| M-47 | exp_configs | RED-4/cfg | Redundancy | `stage/stage3_diversity.yaml`, `stage4_benchmarks.yaml`, `stage5_full.yaml` | ~~RESOLVED (`007fe4ef`)~~ Stage 3-5 now inherit one shared runtime config for optimizer, scheduler, and default noise levels |
| M-48 | exp_configs | DC-1 | Dead-Config | `base/logger/grid_wandb.yaml` | ~~RESOLVED (`ce5ac068`)~~ `grid_wandb.yaml` is now live and selected by `grid_search_base.yaml` via Hydra defaults |
| M-49 | exp_configs | DC-2 | Dead-Config | `base/logger/tensorboard_s3.yaml` | ~~RESOLVED (`ce5ac068`)~~ Deleted the dangerous placeholder TensorBoard S3 logger config |
| M-50 | exp_configs | DC-4 | Dead-Config | `data/er_spectral.yaml`, `lfr_spectral.yaml`, `regular_spectral.yaml`, `tree_spectral.yaml` | ~~RESOLVED (`e1de87a7`)~~ Removed the four deprecated `_spectral` data presets and the last public docs reference to `er_spectral` |
| M-51 | exp_configs | HBP-2 | Hydra-Practice | `base_config_discrete_diffusion_generative.yaml:52-82` | ~~RESOLVED (`ce5ac068`)~~ Discrete diffusion now selects `discrete_nll` and `discrete_wandb` through the shared Hydra config groups |
| M-52 | exp_configs | DOC-1 | Documentation | `data/single_graph_base.yaml:27-28` | ~~RESOLVED (`53dcc3bc`)~~ Removed hardcoded noise config; injected by `task/denoising.yaml` |
| M-53 | data | CFG-3 | Config-Drift | `task/denoising.yaml:21-23`, `data_module.py:35-47` | ~~RESOLVED (`80142202`)~~ Denoising noise settings now stay at top level; `cfg.data` no longer receives `noise_type`/`noise_levels`, and Hydra datamodule instantiation is green again |
| M-54 | models | M/S-4 | Correctness | `tests/test_config_composition.py:215-259`, `digress/transformer_model.py:737` | ~~RESOLVED (`80142202`)~~ DiGress denoising now uses the scalar edge-state contract (`E=1`), and `EigenvectorAugmentation` accepts both scalar and categorical edge encodings |

### Nitpick

| # | Source | Original | Category | File(s) | Description |
|---|--------|----------|----------|---------|-------------|
| N-1 | models | M/I-1 | Inconsistency | `gnn/nvgnn.py:28-37` | Google-style `Args:`/`Returns:` docstring in NumPy-style codebase |
| N-2 | models | M/I-6 | Inconsistency | `factory.py:355-358` | Empty `Baselines` section comment |
| N-3 | models | M/D-2 | Dead-Code | `digress/diffusion_utils.py:22-24` | `sample_gaussian` is a trivial `torch.randn` wrapper |
| N-4 | models | M/T-4 | Type-Safety | `base.py:30` | ~~RESOLVED (`922ed3d8`)~~ `parameter_count()` now returns a recursive `ParameterCountTree`; logger formatting and base-model tests consume the typed tree without changing the runtime shape |
| N-5 | experiments | EXP-D-2 | Dead-Code | `eigenstructure_study/analyzer.py:490-513` | ~~RESOLVED (`0798edd7`)~~ Both CLI and Hydra analyze paths now call `SpectralAnalyzer.save_results()` for `analysis.json`; the duplicated direct writes are gone |
| N-6 | experiments | EXP-C-5 | Inconsistency | `grid_search_runner.py:11-15` | `config_path=str(Path(__file__).parent / "exp_configs")` vs `"../exp_configs"` everywhere else |
| N-7 | experiments | EXP-C-6 | Inconsistency | `stages/runner.py:42` | ~~RESOLVED (`0798edd7`)~~ `stages/runner.py` now imports `run_experiment` at module scope like the other runners |
| N-8 | experiments | EXP-T-2 | Type-Safety | `embedding_study/execute.py:91` | `# type: ignore[assignment]` on `OmegaConf.to_container` without narrowing |
| N-9 | exp_configs | DC-3 | Dead-Config | `base/logger/{multi,csv,wandb}.yaml`, `base/lightning_base.yaml` | Unreferenced from any defaults list; may be CLI-selectable alternatives |
| N-10 | exp_configs | DC-5 | Dead-Config | `grid_search_base.yaml:72` | ~~RESOLVED (`0798edd7`)~~ Removed unread `evaluation.final_eval_noise_levels` from `grid_search_base.yaml` |
| N-11 | exp_configs | DC-6 | Dead-Config | `stage/stage3-5` configs | ~~RESOLVED (`0798edd7`)~~ Removed documentation-only `trigger_conditions`, `ablations`, and `statistical_tests` keys from stage 3-5 configs |
| N-12 | exp_configs | HBP-1 | Hydra-Practice | various data configs | ~~RESOLVED (`0798edd7`)~~ All data configs now declare their Hydra package explicitly (`data` or `_global_` as appropriate) |
| N-13 | exp_configs | DOC-2 | Documentation | `base_config_gaussian_diffusion.yaml:7` | Header comment references `python -m tmgg.experiments...` instead of `tmgg-gaussian-gen` CLI |

---

*Individual reports with full descriptions and line numbers: `data-review.md`, `models-review.md`, `shared-utils-review.md`, `experiments-review.md`, `exp-configs-review.md`.*
