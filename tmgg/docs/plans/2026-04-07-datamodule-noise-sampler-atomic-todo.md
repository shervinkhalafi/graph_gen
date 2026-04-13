# DataModule / NoiseProcess / Sampler Refactor - Atomic Todo List

**Source plan:** [2026-04-06-datamodule-noise-sampler-refactor-impl.md](./2026-04-06-datamodule-noise-sampler-refactor-impl.md)  
**Branch state:** post-chunk-1 accessor split, before chunk 2  
**Scope:** break chunks 2 through 8 into small tasks that can sit on a todo list

This file turns the remaining implementation chunks into single-purpose tasks. Each task should be small enough to land in one focused patch. Tasks are ordered. Unless a task says otherwise, treat earlier tasks in the same chunk as prerequisites.

> Execution rule: if a task reaches a point that requires decision-making not already covered here, or if the remaining work needs replanning, stop and ask the user before proceeding.

## Working Rules

- Finish one task completely before starting the next task in the same chunk.
- Keep each task narrow. Do not combine interface rewrites with config or doc work.
- Run the task-local verification before marking a task done.
- If a task deletes public surface, run `rg` first and again after the deletion.

## Chunk 2: Datamodule contract and graph generation

- [ ] **2.1 Base contract:** update [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py) so `graph_type` and `num_nodes` live on the base class, `get_size_distribution()` is concrete, and `get_dataset_info()` is no longer abstract.
  Done when: base class returns `SizeDistribution.fixed(self.num_nodes)` and no abstract metadata hook remains.
  Completion note, 2026-04-07: added `graph_type` and `num_nodes` annotations to `BaseGraphDataModule`, added a concrete base `get_size_distribution(split=None)` that returns `SizeDistribution.fixed(self.num_nodes)`, and removed `get_dataset_info()` from the base abstract contract. Verified with `uv run pytest tests/experiments/test_categorical_datamodule.py::TestSizeDistribution -x -v` (`9 passed`).

- [ ] **2.2 Shared graph-generation entrypoint:** create [src/tmgg/data/data_modules/graph_generation.py](../../src/tmgg/data/data_modules/graph_generation.py) and move the production graph-type dispatch there.
  Files: [src/tmgg/data/data_modules/multigraph_data_module.py](../../src/tmgg/data/data_modules/multigraph_data_module.py), [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py), [src/tmgg/data/data_modules/data_module.py](../../src/tmgg/data/data_modules/data_module.py)
  Done when: those modules no longer keep separate graph-type switch trees.
  Completion note, 2026-04-07: added shared dispatch helpers in `graph_generation.py` for synthetic, SBM, PyG single-graph, and PyG dataset-split loading; rewired `MultiGraphDataModule`, `SingleGraphDataModule`, and `GraphDataModule` to delegate there instead of keeping separate graph-type switch trees. Also removed the stale `@override` annotations on datamodule `get_dataset_info()` methods that became invalid once task `2.1` removed the base abstract hook. Verified with `uv run pytest tests/test_single_graph_datasets.py tests/experiment_utils/test_data_module.py tests/data_modules/test_pyg_conversion.py tests/experiments/test_categorical_datamodule.py -x -v` (`77 passed`) and `uv run ruff check ... && uv run ruff format --check ...` on the touched datamodule files.

- [ ] **2.3 Multigraph migration:** switch [src/tmgg/data/data_modules/multigraph_data_module.py](../../src/tmgg/data/data_modules/multigraph_data_module.py) to the shared generator without changing dataset semantics.
  Done when: train/val/test splits still produce the same shape and graph invariants.
  Completion note, 2026-04-07: moved multigraph split generation, including SBM `fixed` and `enumerated` partition handling, into shared helpers in `graph_generation.py`; `MultiGraphDataModule` now delegates `_generate_and_split()` to that shared entrypoint and keeps only dataloader/setup responsibilities. Verified with `uv run pytest tests/experiment_utils/test_data_module.py tests/experiments/test_categorical_datamodule.py -x -v` (`44 passed`) and `uv run ruff check ... && uv run ruff format --check ...` on the touched multigraph files.

- [ ] **2.4 Single-graph migration:** switch [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py) to the shared generator without keeping a private dispatch path.
  Done when: same-graph and different-graph split behavior still work.
  Completion note, 2026-04-07: no additional production edit was needed beyond the `2.2` shared-dispatch patch, because `SingleGraphDataModule` already delegated graph generation to `graph_generation.generate_single_graph()` and no longer kept a private PyG or synthetic dispatch path. Re-verified the acceptance criteria with `uv run pytest tests/test_single_graph_datasets.py -x -v` (`24 passed`), covering both `same_graph_all_splits=True` and `same_graph_all_splits=False` behavior.

- [ ] **2.5 Noise-field removal:** remove datamodule-owned `noise_level`, `noise_levels`, and `noise_type` constructor fields from [src/tmgg/data/data_modules/data_module.py](../../src/tmgg/data/data_modules/data_module.py) and [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py).
  Done when: constructors fail loudly on the old arguments and no production caller still passes them.
  Completion note, 2026-04-07: removed `noise_levels` and `noise_type` from the `GraphDataModule` and `SingleGraphDataModule` constructor signatures and deleted the stale compatibility docstrings. Updated Python callers and regression tests so old kwargs now raise `TypeError` immediately instead of being accepted and discarded. Verified with `uv run pytest tests/experiment_utils/test_data_module.py tests/test_single_graph_datasets.py tests/test_data_split_reproducibility.py tests/test_training_integration.py::TestDataModuleIntegration::test_datamodule_rejects_legacy_noise_params -x -v` (`47 passed`), direct fixture instantiation for the updated experiment test files (`GraphDataModule GraphDataModule GraphDataModule`), and `uv run ruff check ... && uv run ruff format --check ...` on the touched files. By explicit user decision, stale YAML/config callers remain deferred to chunk `6.4` rather than being migrated in this step.

- [ ] **2.6 Synthetic categorical cleanup:** strip marginal-specific responsibilities from [src/tmgg/data/data_modules/synthetic_categorical.py](../../src/tmgg/data/data_modules/synthetic_categorical.py).
  Done when: the class only exposes batch generation plus size-distribution behavior that still belongs on the datamodule.
  Completion note, 2026-04-07: removed `_compute_marginals()`, `_node_marginals`, `_edge_marginals`, `node_marginals`, and `edge_marginals` from [src/tmgg/data/data_modules/synthetic_categorical.py](../../src/tmgg/data/data_modules/synthetic_categorical.py); `setup()` now delegates entirely to the parent graph-generation path and the datamodule only owns categorical batch generation plus size-distribution behavior. Rewrote the remaining datamodule tests in [tests/test_datamodule_contracts.py](../../tests/test_datamodule_contracts.py) and [tests/experiments/test_categorical_datamodule.py](../../tests/experiments/test_categorical_datamodule.py) to assert train-batch one-hot contracts, reference graphs, and reproducibility instead of cached marginal vectors. Verified with `rg -n "node_marginals|edge_marginals|_compute_marginals" src/tmgg tests -g '*.py'` (no matches) and `uv run pytest tests/test_datamodule_contracts.py tests/experiments/test_categorical_datamodule.py -x -v` (`45 passed`).

- [ ] **2.7 Dead datamodule API removal:** remove `get_dataset_info()`, `get_sample_adjacency_matrix()`, `get_sample_graph()`, `get_train_graph()`, `get_val_graph()`, `get_test_graph()`, and `sample_n_nodes()` from production datamodules.
  Files: [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py), [src/tmgg/data/data_modules/multigraph_data_module.py](../../src/tmgg/data/data_modules/multigraph_data_module.py), [src/tmgg/data/data_modules/data_module.py](../../src/tmgg/data/data_modules/data_module.py), [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py), [src/tmgg/data/__init__.py](../../src/tmgg/data/__init__.py), [src/tmgg/data/data_modules/__init__.py](../../src/tmgg/data/data_modules/__init__.py)
  Done when: the production `rg` sweep for those names returns no matches.
  Completion note, 2026-04-07: removed the dead datamodule accessor surface from `GraphDataModule`, `MultiGraphDataModule`, `SingleGraphDataModule`, and `SyntheticCategoricalDataModule`, including the now-unused dense-adjacency reconstruction helpers and sample-selection RNG. The package `__init__` files needed no edit because they only re-export datamodule classes, not those deleted methods. Verified with `rg -n "\\bget_dataset_info\\(|\\bget_sample_adjacency_matrix\\(|\\bget_sample_graph\\(|\\bget_train_graph\\(|\\bget_val_graph\\(|\\bget_test_graph\\(|\\bsample_n_nodes\\(" src/tmgg -g '*.py'` (no matches) and `uv run ruff check ... && uv run ruff format --check ...` on the touched production files. By task ordering, the datamodule contract tests that still asserted the removed surface were left for `2.8`.

- [ ] **2.8 Datamodule tests:** rewrite [tests/test_datamodule_contracts.py](../../tests/test_datamodule_contracts.py), [tests/test_single_graph_datasets.py](../../tests/test_single_graph_datasets.py), [tests/experiments/test_categorical_datamodule.py](../../tests/experiments/test_categorical_datamodule.py), and [tests/experiment_utils/test_data_module.py](../../tests/experiment_utils/test_data_module.py) around dataloaders, `get_reference_graphs()`, and `get_size_distribution()`.
  Verification:
  `uv run pytest tests/test_datamodule_contracts.py tests/test_single_graph_datasets.py tests/experiments/test_categorical_datamodule.py tests/experiment_utils/test_data_module.py -x -v`
  Completion note, 2026-04-07: rewrote the remaining datamodule-contract tests to stop calling deleted helpers and instead read train graphs from dataloaders, val/test graphs from `get_reference_graphs()`, and fixed-size sampling expectations from `get_size_distribution().sample(...)`. This touched the four planned files only and also updated the categorical size-distribution rationale text so it no longer mentions `sample_n_nodes()`. Verified with `rg -n "\\bget_dataset_info\\(|\\bget_sample_adjacency_matrix\\(|\\bget_sample_graph\\(|\\bget_train_graph\\(|\\bget_val_graph\\(|\\bget_test_graph\\(|\\bsample_n_nodes\\(" tests/test_datamodule_contracts.py tests/test_single_graph_datasets.py tests/experiments/test_categorical_datamodule.py tests/experiment_utils/test_data_module.py` (no matches) and the targeted pytest sweep for those four files, which ran through all 92 collected tests without any failure output before pytest exited in this environment.

## Chunk 3: Functional `NoiseProcess`

- [ ] **3.1 Interface replacement:** replace the abstract surface in [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) with the new `NoiseProcess` and `ExactDensityNoiseProcess` contracts.
  Done when: old public methods like `apply()` and `get_posterior()` are gone from the abstract API.
  Completion note, 2026-04-07: replaced the abstract `NoiseProcess` surface with `timesteps`, `initialize_from_data()`, `sample_prior()`, `forward_sample()`, and `posterior_sample()`, and added the public `ExactDensityNoiseProcess` contract to [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py). By explicit user decision, this step also pulled categorical prior/posterior sampling logic forward out of the current sampler so the new contract is real rather than shimmed: [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py) now consumes process-owned prior/posterior methods, and [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py) uses `forward_sample()` directly. The still-live one-step denoising path in [src/tmgg/training/lightning_modules/denoising_module.py](../../src/tmgg/training/lightning_modules/denoising_module.py) keeps using the concrete continuous `apply(noise_level=...)` helper until chunk `5` removes that hierarchy; the old names are no longer part of the abstract API. Verified with `uv run ruff check ...` on the touched diffusion/module files, `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py -x -v` (`29 passed`), and `uv run pytest tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py tests/experiments/test_discrete_diffusion_module.py -x -v` (`65 passed`).

- [ ] **3.2 Continuous constructor cleanup:** rename continuous-process constructor arguments to `definition` and `schedule`, remove the `generator` alias, and make `timesteps` come from the owned schedule.
  Done when: no production constructor passes `generator=`.
  Completion note, 2026-04-07: renamed `ContinuousNoiseProcess.__init__` in [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) to `definition` / `schedule`, removed the deprecated `.generator` alias entirely, and kept `timesteps` sourced from the owned schedule module. Updated every explicit `ContinuousNoiseProcess(...)` keyword call site in the fast diffusion tests plus the slower integration/full-flow test files, and rewired the per-element noise instrumentation in [tests/test_generative_integration.py](../../tests/test_generative_integration.py) to patch `.definition.add_noise` directly. Verified with `rg -n "ContinuousNoiseProcess\\(.*generator=|\\.generator\\b" src/tmgg tests -g '*.py'` (no process-surface matches), `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py -x -v` (`75 passed`), and `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py tests/test_generative_integration.py -k 'invalid_loss_type_raises or per_element_noise_via_noise_process' -x -v` (`2 passed`). Per user instruction, the slow full-flow suite was updated for API consistency but not executed here.

- [ ] **3.3 Continuous forward sampling:** rewrite continuous forward diffusion to operate only through `to_edge_state()` and `from_edge_state()`.
  Files: [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py), [src/tmgg/diffusion/diffusion_sampling.py](../../src/tmgg/diffusion/diffusion_sampling.py)
  Done when: no binary projection remains in the continuous forward path.
  Completion note, 2026-04-07: simplified the continuous-process state lift in [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) so `_continuous_edge_state()` now delegates only to `to_edge_state()`, with no `to_binary_adjacency()` fallback left in the continuous forward or reverse path. To make that contract real, updated [src/tmgg/data/datasets/graph_types.py](../../src/tmgg/data/datasets/graph_types.py) so binary-topology graphs expose their explicit edge-indicator channel through `to_edge_state()` while edge-state graphs still round-trip losslessly; no change was needed in `diffusion_sampling.py` after the sweep because the binary projection lived in the process helper, not in the sampling utilities. Added a regression test in [tests/experiment_utils/test_conversions.py](../../tests/experiment_utils/test_conversions.py) covering binary-topology lifting into edge-state space. Verified with `rg -n "to_binary_adjacency\\(" src/tmgg/diffusion/noise_process.py` (no matches) and `uv run pytest tests/experiment_utils/test_conversions.py tests/diffusion/test_noise_process.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py -x -v` (`94 passed`).

- [ ] **3.4 Continuous posterior sampling:** rewrite `ContinuousNoiseProcess.posterior_sample()` to return sampled `GraphData` at timestep `s`, not Gaussian parameter dicts.
  Done when: reverse sampling callers consume only `GraphData`.
  Completion note, 2026-04-07: no additional production patch was needed here because the behavioral rewrite already landed with task `3.1`: [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) returns `GraphData` from `ContinuousNoiseProcess.posterior_sample()`, and the reverse-diffusion callers in [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py) consume that `GraphData` directly. The remaining Gaussian-parameter dict surface is the private `_posterior_parameters()` helper, which is still used only for metric collection and schedule-sensitive regression checks, not for reverse sampling. Re-verified with `rg -n -C 1 "posterior_sample\\(" src/tmgg/diffusion/sampler.py tests/diffusion/test_noise_process.py tests/diffusion/test_sampler.py -g '*.py'` and `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_sampler.py -k 'posterior_sample or ContinuousSamplerSample' -x -v` (`8 passed`).

- [ ] **3.5 Continuous exact-density helpers:** move any continuous VLB or NLL math onto the exact-density interface or private helpers behind it.
  Done when: no old method names such as `kl_prior`, `compute_Lt`, or `reconstruction_logp` remain public on the process.
  Completion note, 2026-04-07: by explicit user decision, pulled the remaining categorical VLB cleanup forward into this task because the old public helper names were the only live exact-density surface still in use. [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) now makes `CategoricalNoiseProcess` implement `ExactDensityNoiseProcess` directly through `forward_log_prob()`, `posterior_log_prob()`, and `prior_log_prob()`, and deletes the public `kl_prior()`, `compute_Lt()`, and `reconstruction_logp()` methods. [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py) now composes VLB estimates from exact-density log-probability differences and keeps reconstruction log-probability in the training module via a private helper, as required by the spec. No additional continuous-process VLB surface needed migration here because the current continuous training path does not expose public VLB/NLL helpers. Updated the categorical exact-density and module-VLB tests in [tests/diffusion/test_noise_process.py](../../tests/diffusion/test_noise_process.py), [tests/diffusion/test_noise_process_vlb.py](../../tests/diffusion/test_noise_process_vlb.py), [tests/experiment_utils/test_diffusion_module.py](../../tests/experiment_utils/test_diffusion_module.py), and [tests/experiments/test_discrete_diffusion_module.py](../../tests/experiments/test_discrete_diffusion_module.py). Verified with `rg -n "kl_prior\\(|compute_Lt\\(|reconstruction_logp\\(" src/tmgg tests -g '*.py'` (no matches) and `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/experiment_utils/test_diffusion_module.py tests/experiments/test_discrete_diffusion_module.py -x -v` (`75 passed`).

- [ ] **3.6 Categorical constructor rewrite:** replace transition-model injection with direct categorical process configuration: `schedule`, `x_classes`, `e_classes`, and `limit_distribution`.
  Done when: no public transition-model argument survives.
  Completion note, 2026-04-07: rewrote [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) so `CategoricalNoiseProcess` now takes `schedule`, `x_classes`, `e_classes`, and `limit_distribution` instead of a public `transition_model` argument. `limit_distribution="uniform"` now builds the uniform transition internally at construction time, while `limit_distribution="empirical_marginal"` leaves the internal transition unset for the existing deferred setup path. The temporary `set_transition_model()` bridge remains in place for tasks `3.7` and `3.8`, but the public constructor surface no longer exposes transition-model injection. Updated all categorical constructor call sites across the process, sampler, diffusion-module, discrete-runner, and slow full-flow experiment tests to use the new surface. Verified with `rg -n "transition_model=|CategoricalNoiseProcess\\(.*noise_schedule=" src/tmgg tests -g '*.py'` (no matches) and `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py tests/experiments/test_discrete_diffusion_module.py tests/experiments/test_discrete_diffusion_runner.py -x -v` (`110 passed`). Per user instruction, [tests/experiments/test_all_experiments_full_flow.py](../../tests/experiments/test_all_experiments_full_flow.py) was updated for API consistency but not executed.

- [ ] **3.7 Empirical-marginal initialization:** implement `initialize_from_data(train_loader)` for categorical marginal estimation.
  Rules: count only real nodes, count only valid real-node edges, use the strict upper triangle for undirected graphs, fall back to uniform PMFs on zero-count domains.
  Completion note, 2026-04-07: implemented [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py)::`CategoricalNoiseProcess.initialize_from_data()` to estimate empirical node and edge marginals from the training loader. The implementation counts node classes only at real positions, counts edge classes only on valid real-node pairs in the strict upper triangle, and falls back to uniform PMFs on zero-count domains before constructing the internal `MarginalUniformTransition`. [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py) now initialises empirical categorical processes from `datamodule.train_dataloader()` instead of reading `node_marginals` / `edge_marginals` directly. Added regression coverage for strict-upper-triangle counting and zero-edge fallback in [tests/diffusion/test_noise_process.py](../../tests/diffusion/test_noise_process.py), and updated the setup wiring tests in [tests/experiment_utils/test_diffusion_module.py](../../tests/experiment_utils/test_diffusion_module.py) plus the discrete-module slice to use loader-driven setup. Verified with `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py tests/experiments/test_discrete_diffusion_module.py tests/experiments/test_discrete_diffusion_runner.py -x -v` (`112 passed`). Remaining `node_marginals` / `edge_marginals` references under datamodule tests and `synthetic_categorical.py` are later chunk work tied to task `2.6`.

- [ ] **3.8 Transition-surface deletion:** delete [src/tmgg/diffusion/categorical_noise.py](../../src/tmgg/diffusion/categorical_noise.py), [src/tmgg/diffusion/protocols.py](../../src/tmgg/diffusion/protocols.py), [src/tmgg/diffusion/transitions.py](../../src/tmgg/diffusion/transitions.py), and [src/tmgg/diffusion/diffusion_graph_types.py](../../src/tmgg/diffusion/diffusion_graph_types.py), then update [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py).
  Done when: the transition-related `rg` sweep returns no production matches.
  Completion note, 2026-04-07: deleted the four public transition-surface files and rewrote [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py) to own categorical stationary PMFs and reverse-process kernel construction directly. [src/tmgg/diffusion/diffusion_sampling.py](../../src/tmgg/diffusion/diffusion_sampling.py) now samples categorical priors from raw PMFs instead of deleted wrapper types, [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py) no longer re-exports the removed surface, and the categorical setup checks in [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py) plus the directly affected diffusion/module tests now assert loader-initialised stationary PMFs instead of transition objects. Verified with `rg -n "transition_model|set_transition_model|get_posterior\\(|CategoricalNoiseDefinition|TransitionModel|LimitDistribution|TransitionMatrices|DiscreteUniformTransition|MarginalUniformTransition" src/tmgg -g '*.py'` (no matches) and `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py tests/experiments/test_discrete_diffusion_module.py tests/experiments/test_discrete_diffusion_runner.py -x -v` (`109 passed`).

- [ ] **3.9 Process tests:** rewrite [tests/diffusion/test_noise_process.py](../../tests/diffusion/test_noise_process.py), [tests/diffusion/test_noise_process_vlb.py](../../tests/diffusion/test_noise_process_vlb.py), [tests/models/test_noise_schedule.py](../../tests/models/test_noise_schedule.py), and [tests/models/test_graph_types.py](../../tests/models/test_graph_types.py) around the new process contract.
  Verification:
  `uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/models/test_noise_schedule.py tests/models/test_graph_types.py -x -v`
  Completion note, 2026-04-07: rewrote [tests/diffusion/test_noise_process.py](../../tests/diffusion/test_noise_process.py) to assert stationary-PMF initialization and forward-sampling behavior instead of deleted transition objects, and replaced the removed transition/dataclass coverage in [tests/models/test_noise_schedule.py](../../tests/models/test_noise_schedule.py) and [tests/models/test_graph_types.py](../../tests/models/test_graph_types.py) with schedule-plus-process tests and explicit graph-boundary accessor tests. [tests/diffusion/test_noise_process_vlb.py](../../tests/diffusion/test_noise_process_vlb.py) already matched the exact-density process contract and needed no code change. Also tightened the setup expectations in [tests/experiment_utils/test_diffusion_module.py](../../tests/experiment_utils/test_diffusion_module.py) so repo-wide type checking no longer relies on deleted imports or optional PMF access. Verified with `uv run pytest tests/models/test_noise_schedule.py tests/models/test_graph_types.py tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/diffusion/test_sampler.py tests/experiment_utils/test_diffusion_module.py tests/experiments/test_discrete_diffusion_module.py tests/experiments/test_discrete_diffusion_runner.py -x -v` (`130 passed`).

## Chunk 4: Unified sampler and `DiffusionState`

- [ ] **4.1 Add `DiffusionState`:** create the public dataclass in [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py) with strict validation for timestep, mask shape, and graph state.
  Completion note, 2026-04-07: added the public frozen `DiffusionState` dataclass to [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py) and exported it from [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py). The validator now fails loudly on non-`GraphData` inputs, non-integer `t`/`max_t`, invalid timestep ranges, unbatched graph states, and any mismatch between `node_mask` and the graph tensor extents. Added direct validation coverage in [tests/diffusion/test_sampler.py](../../tests/diffusion/test_sampler.py) for valid construction, bad timestep ranges, unbatched warm-start graphs, and mismatched mask shapes. Verified with `uv run pytest tests/diffusion/test_sampler.py -x -v` (`24 passed`).

- [ ] **4.2 Replace sampler subclasses:** delete the sampler ABC plus `CategoricalSampler` and `ContinuousSampler`, then build one concrete `Sampler` that depends only on `NoiseProcess`.
  Done when: no subtype-specific sampler classes remain in production code.

- [ ] **4.3 Warm-start rewrite:** remove `start_timestep` and replace it with `start_from: DiffusionState`.
  Done when: no production or test caller still uses `start_timestep`.

- [ ] **4.4 Reverse-loop implementation:** implement the exact reverse loop in [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py): prior or warm start, model call, `posterior_sample`, new `DiffusionState`.
  Done when: the sampler never reads transition matrices, Gaussian posterior dicts, or schedule buffers directly.

- [ ] **4.5 Final graph boundary:** keep continuous intermediate states as `GraphData` until the explicit final extraction boundary, then use binary topology only for returned graphs.

- [ ] **4.6 Sampler tests:** rewrite [tests/diffusion/test_sampler.py](../../tests/diffusion/test_sampler.py) and the sampler-facing parts of [tests/test_generative_integration.py](../../tests/test_generative_integration.py) to cover `DiffusionState`, warm starts, and `T=1`.
  Verification:
  `uv run pytest tests/diffusion/test_sampler.py tests/test_generative_integration.py -x -v`

## Chunk 5: `DiffusionModule` rewrite and `T=1` denoising

- [ ] **5.1 Constructor contraction:** remove `sampler` and `noise_schedule` from [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py), and instantiate `Sampler(self.noise_process)` internally.
  Done when: `self.T` derives from `self.noise_process.timesteps`.

- [ ] **5.2 Setup contract:** make `setup()` always initialize the process from `dm.train_dataloader()` and fetch `dm.get_size_distribution("train")`.
  Done when: no datamodule duck typing or marginal extraction remains.

- [ ] **5.3 Training-step rewrite:** switch `training_step()` to `noise_process.forward_sample()` and make exact-density calls fail loudly unless the process implements the required interface.

- [ ] **5.4 T=1 process support:** extend the concrete process constructors so `timesteps == 1` represents denoising without a parallel LightningModule hierarchy.
  Done when: one-step denoising uses `DiffusionModule` plus a `T=1` process.

- [ ] **5.5 Delete `SingleStepDenoisingModule`:** remove [src/tmgg/training/lightning_modules/denoising_module.py](../../src/tmgg/training/lightning_modules/denoising_module.py) and update [src/tmgg/training/lightning_modules/__init__.py](../../src/tmgg/training/lightning_modules/__init__.py), [src/tmgg/training/__init__.py](../../src/tmgg/training/__init__.py), and experiment package re-exports.

- [ ] **5.6 Training-module tests:** replace denoising-module tests with `T=1` diffusion tests in [tests/experiment_utils/test_diffusion_module.py](../../tests/experiment_utils/test_diffusion_module.py), [tests/experiment_utils/test_denoising_module.py](../../tests/experiment_utils/test_denoising_module.py), [tests/test_training_integration.py](../../tests/test_training_integration.py), [tests/experiments/test_baseline_denoising_module.py](../../tests/experiments/test_baseline_denoising_module.py), [tests/experiments/test_digress_denoising_module.py](../../tests/experiments/test_digress_denoising_module.py), and [tests/experiments/test_all_experiments_full_flow.py](../../tests/experiments/test_all_experiments_full_flow.py).
  Verification:
  `uv run pytest tests/experiment_utils/test_diffusion_module.py tests/experiment_utils/test_denoising_module.py tests/test_training_integration.py tests/experiments/test_baseline_denoising_module.py tests/experiments/test_digress_denoising_module.py tests/experiments/test_all_experiments_full_flow.py -x -v`

## Chunk 6: Hydra config and experiment packaging migration

- [ ] **6.1 Discrete generative configs:** remove top-level `sampler` and `noise_schedule` from the discrete generative config set, nest the schedule under `model.noise_process`, and replace `transition_type: marginal` with `limit_distribution: empirical_marginal`.

- [ ] **6.2 Gaussian generative configs:** remove top-level `sampler` and `noise_schedule`, rename `generator` to `definition`, and nest the schedule under `model.noise_process`.

- [ ] **6.3 Denoising task configs:** move denoising runs to `DiffusionModule`, instantiate a `T=1` process, and replace plural noise-level fields with singular `noise_level`.

- [ ] **6.4 Data-config cleanup:** remove noise fields from datamodule-facing config files so the data layer no longer owns denoising parameters.

- [ ] **6.5 Sweep migration:** rewrite stage or multirun configs that currently encode several denoising levels inside one config object.
  Done when: each run still receives exactly one `noise_level`.

- [ ] **6.6 Scaffold update:** update [src/tmgg/experiments/_scaffold/model_config.yaml.j2](../../src/tmgg/experiments/_scaffold/model_config.yaml.j2) so newly scaffolded configs use the new process shape by default.

- [ ] **6.7 Config tests:** update [tests/experiments/test_discrete_diffusion_runner.py](../../tests/experiments/test_discrete_diffusion_runner.py) to assert the new config shape and the deletion of `SingleStepDenoisingModule`.
  Verification:
  `uv run pytest tests/experiments/test_discrete_diffusion_runner.py -x -v`

## Chunk 7: Contract-focused test sweep

- [ ] **7.1 Remove dead surface assertions:** delete any remaining test assertions about datamodule marginals, transition models, `get_posterior()`, `start_timestep`, `SingleStepDenoisingModule`, and ambiguous adjacency helpers.

- [ ] **7.2 Add process-behavior tests:** add or rewrite tests for `initialize_from_data()`, `sample_prior()`, `forward_sample()`, `posterior_sample()`, and `DiffusionState` warm starts.

- [ ] **7.3 Add datamodule-behavior tests:** add direct tests for fixed-size default `get_size_distribution()` and the cleaned datamodule public contract.

- [ ] **7.4 Add T=1 denoising tests:** make one-step denoising behavior explicit as `DiffusionModule + T=1 process`.

- [ ] **7.5 Broad test sweep:** run the targeted suites for `tests/diffusion`, `tests/experiment_utils`, `tests/experiments`, [tests/test_datamodule_contracts.py](../../tests/test_datamodule_contracts.py), [tests/test_single_graph_datasets.py](../../tests/test_single_graph_datasets.py), [tests/test_generative_integration.py](../../tests/test_generative_integration.py), and [tests/test_training_integration.py](../../tests/test_training_integration.py).
  Verification:
  `uv run pytest tests/diffusion tests/experiment_utils tests/experiments tests/test_datamodule_contracts.py tests/test_single_graph_datasets.py tests/test_generative_integration.py tests/test_training_integration.py -x -v`

## Chunk 8: Docs, exports, and final sweeps

- [ ] **8.1 Public docs update:** rewrite [docs/extending.md](../extending.md), [docs/configuration.md](../configuration.md), [docs/experiments.md](../experiments.md), [docs/data.md](../data.md), and [docs/architecture.md](../architecture.md) so examples use one `NoiseProcess`, nested schedules, `DiffusionModule` for both multi-step and `T=1` denoising, and explicit binary versus edge-state helpers.

- [ ] **8.2 Export cleanup:** remove deleted exports from [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py), [src/tmgg/training/__init__.py](../../src/tmgg/training/__init__.py), and [src/tmgg/training/lightning_modules/__init__.py](../../src/tmgg/training/lightning_modules/__init__.py).

- [ ] **8.3 Experiment package cleanup:** update experiment package `__init__` files so they no longer re-export `SingleStepDenoisingModule`.

- [ ] **8.4 Final `rg` sweeps:** run the removal sweeps from the source plan and require zero production matches before calling the branch done.

- [ ] **8.5 Final branch validation:** run the broad test pass for the branch after the `rg` sweeps are clean.

## Suggested Todo Ordering

1. Do chunk 2 completely before touching chunk 3.
2. Do chunk 3 before chunk 4. The unified sampler depends on the new process contract.
3. Do chunk 4 before chunk 5. The training module should consume the final sampler shape.
4. Do chunk 5 before chunk 6. Config rewrites should target the final training interface.
5. Do chunk 6 before chunk 7. Test rewrites are easier once the config and class surfaces stop moving.
6. Leave chunk 8 for the end. The docs and export sweeps should describe the final code, not a moving target.
