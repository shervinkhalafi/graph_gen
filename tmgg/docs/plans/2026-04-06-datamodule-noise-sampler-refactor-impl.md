# DataModule / NoiseProcess / Sampler Refactor — Implementation Plan

**Primary spec:** [2026-04-06-datamodule-noise-sampler-refactor-spec.md](./2026-04-06-datamodule-noise-sampler-refactor-spec.md)  
**Rationale review:** [2026-04-05-datamodule-noise-interface-review.md](../reports/2026-04-05-datamodule-noise-interface-review.md)
**Atomic todo breakdown:** [2026-04-07-datamodule-noise-sampler-atomic-todo.md](./2026-04-07-datamodule-noise-sampler-atomic-todo.md)

This document turns the approved spec into an implementation sequence against the current code base. Follow the chunks in order. Each chunk assumes all earlier chunks are complete and green. Do not add compatibility shims, deprecated aliases, `hasattr`/`getattr` fallbacks, or dual-path APIs. When the spec and the current code disagree, the spec wins.

## Working Rules

1. Before changing implementation code, create an annotated rollback tag on the current branch state.
2. End every chunk with passing targeted tests and a commit.
3. Re-run `rg` before deleting any API that the spec marks dead.
4. Crash early on unsupported combinations. Do not add graceful fallback paths.

## Preflight

Run these first and save the outputs in the implementation notes for the branch:

```bash
git status --short
git tag -a datamodule-noise-sampler-refactor-pre -m "pre refactor rollback point"
rg -n "transition_model|set_transition_model|get_posterior\\(|node_marginals|edge_marginals" src tests
rg -n "get_dataset_info\\(|get_sample_adjacency_matrix\\(|get_sample_graph\\(|get_train_graph\\(|get_val_graph\\(|get_test_graph\\(|sample_n_nodes\\(" src tests
rg -n "from_adjacency\\(|to_adjacency\\(|edge_features_to_adjacency\\(" src tests
rg -n "start_timestep" src tests
rg -n "_target_: .*SingleStepDenoisingModule" src/tmgg/experiments/exp_configs
rg -n "noise_schedule:|sampler:" src/tmgg/experiments/exp_configs
```

Suggested commit after the preflight tag is not necessary. The tag is the rollback point.

## Migration Surface

### Core diffusion files

- [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py)
- [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py)
- [src/tmgg/diffusion/schedule.py](../../src/tmgg/diffusion/schedule.py)
- [src/tmgg/diffusion/diffusion_sampling.py](../../src/tmgg/diffusion/diffusion_sampling.py)
- [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py)

### Diffusion files slated for deletion

- [src/tmgg/diffusion/categorical_noise.py](../../src/tmgg/diffusion/categorical_noise.py)
- [src/tmgg/diffusion/protocols.py](../../src/tmgg/diffusion/protocols.py)
- [src/tmgg/diffusion/transitions.py](../../src/tmgg/diffusion/transitions.py)
- [src/tmgg/diffusion/diffusion_graph_types.py](../../src/tmgg/diffusion/diffusion_graph_types.py)
- [src/tmgg/training/lightning_modules/denoising_module.py](../../src/tmgg/training/lightning_modules/denoising_module.py)

### Datamodule files

- [src/tmgg/data/datasets/graph_types.py](../../src/tmgg/data/datasets/graph_types.py)
- [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py)
- [src/tmgg/data/data_modules/multigraph_data_module.py](../../src/tmgg/data/data_modules/multigraph_data_module.py)
- [src/tmgg/data/data_modules/data_module.py](../../src/tmgg/data/data_modules/data_module.py)
- [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py)
- [src/tmgg/data/data_modules/synthetic_categorical.py](../../src/tmgg/data/data_modules/synthetic_categorical.py)
- Create [src/tmgg/data/data_modules/graph_generation.py](../../src/tmgg/data/data_modules/graph_generation.py)

### Continuous-path production call sites that must stop using ambiguous adjacency helpers

- [src/tmgg/models/attention/attention.py](../../src/tmgg/models/attention/attention.py)
- [src/tmgg/models/baselines/linear.py](../../src/tmgg/models/baselines/linear.py)
- [src/tmgg/models/baselines/mlp.py](../../src/tmgg/models/baselines/mlp.py)
- [src/tmgg/models/gnn/gnn.py](../../src/tmgg/models/gnn/gnn.py)
- [src/tmgg/models/gnn/gnn_sym.py](../../src/tmgg/models/gnn/gnn_sym.py)
- [src/tmgg/models/gnn/nvgnn.py](../../src/tmgg/models/gnn/nvgnn.py)
- [src/tmgg/models/hybrid/hybrid.py](../../src/tmgg/models/hybrid/hybrid.py)
- [src/tmgg/models/spectral_denoisers/base_spectral.py](../../src/tmgg/models/spectral_denoisers/base_spectral.py)
- [src/tmgg/models/spectral_denoisers/bilinear.py](../../src/tmgg/models/spectral_denoisers/bilinear.py)
- [src/tmgg/utils/noising/noise.py](../../src/tmgg/utils/noising/noise.py)
- [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py)
- [src/tmgg/training/orchestration/sanity_check.py](../../src/tmgg/training/orchestration/sanity_check.py)

### Topology-boundary callers that should use explicit binary accessors

- [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py)
- [src/tmgg/models/digress/transformer_model.py](../../src/tmgg/models/digress/transformer_model.py)
- [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py)
- [src/tmgg/data/datasets/graph_types.py](../../src/tmgg/data/datasets/graph_types.py)

### Config and docs surface

- [src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml](../../src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml)
- [src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml](../../src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml)
- [src/tmgg/experiments/exp_configs/task/denoising.yaml](../../src/tmgg/experiments/exp_configs/task/denoising.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml)
- [src/tmgg/experiments/exp_configs/models/digress/digress_base.yaml](../../src/tmgg/experiments/exp_configs/models/digress/digress_base.yaml)
- [src/tmgg/experiments/exp_configs/models/gnn/standard_gnn.yaml](../../src/tmgg/experiments/exp_configs/models/gnn/standard_gnn.yaml)
- [src/tmgg/experiments/exp_configs/models/gnn/symmetric_gnn.yaml](../../src/tmgg/experiments/exp_configs/models/gnn/symmetric_gnn.yaml)
- [src/tmgg/experiments/exp_configs/models/gnn/nodevar_gnn.yaml](../../src/tmgg/experiments/exp_configs/models/gnn/nodevar_gnn.yaml)
- [src/tmgg/experiments/exp_configs/models/hybrid/hybrid_gnn_only.yaml](../../src/tmgg/experiments/exp_configs/models/hybrid/hybrid_gnn_only.yaml)
- [src/tmgg/experiments/exp_configs/models/hybrid/hybrid_with_transformer.yaml](../../src/tmgg/experiments/exp_configs/models/hybrid/hybrid_with_transformer.yaml)
- [src/tmgg/experiments/exp_configs/models/baselines/linear.yaml](../../src/tmgg/experiments/exp_configs/models/baselines/linear.yaml)
- [src/tmgg/experiments/exp_configs/models/baselines/mlp.yaml](../../src/tmgg/experiments/exp_configs/models/baselines/mlp.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/filter_bank.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/filter_bank.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/linear_pe.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/linear_pe.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/multilayer_self_attention.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/multilayer_self_attention.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/self_attention.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/self_attention.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/self_attention_mlp.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/self_attention_mlp.yaml)
- [src/tmgg/experiments/_scaffold/model_config.yaml.j2](../../src/tmgg/experiments/_scaffold/model_config.yaml.j2)
- [docs/extending.md](../extending.md)
- [docs/configuration.md](../configuration.md)
- [docs/experiments.md](../experiments.md)
- [docs/data.md](../data.md)
- [docs/architecture.md](../architecture.md)

## Chunk 1: Split `GraphData` accessors and migrate continuous-state callers

### Goal

Make `GraphData` explicit about meaning:

- binary topology uses `from_binary_adjacency()` / `to_binary_adjacency()`
- dense latent edge state uses `from_edge_state()` / `to_edge_state()`

Continuous and denoising paths must stop round-tripping through binary adjacency. The current `from_adjacency()` / `to_adjacency()` / `edge_features_to_adjacency()` helpers are ambiguous and must disappear from production code by the end of this chunk.

### Files to change

- [src/tmgg/data/datasets/graph_types.py](../../src/tmgg/data/datasets/graph_types.py)
- [src/tmgg/models/attention/attention.py](../../src/tmgg/models/attention/attention.py)
- [src/tmgg/models/baselines/linear.py](../../src/tmgg/models/baselines/linear.py)
- [src/tmgg/models/baselines/mlp.py](../../src/tmgg/models/baselines/mlp.py)
- [src/tmgg/models/gnn/gnn.py](../../src/tmgg/models/gnn/gnn.py)
- [src/tmgg/models/gnn/gnn_sym.py](../../src/tmgg/models/gnn/gnn_sym.py)
- [src/tmgg/models/gnn/nvgnn.py](../../src/tmgg/models/gnn/nvgnn.py)
- [src/tmgg/models/hybrid/hybrid.py](../../src/tmgg/models/hybrid/hybrid.py)
- [src/tmgg/models/spectral_denoisers/base_spectral.py](../../src/tmgg/models/spectral_denoisers/base_spectral.py)
- [src/tmgg/models/spectral_denoisers/bilinear.py](../../src/tmgg/models/spectral_denoisers/bilinear.py)
- [src/tmgg/utils/noising/noise.py](../../src/tmgg/utils/noising/noise.py)
- [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py)
- [src/tmgg/training/orchestration/sanity_check.py](../../src/tmgg/training/orchestration/sanity_check.py)
- [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py)
- [src/tmgg/models/digress/transformer_model.py](../../src/tmgg/models/digress/transformer_model.py)

### Instructions

1. In [src/tmgg/data/datasets/graph_types.py](../../src/tmgg/data/datasets/graph_types.py), add the four explicit accessors from the spec.
2. Keep the binary-topology helpers semantically identical to the old adjacency helpers. They should build or recover discrete graph structure and may zero the diagonal.
3. Make the edge-state helpers lossless. They must not call `argmax`, `round`, `clamp`, `threshold`, or convert a dense latent state into a two-class categorical encoding.
4. Use the edge-state helpers in every continuous or denoising model that currently calls `to_adjacency()` or `from_adjacency()`. The files above are the current production call sites.
5. Use binary-topology helpers only at explicit graph boundaries:
   - NetworkX or PyG export
   - reference-graph extraction
   - categorical transformer structural context
   - final generated-graph extraction
6. In [src/tmgg/models/digress/transformer_model.py](../../src/tmgg/models/digress/transformer_model.py), stop decoding raw `E_cat` through a static helper. Build binary adjacency from the input `GraphData` object before the hidden-state MLPs, so the caller never depends on `GraphData.E` channel layout.
7. Update [src/tmgg/utils/noising/noise.py](../../src/tmgg/utils/noising/noise.py) so `NoiseDefinition.apply_noise()` operates on edge state rather than binary adjacency.
8. Remove the ambiguous helpers from production code. If needed, remove them from `GraphData` entirely in this chunk, not later.

### Tests to update first

- [tests/experiment_utils/test_conversions.py](../../tests/experiment_utils/test_conversions.py)
- [tests/models/test_attention.py](../../tests/models/test_attention.py)
- [tests/models/test_gnn.py](../../tests/models/test_gnn.py)
- [tests/models/test_hybrid.py](../../tests/models/test_hybrid.py)
- [tests/models/test_self_attention_denoiser.py](../../tests/models/test_self_attention_denoiser.py)

Add explicit tests for:

- binary topology round-trip
- edge-state round-trip with non-binary floating values
- no binary projection inside continuous model forward passes

### Verification

```bash
uv run pytest tests/experiment_utils/test_conversions.py tests/models/test_attention.py tests/models/test_gnn.py tests/models/test_hybrid.py tests/models/test_self_attention_denoiser.py -x -v
rg -n "from_adjacency\\(|to_adjacency\\(|edge_features_to_adjacency\\(" src
```

Suggested commit: `refactor(graphdata): split binary topology and edge-state accessors`

## Chunk 2: Rewrite the datamodule contract and deduplicate graph generation

### Goal

Make `BaseGraphDataModule` the full training-facing contract, add `get_size_distribution()` on the base class, move `graph_type` and `num_nodes` to the base contract, and remove datamodule-owned noise and marginal responsibilities.

### Files to change

- [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py)
- [src/tmgg/data/data_modules/multigraph_data_module.py](../../src/tmgg/data/data_modules/multigraph_data_module.py)
- [src/tmgg/data/data_modules/data_module.py](../../src/tmgg/data/data_modules/data_module.py)
- [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py)
- [src/tmgg/data/data_modules/synthetic_categorical.py](../../src/tmgg/data/data_modules/synthetic_categorical.py)
- Create [src/tmgg/data/data_modules/graph_generation.py](../../src/tmgg/data/data_modules/graph_generation.py)
- [src/tmgg/data/data_modules/__init__.py](../../src/tmgg/data/data_modules/__init__.py)
- [src/tmgg/data/__init__.py](../../src/tmgg/data/__init__.py)

### Instructions

1. Add `graph_type` and `num_nodes` annotations to [src/tmgg/data/data_modules/base_data_module.py](../../src/tmgg/data/data_modules/base_data_module.py), and give the base class a concrete `get_size_distribution()` that returns `SizeDistribution.fixed(self.num_nodes)`.
2. Remove `get_dataset_info()` from the base-class abstract contract.
3. Create `generate_graph_adjacencies()` in [src/tmgg/data/data_modules/graph_generation.py](../../src/tmgg/data/data_modules/graph_generation.py). This becomes the only production dispatch point for:
   - SBM generation
   - `SyntheticGraphDataset`
   - PyG dataset loading and graph selection
4. Make [src/tmgg/data/data_modules/multigraph_data_module.py](../../src/tmgg/data/data_modules/multigraph_data_module.py) and [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py) call the shared helper instead of keeping separate graph-type branching.
5. Remove datamodule constructor noise fields from [src/tmgg/data/data_modules/data_module.py](../../src/tmgg/data/data_modules/data_module.py) and [src/tmgg/data/data_modules/single_graph_data_module.py](../../src/tmgg/data/data_modules/single_graph_data_module.py). They should no longer accept `noise_level`, `noise_levels`, or `noise_type`.
6. Strip marginal computation from [src/tmgg/data/data_modules/synthetic_categorical.py](../../src/tmgg/data/data_modules/synthetic_categorical.py). After this chunk, that class should only generate categorical graph batches and expose `get_size_distribution()` if it still adds value.
7. Remove the dead and tests-only datamodule APIs from production code:
   - `get_dataset_info()`
   - `get_sample_adjacency_matrix()`
   - `get_sample_graph()`
   - `get_train_graph()`
   - `get_val_graph()`
   - `get_test_graph()`
   - `sample_n_nodes()`
8. Rewrite tests to use only the public contract:
   - dataloaders
   - `get_reference_graphs()`
   - `get_size_distribution()`

### Tests to update first

- [tests/test_datamodule_contracts.py](../../tests/test_datamodule_contracts.py)
- [tests/test_single_graph_datasets.py](../../tests/test_single_graph_datasets.py)
- [tests/experiments/test_categorical_datamodule.py](../../tests/experiments/test_categorical_datamodule.py)
- [tests/experiment_utils/test_data_module.py](../../tests/experiment_utils/test_data_module.py)

### Verification

```bash
uv run pytest tests/test_datamodule_contracts.py tests/test_single_graph_datasets.py tests/experiments/test_categorical_datamodule.py tests/experiment_utils/test_data_module.py -x -v
rg -n "get_dataset_info\\(|get_sample_adjacency_matrix\\(|get_sample_graph\\(|get_train_graph\\(|get_val_graph\\(|get_test_graph\\(|sample_n_nodes\\(" src
```

Suggested commit: `refactor(data): unify datamodule contract and graph generation`

## Chunk 3: Rewrite `NoiseProcess` around functional process operations

### Goal

Replace `apply()` / `get_posterior()` / transition-model injection with the new timestep-based contract. Move categorical stationary-distribution logic inside `CategoricalNoiseProcess`. Keep `NoiseSchedule` as an implementation detail owned by the concrete process.

### Files to change

- [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py)
- [src/tmgg/diffusion/diffusion_sampling.py](../../src/tmgg/diffusion/diffusion_sampling.py)
- [src/tmgg/diffusion/schedule.py](../../src/tmgg/diffusion/schedule.py)
- [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py)
- Delete [src/tmgg/diffusion/categorical_noise.py](../../src/tmgg/diffusion/categorical_noise.py)
- Delete [src/tmgg/diffusion/protocols.py](../../src/tmgg/diffusion/protocols.py)
- Delete [src/tmgg/diffusion/transitions.py](../../src/tmgg/diffusion/transitions.py)
- Delete [src/tmgg/diffusion/diffusion_graph_types.py](../../src/tmgg/diffusion/diffusion_graph_types.py)

### Instructions

1. In [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py), replace the current abstract base with:
   - `NoiseProcess`
   - `ExactDensityNoiseProcess`
2. The base contract must expose:
   - `timesteps`
   - `initialize_from_data(train_loader)`
   - `sample_prior(node_mask)`
   - `forward_sample(x_0, t)`
   - `posterior_sample(z_t, x0_param, t, s)`
3. `ContinuousNoiseProcess` should own its schedule privately. Rename constructor args to `definition` and `schedule`. Remove the deprecated `generator` alias entirely.
4. Make continuous forward and reverse math operate in edge-state space through `to_edge_state()` / `from_edge_state()`.
5. `ContinuousNoiseProcess.posterior_sample()` must return the sampled `GraphData` at timestep `s`, not a dict of Gaussian parameters. Keep any mean/std helpers private.
6. Implement exact-density methods for the scheduled Gaussian process in the same class or a shared private helper.
7. Replace the categorical transition-model pipeline with direct process math. The public constructor should use:
   - `schedule`
   - `x_classes`
   - `e_classes`
   - `limit_distribution: "uniform" | "empirical_marginal"`
8. `CategoricalNoiseProcess.initialize_from_data()` must compute node and edge PMFs directly from `train_loader` when `limit_distribution == "empirical_marginal"`.
9. For marginal estimation:
   - count only real nodes
   - count only valid real-node edges
   - use the strict upper triangle for undirected graphs
   - fall back to uniform PMF on zero-count domains
10. Delete all transition-model public types and public categorical-noise wrappers after moving the math.
11. Rewrite any old `kl_prior`, `compute_Lt`, or `reconstruction_logp` logic to use the exact-density interface. Do not keep the old method names.

### Tests to update first

- [tests/diffusion/test_noise_process.py](../../tests/diffusion/test_noise_process.py)
- [tests/diffusion/test_noise_process_vlb.py](../../tests/diffusion/test_noise_process_vlb.py)
- [tests/models/test_noise_schedule.py](../../tests/models/test_noise_schedule.py)
- [tests/models/test_graph_types.py](../../tests/models/test_graph_types.py)

The transition-specific tests should become process-level tests. Delete tests that only assert properties of removed transition classes.

### Verification

```bash
uv run pytest tests/diffusion/test_noise_process.py tests/diffusion/test_noise_process_vlb.py tests/models/test_noise_schedule.py tests/models/test_graph_types.py -x -v
rg -n "transition_model|set_transition_model|get_posterior\\(|CategoricalNoiseDefinition|TransitionModel|LimitDistribution|TransitionMatrices|DiscreteUniformTransition|MarginalUniformTransition" src
```

Suggested commit: `refactor(diffusion): move process math behind functional noise interface`

## Chunk 4: Replace sampler subclasses with one `Sampler` and `DiffusionState`

### Goal

Delete `CategoricalSampler` and `ContinuousSampler`. The sampler must become a single reverse-loop controller that knows nothing about subtype internals.

### Files to change

- [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py)
- [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py)
- [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py)

### Instructions

1. Add `DiffusionState` as a public dataclass at the top of [src/tmgg/diffusion/sampler.py](../../src/tmgg/diffusion/sampler.py).
2. Replace the current ABC plus two subclasses with one concrete `Sampler` that accepts only a `NoiseProcess`.
3. Implement the reverse loop exactly as the spec describes:
   - construct `node_mask`
   - call `noise_process.sample_prior(node_mask)` if no warm start
   - validate `DiffusionState`
   - loop `t -> s = t-1`
   - call the model on `current_state.graph`
   - call `noise_process.posterior_sample(...)`
   - wrap the result back into `DiffusionState`
4. Remove `start_timestep`. Warm starts must use `start_from: DiffusionState`.
5. The sampler must not inspect schedule buffers, transition matrices, categorical class counts, or Gaussian posterior dicts.
6. Continuous intermediate states must remain in `GraphData` form until the explicit final graph-extraction boundary.

### Tests to update first

- [tests/diffusion/test_sampler.py](../../tests/diffusion/test_sampler.py)
- [tests/test_generative_integration.py](../../tests/test_generative_integration.py)

Add direct tests for:

- `DiffusionState` validation
- warm-start sampling via `DiffusionState`
- `T=1` reverse chain (`t=1 -> s=0`)

### Verification

```bash
uv run pytest tests/diffusion/test_sampler.py tests/test_generative_integration.py -x -v
rg -n "CategoricalSampler|ContinuousSampler|start_timestep" src tests
```

Suggested commit: `refactor(diffusion): unify reverse sampling and add DiffusionState`

## Chunk 5: Rewrite `DiffusionModule` and absorb `T=1` denoising into the process contract

### Goal

Make `DiffusionModule` accept one `NoiseProcess`, construct its own `Sampler`, initialize the process from data in `setup()`, and treat one-step denoising as the `T=1` case of the same contract. Delete `SingleStepDenoisingModule`.

### Files to change

- [src/tmgg/training/lightning_modules/diffusion_module.py](../../src/tmgg/training/lightning_modules/diffusion_module.py)
- Delete [src/tmgg/training/lightning_modules/denoising_module.py](../../src/tmgg/training/lightning_modules/denoising_module.py)
- [src/tmgg/training/lightning_modules/__init__.py](../../src/tmgg/training/lightning_modules/__init__.py)
- [src/tmgg/training/__init__.py](../../src/tmgg/training/__init__.py)
- [src/tmgg/experiments/lin_mlp_baseline_denoising/__init__.py](../../src/tmgg/experiments/lin_mlp_baseline_denoising/__init__.py)
- [src/tmgg/experiments/spectral_arch_denoising/__init__.py](../../src/tmgg/experiments/spectral_arch_denoising/__init__.py)
- Search and update any other experiment package that re-exports `SingleStepDenoisingModule`

### Instructions

1. Remove `sampler` and `noise_schedule` from the `DiffusionModule` constructor. Instantiate `self.sampler = Sampler(self.noise_process)` internally.
2. Make `self.T` derive from `self.noise_process.timesteps`.
3. In `setup()`, always do exactly:
   - `self.noise_process.initialize_from_data(dm.train_dataloader())`
   - `self._size_distribution = dm.get_size_distribution("train")`
4. Remove all datamodule duck typing from `setup()`. No `node_marginals`, `edge_marginals`, or `hasattr(get_size_distribution)`.
5. Update `training_step()` to use `noise_process.forward_sample()`.
6. Rewrite VLB or NLL code to use `ExactDensityNoiseProcess` methods directly. If the configured process does not implement the exact-density interface and the code path needs it, fail immediately.
7. Keep `reconstruction_logp` logic on the training module only when it depends on model-output parameterization. Do not reach back into categorical-only process helpers.
8. Delete `SingleStepDenoisingModule`. Do not replace it with another Lightning subclass.
9. Represent denoising as `DiffusionModule + T=1 noise process`:
   - the process owns `noise_level`
   - `timesteps == 1`
   - `forward_sample(x_0, t=1)` applies the configured corruption level
   - `sample_prior(node_mask)` draws from the induced noisy prior
10. Prefer extending the concrete process constructors so they can represent both scheduled `T>1` and fixed-level `T=1` cases. That keeps the public surface smaller than adding a parallel denoising-process hierarchy.

### Tests to update first

- [tests/experiment_utils/test_diffusion_module.py](../../tests/experiment_utils/test_diffusion_module.py)
- [tests/experiment_utils/test_denoising_module.py](../../tests/experiment_utils/test_denoising_module.py)
- [tests/test_training_integration.py](../../tests/test_training_integration.py)
- [tests/experiments/test_baseline_denoising_module.py](../../tests/experiments/test_baseline_denoising_module.py)
- [tests/experiments/test_digress_denoising_module.py](../../tests/experiments/test_digress_denoising_module.py)
- [tests/experiments/test_all_experiments_full_flow.py](../../tests/experiments/test_all_experiments_full_flow.py)

Replace the denoising-module tests with `T=1` diffusion tests. Do not keep a thin wrapper test just to preserve the deleted class name.

### Verification

```bash
uv run pytest tests/experiment_utils/test_diffusion_module.py tests/experiment_utils/test_denoising_module.py tests/test_training_integration.py tests/experiments/test_baseline_denoising_module.py tests/experiments/test_digress_denoising_module.py tests/experiments/test_all_experiments_full_flow.py -x -v
rg -n "SingleStepDenoisingModule|noise_schedule=|sampler=" src tests
```

Suggested commit: `refactor(training): unify denoising and diffusion under DiffusionModule`

## Chunk 6: Migrate Hydra configs, scaffold template, and experiment packaging

### Goal

Make configs instantiate one `NoiseProcess` object, nest schedules under that process, and move denoising runs to `DiffusionModule` with a singular `noise_level`.

### Files to change

- [src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml](../../src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml)
- [src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml](../../src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml)
- [src/tmgg/experiments/exp_configs/task/denoising.yaml](../../src/tmgg/experiments/exp_configs/task/denoising.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml)
- [src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml](../../src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml)
- [src/tmgg/experiments/exp_configs/models/digress/digress_base.yaml](../../src/tmgg/experiments/exp_configs/models/digress/digress_base.yaml)
- [src/tmgg/experiments/exp_configs/models/gnn/standard_gnn.yaml](../../src/tmgg/experiments/exp_configs/models/gnn/standard_gnn.yaml)
- [src/tmgg/experiments/exp_configs/models/gnn/symmetric_gnn.yaml](../../src/tmgg/experiments/exp_configs/models/gnn/symmetric_gnn.yaml)
- [src/tmgg/experiments/exp_configs/models/gnn/nodevar_gnn.yaml](../../src/tmgg/experiments/exp_configs/models/gnn/nodevar_gnn.yaml)
- [src/tmgg/experiments/exp_configs/models/hybrid/hybrid_gnn_only.yaml](../../src/tmgg/experiments/exp_configs/models/hybrid/hybrid_gnn_only.yaml)
- [src/tmgg/experiments/exp_configs/models/hybrid/hybrid_with_transformer.yaml](../../src/tmgg/experiments/exp_configs/models/hybrid/hybrid_with_transformer.yaml)
- [src/tmgg/experiments/exp_configs/models/baselines/linear.yaml](../../src/tmgg/experiments/exp_configs/models/baselines/linear.yaml)
- [src/tmgg/experiments/exp_configs/models/baselines/mlp.yaml](../../src/tmgg/experiments/exp_configs/models/baselines/mlp.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/filter_bank.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/filter_bank.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/linear_pe.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/linear_pe.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/multilayer_self_attention.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/multilayer_self_attention.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/self_attention.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/self_attention.yaml)
- [src/tmgg/experiments/exp_configs/models/spectral/self_attention_mlp.yaml](../../src/tmgg/experiments/exp_configs/models/spectral/self_attention_mlp.yaml)
- [src/tmgg/experiments/_scaffold/model_config.yaml.j2](../../src/tmgg/experiments/_scaffold/model_config.yaml.j2)

### Instructions

1. Discrete generative configs:
   - remove `sampler:`
   - remove top-level `noise_schedule:`
   - nest the schedule under `model.noise_process.schedule`
   - replace `transition_type: marginal` with `limit_distribution: empirical_marginal`
2. Gaussian generative config:
   - remove `sampler:` and top-level `noise_schedule:`
   - rename `generator:` to `definition:`
   - nest schedule under `model.noise_process.schedule`
3. Denoising configs:
   - point the outer `_target_` at `DiffusionModule`
   - instantiate a `T=1` process under `model.noise_process`
   - use singular `noise_level`, not `noise_levels`
   - remove `eval_noise_levels`
4. Remove noise fields from all data configs. The datamodule should no longer accept or store them.
5. For stage and sweep configs that currently compare several denoising levels, move the list out of one-module config shape. Use Hydra multirun or stage-level sweep parameters so each run still has exactly one `noise_level`.
6. Update the scaffold template so new experiment model configs generate the new shape by default.

### Tests to update first

- [tests/experiments/test_discrete_diffusion_runner.py](../../tests/experiments/test_discrete_diffusion_runner.py)

Add config-shape assertions for:

- no `sampler:` under the model section
- no top-level `model.noise_schedule`
- no `SingleStepDenoisingModule`
- denoising model configs use a singular `noise_level`

### Verification

```bash
uv run pytest tests/experiments/test_discrete_diffusion_runner.py -x -v
rg -n "_target_: .*SingleStepDenoisingModule" src/tmgg/experiments/exp_configs
rg -n "sampler:|noise_schedule:" src/tmgg/experiments/exp_configs
rg -n "noise_levels:|eval_noise_levels:" src/tmgg/experiments/exp_configs/models src/tmgg/experiments/exp_configs/task
```

Suggested commit: `refactor(config): simplify process wiring and T=1 denoising configs`

## Chunk 7: Rewrite tests around the new contracts and remove dead test coverage

### Goal

Update the test suite so it validates the new public surface rather than the removed helper APIs.

### Priority files

- [tests/diffusion/test_noise_process.py](../../tests/diffusion/test_noise_process.py)
- [tests/diffusion/test_noise_process_vlb.py](../../tests/diffusion/test_noise_process_vlb.py)
- [tests/diffusion/test_sampler.py](../../tests/diffusion/test_sampler.py)
- [tests/experiment_utils/test_conversions.py](../../tests/experiment_utils/test_conversions.py)
- [tests/experiment_utils/test_diffusion_module.py](../../tests/experiment_utils/test_diffusion_module.py)
- [tests/experiment_utils/test_denoising_module.py](../../tests/experiment_utils/test_denoising_module.py)
- [tests/experiments/test_categorical_datamodule.py](../../tests/experiments/test_categorical_datamodule.py)
- [tests/test_datamodule_contracts.py](../../tests/test_datamodule_contracts.py)
- [tests/test_single_graph_datasets.py](../../tests/test_single_graph_datasets.py)
- [tests/test_generative_integration.py](../../tests/test_generative_integration.py)
- [tests/test_training_integration.py](../../tests/test_training_integration.py)

### Required rewrites

1. Remove all assertions about:
   - datamodule marginals
   - transition-model classes
   - `get_posterior()`
   - `start_timestep`
   - `SingleStepDenoisingModule`
   - ambiguous adjacency helpers
2. Add direct tests for:
   - `NoiseProcess.initialize_from_data()`
   - `NoiseProcess.sample_prior()`
   - `Sampler.sample()`
   - `DiffusionState` warm starts
   - fixed-size default `get_size_distribution()`
   - `T=1` denoising process behavior
   - edge-state round-trip without projection
3. Delete tests that only cover removed public types. Replace them with behavior tests against the new contracts, not one-to-one class-name replacements.

### Verification

Run targeted suites first, then the broader sweep:

```bash
uv run pytest tests/diffusion tests/experiment_utils tests/experiments tests/test_datamodule_contracts.py tests/test_single_graph_datasets.py tests/test_generative_integration.py tests/test_training_integration.py -x -v
```

Suggested commit: `test(refactor): move coverage to new datamodule and diffusion contracts`

## Chunk 8: Docs, exports, and final sweeps

### Goal

Bring the public documentation and package exports in line with the refactor, then prove the removed interfaces are gone.

### Files to change

- [docs/extending.md](../extending.md)
- [docs/configuration.md](../configuration.md)
- [docs/experiments.md](../experiments.md)
- [docs/data.md](../data.md)
- [docs/architecture.md](../architecture.md)
- [src/tmgg/diffusion/__init__.py](../../src/tmgg/diffusion/__init__.py)
- [src/tmgg/training/__init__.py](../../src/tmgg/training/__init__.py)
- [src/tmgg/training/lightning_modules/__init__.py](../../src/tmgg/training/lightning_modules/__init__.py)

### Instructions

1. Update docs so every example uses:
   - one `NoiseProcess`
   - nested process-owned schedules
   - `DiffusionModule` for both multi-step and `T=1` denoising
   - explicit binary vs edge-state accessors
2. Remove exports of deleted types and classes from package `__init__` files.
3. Update experiment package `__init__` files so they no longer re-export `SingleStepDenoisingModule`.

### Final call-site sweeps

These must return no production callers:

```bash
rg -n "transition_model|set_transition_model|get_posterior\\(|node_marginals|edge_marginals" src
rg -n "get_dataset_info\\(|get_sample_adjacency_matrix\\(|get_sample_graph\\(|get_train_graph\\(|get_val_graph\\(|get_test_graph\\(|sample_n_nodes\\(" src
rg -n "hasattr\\(.*get_size_distribution|getattr\\(.*noise_level|getattr\\(.*noise_levels" src
rg -n "from_adjacency\\(|to_adjacency\\(|edge_features_to_adjacency\\(" src
rg -n "start_timestep" src
rg -n "_target_: .*SingleStepDenoisingModule" src/tmgg/experiments/exp_configs
rg -n "noise_schedule:" src/tmgg/experiments/exp_configs | rg "DiffusionModule|sampler"
```

### Full test suite

Run the targeted suites first. When they pass, run the full suite used for this branch.

Suggested final commit: `docs(refactor): update public guidance for unified process and sampler contracts`

## Suggested Commit Boundaries

1. `refactor(graphdata): split binary topology and edge-state accessors`
2. `refactor(data): unify datamodule contract and graph generation`
3. `refactor(diffusion): move process math behind functional noise interface`
4. `refactor(diffusion): unify reverse sampling and add DiffusionState`
5. `refactor(training): unify denoising and diffusion under DiffusionModule`
6. `refactor(config): simplify process wiring and T=1 denoising configs`
7. `test(refactor): move coverage to new datamodule and diffusion contracts`
8. `docs(refactor): update public guidance for unified process and sampler contracts`

## Notes for the Implementer

- Do not preserve checkpoint compatibility unless it remains trivial after the refactor.
- If a piece of logic is still needed but only for one concrete process, keep it private to that process. Do not rebuild the removed public abstraction under a different name.
- If a test only proves a removed helper existed, delete or replace it. Do not keep dead API surface alive for test convenience.
- Do not finish the branch without running the final `rg` sweeps and a broad test pass.
