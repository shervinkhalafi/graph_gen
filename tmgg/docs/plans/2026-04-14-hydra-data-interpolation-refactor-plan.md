# Hydra Data Interpolation Refactor Plan

> **For agentic workers:** use `subagent-driven-development` or `executing-plans` if this plan is executed later. This file is a design and edit plan only; it is not an implementation log.

**Goal:** Make datamodule selection orthogonal to graph-family, protocol, and loader settings by restructuring Hydra config ownership around interpolation-based composition.

**Architecture:** Replace the current monolithic `/data` presets with four orthogonal config groups: `data_module`, `data_source`, `data_protocol`, and `data_loader`. `data_module` will own the final assembled `data:` subtree and `_target_`; the other three groups will expose namespaced values consumed by interpolation. This keeps the refactor Hydra-native while preventing `_target_` collisions.

**Tech Stack:** Hydra, OmegaConf interpolation, existing experiment CLIs (`tmgg-experiment`, `tmgg-gaussian-gen`, `tmgg-discrete-gen`, `tmgg-modal run`).

---

## Problem Statement

The current config layout mixes structural and semantic concerns under the same `/data` group.

- Some base configs define `data:` inline, for example [base_config_discrete_diffusion_generative.yaml](../../src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml).
- Some data presets own `_target_`, for example [base_dataloader.yaml](../../src/tmgg/experiments/exp_configs/data/base_dataloader.yaml).
- Some stage configs replace the full `/data` group, for example [stage1_poc.yaml](../../src/tmgg/experiments/exp_configs/stage/stage1_poc.yaml).
- Single-graph, multigraph denoising, gaussian generative, and categorical generative datamodules have different constructor signatures, so a blind flat merge into one `data` namespace is fragile.

The result is that overrides intended to change graph-family settings can silently swap the datamodule class. The `+data=sbm_digress` case is one concrete example of this failure mode.

## Target State

The end state should obey one ownership rule:

- `data_module` owns the datamodule class and the shape of the final `data:` subtree.
- `data_source` owns graph-family semantics such as `graph_type`, `num_nodes`, and `graph_config`.
- `data_protocol` owns split or protocol semantics such as `train_ratio`, `val_ratio`, `samples_per_graph`, or `same_graph_all_splits`.
- `data_loader` owns only loader and runtime knobs such as `batch_size`, `num_workers`, and `pin_memory`.

The crucial design choice is that `data_module` does not merely expose `_target_`. It also assembles the final `data:` subtree through interpolation so that different datamodule classes can keep different constructor shapes without introducing a factory.

Example shape:

```yaml
defaults:
  - data_module: denoising_multigraph
  - data_source: sbm_partitioned
  - data_protocol: multigraph_default
  - data_loader: denoising_default
  - _self_
```

```yaml
# data_module/denoising_multigraph.yaml
# @package _global_

data:
  _target_: tmgg.data.GraphDataModule
  graph_type: ${data_source.graph_type}
  graph_config: ${data_source.graph_config}
  samples_per_graph: ${data_protocol.samples_per_graph}
  val_samples_per_graph: ${data_protocol.val_samples_per_graph}
  train_ratio: ${data_protocol.train_ratio}
  val_ratio: ${data_protocol.val_ratio}
  batch_size: ${data_loader.batch_size}
  num_workers: ${data_loader.num_workers}
  pin_memory: ${data_loader.pin_memory}
  seed: ${seed}
```

```yaml
# data_source/sbm_partitioned.yaml
# @package _global_

data_source:
  graph_type: sbm
  graph_config:
    num_nodes: 20
    p_intra: 1.0
    p_inter: 0.0
    num_train_partitions: 10
    num_test_partitions: 10
```

This preserves Hydra composition while making each group's responsibility explicit.

## Non-Goals

- Do not introduce a Python datamodule factory in this refactor.
- Do not keep the old `/data` group as a long-lived compatibility layer.
- Do not redesign datamodule constructor APIs in the same change unless a specific constructor mismatch blocks interpolation-based assembly.

## Config Group Layout

Create these new directories under [src/tmgg/experiments/exp_configs](../../src/tmgg/experiments/exp_configs/):

- `data_module/`
- `data_source/`
- `data_protocol/`
- `data_loader/`

Recommended initial members:

- `data_module/denoising_multigraph.yaml`
- `data_module/denoising_single_graph.yaml`
- `data_module/gaussian_multigraph.yaml`
- `data_module/discrete_categorical.yaml`
- `data_source/sbm_partitioned.yaml`
- `data_source/sbm_single_graph.yaml`
- `data_source/erdos_renyi.yaml`
- `data_source/regular.yaml`
- `data_source/tree.yaml`
- `data_source/ring_of_cliques.yaml`
- `data_source/lfr.yaml`
- `data_source/pyg_enzymes.yaml`
- `data_source/pyg_proteins.yaml`
- `data_source/pyg_qm9.yaml`
- `data_protocol/multigraph_default.yaml`
- `data_protocol/multigraph_grid.yaml`
- `data_protocol/single_graph_same.yaml`
- `data_protocol/single_graph_cross.yaml`
- `data_protocol/discrete_sbm_digress.yaml`
- `data_loader/denoising_default.yaml`
- `data_loader/single_graph_small.yaml`
- `data_loader/grid_default.yaml`
- `data_loader/discrete_digress.yaml`

## File Mapping From The Current Tree

The old `data/*.yaml` files should be split by responsibility rather than copied mechanically.

| Current file | New ownership |
|---|---|
| `data/base_dataloader.yaml` | `data_loader/denoising_default.yaml` |
| `data/single_graph_base.yaml` | split across `data_module/denoising_single_graph.yaml`, `data_protocol/single_graph_same.yaml`, and `data_loader/single_graph_small.yaml` |
| `data/sbm_default.yaml` | `data_source/sbm_partitioned.yaml` plus `data_protocol/multigraph_default.yaml` |
| `data/sbm_single_graph.yaml` | `data_source/sbm_single_graph.yaml` plus `data_protocol/single_graph_same.yaml` |
| `data/sbm_digress.yaml` | `data_protocol/discrete_sbm_digress.yaml` and `data_loader/discrete_digress.yaml`; the datamodule class must move to `data_module/discrete_categorical.yaml` |
| `data/grid_base.yaml` | split into `data_source/sbm_partitioned.yaml`, `data_protocol/multigraph_grid.yaml`, `data_loader/grid_default.yaml`, and keep the noise contract outside data |
| `data/pyg_*` files | one `data_source/pyg_*.yaml` each plus whichever module and protocol they require |

## Migration Surface

### Config files to create

- [src/tmgg/experiments/exp_configs/data_module/](../../src/tmgg/experiments/exp_configs)
- [src/tmgg/experiments/exp_configs/data_source/](../../src/tmgg/experiments/exp_configs)
- [src/tmgg/experiments/exp_configs/data_protocol/](../../src/tmgg/experiments/exp_configs)
- [src/tmgg/experiments/exp_configs/data_loader/](../../src/tmgg/experiments/exp_configs)

### Config files to modify

- [src/tmgg/experiments/exp_configs/base_config_denoising.yaml](../../src/tmgg/experiments/exp_configs/base_config_denoising.yaml)
- [src/tmgg/experiments/exp_configs/base_config_spectral_arch.yaml](../../src/tmgg/experiments/exp_configs/base_config_spectral_arch.yaml)
- [src/tmgg/experiments/exp_configs/base_config_gnn.yaml](../../src/tmgg/experiments/exp_configs/base_config_gnn.yaml)
- [src/tmgg/experiments/exp_configs/base_config_gnn_transformer.yaml](../../src/tmgg/experiments/exp_configs/base_config_gnn_transformer.yaml)
- [src/tmgg/experiments/exp_configs/base_config_digress.yaml](../../src/tmgg/experiments/exp_configs/base_config_digress.yaml)
- [src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml](../../src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml)
- [src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml](../../src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml)
- [src/tmgg/experiments/exp_configs/grid_search_base.yaml](../../src/tmgg/experiments/exp_configs/grid_search_base.yaml)
- [src/tmgg/experiments/exp_configs/task/denoising.yaml](../../src/tmgg/experiments/exp_configs/task/denoising.yaml)
- [src/tmgg/experiments/exp_configs/stage/stage1_poc.yaml](../../src/tmgg/experiments/exp_configs/stage/stage1_poc.yaml)
- [src/tmgg/experiments/exp_configs/stage/stage1_sanity.yaml](../../src/tmgg/experiments/exp_configs/stage/stage1_sanity.yaml)
- [src/tmgg/experiments/exp_configs/stage/stage2_validation.yaml](../../src/tmgg/experiments/exp_configs/stage/stage2_validation.yaml)
- [src/tmgg/experiments/exp_configs/README.md](../../src/tmgg/experiments/exp_configs/README.md)
- [README.md](../../README.md)
- [docs/experiments.md](../experiments.md)
- [docs/cloud.md](../cloud.md)
- [docs/configuration.md](../configuration.md)
- [docs/data.md](../data.md)

### Tests to modify or add

- [tests/test_config_composition.py](../../tests/test_config_composition.py)
- [tests/experiments/test_discrete_diffusion_runner.py](../../tests/experiments/test_discrete_diffusion_runner.py)
- [tests/modal/test_config_resolution.py](../../tests/modal/test_config_resolution.py)
- Create `tests/test_data_config_ownership.py`

## Task 1: Introduce Orthogonal Config Groups

**Files:**
- Create the `data_module`, `data_source`, `data_protocol`, and `data_loader` directories and the initial YAML files listed above.

- [ ] Create the four new config-group directories.
- [ ] Add one seed config to each new directory so Hydra can compose them immediately.
- [ ] Use `@package _global_` for the new groups and keep their payloads under top-level namespaces `data_source`, `data_protocol`, and `data_loader`.
- [ ] Keep `data_module/*` responsible for assembling the final `data:` subtree.

Rationale:

- This avoids field collisions because `data_source.*` and `data_loader.*` never merge directly into `data`.
- It allows datamodules with different constructor signatures because each `data_module/*` config can interpolate a different final shape.

## Task 2: Refactor Base Configs To Compose Through Interpolation

**Files:**
- [src/tmgg/experiments/exp_configs/base_config_denoising.yaml](../../src/tmgg/experiments/exp_configs/base_config_denoising.yaml)
- [src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml](../../src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml)
- [src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml](../../src/tmgg/experiments/exp_configs/base_config_discrete_diffusion_generative.yaml)
- [src/tmgg/experiments/exp_configs/task/denoising.yaml](../../src/tmgg/experiments/exp_configs/task/denoising.yaml)

- [ ] Add defaults entries for the new config groups in each experiment family.
- [ ] Remove inline `data:` definitions from the two generative base configs.
- [ ] Remove `/data` defaults from `task/denoising.yaml`; that file should own only denoising-task semantics such as loss and noise settings.
- [ ] Ensure each base config still exposes a fully instantiable `cfg.data`.

Recommended family defaults:

- Denoising multigraph defaults:
  - `data_module: denoising_multigraph`
  - `data_source: sbm_partitioned`
  - `data_protocol: multigraph_default`
  - `data_loader: denoising_default`
- Denoising single-graph stages override:
  - `data_module: denoising_single_graph`
  - `data_source: sbm_single_graph`
  - `data_protocol: single_graph_same`
  - `data_loader: single_graph_small`
- Gaussian generative defaults:
  - `data_module: gaussian_multigraph`
  - `data_source: sbm_partitioned`
  - `data_protocol: multigraph_default`
  - `data_loader: denoising_default`
- Discrete generative defaults:
  - `data_module: discrete_categorical`
  - `data_source: sbm_partitioned`
  - `data_protocol: discrete_sbm_digress`
  - `data_loader: discrete_digress`

## Task 3: Rewrite Stage And Study Overrides

**Files:**
- [src/tmgg/experiments/exp_configs/stage/stage1_poc.yaml](../../src/tmgg/experiments/exp_configs/stage/stage1_poc.yaml)
- [src/tmgg/experiments/exp_configs/stage/stage1_sanity.yaml](../../src/tmgg/experiments/exp_configs/stage/stage1_sanity.yaml)
- [src/tmgg/experiments/exp_configs/stage/stage2_validation.yaml](../../src/tmgg/experiments/exp_configs/stage/stage2_validation.yaml)
- [src/tmgg/experiments/exp_configs/grid_search_base.yaml](../../src/tmgg/experiments/exp_configs/grid_search_base.yaml)

- [ ] Replace `override /data: ...` patterns with separate overrides to `data_module`, `data_source`, `data_protocol`, and `data_loader`.
- [ ] Make stage configs express protocol switches explicitly.
- [ ] Keep noise-level and optimizer overrides outside the new data groups.

Examples:

- `stage1_poc` should no longer say `override /data: sbm_single_graph`. It should override the single-graph module and protocol groups explicitly.
- `grid_search_base` should stop embedding dataloader, split, and source fields in one monolithic block.

## Task 4: Remove The Old `/data` Group After Migration

**Files:**
- All files in [src/tmgg/experiments/exp_configs/data/](../../src/tmgg/experiments/exp_configs/data)

- [ ] Run a repo-wide search for `/data:` overrides and `+data=` usage.
- [ ] Delete or rewrite the legacy `data/*.yaml` files once no live config depends on them.
- [ ] Do not keep compatibility aliases that preserve the old ambiguity.

Verification search:

```bash
rg -n "override /data:|\\+data=| data=" src/tmgg/experiments/exp_configs tests docs README.md
```

Expected end state:

- The search should show only documentation describing the historical migration, not active config composition.

## Task 5: Add Regression Tests For Ownership And Composition

**Files:**
- [tests/test_config_composition.py](../../tests/test_config_composition.py)
- Create `tests/test_data_config_ownership.py`
- [tests/experiments/test_discrete_diffusion_runner.py](../../tests/experiments/test_discrete_diffusion_runner.py)

- [ ] Add a regression test that changing `data_source` does not change `cfg.data._target_`.
- [ ] Add a regression test that changing `data_protocol` does not change `cfg.data._target_`.
- [ ] Add a regression test that changing `data_loader` does not change `cfg.data._target_`.
- [ ] Add a positive test that changing `data_module` does change `cfg.data._target_`.
- [ ] Add a discrete generative regression test covering the former `+data=sbm_digress` failure mode, rewritten in the new group vocabulary.

Suggested assertions:

- `cfg.data._target_ == "tmgg.data.data_modules.synthetic_categorical.SyntheticCategoricalDataModule"` remains stable across `data_source` and `data_protocol` overrides in the discrete path.
- `cfg.data._target_ == "tmgg.data.SingleGraphDataModule"` only when the single-graph module group is selected.

## Task 6: Update Documentation To Match The New Mental Model

**Files:**
- [src/tmgg/experiments/exp_configs/README.md](../../src/tmgg/experiments/exp_configs/README.md)
- [docs/experiments.md](../experiments.md)
- [docs/cloud.md](../cloud.md)
- [docs/configuration.md](../configuration.md)
- [docs/data.md](../data.md)
- [README.md](../../README.md)

- [ ] Explain that `data_module` owns class selection and constructor shape.
- [ ] Explain that `data_source`, `data_protocol`, and `data_loader` are safe overrides that cannot swap datamodule class.
- [ ] Replace examples using `data=...` and `+data=...` with the new group syntax.
- [ ] Add one explicit before/after example for the discrete generative SBM path.

## Verification Checklist

Run these after the refactor:

```bash
uv run pytest tests/test_config_composition.py tests/experiments/test_discrete_diffusion_runner.py tests/modal/test_config_resolution.py -x -v
uv run tmgg-discrete-gen --cfg job
uv run tmgg-gaussian-gen --cfg job
uv run tmgg-experiment +stage=stage1_poc --cfg job
uv run tmgg-modal run tmgg-discrete-gen --dry-run
```

Add one representative explicit override check per family:

```bash
uv run tmgg-discrete-gen --cfg job \
  data_source=sbm_partitioned \
  data_protocol=discrete_sbm_digress \
  data_loader=discrete_digress

uv run tmgg-experiment +stage=stage1_poc --cfg job \
  data_module=denoising_single_graph \
  data_source=sbm_single_graph \
  data_protocol=single_graph_same \
  data_loader=single_graph_small
```

Expected invariants:

- `cfg.data` is always instantiable.
- Only `data_module` changes `cfg.data._target_`.
- Stage configs remain readable because protocol intent is explicit.
- Generative configs no longer define inline `data:` blocks.

## Recommended Execution Order

1. Create the four new config groups and seed files.
2. Refactor the three base families to use interpolation assembly.
3. Rewrite stage and study overrides to the new group vocabulary.
4. Add the ownership regression tests.
5. Delete the old `/data` group.
6. Update docs after the config surface is final.

## Risks And Failure Modes

- If `data_module/*` does not own the final `data:` assembly, constructor-shape drift will reappear immediately.
- If the old `/data` group is kept alive in parallel, humans will keep using it and the ambiguity will return.
- If stage configs override only one of the orthogonal groups when they really need multiple, the composed config will be syntactically valid but semantically wrong.
- If docs are not updated in the same change, users will continue trying `data=...` and `+data=...`.

## Success Criteria

The refactor is successful when:

- no live config file overrides `/data` directly
- no live docs tell users to use `data=...` or `+data=...`
- the discrete generative path can switch graph-family and split presets without changing datamodule class
- the denoising single-graph protocol is expressed as an explicit module-plus-protocol choice rather than a monolithic data preset
