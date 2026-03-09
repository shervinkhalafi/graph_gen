# Training Loop & Model Unification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge three independent LightningModule trees into a unified BaseGraphModule → DiffusionModule → SingleStepDenoisingModule hierarchy, with GraphData everywhere, composable noise processes / samplers / evaluators, and all experiments as Hydra configs.

**Architecture:** Bottom-up: new component interfaces → model migration → training loop → experiment migration → dead code removal. Each task ends with a commit and passing tests. The existing test suite is the regression safety net.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra, graph-tool, NetworkX, numpy

**Test command:** `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v`

**Design doc:** `docs/plans/2026-03-02-training-loop-unification-design.md`

---

## Phase 1: New Component Interfaces

Foundation layer. Create the new abstractions that everything else builds on. Old code continues working alongside — we don't remove anything yet.

---

### Task 1: GraphModel base class

Replace `DenoisingModel` / `AdjacencyDenoisingModel` / `CategoricalDenoisingModel` with a single `GraphModel` that accepts and returns `GraphData`.

**Files:**
- Modify: `src/tmgg/models/base.py`
- Test: `tests/models/test_graph_model_base.py` (create)

**Step 1: Write tests for the new interface**

```python
# tests/models/test_graph_model_base.py
"""Tests for the unified GraphModel base class."""
import pytest
import torch
from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.base import BaseModel, GraphModel


class DummyGraphModel(GraphModel):
    """Minimal concrete implementation for testing the interface."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        return GraphData(X=data.X, E=self.linear(data.E), y=data.y, node_mask=data.node_mask)

    def get_config(self) -> dict:
        return {"type": "dummy"}


def _make_batch(bs: int = 2, n: int = 4) -> GraphData:
    return GraphData.from_adjacency(torch.randint(0, 2, (bs, n, n)).float())


class TestGraphModelInterface:
    def test_forward_accepts_graphdata_returns_graphdata(self):
        model = DummyGraphModel()
        batch = _make_batch()
        result = model.forward(batch)
        assert isinstance(result, GraphData)

    def test_forward_with_timestep(self):
        model = DummyGraphModel()
        batch = _make_batch()
        t = torch.tensor([0.5, 0.3])
        result = model.forward(batch, t=t)
        assert isinstance(result, GraphData)

    def test_inherits_parameter_count(self):
        model = DummyGraphModel()
        counts = model.parameter_count()
        assert counts["total"] > 0

    def test_get_config(self):
        model = DummyGraphModel()
        assert model.get_config() == {"type": "dummy"}

    def test_abstract_forward_enforced(self):
        with pytest.raises(TypeError):
            GraphModel()  # type: ignore[abstract]
```

**Step 2: Run test — expect failure** (GraphModel doesn't exist yet)

```bash
uv run pytest tests/models/test_graph_model_base.py -v
```

**Step 3: Implement GraphModel in base.py**

Add `GraphModel(BaseModel, ABC)` with `forward(data: GraphData, t: Tensor | None = None) -> GraphData` as abstract method. Keep `AdjacencyDenoisingModel`, `CategoricalDenoisingModel`, `DenoisingModel` alive for now — they'll be removed in Phase 4.

**Step 4: Run tests — expect pass**

```bash
uv run pytest tests/models/test_graph_model_base.py -v
```

**Step 5: Run full suite to verify no regressions**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```

**Step 6: Commit**

```
feat: add GraphModel base class with unified GraphData interface
```

---

### Task 2: NoiseProcess hierarchy

Unify `NoiseGenerator` (adjacency-based, `data/noise.py`) and transition models (`models/digress/noise_schedule.py`) under a common `NoiseProcess` interface operating on `GraphData`.

**Files:**
- Create: `src/tmgg/diffusion/__init__.py`
- Create: `src/tmgg/diffusion/noise_process.py`
- Test: `tests/diffusion/test_noise_process.py` (create)
- Modify: `tach.toml` (add `tmgg.diffusion` module boundary)

**Step 1: Write tests**

Test both continuous and categorical noise processes:
- `ContinuousNoiseProcess.apply(data, t)` returns `GraphData` with noise applied to edge features
- `CategoricalNoiseProcess.apply(data, t)` returns `GraphData` with categorical noise via transition matrices
- `CategoricalNoiseProcess.setup(datamodule)` initializes transition model from marginals
- `CategoricalNoiseProcess.kl_prior(X, E, node_mask)` returns tensor
- Both `get_posterior()` methods return the right shapes
- Construction-time: verify existing noise generators (Gaussian, EdgeFlip, DiGress, Rotation, Logit) all work through `ContinuousNoiseProcess`

**Step 2: Implement**

`ContinuousNoiseProcess` wraps existing `NoiseGenerator` subclasses. Internally: extract adjacency from `GraphData.E`, apply noise, wrap back into `GraphData`. The `get_posterior()` method computes the Gaussian posterior for binary data (new math).

`CategoricalNoiseProcess` wraps existing `DiscreteUniformTransition` / `MarginalUniformTransition`. The `apply()` method is the forward diffusion (current `DiscreteDiffusionLightningModule.apply_noise()` logic, relocated). VLB methods (`kl_prior`, `compute_Lt`, `reconstruction_logp`) relocated from `DiscreteDiffusionLightningModule`.

Keep existing `NoiseGenerator` classes and transition models untouched — new code wraps them.

**Step 3: Update tach.toml**

```toml
[[modules]]
path = "tmgg.diffusion"
depends_on = ["tmgg.data", "tmgg.models"]
```

**Step 4: Run tests, verify pass, run full suite**

**Step 5: Commit**

```
feat: add NoiseProcess hierarchy (continuous + categorical)
```

---

### Task 3: Unified NoiseSchedule

Merge `PredefinedNoiseScheduleDiscrete` and the generative module's schedule lookup into a single `NoiseSchedule` interface.

**Files:**
- Create: `src/tmgg/diffusion/schedule.py`
- Test: `tests/diffusion/test_schedule.py` (create)

**Step 1: Write tests**

- `NoiseSchedule(schedule_type, timesteps)` constructs schedule
- `schedule.get_noise_level(t_int)` returns noise level at integer timestep
- `schedule.get_alpha_bar(t_int)` returns cumulative product
- Supports "cosine", "linear" schedule types
- Works for both discrete diffusion (large T) and denoising (T=1 with explicit noise levels)

**Step 2: Implement**

Thin wrapper that delegates to existing `PredefinedNoiseScheduleDiscrete` internals. For continuous experiments, also support the simpler `noise_levels` list lookup currently in `GenerativeLightningModule`.

**Step 3: Run tests, full suite, commit**

```
feat: add unified NoiseSchedule
```

---

### Task 4: GraphEvaluator

Merge `MMDEvaluator` + `SamplingEvaluator` into one class that always computes all metrics. `graph-tool` becomes a hard import.

**Files:**
- Create: `src/tmgg/experiments/_shared_utils/graph_evaluator.py`
- Test: `tests/experiment_utils/test_graph_evaluator.py` (create)

**Step 1: Write tests**

- `GraphEvaluator(eval_num_samples, kernel, sigma)` constructs
- `accumulate(graph: nx.Graph)` stores refs
- `evaluate(generated: list[nx.Graph])` returns `EvaluationResults` with all fields: degree_mmd, clustering_mmd, spectral_mmd, orbit_mmd, sbm_accuracy, planarity_accuracy, uniqueness, novelty
- `setup(train_graphs)` stores training graphs for novelty computation
- `clear()` resets state
- One flat results dataclass (no nesting)
- `graph-tool` import at module level

**Step 2: Implement**

Combine logic from `MMDEvaluator` (accumulation lifecycle) and `SamplingEvaluator` (structural metrics). One class, always computes everything. Move `graph_tool` import to top of file.

**Step 3: Run tests, full suite, commit**

```
feat: add GraphEvaluator merging MMD + structural metrics
```

---

### Task 5: Sampler hierarchy

Create `CategoricalSampler` (relocated from DiscreteDiffusionLM) and `ContinuousSampler` (new Gaussian posterior math).

**Files:**
- Create: `src/tmgg/diffusion/sampler.py`
- Test: `tests/diffusion/test_sampler.py` (create)

**Step 1: Write tests**

- `CategoricalSampler(noise_process, schedule)` — asserts `isinstance(noise_process, CategoricalNoiseProcess)` at construction, raises `TypeError` otherwise
- `ContinuousSampler(noise_process, schedule)` — same for `ContinuousNoiseProcess`
- `CategoricalSampler.sample(model, num_graphs=4, num_nodes=10)` returns `list[GraphData]` of length 4
- `ContinuousSampler.sample(model, num_graphs=4, num_nodes=10)` returns `list[GraphData]` of length 4
- Generated graphs have correct shapes
- Smoke test: generated graphs are valid (symmetric edges, zero diagonal)

**Step 2: Implement**

`CategoricalSampler.sample()`: Relocate logic from `DiscreteDiffusionLightningModule.sample_batch()` and `_sample_p_zs_given_zt()`. Convert to use `NoiseProcess.get_posterior()`.

`ContinuousSampler.sample()`: New implementation. Start from noise process limit distribution, iterate reverse steps using Gaussian posterior. At each step: predict x_0 via model, compute posterior p(z_s | z_t, x_0), sample z_s. Threshold and symmetrize at the end.

**Step 3: Run tests, full suite, commit**

```
feat: add Sampler hierarchy (categorical + continuous)
```

---

## Phase 2: Model Layer Unification

Migrate all models to the `GraphModel` interface. Update the factory.

---

### Task 6: Unify transformer wrappers

Merge `GraphTransformer` and `DiscreteGraphTransformer` into one class. Remove `assume_adjacency_input` from `_GraphTransformer`.

**Files:**
- Modify: `src/tmgg/models/digress/transformer_model.py`
- Delete: `src/tmgg/models/digress/discrete_transformer.py`
- Modify: `src/tmgg/models/factory.py`
- Modify: `tests/models/test_discrete_transformer.py`
- Create: `tests/models/test_unified_transformer.py`

**Step 1: Write tests for unified wrapper**

- Accepts `GraphData` with 2-class edge features (binary adjacency), returns `GraphData`
- Accepts `GraphData` with >2-class edge features (categorical), returns `GraphData`
- Eigenvector extraction works for both cases
- With `use_eigenvectors=True`: concatenates eigenvectors with X features
- Without eigenvectors: passes X features through directly
- Registered in factory under `"graph_transformer"` (replaces both `"discrete_graph_transformer"` and any direct instantiation)

**Step 2: Implement unified `UnifiedGraphTransformer(GraphModel)`**

- Takes `GraphData`, extracts adjacency from `E` for eigenvector computation if needed
- Always passes `(X, E, y, node_mask)` to `_GraphTransformer` core
- Remove `assume_adjacency_input` flag and its conditional branch from `_GraphTransformer.forward()`
- `_GraphTransformer.forward()` always receives node features as X and edge features from E
- Delete `discrete_transformer.py`
- Register as `"graph_transformer"` in factory

**Step 3: Update existing transformer tests, run full suite, commit**

```
refactor: unify GraphTransformer + DiscreteGraphTransformer into single class
```

---

### Task 7: Migrate non-transformer models to GraphModel

Update GNN, spectral, baseline, hybrid, attention models to implement `GraphModel.forward(GraphData, t) → GraphData`. Update their internals to work with `GraphData.E` directly.

**Files:**
- Modify: `src/tmgg/models/spectral_denoisers/base_spectral.py` and all spectral subclasses
- Modify: `src/tmgg/models/gnn/gnn.py`, `gnn_sym.py`, `nvgnn.py`
- Modify: `src/tmgg/models/baselines/linear.py`, `mlp.py`
- Modify: `src/tmgg/models/hybrid/hybrid.py`
- Modify: `src/tmgg/models/attention/attention.py`
- Modify: `src/tmgg/models/factory.py` (return type: `GraphModel`)
- Modify: all corresponding test files

**Step 1: Update SpectralDenoiser base and subclasses**

`SpectralDenoiser` base: change from `AdjacencyDenoisingModel` to `GraphModel`. `forward(data: GraphData, t) → GraphData`. Internally: spectral models need eigendecomposition, which they compute from the edge features. Update all subclasses (FilterBank, LinearPE, SelfAttention, Bilinear variants).

**Step 2: Update GNN models**

`GNN`, `GNNSymmetric`, `NodeVarGNN`: natural fit for GraphData — they already do message passing. Update to read node/edge features from `data.X`, `data.E`.

**Step 3: Update baselines**

`LinearBaseline`, `MLPBaseline`: operate on flattened edge features `data.E`.

**Step 4: Update hybrid and attention**

`SequentialDenoisingModel`, `MultiLayerAttention`: compose other models, so they follow the same pattern.

**Step 5: Update factory return types**

`create_model()` returns `GraphModel` instead of `DenoisingModel`. Update all factory functions.

**Step 6: Update all model tests to use GraphData**

Every test that constructs a raw adjacency tensor and passes it to a model needs to wrap it in `GraphData.from_adjacency()` first, and unwrap the result.

**Step 7: Run full test suite, commit**

```
refactor: migrate all models to GraphModel interface with GraphData
```

---

## Phase 3: Training Loop Unification

Build the new LightningModule hierarchy.

---

### Task 8: BaseGraphModule

Shared infrastructure for all graph learning experiments.

**Files:**
- Create: `src/tmgg/experiments/_shared_utils/base_graph_module.py`
- Test: `tests/experiment_utils/test_base_graph_module.py` (create)

**Step 1: Write tests**

- `_make_parametrized_model()` hook creates model via factory from `model_type` + `model_config` hparams
- `configure_optimizers()` delegates to `configure_optimizers_from_config()` with stored params
- `transfer_batch_to_device()` moves `GraphData` to device
- `on_fit_start()` logs parameter count
- `get_model_name()` returns `model_type` by default
- `get_model_config()` delegates to `model.get_config()`

**Step 2: Implement**

Extract shared logic from current `DenoisingLightningModule.__init__` (optimizer params, model creation via `_make_parametrized_model()`), `configure_optimizers` (from any of the 3 copies), `transfer_batch_to_device`, `on_fit_start`. Pure infrastructure, no training logic.

**Step 3: Run tests, full suite, commit**

```
feat: add BaseGraphModule with shared LightningModule infrastructure
```

---

### Task 9: DiffusionModule

The unified training loop for multi-step diffusion (both categorical and continuous).

**Files:**
- Create: `src/tmgg/experiments/_shared_utils/diffusion_module.py`
- Test: `tests/experiment_utils/test_diffusion_module.py` (create)

**Step 1: Write tests**

Training loop tests:
- `training_step` samples timestep, applies noise via `noise_process`, forwards through model, computes loss
- Loss types: "mse", "bce", "cross_entropy" all work
- `validation_step` accumulates refs in evaluator
- `on_validation_epoch_end` generates via sampler, evaluates via evaluator, logs results
- With `CategoricalNoiseProcess`: VLB components are computed and logged during validation
- Without sampler (`sampler=None`): no generative eval at epoch end
- `setup()` calls `noise_process.setup(datamodule)` and `evaluator.setup(train_graphs)`

Integration tests:
- Construct `DiffusionModule` with `ContinuousNoiseProcess` + `ContinuousSampler` + mock model → training step runs
- Construct with `CategoricalNoiseProcess` + `CategoricalSampler` + mock model → training step runs
- Full Lightning `Trainer.fit()` smoke test (1 epoch, tiny data)

**Step 2: Implement**

```python
class DiffusionModule(BaseGraphModule):
    def __init__(self, *, model_type, model_config,
                 noise_process: NoiseProcess, schedule: NoiseSchedule,
                 loss_type: str, sampler: Sampler | None,
                 evaluator: GraphEvaluator,
                 eval_noise_levels: list[float] | None = None,
                 diffusion_steps: int, ...optimizer_params...): ...

    def training_step(self, batch: GraphData, batch_idx: int) -> dict:
        t = self._sample_timestep(batch.X.size(0))
        noisy = self.noise_process.apply(batch, t)
        prediction = self.model(noisy, t)
        loss = self._compute_loss(prediction, batch)
        ...

    def validation_step(self, batch: GraphData, batch_idx: int) -> dict:
        self.evaluator.accumulate_from_graphdata(batch)
        if isinstance(self.noise_process, CategoricalNoiseProcess):
            self._accumulate_vlb(batch)
        ...

    def on_validation_epoch_end(self) -> None:
        if self.sampler is not None:
            generated = self.sampler.sample(self.model, ...)
            results = self.evaluator.evaluate(generated_as_networkx)
            self._log_eval_results("val", results)
        if isinstance(self.noise_process, CategoricalNoiseProcess):
            self._log_vlb("val")
```

**Step 3: Run tests, full suite, commit**

```
feat: add DiffusionModule — unified multi-step diffusion training loop
```

---

### Task 10: SingleStepDenoisingModule

Semantic subclass that hardcodes T=1 simplifications.

**Files:**
- Create: `src/tmgg/experiments/_shared_utils/denoising_module.py`
- Test: `tests/experiment_utils/test_denoising_module.py` (create)

**Step 1: Write tests**

- Inherits from `DiffusionModule`
- `T=1` hardcoded, `sampler=None` hardcoded
- `training_step`: samples noise level from configured set (not timestep from schedule)
- `validation_step`: evaluates at each `eval_noise_levels`, logs per-level loss + accuracy
- Spectral delta logging when `log_spectral_deltas=True`
- No generative sampling at epoch end
- Full `Trainer.fit()` smoke test

**Step 2: Implement**

Override `training_step` to skip timestep sampling (sample noise level directly), override `validation_step` to iterate over `eval_noise_levels`, override epoch-end to skip sampling.

Relocate spectral delta logic from `DenoisingLightningModule._log_spectral_deltas()` and visualization hooks from `_log_visualizations()`.

**Step 3: Run tests, full suite, commit**

```
feat: add SingleStepDenoisingModule for single-step denoising experiments
```

---

## Phase 4: Experiment Migration & Dead Code Removal

Switch all experiments to the new modules, then remove everything old.

---

### Task 11: Migrate denoising experiments

Update configs for spectral, GNN, baseline, hybrid, DiGress denoising experiments.

**Files:**
- Modify: `src/tmgg/experiments/exp_configs/` — all denoising config YAML files
- Modify: experiment `__init__.py` files to re-export from new module
- Modify: tests that instantiate denoising LightningModules

**Step 1: Update configs**

Each denoising config changes `_target_` from the experiment-specific LightningModule to `tmgg.experiments._shared_utils.denoising_module.SingleStepDenoisingModule`, with component params (noise_type, model_type, model_config, loss_type, eval_noise_levels, etc.).

**Step 2: Update tests**

Tests that directly instantiate `SpectralDenoisingLightningModule`, `GNNDenoisingLightningModule`, etc. switch to instantiating `SingleStepDenoisingModule` with appropriate component config.

**Step 3: Verify all denoising experiments work**

Run the full test suite. If specific integration tests exist for denoising experiments, run those explicitly.

**Step 4: Commit**

```
refactor: migrate all denoising experiments to SingleStepDenoisingModule
```

---

### Task 12: Migrate generative experiments

Update configs for gaussian diffusion and discrete diffusion generative experiments.

**Files:**
- Modify: `src/tmgg/experiments/exp_configs/` — generative config YAML files
- Modify: tests that instantiate generative LightningModules

**Step 1: Update configs**

Gaussian generative: `_target_` → `DiffusionModule`, with `ContinuousNoiseProcess`, `ContinuousSampler`, `GraphEvaluator`.

Discrete generative: `_target_` → `DiffusionModule`, with `CategoricalNoiseProcess`, `CategoricalSampler`, `GraphEvaluator`.

**Step 2: Update tests, verify, commit**

```
refactor: migrate generative experiments to DiffusionModule
```

---

### Task 13: Remove dead code

Remove all old LightningModules, old model base classes, old evaluators, heuristic samplers.

**Files to delete:**
- `src/tmgg/experiments/_shared_utils/base_lightningmodule.py` (700 lines)
- `src/tmgg/experiments/_shared_utils/mmd_evaluator.py` (127 lines)
- `src/tmgg/experiments/_shared_utils/sampling_evaluator.py` (194 lines)
- `src/tmgg/experiments/digress_denoising/lightning_module.py` (316 lines)
- `src/tmgg/experiments/gnn_denoising/lightning_module.py` (112 lines)
- `src/tmgg/experiments/spectral_arch_denoising/lightning_module.py` (182 lines)
- `src/tmgg/experiments/lin_mlp_baseline_denoising/lightning_module.py` (126 lines)
- `src/tmgg/experiments/gnn_transformer_denoising/lightning_module.py` (127 lines)
- `src/tmgg/experiments/gaussian_diffusion_generative/lightning_module.py` (482 lines)
- `src/tmgg/experiments/discrete_diffusion_generative/lightning_module.py` (1014 lines)
- `src/tmgg/models/digress/discrete_transformer.py` (153 lines, already deleted in Task 6)

**Classes to remove from base.py:**
- `DenoisingModel`
- `AdjacencyDenoisingModel`
- `CategoricalDenoisingModel`

**Methods to remove if no remaining callers:**
- `transform_for_loss()`, `logits_to_graph()`, `predict()`, `_zero_diagonal()` — check callers first. If sampler/evaluator handle the thresholding/symmetry/diagonal logic, these are dead.

**Step 1: Remove old LightningModule files**

Delete the files listed above. Update any `__init__.py` re-exports.

**Step 2: Remove old model base classes from base.py**

Remove `DenoisingModel`, `AdjacencyDenoisingModel`, `CategoricalDenoisingModel`. Only `BaseModel` and `GraphModel` remain.

**Step 3: Remove old evaluator classes**

Delete `mmd_evaluator.py` and `sampling_evaluator.py`. Update imports everywhere.

**Step 4: Check for and remove orphaned utility methods**

Search for callers of `logits_to_graph`, `predict`, `_zero_diagonal`, `transform_for_loss`. Remove any that have zero callers.

**Step 5: Update `__init__.py` files and re-exports**

`src/tmgg/models/__init__.py`, `src/tmgg/experiments/__init__.py`, etc.

**Step 6: Remove or update stale tests**

Tests that tested deleted classes: `test_base_lightningmodule.py` (588 lines), `test_mmd_evaluator.py` (159 lines), `test_discrete_transformer.py`. Either delete or rewrite to test the replacement.

**Step 7: Run full test suite — must pass clean**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```

**Step 8: Run basedpyright**

```bash
uv run basedpyright src/tmgg/
```

**Step 9: Commit**

```
refactor: remove old LightningModules, model base classes, evaluators (~3000 lines)
```

---

### Task 14: Final cleanup

Polish pass: update imports, tach boundaries, documentation.

**Files:**
- Modify: `tach.toml` — update dependencies for new module structure
- Modify: `src/tmgg/models/__init__.py` — export `GraphModel` instead of old classes
- Modify: `src/tmgg/experiments/__init__.py` — export new modules
- Modify: `docs/extending.md`, `docs/experiments.md` if they reference old classes
- Review: `data/noise.py` — if `NoiseGenerator` classes are now only used internally by `ContinuousNoiseProcess`, consider whether they should be private

**Step 1: Update tach.toml boundaries**

Remove dependencies on deleted modules, add `tmgg.diffusion`.

**Step 2: Update __init__.py exports**

Ensure public API exports `GraphModel`, `BaseGraphModule`, `DiffusionModule`, `SingleStepDenoisingModule`, `NoiseProcess`, `Sampler`, `GraphEvaluator`.

**Step 3: Update documentation**

Any docs referencing old class names or architecture.

**Step 4: Final full test suite + basedpyright + tach check**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
uv run basedpyright src/tmgg/
tach check
```

**Step 5: Commit**

```
chore: final cleanup — tach boundaries, exports, documentation
```

---

## Execution Notes

**Parallelizable tasks:** Tasks 1-5 (Phase 1) have no dependencies on each other and can run in parallel via subagents. However, Tasks 2, 3, and 5 share the `tmgg.diffusion` package, so coordinate file creation.

**Critical ordering:** Task 6 depends on Task 1 (GraphModel). Task 7 depends on Tasks 1 and 6. Tasks 8-10 depend on Tasks 1-5. Tasks 11-12 depend on Tasks 8-10. Task 13 depends on Tasks 11-12. Task 14 is last.

**Risk points:**
- Task 7 (migrating all models) touches the most files and is most likely to surface unexpected breakage
- Task 9 (DiffusionModule) is the largest single implementation — the training loop unification
- Task 5 (ContinuousSampler) requires new math — Gaussian posterior for binary data

**Estimated total:** ~14 tasks, each 30-90 minutes depending on scope. Phases 1-2 are foundation work. Phase 3 is the core logic. Phase 4 is cleanup.
