# Training Loop & Model Unification Design

**Date:** 2026-03-02
**Status:** Approved
**Scope:** Merge the three independent LightningModule inheritance trees, unify the model interface around GraphData, and make training regime / noise framework / architecture fully orthogonal.

## Problem

Path-dependent development produced three separate training loop implementations that share most of their logic but duplicate it independently:

1. **DenoisingLightningModule** (abstract base, 700 lines) with 5 subclasses (spectral, GNN, baseline, hybrid, DiGress denoising) — single-step adjacency-based denoising.
2. **DiscreteDiffusionLightningModule** (standalone, 984 lines) — multi-step categorical diffusion with principled ancestral sampling.
3. **GenerativeLightningModule** (standalone, 482 lines) — multi-step continuous diffusion with heuristic sampling.

These duplicate: `configure_optimizers` (3 copies), `transfer_batch_to_device` (3 copies), `on_fit_start` (3 copies), MMD evaluation wiring (3 copies), and sampling code (2 heuristic + 1 principled).

The model layer has a parallel split: `AdjacencyDenoisingModel` (tensor in/out) vs `CategoricalDenoisingModel` (GraphData in/out), with two transformer wrappers (`GraphTransformer` and `DiscreteGraphTransformer`) around the same `_GraphTransformer` core.

## Design Principles

- **Remove code, don't add abstractions.** Every component exists because it eliminates duplication, not because it looks clean on a diagram.
- **Three orthogonal axes.** Training regime (multi-step diffusion vs single-step denoising), noise framework (categorical vs continuous), and architecture (transformer, GNN, spectral, etc.) are independently selectable.
- **GraphData everywhere.** One data format flows through the entire system. No raw adjacency tensors in training loops or model interfaces.
- **Fail loud.** No graceful fallback, no `hasattr` checks, no silent `None` returns. Class hierarchy and `isinstance` for type dispatch. Missing dependencies crash at import time.

## Architecture

### Data Format: GraphData as Universal Representation

`GraphData` (already exists: `X`, `E`, `y`, `node_mask`) becomes the only data format. Binary adjacency graphs use 2-class categorical encoding (`de=2`: no-edge/edge). Continuous noise produces soft (non-one-hot) edge features, which models handle naturally through MLP processing.

DataModules always yield `GraphData`. The `to_adjacency()` method stays for visualization, NetworkX conversion, and evaluation output only.

### Unified Model Interface

All model base classes (`DenoisingModel`, `AdjacencyDenoisingModel`, `CategoricalDenoisingModel`) are replaced by a single contract:

```python
class GraphModel(BaseModel):
    @abstractmethod
    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData: ...
```

`BaseModel` (parameter counting + `get_config()`) stays as the root.

**Transformer unification:** `GraphTransformer` and `DiscreteGraphTransformer` merge into one class. The `assume_adjacency_input` flag and its conditional branch inside `_GraphTransformer.forward()` disappear — one code path. Eigenvectors are always concatenated with whatever node features `X` contains. Registered in the factory under a single name.

**Non-transformer models** (GNN, spectral, baseline, hybrid): internal math updated to operate on `GraphData.E` fields directly. No `to_adjacency()` conversion inside model `forward()`.

**Removed:** `transform_for_loss()` (was identity), `logits_to_graph()`, `predict()`, `_zero_diagonal()` become standalone utility functions — they're graph operations, not model concerns.

### Noise Process Hierarchy

Two currently separate noise systems merge into one hierarchy:

```
NoiseProcess (abstract base)
    apply(data: GraphData, t: Tensor) -> GraphData
    get_posterior(z_t, z_0, t, s) -> ...       # for principled sampling

ContinuousNoiseProcess(NoiseProcess)
    # Wraps existing generators (Gaussian, EdgeFlip, DiGress, Rotation, Logit)
    # Internal math unchanged, interface widened to GraphData

CategoricalNoiseProcess(NoiseProcess)
    # Wraps existing transition models (DiscreteUniform, MarginalUniform)
    # VLB methods: kl_prior(), compute_Lt(), reconstruction_logp()
    # setup(datamodule) for deferred initialization from marginals
```

The existing `NoiseGenerator` subclasses and `DiscreteUniformTransition` / `MarginalUniformTransition` keep their internal math but implement the unified interface. `PredefinedNoiseScheduleDiscrete` and the generative module's separate schedule merge into a single `NoiseSchedule` abstraction (a `t -> noise_level` mapping).

### Sampler Hierarchy

Drop the heuristic interpolation sampler (no theoretical grounding). Keep only principled ancestral sampling (DiGress-style), adapted for both categorical and continuous cases:

```
Sampler (abstract base)
    __init__(noise_process, schedule)   # validates compatibility at construction
    sample(model, num_graphs, num_nodes) -> list[GraphData]

CategoricalSampler(Sampler)
    # Asserts isinstance(noise_process, CategoricalNoiseProcess)
    # Relocated from DiscreteDiffusionLightningModule.sample_batch()

ContinuousSampler(Sampler)
    # Asserts isinstance(noise_process, ContinuousNoiseProcess)
    # New: Gaussian posterior math for principled continuous reverse sampling
```

Construction-time validation ensures incompatible noise process / sampler combinations fail immediately, not mid-training.

### Loss

No new hierarchy. Training step applies a standard PyTorch loss (MSE, BCE, cross-entropy) to GraphData fields, masked by `node_mask`. Configured by string. `TrainLossDiscrete` (masked cross-entropy with edge weighting) stays as a function.

### Evaluation

`MMDEvaluator` and `SamplingEvaluator` merge into a single `GraphEvaluator` class. Always computes all metrics (degree, clustering, spectral, orbit, SBM, planarity, uniqueness, novelty). No flags — one class, one results dataclass with all fields. `graph-tool` becomes a hard dependency (import at module level, crash immediately if missing).

The evaluator uses the accumulate-then-evaluate lifecycle: accumulate reference graphs during validation steps, generate + evaluate at epoch end. Training graphs for novelty come from the datamodule at `setup()` time.

VLB computation is not part of the evaluator — it's methods on `CategoricalNoiseProcess`, called by the DiffusionModule's validation step when the noise process is categorical (`isinstance` check).

### Class Hierarchy

```
BaseGraphModule(pl.LightningModule)
    _make_parametrized_model()      # overridable hook, default: model factory
    configure_optimizers()           # shared, delegates to configure_optimizers_from_config
    transfer_batch_to_device()       # shared: batch.to(device)
    on_fit_start()                   # shared: log param count, scheduler info
    get_model_name() -> str          # overridable
    get_model_config() -> dict       # delegates to model.get_config()

DiffusionModule(BaseGraphModule)
    # Composes: noise_process, schedule, loss_type, sampler, evaluator
    # Handles both categorical and continuous diffusion
    training_step()                  # sample t, apply noise, forward, loss
    validation_step()                # accumulate refs, per-noise-level eval, VLB if categorical
    on_validation_epoch_end()        # generate via sampler, evaluate via evaluator
    setup()                          # deferred init for noise process + evaluator

SingleStepDenoisingModule(DiffusionModule)
    # Semantic subclass: hardcodes T=1, sampler=None
    # Simplifies: no timestep sampling, no generative sampling
    # Adds: per-noise-level reconstruction evaluation, spectral delta logging
```

### Experiment Mapping

Every current experiment becomes a Hydra config pointing at one of two classes:

| Training regime | Noise | Architecture (config) | Class |
|---|---|---|---|
| Single-step denoising | continuous (gaussian) | spectral | `SingleStepDenoisingModule` |
| Single-step denoising | continuous (gaussian) | gnn | `SingleStepDenoisingModule` |
| Single-step denoising | continuous (gaussian) | linear / mlp | `SingleStepDenoisingModule` |
| Single-step denoising | continuous (gaussian) | hybrid | `SingleStepDenoisingModule` |
| Single-step denoising | any | graph_transformer | `SingleStepDenoisingModule` |
| Multi-step diffusion | continuous (gaussian/digress/...) | any | `DiffusionModule` |
| Multi-step diffusion | categorical (uniform/marginal) | any | `DiffusionModule` |

The 7 current LightningModule subclasses become 7 Hydra config files.

## Line Count Estimate

| Category | Lines |
|---|---|
| Removed (classes, duplication, heuristic samplers) | ~3800 |
| Added (relocated logic + new code) | ~1200 |
| **Net reduction** | **~2600** |

Genuinely new code (~380 lines): `ContinuousSampler` with Gaussian posterior math (~80), noise process GraphData adapters (~100), unified transformer wrapper (~100), non-transformer model GraphData internals (~100).

## Key Decisions

1. **GraphData everywhere** — no raw adjacency tensors in training loops or models. Models operate on `GraphData.E` fields internally.
2. **Drop heuristic sampler** — principled ancestral sampling only (DiGress-style), adapted for continuous case.
3. **Single evaluator, always compute all metrics** — `graph-tool` as hard dependency. No flags, no optional metrics.
4. **Class hierarchy for noise processes** — `ContinuousNoiseProcess` vs `CategoricalNoiseProcess`. VLB methods on categorical. `isinstance` dispatch in DiffusionModule.
5. **Sampler validates noise process compatibility at construction** — fail at wiring time, not mid-training.
6. **SingleStepDenoisingModule as semantic subclass** — hardcodes T=1 simplifications rather than configuring them away.
7. **Prune utility methods aggressively** — `logits_to_graph`, `predict`, `_zero_diagonal`, `transform_for_loss` may become unnecessary once models return GraphData, since thresholding/symmetry/diagonal logic can fold into the sampler or evaluator. Don't pre-commit to keeping them as standalone utilities; remove during implementation if they have no remaining callers.
