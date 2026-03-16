# Evaluator + Sampling API Unification

**Date:** 2026-03-15
**Branch:** `cleanup-2`
**Findings:** G50 (dual `eval_num_samples`), plus architectural cleanup of evaluation and sampling paths

---

## Problem

The evaluation architecture has two independent paths that do conceptually the same thing — compare model-produced graphs against reference graphs — through completely different plumbing:

**Generative (DiffusionModule):** Accumulates validation-set graphs into `GraphEvaluator.ref_graphs` during `validation_step` via `evaluator.accumulate()`. At epoch end, generates graphs via `sampler.sample()` (full reverse chain from noise), calls `evaluator.evaluate(generated)`, clears.

**Denoising (SingleStepDenoisingModule):** Never calls `accumulate()`. Maintains its own `_clean_graphs_by_eps` and `_denoised_graphs_by_eps` dicts with its own `_eval_num_samples` cap. At epoch end, directly overwrites `evaluator.ref_graphs = clean_list` per noise level, calls `evaluator.evaluate(denoised)`.

This creates several problems:
- `eval_num_samples` exists in two places (evaluator constructor + denoising module constructor) with identical values, documented as G50.
- `GraphEvaluator.accumulate()` is dead code in the denoising path.
- The denoising module mutates evaluator internals (`evaluator.ref_graphs = ...`).
- `setup(train_graphs)` is a stateful init-after-init smell.
- The `Sampler` API is hardcoded to start from pure noise — no way to start from an arbitrary timestep or collect per-step metrics during the reverse chain.
- Validation-set reference graphs are accumulated graph-by-graph during `validation_step` when the datamodule already holds them.

## Design

### Part 1: GraphEvaluator — stateless metric computer

**Remove** all accumulation and lifecycle state:
- Delete `accumulate()`, `clear()`, `ref_graphs` attribute.
- Delete `setup()` — pass `train_graphs` to the constructor instead.

**Constructor:**

```python
def __init__(
    self,
    eval_num_samples: int,
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 1.0,
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    skip_metrics: set[str] | frozenset[str] | list[str] | None = None,
    train_graphs: list[nx.Graph] | None = None,
) -> None:
```

`eval_num_samples` stays on the evaluator — it owns the "how many graphs to compare" policy. Callers pass all available graphs; the evaluator truncates internally.

`train_graphs` replaces `setup()`. Passed once at construction for novelty computation. No mutable state, no two-phase initialization.

**`evaluate()` signature:**

```python
def evaluate(
    self,
    refs: list[nx.Graph],
    generated: list[nx.Graph],
) -> EvaluationResults | None:
```

Internally caps both lists to `self.eval_num_samples` before computing metrics. Returns `None` if either truncated list has fewer than 2 graphs.

**Dependency checks:** ORCA availability (`_orca_is_available()`) becomes a module-level `@cache`'d constant, matching `_GRAPH_TOOL_AVAILABLE` which is already module-level. Warnings fire once at import time, not per-instance. No `_warn` flag needed — constructing 100 evaluators has zero warning overhead.

**`eval_num_samples` lives here only.** Neither `DiffusionModule` nor `SingleStepDenoisingModule` has this parameter. The YAML config sets it on the evaluator's `_target_` block. One value, one location.

### Part 2: Unified Sampler API

Extend `Sampler.sample()` to accept an optional starting point and per-step metric collector:

```python
class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        model: GraphModel,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
        *,
        start_from: GraphData | None = None,
        start_timestep: int | None = None,
        collector: StepMetricCollector | None = None,
    ) -> list[GraphData]:
        ...
```

**When `start_from` is None** (default): sample from the marginal/limit distribution at `t=T`, as today.

**When `start_from` is provided** with `start_timestep`: begin the reverse chain from that point.

**`StepMetricCollector` protocol:**

```python
class StepMetricCollector(Protocol):
    def record(self, t: int, s: int, metrics: dict[str, float]) -> None:
        """Record per-step metrics during reverse diffusion.

        Parameters
        ----------
        t
            Current timestep (before this step).
        s
            Target timestep (after this step).
        metrics
            Step-level metrics (e.g., KL divergence, log-likelihood
            contribution, reconstruction error).
        """
        ...
```

**Concrete implementation: `DiffusionLikelihoodCollector`.** Collects per-step KL divergence and log-likelihood contributions during the full reverse chain. At chain completion, computes the full variational lower bound (VLB) via Bayesian composition of per-step likelihoods. This replaces the current single-random-timestep VLB estimator in `DiffusionModule` (five manual accumulator lists) with a proper full-chain computation that can be run at evaluation time.

The collector is wired into `CategoricalSampler.sample()` and `ContinuousSampler.sample()`. At each reverse step, the sampler calls `collector.record(t, s, {"kl": ..., "log_likelihood": ...})` with the step's KL divergence.

**SingleStepDenoisingModule** does not use `Sampler`. Since T=1, its "generation" is just `model.forward()`. The `Sampler` abstraction exists for iterative reverse chains; forcing single-step through it adds indirection for no benefit.

### Part 3: No accumulation during validation steps

The key insight: **neither path needs to accumulate graphs during `validation_step`.**

**Generative path:** Currently accumulates validation-set graphs as "references" during `validation_step`, then generates at epoch end. But the validation set is already in the datamodule — no need to re-collect it batch by batch.

**Denoising path:** Currently accumulates (clean, denoised) pairs per noise level during `validation_step`. But the clean graphs are in the datamodule, and the denoised graphs can be produced on-demand at epoch end by running forward passes on a subset of validation data at each noise level. The forward pass cost is bounded by `eval_num_samples` (typically 128), which is small relative to a training epoch.

Both paths become: **`validation_step` computes loss only; `on_validation_epoch_end` produces everything for distributional evaluation on-demand.**

#### Datamodule accessor

Add a concrete public method to `BaseGraphDataModule`:

```python
def get_reference_graphs(self, stage: str, max_graphs: int) -> list[nx.Graph]:
    """Extract up to max_graphs from a dataset split as NetworkX graphs.

    Parameters
    ----------
    stage
        One of "val" or "test".
    max_graphs
        Maximum number of graphs to return.

    Returns
    -------
    list[nx.Graph]
        NetworkX graphs extracted from the specified split.

    Raises
    ------
    RuntimeError
        If the datamodule has not been set up yet.
    """
```

Default implementation iterates the appropriate dataloader (`val_dataloader()` / `test_dataloader()`), extracts adjacency matrices from `GraphData` batches, converts to NetworkX, and stops after `max_graphs`. This works generically for all datamodule subclasses since they all produce `GraphData` batches. Subclasses can override for efficiency (e.g., reading directly from `_val_data` tensors without going through the dataloader), but the default is correct for all.

#### DiffusionModule — epoch-end only

**Before:**
1. `validation_step`: compute loss + accumulate validation graphs
2. `on_validation_epoch_end`: generate graphs, evaluate, clear

**After:**
1. `validation_step`: compute loss only. VLB estimation at random timestep stays here (it's a per-batch computation, not a full-chain one).
2. `on_validation_epoch_end`:
   - Skip if `global_step % eval_every_n_steps != 0` (bare `return`, no `clear()` needed).
   - Pull refs via `self.trainer.datamodule.get_reference_graphs("val", self.evaluator.eval_num_samples)`.
   - Generate graphs via `generate_graphs(len(refs))`.
   - Call `self.evaluator.evaluate(refs=refs, generated=generated)`.
   - Log metrics.

**`on_test_epoch_end`:** Same but `get_reference_graphs("test", ...)`.

#### SingleStepDenoisingModule — epoch-end only

**Before:**
1. `_val_or_test`: forward pass + loss + accumulate (clean, denoised) pairs per noise level
2. `on_validation_epoch_end`: loop over noise levels, hack evaluator, evaluate, average, clear

**After:**
1. `_val_or_test` / `validation_step`: compute per-noise-level loss and per-batch metrics only. No graph accumulation.
2. `on_validation_epoch_end`:
   - Pull reference (clean) graphs from the datamodule once.
   - For each noise level:
     - Apply noise to the reference graphs (up to `eval_num_samples`).
     - Run forward pass to get denoised predictions.
     - Convert denoised predictions to NetworkX graphs.
     - Call `evaluator.evaluate(refs=ref_graphs, generated=denoised_graphs)`.
     - Log per-noise-level metrics.
   - Average across noise levels, log.

This eliminates `_clean_graphs_by_eps`, `_denoised_graphs_by_eps`, `_accumulate_graphs()`, and `_eval_num_samples` entirely from the denoising module. The forward pass at epoch end costs `eval_num_samples × len(noise_levels)` inferences — typically 128 × 5 = 640, a fraction of a training epoch.

**`on_test_epoch_end`:** Same but reads from `"test"` split.

### Part 4: eval_num_samples — single owner

`eval_num_samples` lives on `GraphEvaluator` only. The YAML configs set it on the evaluator's `_target_` block. No Lightning module has this parameter. The evaluator internally truncates `refs` and `generated` in `evaluate()`.

The denoising YAML currently has both `model.evaluator.eval_num_samples` and `model.eval_num_samples`. After this change, only `model.evaluator.eval_num_samples` exists. `model.eval_num_samples` is deleted from all configs.

### Part 5: VLB collector

The current VLB estimation uses five manual lists populated at a single random timestep per validation batch. This gives an unbiased but high-variance estimate of the full VLB.

The `DiffusionLikelihoodCollector` enables a proper full-chain VLB computation: run the complete reverse diffusion, collect per-step KL at every timestep, sum to get the exact VLB for those specific graphs. This can be run at evaluation time (not every validation step — it's T× more expensive).

The existing single-timestep VLB estimation in `validation_step` can coexist or be replaced. Keeping it as a cheap per-epoch diagnostic while using the collector for periodic exact evaluation is reasonable.

---

## Files to modify

| File | Change |
|------|--------|
| `evaluation_metrics/graph_evaluator.py` | Remove `accumulate()`, `clear()`, `ref_graphs`, `setup()`. Accept `train_graphs` in constructor. `evaluate()` takes `refs` + `generated`, truncates to `eval_num_samples`. Make ORCA check module-level `@cache`. |
| `data/data_modules/base_data_module.py` | Add concrete `get_reference_graphs(stage, max_graphs)` method using dataloader iteration. |
| `lightning_modules/diffusion_module.py` | Remove graph accumulation from `validation_step`. Epoch-end pulls refs from datamodule, generates, evaluates. Remove `eval_num_samples` param. Remove stale `clear()` calls. |
| `lightning_modules/denoising_module.py` | Remove `_clean_graphs_by_eps`, `_denoised_graphs_by_eps`, `_accumulate_graphs()`, `_eval_num_samples`. Epoch-end pulls refs from datamodule, applies noise + forward pass, evaluates per noise level. |
| `diffusion/sampler.py` | Add `start_from`, `start_timestep`, `collector` to `Sampler.sample()`. Implement in `CategoricalSampler` and `ContinuousSampler`. Add `StepMetricCollector` protocol. |
| `diffusion/collectors.py` (new) | `DiffusionLikelihoodCollector` implementing `StepMetricCollector`. |
| `exp_configs/task/denoising.yaml` | Remove `model.eval_num_samples`. Keep `model.evaluator.eval_num_samples`. |
| `exp_configs/base_config_gaussian_diffusion.yaml` | Keep `eval_num_samples` on evaluator only. |
| `exp_configs/models/discrete/*.yaml` | Keep `eval_num_samples` on evaluator only. |

## Test changes

| Test file | Change |
|-----------|--------|
| `test_graph_evaluator.py` | Delete `TestAccumulation` (4 tests). Update `evaluate()` calls to pass `refs` + `generated`. Add test for internal truncation to `eval_num_samples`. Remove `setup()` tests, verify `train_graphs` constructor arg. |
| `test_diffusion_module.py` | Delete `test_accumulates_reference_graphs`. Rework epoch-end tests to verify datamodule-based ref retrieval. Add `on_test_epoch_end` test. |
| `test_denoising_module.py` | Remove tests for `_clean_graphs_by_eps`, `_denoised_graphs_by_eps`, `_accumulate_graphs()`. Add tests verifying epoch-end forward pass produces distributional metrics. |
| `test_data_modules.py` (new or extended) | Test `get_reference_graphs()` on each datamodule: not-set-up error, dataset smaller than max_graphs, val vs test stage. |
| `test_collectors.py` (new) | Test `DiffusionLikelihoodCollector` per-step recording and VLB summary. |

## What stays the same

- `EvaluationResults` dataclass — unchanged.
- All metric computation functions (`compute_mmd_metrics`, `compute_orbit_mmd`, `compute_sbm_accuracy`, etc.) — unchanged.
- Metric config parameters (`kernel`, `sigma`, `p_intra`, `p_inter`, `skip_metrics`) — unchanged.
- `generate_graphs()` on `DiffusionModule` — unchanged (wraps `sampler.sample()`, the only call site).

## Migration risk

- **Breaking change to `GraphEvaluator` API.** `accumulate()`, `clear()`, `setup()` removed; `evaluate()` signature changes. All callers are in the two Lightning modules — no external consumers.
- **Breaking change to `Sampler.sample()` signature.** New keyword-only parameters with defaults, so `generate_graphs()` (the only caller) is unaffected.
- **YAML configs:** `model.eval_num_samples` removed from denoising configs. `model.evaluator.eval_num_samples` stays. Hydra will error loudly on the removed key if old configs are used (good).
- **Datamodule subclasses:** `get_reference_graphs()` has a concrete default implementation, so existing subclasses work without changes. Only override if efficiency matters.
