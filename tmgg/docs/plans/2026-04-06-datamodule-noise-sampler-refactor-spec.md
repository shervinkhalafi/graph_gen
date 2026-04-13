# DataModule / NoiseProcess / Sampler Refactor Specification

**Date:** 2026-04-06  
**Status:** Approved  
**Scope:** `tmgg.data.data_modules`, `tmgg.diffusion`, `tmgg.training.lightning_modules`, Hydra experiment configs, affected tests and docs  
**Reference review:** [docs/reports/2026-04-05-datamodule-noise-interface-review.md](../reports/2026-04-05-datamodule-noise-interface-review.md)

This document is the implementation source of truth for the refactor. The review document remains the rationale and current-state analysis. When they differ, this specification wins.

## 1. Outcome

The refactor will produce four clean boundaries:

| Component | Owns | Must not own |
|---|---|---|
| `BaseGraphDataModule` and subclasses | dataset generation, splitting, dataloaders, size distribution, reference graphs | marginals, transition models, noise configuration |
| `NoiseProcess` and subclasses | schedule, forward process, reverse posterior, prior, data-dependent initialization | datamodule-specific fields, sampler loop control |
| `Sampler` | reverse-chain loop, timestep bookkeeping, batch assembly, collector hooks | transition matrices, categorical class math, Gaussian posterior math |
| `DiffusionModule` | training orchestration, loss computation, evaluation wiring | duck-typed datamodule access, subtype-specific diffusion math |

This is a breaking change. The implementation will not preserve the current public APIs, config shape, or tests that only exercise legacy helper methods.

## 2. Decisions

1. The refactor is one pass. We are not doing a staged compatibility migration. We commit the repo state before and tag it to ensure we can roll back.
2. Dead, zero-caller, and tests-only APIs are removed aggressively. Before deletion, implementation must re-run `rg` to confirm the call set has not changed.
3. There is one public reverse sampler. `CategoricalSampler` and `ContinuousSampler` are deleted.
4. `NoiseProcess` owns timestep semantics. Scheduled multi-step processes may use a private `NoiseSchedule`, but `DiffusionModule` and `Sampler` no longer accept a second `noise_schedule` object.
5. `CategoricalNoiseProcess` owns the categorical kernel. Its public config uses `limit_distribution: uniform | empirical_marginal`, not public transition-model classes.
6. `DataModule` never computes marginals for public consumption. `CategoricalNoiseProcess.initialize_from_data()` computes them from `train_dataloader()` only when `limit_distribution == "empirical_marginal"`.
7. There are no compatibility shims, no deprecated aliases, no `hasattr`/`getattr` fallbacks, and no legacy dual-path noising API.
8. A fixed-level one-step denoiser is modeled as the `T = 1` case of the same latent-corruption process family. Multi-step processes may use a schedule internally; fixed-level `T = 1` processes map `t = 1` directly to their configured corruption level.
9. Hydra configs are simplified to instantiate one `NoiseProcess` object. The sampler is created inside `DiffusionModule`; configs do not instantiate a second noise process for the sampler.
10. `GraphData` remains the shared transport container in this refactor, but adjacency access splits by meaning: binary-topology accessors are distinct from lossless dense edge-state accessors. We are not introducing `ContinuousGraphData` yet; this accessor split is the preparatory step for a later sibling-type or thinner-base refactor if we still want it.
11. This refactor introduces explicit edge-state accessors first because the current continuous paths operate on edge-valued latent states. Later work may extend the same separation between binary topology and lossless latent state to node and global state without changing the `NoiseProcess` or sampler contracts.

## 3. Public Contracts

### 3.1 DataModule Contract

`BaseGraphDataModule` becomes the complete training-facing interface:

```python
class BaseGraphDataModule(pl.LightningDataModule, ABC):
    graph_type: str
    num_nodes: int

    def setup(self, stage: str | None = None) -> None: ...
    def train_dataloader(self) -> DataLoader[GraphData]: ...
    def val_dataloader(self) -> DataLoader[GraphData]: ...
    def test_dataloader(self) -> DataLoader[GraphData]: ...

    def get_reference_graphs(self, stage: str, max_graphs: int) -> list[nx.Graph]: ...
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution: ...
```

Required behavior:

- `graph_type` and `num_nodes` live on the base class so training code can treat datamodules uniformly.
- `get_size_distribution()` is defined on the base class. The default is `SizeDistribution.fixed(self.num_nodes)`.
- Variable-size datasets override `get_size_distribution()` to return empirical distributions.
- No datamodule exposes `node_marginals`, `edge_marginals`, `noise_level`, `noise_levels`, `noise_type`, or transition-related state.
- `get_reference_graphs()` remains the only graph-extraction helper in the public contract.

### 3.2 NoiseProcess Contract

`NoiseProcess` is the single polymorphic latent-corruption interface:

```python
class NoiseProcess(ABC, nn.Module):
    @property
    def timesteps(self) -> int: ...

    def initialize_from_data(self, train_loader: DataLoader[GraphData]) -> None: ...

    @abstractmethod
    def sample_prior(self, node_mask: Tensor) -> GraphData: ...

    @abstractmethod
    def forward_sample(self, x_0: GraphData, t: Tensor) -> GraphData: ...

    @abstractmethod
    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData: ...
```

For processes whose relevant densities are tractable exactly, the stronger contract is:

```python
class ExactDensityNoiseProcess(NoiseProcess, ABC):

    @abstractmethod
    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor: ...

    @abstractmethod
    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor: ...

    @abstractmethod
    def prior_log_prob(self, x: GraphData) -> Tensor: ...
```

Required behavior:

- `t` and `s` are integer timestep tensors of shape `(bs,)` with `0 <= s < t <= T`.
- `NoiseProcess` owns timestep semantics. A concrete implementation may use a private `NoiseSchedule`, a fixed `noise_level`, or another internal timestep-to-parameter map. Callers do not inspect that internal representation.
- `sample_prior(node_mask)` returns a valid state at timestep `T` for the given real-node mask.
- `forward_sample(x_0, t)` samples `x_t ~ q(x_t | x_0)`.
- `posterior_sample(z_t, x0_param, t, s)` samples `x_s ~ q(x_s | z_t, x0_param)` or the process-equivalent reverse posterior.
- No public method may expose transition matrices, limit-distribution structs, posterior parameter dicts, or other subtype-specific internal representations.
- `initialize_from_data()` is always called by `DiffusionModule.setup()`. Continuous scheduled processes may implement it as a no-op. Categorical processes use it to compute empirical marginals when needed. `T = 1` empirical-prior processes may use it to build their clean-prior or noisy-prior samplers from the training loader.
- For `T = 1`, the only valid reverse step is `t = 1`, `s = 0`. In that case `posterior_sample(z_1, x0_param, 1, 0)` is the single denoising step.
- `ExactDensityNoiseProcess.forward_log_prob`, `posterior_log_prob`, and `prior_log_prob` return per-sample log probabilities of shape `(bs,)`. They are required only for processes that claim exact-density support.
- Paths that compute VLB or NLL terms call the exact-density methods directly. If a process does not implement `ExactDensityNoiseProcess`, those paths fail immediately rather than branching to a fallback or degraded approximation.

`GraphData` remains the common transport type, but its state accessors split explicitly by semantics:

```python
class GraphData:
    @classmethod
    def from_binary_adjacency(cls, adj: Tensor) -> GraphData: ...

    def to_binary_adjacency(self) -> Tensor: ...

    @classmethod
    def from_edge_state(
        cls,
        edge_state: Tensor,
        *,
        node_mask: Tensor | None = None,
    ) -> GraphData: ...

    def to_edge_state(self) -> Tensor: ...
```

Required behavior:

- `from_binary_adjacency()` and `to_binary_adjacency()` are graph-topology helpers. They are for ingesting discrete graphs, exporting graphs to NetworkX or PyG, and any other boundary where the caller explicitly wants binary topology.
- `from_edge_state()` and `to_edge_state()` are lossless state-transport helpers. They are for continuous diffusion states, one-step denoising states, and any other dense edge-valued latent state that must not be collapsed to binary topology.
- Intermediate continuous states must round-trip through `GraphData` without `argmax`, thresholding, clamping, or rounding.
- The accessor split is the public contract. Callers must not rely on the raw channel layout inside `GraphData.E`; the implementation may still use a simple single-channel storage scheme now and move to a thinner neutral base or sibling state types later.
- The ambiguous helpers `from_adjacency()`, `to_adjacency()`, and `edge_features_to_adjacency()` are removed from production code.
- `GraphData` does not carry timestep or schedule metadata. Sampler-facing APIs use a separate wrapper when they need to move a graph state together with its diffusion-step metadata.

`x0_param` is the process's clean-data parameterization:

- For continuous processes it is a clean-state estimate in `GraphData` form, consumed through `to_edge_state()` rather than through binary-topology helpers.
- For categorical processes it may be a one-hot clean graph or model logits over clean categories. The categorical process is responsible for normalizing logits internally before using them.

### 3.3 Sampler Contract

Warm-start and in-flight sampler state use an explicit wrapper:

```python
@dataclass(frozen=True)
class DiffusionState:
    graph: GraphData
    t: int
    max_t: int
```

Required behavior:

- `graph` is the latent graph state at timestep `t`.
- `t` is the current timestep of that state. For the current sampler contract, one `DiffusionState` represents a homogeneous batch, so all graphs in `graph` share the same integer timestep.
- `max_t` is the total number of timesteps for the process that produced this state. For states consumed by a sampler, `max_t` must equal `noise_process.timesteps`.
- `DiffusionState` exists to make sampler state explicit. It does not replace `GraphData` as the shared tensor container for models, dataloaders, evaluators, or diffusion-process methods.

There is one public sampler class:

```python
class Sampler:
    def __init__(self, noise_process: NoiseProcess) -> None: ...

    @torch.no_grad()
    def sample(
        self,
        model: GraphModel,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
        *,
        start_from: DiffusionState | None = None,
        collector: StepMetricCollector | None = None,
    ) -> list[GraphData]: ...
```

The sampler algorithm is:

1. Build `node_mask` from `num_nodes` unless `start_from.graph` already provides it.
2. If `start_from` is `None`, call `noise_process.sample_prior(node_mask)` and wrap the result as `DiffusionState(graph=..., t=T, max_t=T)`, where `T = noise_process.timesteps`.
3. If `start_from` is provided:
   - validate `0 <= start_from.t <= start_from.max_t`
   - validate `start_from.max_t == noise_process.timesteps`
   - use `start_from` as the initial sampler state
4. For `s = current_state.t - 1 ... 0`:
   - set `t = s + 1`
   - call the model on `current_state.graph`
   - call `noise_process.posterior_sample(current_state.graph, pred, t, s)`
   - wrap the result as `DiffusionState(graph=z_s, t=s, max_t=current_state.max_t)`
   - optionally record step metrics
5. Trim each final graph to its real node count and return `list[GraphData]`.

Hard rules:

- The sampler does not use `isinstance`.
- The sampler does not access `.transition_model`, `.x_classes`, `.e_classes`, Gaussian posterior dicts, or any other subtype internals.
- The sampler does not accept a second `NoiseSchedule`.
- The sampler does not implement categorical or Gaussian math directly.
- The sampler keeps latent states in `GraphData` form throughout the reverse loop. It does not discretize intermediate continuous states; any projection to binary graph topology happens only at an explicit graph-extraction boundary outside the process math.
- The sampler does not expose or accept a separate `start_timestep` argument. Warm starts use `DiffusionState`.

### 3.4 Training Module Contract

`DiffusionModule` keeps orchestration responsibilities only.

Constructor shape after the refactor:

```python
class DiffusionModule(BaseGraphModule):
    def __init__(
        self,
        *,
        model: GraphModel,
        noise_process: NoiseProcess,
        evaluator: GraphEvaluator | None = None,
        loss_type: str = "cross_entropy",
        lambda_E: float = 5.0,
        num_nodes: int = 20,
        eval_every_n_steps: int = 5000,
        ...
    ) -> None: ...
```

Required behavior:

- `DiffusionModule` does not accept `noise_schedule`.
- `DiffusionModule` constructs `self.sampler = Sampler(self.noise_process)` internally.
- `setup()` always does exactly two datamodule reads:
  - `self.noise_process.initialize_from_data(dm.train_dataloader())`
  - `self._size_distribution = dm.get_size_distribution("train")`
- `training_step()` uses `noise_process.forward_sample()`.
- Generative evaluation uses the unified sampler.
- VLB terms are composed from generic exact-density primitives when `noise_process` implements `ExactDensityNoiseProcess`:
  - `kl_prior = E[log q(z_T | x_0) - log p(z_T)]`
  - `L_t = E[log q(z_s | z_t, x_0) - log q(z_s | z_t, x0_pred)]`
- `reconstruction_logp` remains on the training module because it depends on model output parameterization, not just the diffusion process.
- The module does not use `isinstance` to access categorical-only diffusion methods.
- VLB- or NLL-specific code paths call the exact-density methods directly and fail fast when the configured process does not support them. The refactor does not add fallback logic for non-exact-density processes.

`T = 1` denoising uses the ordinary `DiffusionModule` contract with a `T = 1` process and denoising-specific evaluator or reporting config:

- A denoising process uses `timesteps = 1`, with `t = 0` for clean data and `t = 1` for the configured noisy state.
- Denoising configs use a singular `noise_level` for each process or run. Comparisons across multiple levels are expressed as multiple process instances or multiple runs, not as one process with a heterogeneous list of unrelated levels.
- `initialize_from_data()` defines the empirical prior operationally from the clean training distribution. The process samples the induced noisy prior by first sampling `x_0` from the clean training distribution and then sampling `z_1 ~ q(z_1 | x_0)` at the configured corruption level.
- The implementation may realize that empirical prior lazily, by drawing clean examples and corrupting them on demand, or eagerly, by caching or fitting an equivalent sampler. Both implementations are acceptable only when they preserve the same prior semantics.
- `sample_prior(node_mask)` samples from that `t = 1` noisy prior.
- `forward_sample(x_0, t = 1)` applies the one configured corruption step.
- `posterior_sample(z_1, x0_param, t = 1, s = 0)` performs the one denoising step.
- Reconstruction training optimizes the model's clean-data prediction from `z_1`. Exact log-probability methods are optional unless the chosen one-step process also implements `ExactDensityNoiseProcess`.
- Exact `prior_log_prob(z_1)` is not required for this empirical prior. A `T = 1` process implements `ExactDensityNoiseProcess` only when its prior representation and corruption family admit exact tractable log probabilities.
- The datamodule no longer carries `noise_level`, `noise_levels`, or `noise_type` for it. Denoising-specific orchestration lives in experiment config, evaluator wiring, or other composition code, not in a dedicated training-module type.

## 4. Concrete Behavior by NoiseProcess Type

| Operation | `ContinuousNoiseProcess` | `CategoricalNoiseProcess` |
|---|---|---|
| `initialize_from_data` | no-op | if `limit_distribution == "empirical_marginal"`, iterate the training loader, accumulate node and edge class counts over real positions, normalize to PMFs; if `limit_distribution == "uniform"`, no-op |
| `sample_prior` | symmetric Gaussian state with zero diagonal over real nodes, wrapped through `GraphData.from_edge_state()` | sample from the configured stationary categorical distribution over real nodes and valid edges |
| `forward_sample` | existing continuous noise definition applied in edge-state space via `to_edge_state()` / `from_edge_state()` | pointwise mixture using `alpha_bar(t)` and the configured stationary categorical distribution |
| `posterior_sample` | closed-form Gaussian posterior over the continuous edge state | categorical posterior computed directly from the process, without materialized transition-model objects |
| `forward_log_prob` | log-density under the forward Gaussian | log-probability under the discrete forward process |
| `posterior_log_prob` | log-density of the Gaussian posterior | log-probability of the categorical posterior |
| `prior_log_prob` | log-density under the stationary Gaussian prior | log-probability under the stationary categorical prior |

`CategoricalNoiseProcess` accepts:

- `limit_distribution="uniform"` for a uniform stationary PMF over node and edge classes.
- `limit_distribution="empirical_marginal"` for a stationary PMF estimated from the training loader.

Rules for categorical marginal computation when `limit_distribution == "empirical_marginal"`:

- Node counts use only `node_mask == True` positions.
- Edge counts use only valid real-node pairs.
- For undirected graphs, edge counts use the strict upper triangle only.
- Zero-count classes normalize to a valid PMF by falling back to a uniform distribution over the affected domain.

The categorical implementation may use private helpers for reusable math, but it will not retain the current public transition-model hierarchy.

Exact-density note:

- Multi-step Gaussian and categorical diffusion processes implement `ExactDensityNoiseProcess`.
- A `T = 1` denoising process may or may not implement `ExactDensityNoiseProcess`, depending on whether its chosen prior and corruption family admit exact tractable log probabilities.

GraphData note:

- Continuous processes, including `T = 1` denoisers that operate on dense edge states, use `to_edge_state()` and `from_edge_state()` for intermediate states.
- Binary-topology accessors are reserved for discrete graph boundaries: dataset ingestion of actual graphs, final graph extraction, export, and categorical structural features that explicitly require binary adjacency.
- This accessor split is the immediate refactor target. A later LSP-oriented type split, such as a thinner neutral base with categorical and continuous siblings, remains possible and should preserve these semantics rather than reintroduce ambiguous adjacency helpers.
- The current explicit accessors are edge-state accessors because the current continuous implementations diffuse edges. If later work introduces continuous node or global state, it should extend the same semantic split rather than restore one overloaded adjacency API.

## 5. Timestep Semantics, Optional Schedules, and `T = 1` Denoising

`NoiseSchedule` remains a standalone type, but it is an implementation detail of scheduled multi-step processes rather than a mandatory public member of every `NoiseProcess`. The public contract is timestep-based; each concrete process decides how integer timesteps map to its internal parameters.

Required changes:

- `ContinuousNoiseProcess(definition=..., schedule=...)`
- `CategoricalNoiseProcess(schedule=..., ...)`
- a fixed-level `T = 1` process may use `noise_level=...` internally instead of a public `NoiseSchedule`
- `DiffusionModule` does not accept `noise_schedule`
- `Sampler` does not accept `noise_schedule`

Scheduled multi-step processes interpret timesteps through their private schedule:

- `t` is mapped internally to `alpha_bar(t)`, `beta(t)`, or the process-equivalent parameters.
- The process may still be configured from a `NoiseSchedule` object, but callers rely only on integer timesteps.

`T = 1` denoising processes interpret timesteps directly:

- `t = 1` means “apply the one configured corruption level for this process.”
- `sample_prior()` returns the noisy prior at that level.
- The single reverse step maps `z_1` back toward `x_0`.
- Logging may still expose the configured scalar `noise_level` for readability.

## 6. DataModule Changes

### 6.1 Required public interface

- Add `get_size_distribution()` to `BaseGraphDataModule` with fixed-size default.
- Move `graph_type` and `num_nodes` onto the base datamodule contract.

### 6.2 Shared generation

Graph-generation dispatch is centralized in one helper:

```python
def generate_graph_adjacencies(
    *,
    graph_type: str,
    num_nodes: int,
    num_graphs: int,
    graph_config: dict[str, Any],
    seed: int,
) -> NDArray[np.float32]: ...
```

This helper becomes the single dispatch point for:

- SBM generation
- `SyntheticGraphDataset`
- PyG dataset loading and graph selection

`MultiGraphDataModule` and `SingleGraphDataModule` both use it. There will be no duplicated graph-type branching between those classes after the refactor.

### 6.3 Removed datamodule API

The following datamodule methods or fields are deleted:

| API | Reason |
|---|---|
| `BaseGraphDataModule.get_dataset_info()` | zero production callers |
| `SyntheticCategoricalDataModule.node_marginals` | noise-process concern |
| `SyntheticCategoricalDataModule.edge_marginals` | noise-process concern |
| `SyntheticCategoricalDataModule._compute_marginals()` | noise-process concern |
| `GraphDataModule.get_sample_adjacency_matrix()` | tests-only |
| `SingleGraphDataModule.get_sample_adjacency_matrix()` | tests-only |
| `MultiGraphDataModule.get_sample_graph()` | zero production callers |
| `SingleGraphDataModule.get_train_graph()` | tests-only |
| `SingleGraphDataModule.get_val_graph()` | tests-only |
| `SingleGraphDataModule.get_test_graph()` | tests-only |
| `SyntheticCategoricalDataModule.sample_n_nodes()` | tests-only |
| datamodule constructor args `noise_level` / `noise_levels` / `noise_type` | training-module concern |

Tests that currently depend on these helpers must be rewritten against the public contracts: dataloaders, `get_reference_graphs()`, and `get_size_distribution()`.

## 7. Diffusion and Sampler Deletions

The following public types are deleted:

| Type or API | File |
|---|---|
| `TransitionModel` | `src/tmgg/diffusion/protocols.py` |
| `TransitionMatrices` | `src/tmgg/diffusion/diffusion_graph_types.py` |
| `LimitDistribution` | `src/tmgg/diffusion/diffusion_graph_types.py` |
| `DiscreteUniformTransition` | `src/tmgg/diffusion/transitions.py` |
| `MarginalUniformTransition` | `src/tmgg/diffusion/transitions.py` |
| `CategoricalNoiseDefinition` | `src/tmgg/diffusion/categorical_noise.py` |
| `CategoricalSampler` | `src/tmgg/diffusion/sampler.py` |
| `ContinuousSampler` | `src/tmgg/diffusion/sampler.py` |
| `Sampler.sample(..., start_timestep=...)` | `src/tmgg/diffusion/sampler.py` |
| `NoiseProcess.apply()` | `src/tmgg/diffusion/noise_process.py` |
| `NoiseProcess.get_posterior()` | `src/tmgg/diffusion/noise_process.py` |
| `NoiseProcess._apply_noise()` | `src/tmgg/diffusion/noise_process.py` |
| `NoiseProcess._schedule_to_level()` | `src/tmgg/diffusion/noise_process.py` |
| `CategoricalNoiseProcess.set_transition_model()` | `src/tmgg/diffusion/noise_process.py` |
| `CategoricalNoiseProcess.transition_model` | `src/tmgg/diffusion/noise_process.py` |
| `ContinuousNoiseProcess.generator` alias | `src/tmgg/diffusion/noise_process.py` |
| `GraphData.from_adjacency()` | `src/tmgg/data/datasets/graph_types.py` |
| `GraphData.to_adjacency()` | `src/tmgg/data/datasets/graph_types.py` |
| `GraphData.edge_features_to_adjacency()` | `src/tmgg/data/datasets/graph_types.py` |
| `SingleStepDenoisingModule` | `src/tmgg/training/lightning_modules/denoising_module.py` |

Any helper in `diffusion_sampling.py` that exists only to support the removed transition-matrix pipeline must also be deleted or inlined. After the refactor, no public code path may depend on transition matrices. This deletion does not remove the two supported categorical stationary-distribution modes; `CategoricalNoiseProcess` now dispatches internally from `limit_distribution`.

## 8. Config and Wiring Changes

### 8.1 Hydra shape

Generative configs change from:

```yaml
model:
  noise_process: ...
  sampler: ...
  noise_schedule: ...
```

to:

```yaml
model:
  _target_: tmgg.training.lightning_modules.diffusion_module.DiffusionModule
  noise_process:
    _target_: tmgg.diffusion.noise_process.CategoricalNoiseProcess
    schedule:
      _target_: tmgg.diffusion.schedule.NoiseSchedule
      schedule_type: cosine_iddpm
      timesteps: 500
    x_classes: 2
    e_classes: 2
    limit_distribution: empirical_marginal
```

Consequences:

- There is one instantiated `NoiseProcess`, not three copies of the same schedule/process config.
- `sampler:` is removed from experiment configs.
- `noise_schedule:` is removed from `DiffusionModule` configs and nested under `noise_process.schedule`.
- `docs/extending.md`, experiment docs, and config-composition tests must be updated to reflect the new shape.

### 8.2 Denoising configs

Denoising configs instantiate the ordinary `DiffusionModule` with a `T = 1` denoising process. They keep `noise_type` and a singular `noise_level` on the denoising process, not on the datamodule.

If experiments compare multiple denoising levels, they do so by instantiating multiple process configs or running multiple jobs, not by putting an unrelated list of levels into one process contract.

The datamodule configs lose any noise-related compatibility fields.

## 9. Implementation Sequence

1. Create an annotated rollback tag on the current branch state before editing implementation code.
2. Rewrite the public contracts first:
   - datamodule base contract
   - `NoiseProcess`
   - `DiffusionState`
   - unified `Sampler`
   - `DiffusionModule` constructor and setup
3. Split `GraphData` accessors into explicit binary-topology and edge-state helpers, then migrate continuous and denoising call sites to the edge-state helpers before changing process internals.
4. Move marginal computation into `CategoricalNoiseProcess.initialize_from_data()`.
5. Replace categorical and continuous samplers with the unified sampler, including migration from `start_from` plus `start_timestep` to `DiffusionState`.
6. Replace direct schedule ownership in modules/configs with process-owned schedules.
7. Move shared graph generation into one helper and remove duplicate dispatch.
8. Remove all deleted APIs and the dead tests that only exercised them.
9. Update Hydra configs, docs, and compatibility-sensitive tests.
10. Run the final call-site sweeps and the full test suite before merging.

Suggested commit boundaries:

1. interface rewrite
2. sampler unification
3. datamodule cleanup and generation deduplication
4. config and test migration
5. documentation cleanup

## 10. Acceptance Criteria

The refactor is complete only when all of the following are true.

### 10.1 Code shape

- Production code contains exactly one public sampler implementation.
- No production generative diffusion code references `.transition_model`, `.set_transition_model()`, `node_marginals`, `edge_marginals`, or `get_posterior()`.
- No production code uses `hasattr(dm, "get_size_distribution")`, `getattr(dm, "noise_level", ...)`, or `getattr(dm, "noise_levels", ...)`.
- No production caller assumes that every `NoiseProcess` exposes a public `.schedule` attribute.
- No production code uses the ambiguous `GraphData.from_adjacency()`, `to_adjacency()`, or `edge_features_to_adjacency()` helpers.
- No production code uses the old sampler warm-start split of `start_from` plus `start_timestep`; warm starts use `DiffusionState`.
- No production Hydra config instantiates `SingleStepDenoisingModule`.
- No production Hydra config instantiates `sampler:` or top-level `noise_schedule:` for `DiffusionModule`.
- No duplicated graph-generation dispatch remains between `MultiGraphDataModule` and `SingleGraphDataModule`.

### 10.2 Behavioral correctness

- Categorical diffusion still trains, samples, and evaluates on the discrete generative experiment path.
- Continuous diffusion still trains, samples, and evaluates on the Gaussian generative path.
- Single-step denoising still trains and evaluates as a legitimate `T = 1` generative denoising process at its configured `noise_level`.
- `get_size_distribution()` works uniformly for fixed-size and variable-size datamodules.
- `CategoricalNoiseProcess.initialize_from_data()` reproduces the intended marginal computation semantics from the current datamodule implementation.
- Continuous intermediate states remain lossless through `GraphData`; no binary projection happens inside the continuous forward or reverse process.

### 10.3 Test strategy

- Update or replace tests that currently target deleted public APIs.
- Add focused tests for:
  - `NoiseProcess.initialize_from_data()`
  - `NoiseProcess.sample_prior()`
  - unified `Sampler.sample()`
  - `DiffusionState` warm-start sampling
  - fixed-size default `get_size_distribution()`
  - `T = 1` denoising process behavior
  - `GraphData` binary-topology and edge-state accessor round-trips
- Run the relevant targeted tests first, then the full suite.

### 10.4 Required call-site sweeps

Before considering the refactor done, the implementation must verify that these searches return no production callers:

```bash
rg -n "transition_model|set_transition_model|get_posterior\\(|node_marginals|edge_marginals" src
rg -n "get_dataset_info\\(|get_sample_adjacency_matrix\\(|get_sample_graph\\(|get_train_graph\\(|get_val_graph\\(|get_test_graph\\(|sample_n_nodes\\(" src
rg -n "hasattr\\(.*get_size_distribution|getattr\\(.*noise_level|getattr\\(.*noise_levels" src
rg -n "from_adjacency\\(|to_adjacency\\(|edge_features_to_adjacency\\(" src
rg -n "start_timestep" src
rg -n "_target_: .*SingleStepDenoisingModule" src/tmgg/experiments/exp_configs
rg -n "noise_schedule:" src/tmgg/experiments/exp_configs | rg "DiffusionModule|sampler"
```

The exact `rg` expressions may change during implementation, but the final sweep must prove that the removed interfaces have no production callers left. Internal process implementations may still use `noise_level` as a private parameter for `T = 1` denoising, but public training and sampling contracts stay timestep-based.

## 11. Non-goals

- Backward compatibility with the current public API
- Preserving tests that only validate removed helper methods
- Preserving duplicated Hydra wiring for schedule and sampler construction
- Keeping old checkpoint-loading behavior unless it remains cheap and non-invasive after the refactor
