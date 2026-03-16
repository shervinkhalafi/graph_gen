# Evaluator + Sampling API Unification — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify graph evaluation and sampling into a clean, non-redundant architecture: stateless GraphEvaluator, no accumulation during validation steps, unified Sampler with per-step collector, single `eval_num_samples` owner.

**Architecture:** GraphEvaluator becomes a stateless metric computer that accepts `(refs, generated)` and truncates to `eval_num_samples` internally. Both Lightning modules produce graphs on-demand at epoch end rather than accumulating during validation steps. The Sampler ABC gains `start_from`, `start_timestep`, and `collector` parameters for partial reverse chains and per-step metric collection. A `DiffusionLikelihoodCollector` implements the protocol for full-chain VLB computation.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra/OmegaConf, NetworkX, basedpyright

**Spec:** `docs/superpowers/specs/2026-03-15-evaluator-sampling-unification-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py` | Modify | Remove accumulation, accept refs+generated in evaluate(), train_graphs in constructor |
| `src/tmgg/experiments/_shared_utils/evaluation_metrics/__init__.py` | Modify | Update exports if needed |
| `src/tmgg/diffusion/sampler.py` | Modify | Add start_from, start_timestep, collector params to Sampler + subclasses |
| `src/tmgg/diffusion/collectors.py` | Create | StepMetricCollector protocol + DiffusionLikelihoodCollector |
| `src/tmgg/diffusion/__init__.py` | Modify | Export new types |
| `src/tmgg/data/data_modules/base_data_module.py` | Modify | Add concrete get_reference_graphs() method |
| `src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py` | Modify | Remove accumulation from validation_step, epoch-end evaluation from datamodule |
| `src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py` | Modify | Remove all accumulation state, epoch-end noise+denoise+evaluate |
| `src/tmgg/experiments/exp_configs/task/denoising.yaml` | Modify | Remove model.eval_num_samples |
| `src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml` | Modify | Ensure eval_num_samples on evaluator only |
| `src/tmgg/experiments/exp_configs/models/discrete/*.yaml` (4 files) | Modify | Ensure eval_num_samples on evaluator only |
| `tests/experiment_utils/test_graph_evaluator.py` | Modify | Delete accumulation tests, update evaluate() calls |
| `tests/experiment_utils/test_diffusion_module.py` | Modify | Remove accumulation tests, add epoch-end ref-from-datamodule tests |
| `tests/experiment_utils/test_denoising_module.py` | Modify | Remove accumulation tests, add epoch-end denoise+evaluate tests |
| `tests/data_modules/test_reference_graphs.py` | Create | Test get_reference_graphs() |
| `tests/diffusion/test_collectors.py` | Create | Test DiffusionLikelihoodCollector |

---

## Chunk 1: Stateless GraphEvaluator

### Task 1: Make ORCA availability check module-level

**Files:**
- Modify: `src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py:39,48,460-472`

- [ ] **Step 1a: Cache ORCA availability at module level**

In `graph_evaluator.py`, the import at line 39 brings in `is_available as _orca_is_available` — a function that's called per-instance in `__init__`. Make it a module-level constant alongside `_GRAPH_TOOL_AVAILABLE` (line 48):

```python
# Line 48 area — add after _GRAPH_TOOL_AVAILABLE
_GRAPH_TOOL_AVAILABLE = importlib.util.find_spec("graph_tool") is not None
_ORCA_AVAILABLE = _orca_is_available()
```

Then replace all `_orca_is_available()` calls (in `__init__` and `evaluate()`) with `_ORCA_AVAILABLE`.

- [ ] **Step 1b: Move warnings to module level**

Replace the per-instance warnings in `__init__` (lines 461-472) with module-level warnings that fire once at import time:

```python
if not _GRAPH_TOOL_AVAILABLE:
    warnings.warn(
        "graph-tool not installed; sbm_accuracy will be None in GraphEvaluator results.",
        stacklevel=2,
    )
if not _ORCA_AVAILABLE:
    warnings.warn(
        "orca binary not found; orbit_mmd will be None in GraphEvaluator results. "
        "Compile orca.cpp and set ORCA_PATH or place the binary in the expected location.",
        stacklevel=2,
    )
```

Remove the warning logic from `__init__` entirely.

- [ ] **Step 1c: Run basedpyright and tests**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py
uv run pytest tests/experiment_utils/test_graph_evaluator.py -x -v
```

- [ ] **Step 1d: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py
git commit -m "refactor: move ORCA/graph-tool availability checks to module level"
```

### Task 2: Remove accumulation state and accept train_graphs in constructor

**Files:**
- Modify: `src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py`
- Modify: `tests/experiment_utils/test_graph_evaluator.py`

- [ ] **Step 2a: Update tests first — remove accumulation tests, update evaluate() calls**

In `test_graph_evaluator.py`:
- Delete the entire `TestAccumulation` class and `TestSetup` class.
- In all remaining test classes (`TestEvaluateCore`, `TestEvaluateOrbitSBM`, `TestFullLifecycle`), update `evaluate()` calls to pass `refs` and `generated` as arguments instead of relying on `self.ref_graphs`.
- Update constructor calls: replace `GraphEvaluator(eval_num_samples=N)` with `GraphEvaluator(eval_num_samples=N, train_graphs=train_set)` where `setup()` was previously called.
- Remove all `evaluator.accumulate()`, `evaluator.clear()`, `evaluator.setup()` calls.
- Add test for internal truncation: create evaluator with `eval_num_samples=5`, pass 10 refs and 10 generated, verify metrics are computed on 5 each.
- Add test: `train_graphs=None` means novelty is `None` in results.

- [ ] **Step 2b: Run tests to verify they fail**

```bash
uv run pytest tests/experiment_utils/test_graph_evaluator.py -x -v
```

Expected: FAIL — `evaluate()` signature mismatch, `train_graphs` param doesn't exist yet.

- [ ] **Step 2c: Update GraphEvaluator implementation**

In `graph_evaluator.py`:
- Add `train_graphs: list[nx.Graph[Any]] | None = None` to `__init__` signature.
- Store as `self.train_graphs = list(train_graphs) if train_graphs is not None else []` and `self._train_graphs_set = train_graphs is not None`.
- Delete `setup()` method entirely.
- Delete `accumulate()` method entirely.
- Delete `clear()` method entirely.
- Delete `self.ref_graphs: list[nx.Graph[Any]] = []` from `__init__`.
- Change `evaluate()` signature to `evaluate(self, refs: list[nx.Graph[Any]], generated: list[nx.Graph[Any]]) -> EvaluationResults | None`.
- At top of `evaluate()`, add truncation: `refs = refs[:self.eval_num_samples]` and `generated = generated[:self.eval_num_samples]`.
- Replace `self.ref_graphs` references inside `evaluate()` with `refs`.

- [ ] **Step 2d: Run tests to verify they pass**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py
uv run pytest tests/experiment_utils/test_graph_evaluator.py -x -v
```

- [ ] **Step 2e: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py tests/experiment_utils/test_graph_evaluator.py
git commit -m "refactor(G50): make GraphEvaluator stateless — remove accumulation, accept train_graphs in constructor"
```

---

## Chunk 2: Datamodule accessor

### Task 3: Add get_reference_graphs() to BaseGraphDataModule

**Files:**
- Modify: `src/tmgg/data/data_modules/base_data_module.py`
- Create: `tests/data_modules/test_reference_graphs.py`

- [ ] **Step 3a: Write tests for get_reference_graphs()**

Create `tests/data_modules/test_reference_graphs.py` with tests:
- Test with `GraphDataModule` (denoising datamodule): verify returns correct number of nx.Graph objects from val split.
- Test with `max_graphs` smaller than validation set: verify truncation.
- Test with `max_graphs` larger than validation set: verify returns all available.
- Test `stage="test"` returns from test split.
- Test calling before `setup()` raises `RuntimeError`.

Use the existing `GraphDataModule` with `graph_type="sbm"` and small sample counts for fast tests.

- [ ] **Step 3b: Run tests to verify they fail**

```bash
uv run pytest tests/data_modules/test_reference_graphs.py -x -v
```

Expected: FAIL — `get_reference_graphs` doesn't exist.

- [ ] **Step 3c: Implement get_reference_graphs() as concrete method on BaseGraphDataModule**

In `base_data_module.py`, add after the existing abstract methods:

```python
def get_reference_graphs(self, stage: str, max_graphs: int) -> list[nx.Graph]:
    """Extract up to max_graphs from a dataset split as NetworkX graphs.

    Default implementation iterates the appropriate dataloader, converts
    GraphData batches to adjacency matrices, and builds NetworkX graphs.
    Subclasses may override for efficiency.

    Parameters
    ----------
    stage
        ``"val"`` or ``"test"``.
    max_graphs
        Maximum number of graphs to return.

    Returns
    -------
    list[nx.Graph]
        NetworkX graphs from the specified split.

    Raises
    ------
    RuntimeError
        If the datamodule has not been set up.
    ValueError
        If stage is not ``"val"`` or ``"test"``.
    """
    if stage == "val":
        loader = self.val_dataloader()
    elif stage == "test":
        loader = self.test_dataloader()
    else:
        raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")

    graphs: list[nx.Graph] = []
    for batch in loader:
        adj = batch.to_adjacency()  # (B, N, N)
        bs = adj.shape[0]
        for i in range(bs):
            if len(graphs) >= max_graphs:
                return graphs
            n = int(batch.node_mask[i].sum().item())
            A_np = adj[i, :n, :n].cpu().numpy()
            graphs.append(nx.from_numpy_array(A_np))
    return graphs
```

Add `import networkx as nx` to the imports.

- [ ] **Step 3d: Run tests and basedpyright**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/data_modules/base_data_module.py
uv run pytest tests/data_modules/test_reference_graphs.py -x -v
```

- [ ] **Step 3e: Commit**

```bash
git add src/tmgg/data/data_modules/base_data_module.py tests/data_modules/test_reference_graphs.py
git commit -m "feat: add get_reference_graphs() to BaseGraphDataModule"
```

---

## Chunk 3: StepMetricCollector protocol + DiffusionLikelihoodCollector

### Task 4: Create collector protocol and concrete implementation

**Files:**
- Create: `src/tmgg/diffusion/collectors.py`
- Modify: `src/tmgg/diffusion/__init__.py`
- Create: `tests/diffusion/test_collectors.py`

- [ ] **Step 4a: Write tests for DiffusionLikelihoodCollector**

Create `tests/diffusion/test_collectors.py`:
- Test `record(t, s, {"kl": 0.5})` accumulates metrics.
- Test `vlb()` returns sum of all recorded KL values (full-chain VLB = sum of per-step KLs + prior + reconstruction).
- Test `results()` returns dict of summary statistics.
- Test empty collector (no records) returns zero VLB.
- Test collector records are ordered by timestep.

- [ ] **Step 4b: Run tests to verify they fail**

```bash
uv run pytest tests/diffusion/test_collectors.py -x -v
```

- [ ] **Step 4c: Implement collectors.py**

Create `src/tmgg/diffusion/collectors.py`:

```python
"""Per-step metric collectors for reverse diffusion sampling.

Provides the ``StepMetricCollector`` protocol and a concrete
``DiffusionLikelihoodCollector`` that accumulates KL divergence
at each reverse step for full variational lower bound computation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StepMetricCollector(Protocol):
    """Protocol for collecting per-step metrics during reverse diffusion."""

    def record(self, t: int, s: int, metrics: dict[str, float]) -> None:
        """Record metrics for one reverse step from timestep t to s.

        Parameters
        ----------
        t
            Current timestep (before this step).
        s
            Target timestep (after this step).
        metrics
            Step-level metrics (e.g. ``{"kl": 0.5, "log_likelihood": -1.2}``).
        """
        ...


class DiffusionLikelihoodCollector:
    """Collects per-step KL for full variational lower bound computation.

    After running a complete reverse chain with this collector, call
    ``vlb()`` to get the total VLB (sum of per-step KL contributions).

    Examples
    --------
    >>> collector = DiffusionLikelihoodCollector()
    >>> # ... pass to sampler.sample(collector=collector) ...
    >>> print(collector.vlb())
    """

    def __init__(self) -> None:
        self._records: list[tuple[int, int, dict[str, float]]] = []

    def record(self, t: int, s: int, metrics: dict[str, float]) -> None:
        """Record metrics for one reverse step."""
        self._records.append((t, s, metrics))

    def vlb(self) -> float:
        """Total variational lower bound (sum of per-step KL contributions).

        Returns
        -------
        float
            Sum of all ``"kl"`` values across recorded steps. Returns 0.0
            if no steps have been recorded or no ``"kl"`` keys exist.
        """
        return sum(m.get("kl", 0.0) for _, _, m in self._records)

    def results(self) -> dict[str, float]:
        """Summary statistics of the collected metrics.

        Returns
        -------
        dict[str, float]
            Contains ``"vlb"`` (total KL), ``"num_steps"`` (step count),
            and per-metric means (``"mean_{key}"`` for each key seen).
        """
        out: dict[str, float] = {
            "vlb": self.vlb(),
            "num_steps": float(len(self._records)),
        }
        if self._records:
            all_keys: set[str] = set()
            for _, _, m in self._records:
                all_keys.update(m.keys())
            for key in sorted(all_keys):
                values = [m[key] for _, _, m in self._records if key in m]
                if values:
                    out[f"mean_{key}"] = sum(values) / len(values)
        return out

    @property
    def records(self) -> list[tuple[int, int, dict[str, float]]]:
        """Raw recorded data as list of (t, s, metrics) tuples."""
        return list(self._records)
```

- [ ] **Step 4d: Update diffusion/__init__.py exports**

Add `StepMetricCollector` and `DiffusionLikelihoodCollector` to the exports in `src/tmgg/diffusion/__init__.py`.

- [ ] **Step 4e: Run tests and basedpyright**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/diffusion/collectors.py
uv run pytest tests/diffusion/test_collectors.py -x -v
```

- [ ] **Step 4f: Commit**

```bash
git add src/tmgg/diffusion/collectors.py src/tmgg/diffusion/__init__.py tests/diffusion/test_collectors.py
git commit -m "feat: add StepMetricCollector protocol and DiffusionLikelihoodCollector"
```

---

## Chunk 4: Extend Sampler API

### Task 5: Add start_from, start_timestep, collector to Sampler

**Files:**
- Modify: `src/tmgg/diffusion/sampler.py`

- [ ] **Step 5a: Update Sampler ABC signature**

At `sampler.py:50-78`, add keyword-only params after `device`:

```python
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
```

Add import at top of file: `from tmgg.diffusion.collectors import StepMetricCollector`

Update docstring to document the new parameters.

- [ ] **Step 5b: Update CategoricalSampler.sample()**

At `sampler.py:118-287`:
- Add the three new keyword params to the signature.
- When `start_from` is not None and `start_timestep` is not None, use `start_from` as `z_T` and start the reverse loop from `start_timestep` instead of `T`.
- Inside the reverse loop (line ~171), after computing the posterior and sampling `z_s`, call `collector.record(t, s, {"kl": kl_value})` if `collector is not None`. The KL at each step is already implicitly computed during posterior calculation — extract it.
- When `start_from` is None, behavior is unchanged (start from limit distribution at T).

- [ ] **Step 5c: Update ContinuousSampler.sample()**

At `sampler.py:324-424`:
- Same pattern: add keyword params, support `start_from`/`start_timestep`, call collector if present.
- For the Gaussian case, the per-step metric to collect is the log-likelihood contribution from the posterior.

- [ ] **Step 5d: Run existing sampler tests + basedpyright**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/diffusion/sampler.py
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow and not modal" -k "sampler" -v
```

Existing tests should pass since new params have defaults.

- [ ] **Step 5e: Commit**

```bash
git add src/tmgg/diffusion/sampler.py
git commit -m "feat: extend Sampler.sample() with start_from, start_timestep, collector"
```

---

## Chunk 5: DiffusionModule — remove accumulation, epoch-end evaluation

### Task 6: Update DiffusionModule validation flow

**Files:**
- Modify: `src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py`
- Modify: `tests/experiment_utils/test_diffusion_module.py`

- [ ] **Step 6a: Update tests first**

In `test_diffusion_module.py`:
- Delete `test_accumulates_reference_graphs` (or equivalent test that checks `evaluator.ref_graphs`).
- Update any test that constructs `GraphEvaluator(eval_num_samples=N)` — now pass `train_graphs` if novelty tests exist, and update `evaluate()` call signatures.
- Add test: `on_validation_epoch_end` pulls refs from `trainer.datamodule.get_reference_graphs()` and passes them to `evaluator.evaluate(refs=..., generated=...)`.
- Remove any test that calls `evaluator.accumulate()`, `evaluator.clear()`, or `evaluator.setup()`.

- [ ] **Step 6b: Run tests to verify they fail**

```bash
uv run pytest tests/experiment_utils/test_diffusion_module.py -x -v
```

- [ ] **Step 6c: Update DiffusionModule**

In `diffusion_module.py`:
- Remove `eval_num_samples` from `__init__` signature (if it was added in prior work — check current state).
- In `validation_step` (lines 434-440): delete the graph accumulation block entirely (the `if self.evaluator is not None: adj = batch.to_adjacency()...` block).
- In `on_validation_epoch_end` (lines 449-500):
  - Remove `self.evaluator.clear()` calls (two locations: the skip path at line 484 and the end at line 500).
  - Replace `num_samples = len(self.evaluator.ref_graphs)` with ref graph retrieval from datamodule:
    ```python
    refs = self.trainer.datamodule.get_reference_graphs(
        "val", self.evaluator.eval_num_samples
    )
    if len(refs) < 2:
        return
    generated_graphs = self.generate_graphs(len(refs))
    results = self.evaluator.evaluate(refs=refs, generated=generated_graphs)
    ```
  - The `eval_every_n_steps` skip path becomes a bare `return` (no `clear()` needed).
- Update `test_step` to NOT delegate to `validation_step` for the accumulation path (it can still compute loss by delegating). Alternatively, add `on_test_epoch_end` that mirrors `on_validation_epoch_end` but uses `"test"` stage.

- [ ] **Step 6d: Update GraphEvaluator construction in DiffusionModule or YAML**

Ensure the evaluator is constructed with `train_graphs` if novelty computation is desired. Check how the evaluator is instantiated — likely via Hydra `_target_`. If `train_graphs` needs to come from the datamodule, this may need to be wired in `on_fit_start` or similar. For now, pass `train_graphs=None` (novelty disabled) and note this as a follow-up.

- [ ] **Step 6e: Run tests and basedpyright**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py
uv run pytest tests/experiment_utils/test_diffusion_module.py -x -v
```

- [ ] **Step 6f: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py tests/experiment_utils/test_diffusion_module.py
git commit -m "refactor: remove graph accumulation from DiffusionModule validation_step"
```

---

## Chunk 6: SingleStepDenoisingModule — remove accumulation, epoch-end denoise+evaluate

### Task 7: Update SingleStepDenoisingModule validation flow

**Files:**
- Modify: `src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py`
- Modify: `tests/experiment_utils/test_denoising_module.py`

- [ ] **Step 7a: Update tests first**

In `test_denoising_module.py`:
- Remove any test checking `_clean_graphs_by_eps`, `_denoised_graphs_by_eps`, or `_accumulate_graphs`.
- Update GraphEvaluator construction to use new API (no `setup()`, pass `train_graphs` to constructor, `evaluate(refs, generated)`).
- Add test: `on_validation_epoch_end` with an evaluator produces per-noise-level metrics by running forward passes at epoch end.
- Remove `_eval_num_samples` from module constructor calls in tests.

- [ ] **Step 7b: Run tests to verify they fail**

```bash
uv run pytest tests/experiment_utils/test_denoising_module.py -x -v
```

- [ ] **Step 7c: Update SingleStepDenoisingModule**

In `denoising_module.py`:
- Remove from `__init__`:
  - `eval_num_samples` parameter
  - `self._eval_num_samples` assignment (line 163)
  - `self._clean_graphs_by_eps` dict (line 164)
  - `self._denoised_graphs_by_eps` dict (line 165)
- Remove `_accumulate_graphs()` method entirely (lines 476-510).
- In `_val_or_test()` (line 308): remove the `if self.evaluator is not None: self._accumulate_graphs(...)` call (line 368-369). The validation step now computes loss and per-batch metrics only.
- Rewrite `on_validation_epoch_end()` (lines 512-571):

```python
@override
def on_validation_epoch_end(self) -> None:
    if self.evaluator is None:
        return

    dm = self.trainer.datamodule
    ref_graphs = dm.get_reference_graphs("val", self.evaluator.eval_num_samples)
    if len(ref_graphs) < 2:
        return

    # Convert ref graphs back to adjacency tensors for noising
    device = next(self.parameters()).device
    n = max(g.number_of_nodes() for g in ref_graphs)
    ref_adjs = torch.zeros(len(ref_graphs), n, n, device=device)
    for i, g in enumerate(ref_graphs):
        adj_np = nx.to_numpy_array(g)
        ref_adjs[i, :adj_np.shape[0], :adj_np.shape[1]] = torch.from_numpy(adj_np)

    all_results: dict[float, dict[str, float | None]] = {}

    with torch.no_grad():
        for eps in self.eval_noise_levels:
            noisy = self.noise_generator.add_noise(ref_adjs, eps)
            output = self.forward(noisy)
            predictions = (output > 0).float()
            predictions = self._zero_diagonal(predictions)

            # Convert predictions to NetworkX
            denoised_graphs: list[nx.Graph] = []
            for i in range(len(ref_graphs)):
                ng = ref_graphs[i].number_of_nodes()
                A_np = predictions[i, :ng, :ng].cpu().numpy()
                denoised_graphs.append(nx.from_numpy_array(A_np))

            results = self.evaluator.evaluate(
                refs=ref_graphs, generated=denoised_graphs
            )
            if results is None:
                continue

            metrics = results.to_dict()
            all_results[eps] = metrics
            for metric_name, value in metrics.items():
                if value is not None:
                    self.log(f"val_{eps}/{metric_name}", value,
                             on_step=False, on_epoch=True)

    # Noise-level-averaged metrics
    if all_results:
        metric_names = next(iter(all_results.values())).keys()
        for metric_name in metric_names:
            values = [v for r in all_results.values()
                      if (v := r[metric_name]) is not None]
            if values:
                self.log(f"val/{metric_name}", sum(values) / len(values),
                         on_step=False, on_epoch=True)
```

- Update docstring for `__init__` to remove `eval_num_samples` parameter docs.

- [ ] **Step 7d: Run tests and basedpyright**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py
uv run pytest tests/experiment_utils/test_denoising_module.py -x -v
```

- [ ] **Step 7e: Commit**

```bash
git add src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py tests/experiment_utils/test_denoising_module.py
git commit -m "refactor(G50): remove accumulation from SingleStepDenoisingModule, evaluate at epoch end"
```

---

## Chunk 7: YAML config cleanup + integration test

### Task 8: Remove duplicate eval_num_samples from YAML configs

**Files:**
- Modify: `src/tmgg/experiments/exp_configs/task/denoising.yaml`
- Modify: `src/tmgg/experiments/exp_configs/base_config_gaussian_diffusion.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_default.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_small.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_eigenvec.yaml`
- Modify: `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml`

- [ ] **Step 8a: Remove model.eval_num_samples from denoising.yaml**

In `task/denoising.yaml` line 36, delete `eval_num_samples: 128` from the `model:` block (keep the one inside `model.evaluator`).

- [ ] **Step 8b: Verify other configs have eval_num_samples on evaluator only**

Check each discrete config and `base_config_gaussian_diffusion.yaml`. If `eval_num_samples` appears outside the evaluator block, remove it. If it only appears inside the evaluator block, no change needed.

- [ ] **Step 8c: Run full test suite**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow and not modal" -v
```

- [ ] **Step 8d: Run basedpyright on all modified files**

```bash
uv run basedpyright --project pyproject.toml \
  src/tmgg/experiments/_shared_utils/evaluation_metrics/graph_evaluator.py \
  src/tmgg/experiments/_shared_utils/lightning_modules/diffusion_module.py \
  src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py \
  src/tmgg/diffusion/sampler.py \
  src/tmgg/diffusion/collectors.py \
  src/tmgg/data/data_modules/base_data_module.py
```

- [ ] **Step 8e: Commit**

```bash
git add src/tmgg/experiments/exp_configs/
git commit -m "fix(G50): remove duplicate eval_num_samples from YAML configs"
```

### Task 9: Update SUMMARY.md and final verification

**Files:**
- Modify: `docs/reports/2026-03-12-tmgg-review/SUMMARY.md`

- [ ] **Step 9a: Mark G50 as FIXED in SUMMARY.md**

- [ ] **Step 9b: Run full test suite one final time**

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow and not modal" -v
```

- [ ] **Step 9c: Commit**

```bash
git add docs/reports/2026-03-12-tmgg-review/SUMMARY.md
git commit -m "docs: mark G50 FIXED in review summary"
```
