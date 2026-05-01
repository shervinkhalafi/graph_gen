# GraphData-native generation + evaluation pipeline (minispec)

**Date:** 2026-05-01
**Status:** PROPOSED — awaiting review
**Motivation:** unblock molecular post-hoc eval-all *and* in-training
async-eval. Today the discrete-diffusion eval CLI is `nx.Graph`-native
end to end; `MolecularEvaluator` consumes `Sequence[GraphData]`; the
two cannot meet. Today's gap is hidden because no molecular run has
ever invoked the async-eval callback (was missing until c69fb4ed),
but it would fire on the first eval call of any future molecular
training run.

## Goal

Make `GraphData` the canonical currency between sampler and
evaluator. Convert to `nx.Graph` *only at the leaves*: the
`GraphEvaluator` MMD path (already has the helper), the viz layer,
and the edge-list dump. `MolecularEvaluator` then plugs in
unchanged.

## Non-goals

- Reworking `MolecularEvaluator` itself (it's already the right
  shape).
- Adding new metric families.
- Touching the trainer's loss path or per-step diagnostics.
- Backporting the existing 3 finished QM9/MOSES/GuacaMol runs to
  produce gen-val/* (their checkpoints stay readable; once the
  pipeline is fixed the eval-all worker yields full metrics for
  them).

## Concrete change set

### 1. `GraphData.to_networkx()`

**File:** `src/tmgg/data/datasets/graph_types.py`

New instance method:

```python
def to_networkx(self, batch_index: int | None = None) -> nx.Graph[Any]:
    """Convert one graph slice to a NetworkX graph, masked + binarised.

    For 2-D (batched) GraphData pass batch_index. For 1-D (single)
    pass None (the default). Masking honours node_mask; edge
    threshold via binarised_adjacency. Node and edge class indices
    (when present) ride along as 'x_class' / 'e_class' attributes
    so downstream consumers can reason about atom / bond types.
    """
```

Implementation: reuse the existing `binarised_adjacency()`. Pull
``int(node_mask[i].sum())`` to size the slice. If `X_class is not
None`, attach `argmax(-1)` as a per-node attribute. If `E_class is
not None`, attach `argmax(-1)` as a per-edge attribute (only for
edges that survived the binary threshold).

Why on `GraphData` rather than as a helper on the evaluator: the
conversion is data-shape, not evaluator-shape. Future code that wants
"give me the nx view of this batch slice" (sweep tooling, debugging
notebooks) shouldn't have to import an evaluator class to do it.

**Add a single batch helper next to it:**

```python
def to_networkx_list(self) -> list[nx.Graph[Any]]:
    """Convert a batched GraphData to a per-graph NetworkX list."""
```

This subsumes today's `GraphEvaluator.to_networkx_graphs(...)` for
the non-batched-list case.

### 2. `DiffusionModule.generate_graphs` returns `list[GraphData]`

**File:** `src/tmgg/training/lightning_modules/diffusion_module.py`

Signature change:

```python
def generate_graphs(
    self,
    num_graphs: int,
    *,
    collector: StepMetricCollector | None = None,
) -> list[GraphData]:
```

Body change: drop the GraphData→nx conversion that lives at the end
of the method today. Sampler already produces GraphData; that's what
we now return.

Callers (3 in src + 1 in eval-all dump) get nx via
`[gd.to_networkx() for gd in result]` at the call site that needs
nx (viz / edge-list dump). The trainer's `on_validation_epoch_end`
threads `list[GraphData]` directly into `evaluator.evaluate(...)`.

### 3. `BaseDataModule.get_reference_graphs` returns `list[GraphData]`

**File:** `src/tmgg/data/data_modules/base_data_module.py`

**Replace** the existing method (no parallel sibling). After the
change:

```python
def get_reference_graphs(
    self, stage: str, max_graphs: int
) -> list[GraphData]:
    """Pull up to max_graphs reference graphs as per-graph GraphData."""
```

Walks the val/test loader, slices each batched `GraphData` along
the leading dim into a list of single-graph GraphData by re-indexing
the optional split fields (`X_class`/`X_feat`/`E_class`/`E_feat`/`y`)
and `node_mask` per row. Stops when `len(out) >= max_graphs`.

Decision (D2 revised): GraphData is the universal transport
format for the *whole* repo. Keeping a parallel nx-returning
method invites drift; one method, one shape. Callers that need
nx call `gd.to_networkx()` at the consumption site (cheap, local).

**Caller updates** (all single-line):
- `DiffusionModule.on_validation_epoch_end` (line 1955): pass
  GraphData straight through to `evaluator.evaluate(...)`.
- `DenoisingModule` (line 641): unchanged consumer signature once
  `GraphEvaluator.evaluate` accepts GraphData (step 4).
- `FinalSampleDumpCallback` (line 294): callers that need nx for
  matplotlib / chain saving wrap with `[gd.to_networkx() for gd in ...]`.
- `evaluate_cli._collect_reference_graphs` (line 302): pass
  through; CLI is GraphData-native now.
- `analysis/digress-loss-check/validate-gdpo-sbm/validate.py:448`,
  `diagnose_orbit_mmd.py:126`: one-line nx wrap at the consumer.

**Test updates** (mechanical): the ~10 test files asserting
`isinstance(g, nx.Graph)` change to assert `isinstance(g, GraphData)`
or convert and re-assert. Existing semantic checks (node count,
edge count) keep working via `gd.to_networkx()` or via
`gd.binarised_adjacency()` + `gd.node_mask`.

### 4. `GraphEvaluator.evaluate` accepts `list[GraphData]`

**File:** `src/tmgg/evaluation/graph_evaluator.py`

Signature change: both `refs` and `generated` become `list[GraphData]`.
Inside, the first thing it does is convert via the existing
`self.to_networkx_graphs(...)` for the MMD/structure metrics path.
No metric math changes; only the entry-point shape.

`MolecularEvaluator.evaluate` stays as-is (already
`Sequence[GraphData]`).

### 5. CLI: relax type check, route correctly

**File:** `src/tmgg/experiments/discrete_diffusion_generative/evaluate_cli.py`

`_load_graph_evaluator` → `_load_evaluator` (returns
`GraphEvaluator | MolecularEvaluator`). Drop the strict isinstance
check; accept either.

`_collect_reference_graphs` → switches from
`get_reference_graphs(...)` (nx) to
`get_reference_graph_data(...)` (GraphData).

`evaluate_checkpoint`:
- Fetch `ref_graph_data` (list[GraphData]).
- Call `module.generate_graphs(N)` — now returns list[GraphData].
- Call `evaluator.evaluate(refs=ref_graph_data, generated=gen_graph_data)`
  — works for both flavours via duck typing.
- For viz / edge-list dump: build `[gd.to_networkx() for gd in ...]`
  lists alongside (cheap).
- For val-pass: unchanged; that path uses `validation_step`, which
  already operates on raw `GraphData` batches from the dataloader.

### 6. Dump module: nx-from-GraphData at the leaf

**File:** `src/tmgg/experiments/discrete_diffusion_generative/_eval_dump.py`

`dump_eval_artifacts` accepts `list[GraphData]` for both ref and
gen. Internally derives `nx.Graph` lists via `gd.to_networkx()` for
viz + edge-list serialisation. Edge-list serialisation gains an
extra metadata block with X_class / E_class indices (when present)
so molecular outputs let analysis recover atom / bond types
without re-decoding.

### 7. Other callers

- `final_sample_dump.py` callback: today consumes
  `pl_module.generate_graphs(N)`. Switch to `[gd.to_networkx() for
  gd in pl_module.generate_graphs(N)]` at the consumption site.
- `chain_saving.py`: pass-through; sampler chain plumbing already
  works on GraphData.
- `analysis/digress-loss-check/validate-gdpo-sbm/`: legacy script,
  uses generate_graphs; one-line change.

## Files to touch

| File | What |
|---|---|
| `src/tmgg/data/datasets/graph_types.py` | Add `to_networkx`, `to_networkx_list` |
| `src/tmgg/data/data_modules/base_data_module.py` | `get_reference_graphs` returns `list[GraphData]` (replaces nx return) |
| `src/tmgg/training/lightning_modules/diffusion_module.py` | `generate_graphs` → `list[GraphData]`; `on_validation_epoch_end` threads GraphData; viz call updated |
| `src/tmgg/evaluation/graph_evaluator.py` | `evaluate(refs, generated)` accepts `list[GraphData]`; internal convert via existing helper |
| `src/tmgg/experiments/discrete_diffusion_generative/evaluate_cli.py` | `_load_evaluator` (drop isinstance), `_collect_reference_graphs` → GraphData, route both evaluator flavours |
| `src/tmgg/experiments/discrete_diffusion_generative/_eval_dump.py` | Take GraphData; convert to nx at leaf for viz + edge-list |
| `src/tmgg/training/callbacks/final_sample_dump.py` | One-line nx conversion at consumption |
| `analysis/digress-loss-check/validate-gdpo-sbm/validate.py` | One-line nx conversion |
| Tests | New + updated (see below) |

## Tests

**New:**
- `tests/data/test_graph_data_to_networkx.py`: round-trip for a
  hand-crafted GraphData with known X_class / E_class; node_mask
  honoured; batched + non-batched cases.
- `tests/training/test_generate_graphs_returns_graph_data.py`:
  the type contract change is enforced and round-trips through
  `to_networkx`.
- `tests/evaluation/test_graph_evaluator_accepts_graph_data.py`:
  passing a `list[GraphData]` produces the same metrics as today's
  `list[nx.Graph]` path.
- `tests/experiments/test_evaluate_cli_routes_molecular.py`:
  hydra-resolves a MOSES config, the CLI's `_load_evaluator`
  returns a `MolecularEvaluator`, and the `evaluate_checkpoint`
  call path doesn't hit the old TypeError.

**Updated** (all touch `get_reference_graphs` / `generate_graphs`
type assertions; mechanical sweep):
- `tests/data_modules/test_reference_graphs.py`: assertions go
  from `isinstance(g, nx.Graph)` to `isinstance(g, GraphData)`;
  graph-shape checks rewritten via `gd.binarised_adjacency()` /
  `gd.node_mask` or via `gd.to_networkx()`.
- `tests/test_datamodule_contracts.py`, `tests/test_single_graph_datasets.py`,
  `tests/experiment_utils/test_data_module.py`,
  `tests/experiment_utils/test_spectre_sbm_datamodule.py`: same
  treatment.
- `tests/training/test_diffusion_module_*`: any direct
  `generate_graphs` assertions that expected `nx.Graph` now expect
  `GraphData`. About 3 sites by grep.
- `tests/modal/test_eval_all_persistence.py`: dump tests already
  pass nx through `_render_graph_png`; the change in upstream
  produces nx via `to_networkx`. No test change unless we add a
  GraphData-input shape regression.

**Removed:** none.

## Sequencing (commit boundaries, narrow → wide)

1. `feat(graph-data)`: add `GraphData.to_networkx` +
   `to_networkx_list` + tests. No behaviour change anywhere else.
2. `refactor(datamodule)`: switch `get_reference_graphs` to
   return `list[GraphData]`; mechanical update to the ~10 test
   files + 6 src callers (consumer-side `to_networkx()` wrap where
   nx was assumed). This is the second breaking-change commit;
   bundles cleanly with step 3.
3. `refactor(generate-graphs)`: switch `DiffusionModule.generate_graphs`
   to return `list[GraphData]`; update all 3 in-tree callers
   (validation epoch end, FinalSampleDumpCallback, validate-gdpo-sbm).
   Tests updated.
4. `refactor(evaluator)`: `GraphEvaluator.evaluate` accepts
   `list[GraphData]`; internal convert. Update validation epoch end
   to pass GraphData. Tests.
5. `feat(cli)`: relax `_load_evaluator`, switch
   `_collect_reference_graphs` to GraphData; `_eval_dump`
   converts at the leaf. Tests for the molecular CLI route.
6. `chore(deploy)`: redeploy tmgg-eval-all + respawn the 3 detached
   calls.

Each commit is independently revertable. Steps 1-2 are pure
additions. Step 3 is the breaking-change commit; once it lands the
rest cascade.

## Risk + open questions

1. **Memory**: keeping per-batch GraphData objects (instead of
   nx.Graph) carries the categorical tensors around longer.
   Refs at 500 graphs × QM9 × ~30 nodes × 4 bond classes = a few
   hundred MB. Tolerable.
2. **batched vs unbatched GraphData semantics**: today
   `GraphData.__post_init__` accepts both. The new
   `get_reference_graph_data` slices a batched GraphData into a
   list of single-graph GraphData; need to confirm
   `__post_init__` accepts the bs=None form (it does — line
   84-92 explicitly supports 1-D node_mask).
3. **`MolecularEvaluator.evaluate` currently iterates one
   GraphData at a time** in `_decode_all` (line 70). If we hand it
   a list of single-graph GraphData (post-slice), each
   `codec.decode(data)` call sees one graph — matches today's
   trainer-side path that batched single-graph GraphData via the
   sampler. No change.
4. **Edge-list serialisation gains node/edge class fields** for
   molecular runs only (the `X_class`/`E_class` argmax indices
   ride along as nx attributes). For graph-only runs the attrs are
   absent so old analysis scripts keep working.
5. **Pickle safety**: nothing changes; we still write JSON, never
   pickle.

## Out of scope (do not do)

- Adding any new metric to `MolecularEvaluator`.
- Refactoring the Hydra config layout for evaluators.
- Changing the W&B logging keys (`gen-val/*` stays).
- Changing how `_size_distribution` is populated; the trainer
  flow still does it via `setup()`. The CLI flow still patches it
  in via the val-pass helper added in commit `4229fe3e`.

## Time estimate

- Steps 1-2 (additions only): ~1.5h with tests.
- Step 3 (breaking-change): ~1h plus 30 min to fix in-tree callers.
- Step 4: ~45 min.
- Step 5: ~1h plus tests.
- Step 6 (deploy + respawn): 10 min.

**Total: ~half a day.** Plus a session-length re-run on Modal of
the 3 datasets to actually produce gen-val/* metrics.

## Approval

Please flag anything you'd like to change before I start. Key
decisions worth confirming:

- (D1) `to_networkx` lives on `GraphData` (vs in evaluator). I
  picked GraphData for the reasons in §1.
- (D2) **REVISED per review**: `get_reference_graphs` is
  *replaced* (not paralleled) to return `list[GraphData]`. GraphData
  is the universal transport format end-to-end; nx is a leaf
  conversion via `to_networkx()`. Affects 6 src callers + ~10 test
  files (mechanical sweep). Worth flagging that this widens the
  blast radius of step 2 from "pure addition" to "breaking change";
  blast radius is fully in-tree and easy to audit.
- (D3) `GraphEvaluator.evaluate` signature change is breaking for
  any out-of-tree caller. In-tree there are 2 callers (validation
  epoch end + the CLI); both updated in the same commit.
- (D4) Per-graph node/edge class attributes ride along in the
  edge-list JSON for molecular runs. Adds ~30% to the per-ckpt
  graph dump size for QM9 (small) and ~2x for GuacaMol. Acceptable
  IMO; flag if you'd prefer to keep edge-list strictly topological
  and dump categorical indices separately.
