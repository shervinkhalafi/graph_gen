# Unified graph-feature representation: separating edge encoding from edge features

**Status:** Draft
**Type:** Design doc
**Author:** Claude + Igor (2026-04-15 pairing)
**Change class:** Breaking — atomic rewrite of the batch schema. No
compatibility shim, no deprecation window. Every consumer of
`GraphData` in the repository changes in the same commit cluster;
callers that do not are caught by type-check / test failure, not by
runtime warnings.

**Related:**

- `docs/reports/2026-04-15-upstream-digress-parity-audit.md` (motivating
  the need for architecture-task unification)
- `docs/reports/2026-04-15-bug-modal-sigabrt.md` (tangential — why the
  evaluator was exercised end-to-end)

**Normative language.** The key words "MUST", "MUST NOT", "SHOULD",
"SHOULD NOT", and "MAY" are to be interpreted as described in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) when, and only
when, they appear in all capitals.

## Context & scope

`GraphData` currently carries four fields: `X`, `E`, `y`, `node_mask`.
The edge tensor `E` has shape `(bs, n, n, de)`, where callers read `de`
by convention:

- `de = 1` means the last axis carries a *continuous* scalar per edge —
  adjacency weight, noise realisation, regression target. The
  single-step denoising path lives here: every architecture under
  `src/tmgg/models/{gnn,spectral_denoisers,baselines,attention,hybrid}/`
  pulls the scalar via `data.to_edge_state()` and writes the scalar
  back via `GraphData.from_edge_state(...)` (which hardcodes
  `de = 1` at `src/tmgg/data/datasets/graph_types.py:222`).
- `de ≥ 2` means the last axis carries a *categorical distribution*
  over edge classes (typically `{no-edge, edge}` for unweighted
  graphs, or `{no-edge, bond-single, bond-double, ...}` for typed).
  The DiGress discrete-diffusion path lives here. Categorical noise
  processes transform this tensor through stochastic transition
  kernels; the model emits logits and the loss is cross-entropy.

The overload is invisible from the type system. Two architectures with
the same `forward(data: GraphData, t) -> GraphData` signature interpret
`data.E` differently and emit incompatible shapes. The result is two
parallel pipelines, each architecture written against one, with bespoke
conversion at every boundary. The 2026-04-15 parity work established
that this split forces real friction: the existing single-step
denoising architectures (LinearPE, GNN, MultiLayerBilinearDenoiser,
…) cannot participate in a DiGress-style controlled-architecture
comparison without bespoke adapters, and bolting Gaussian-noise
experiments onto the discrete-diffusion path has no clean home.

This spec proposes a single representation that cleanly expresses:

- Discrete categorical encoding of edges and nodes (DiGress-style).
- Continuous per-edge and per-node features (Gaussian diffusion,
  weighted graphs, learned embeddings).
- Both at once, with per-field noise processes composing cleanly.
- Either alone, with the other field `None`.

The scope covers the data structure, the noise-process protocol, the
architecture contract, the evaluator, and the migration plan. It does
**not** cover replacing the Lightning module hierarchy, introducing
new noise paradigms, or touching non-graph datamodules.

## Goals

G1. One `GraphData` definition that supports categorical edges, continuous
edge features, both, or neither, without overloading a single axis.

G2. Architectures observe a stable input/output contract. An
architecture that predicts categorical edge classes SHOULD be usable
for continuous edge regression by only changing its configured output
widths, and vice versa. No wrapper class required.

G3. Noise processes compose by field. A training run MAY apply a
categorical noise process to `E_class` and a Gaussian noise process to
`E_feat` simultaneously. The noise process API MUST NOT assume a
single global field.

G4. Reference tests MUST demonstrate an architecture family (pick one
representative per family) operating on both categorical and continuous
edges with the same class, by config alone.

G5. The refactor lands atomically — one commit cluster, no
compatibility shim — because there are no external consumers of
`GraphData` and no checkpoint artefacts that serialise the batch
schema. Model weights (Lightning `state_dict`) key on parameter
paths, not on `GraphData` field names.

## Non-goals

- Replacing `DiffusionModule` / `SingleStepDenoisingModule` with a
  single Lightning module. They remain distinct — multi-step reverse
  chain is a different training contract from single-shot denoising.
- Introducing new noise process types (flow matching, score-based on
  manifolds, etc.). The spec standardises the existing two
  (`CategoricalNoiseProcess`, Gaussian) and leaves the hook for more.
- Edge directionality. Both `E_class` and `E_feat` remain symmetric
  `(n, n, d_*)` tensors for undirected graphs; directed graph
  support is an orthogonal concern.
- Hypergraphs or higher-order interactions.
- Per-node-pair variable feature widths. Every edge in a graph batch
  MUST have the same `de_class` and `de_feat`.
- Backward compatibility with pre-refactor `GraphData`, pre-refactor
  `.ckpt` files, or pre-refactor wandb dashboards. The refactor is a
  hard cutover. Any artefact from before it lands must be migrated
  in the same commit (for in-repo consumers) or discarded (for
  saved checkpoints and dashboards). The spec does not provide
  transition tooling, aliases, or runtime warnings for legacy
  usage.

## Proposed representation

### New `GraphData` fields

`GraphData` is rewritten with four optional feature tensors; the
old `X` and `E` fields are removed outright (see "Removed fields"
below and the migration plan):

```
GraphData:
  X_class:    Tensor | None     # (bs, n, dx_class)           — categorical node PMF / one-hot
  X_feat:     Tensor | None     # (bs, n, dx_feat)            — continuous node features
  E_class:    Tensor | None     # (bs, n, n, de_class)        — categorical edge PMF / one-hot
  E_feat:     Tensor | None     # (bs, n, n, de_feat)         — continuous edge features / weights
  y:          Tensor            # (bs, dy)                    — global features (unchanged)
  node_mask:  Tensor            # (bs, n)                     — boolean validity (unchanged)
```

**Semantic invariants** (RFC 2119):

- `node_mask` MUST be non-`None` and carries node existence. This is
  the sole required signal of "which positions in the batch are real
  nodes." Graph-theoretic experiments without per-node labels or
  features SHOULD leave all `X_*` fields `None`.
- At construction time, at least one of `E_class` / `E_feat` MUST be
  non-`None`. A graph without edge information has no graph structure
  to reason about and is outside this spec's scope. Node fields are
  entirely optional.
- The `X_*` fields are optional and carry *additional* per-node
  information beyond existence:
  - `X_class` holds categorical node labels — atom type, community
    index, annotated role. DiGress's current "node absent / node
    present" one-hot is, under this spec, a degenerate `X_class`
    with `dx_class=2` that SHOULD NOT be emitted by new datasets;
    use `node_mask` alone and omit `X_class` for structure-only
    graphs. More generally, datasets SHOULD NOT emit an `X_class`
    whose value at every position is uniquely determined by
    `node_mask`.
  - `X_feat` holds continuous per-node attributes — coordinates,
    learned embeddings, precomputed spectral features.
- `E_class[..., 0]` SHOULD encode "no edge" when `E_class` is present.
  This matches DiGress's convention and lets "does this position have
  an edge?" reduce to "is the argmax non-zero?" uniformly.
- Shapes: `node_mask.shape == (bs, n)`. Whenever `X_*` is non-`None`,
  `X_*.shape[:2] == (bs, n)`. Whenever `E_*` is non-`None`,
  `E_*.shape[:3] == (bs, n, n)`.
- `y` is never `None` — an empty graph-level vector is represented as
  `torch.zeros(bs, 0)`, consistent with current practice.
- Symmetry: for undirected graph datasets, both `E_class` and `E_feat`
  MUST satisfy `x[..., i, j, :] == x[..., j, i, :]` at dataset
  construction time. Architectures are not required to preserve
  symmetry in every intermediate tensor; collation layers MUST
  re-symmetrise before emitting the final dataset sample.

**Architectures without explicit node features.** Some architectures
internally require a per-node feature tensor (transformer blocks
fusing X into Q/K/V, positional encoding layers seeded with X). When
all `X_*` fields are `None`, those architectures derive whatever they
need — a constant embedding per real node, a spectral embedding from
the adjacency, a learned parameter — from `node_mask` and the `E_*`
fields. That derivation is an architecture-internal concern; the
data model's job stops at "here is the graph, here is which
positions are real." Datasets MUST NOT re-encode `node_mask` as a
degenerate `X_class` just to feed architectures that currently
assume one.

### Removed fields

`GraphData.X` and `GraphData.E` are *removed*, not deprecated. The
refactor lands atomically — see the migration plan below.
Callsites that previously read `data.X` MUST be rewritten to read
the appropriate `_class` / `_feat` field; reads of `data.E` likewise.
Pyright will enforce this; any missed callsite fails at import or
at first access. No runtime shim, no `DeprecationWarning`, no
heuristic to guess whether an old tensor is "class" or "feat".

### Noise process per-field protocol

The base `NoiseProcess` interface (currently in
`src/tmgg/diffusion/noise_process.py`) is **rewritten** — not
extended — to carry a field declaration. Every method on every
concrete subclass (`CategoricalNoiseProcess`,
`ExactDensityNoiseProcess`, and the new `GaussianNoiseProcess`)
changes signature in the same commit cluster.

```
class NoiseProcess(ABC):
    fields: frozenset[str]  # subset of {"X_class", "X_feat", "E_class", "E_feat"}

    def forward_sample(data: GraphData, t: Tensor) -> GraphData: ...
    # MUST leave non-``fields`` entries untouched (structural copy of
    # references). MUST write the noised value into each field in
    # ``fields``.

    def posterior_sample(z_t, x0_param, t, s) -> GraphData: ...
    def posterior_log_prob(x_s, z_t, x0_param, t, s) -> Tensor: ...
    # Same contract: only read/write the declared fields.
```

Concrete classes:

- `CategoricalNoiseProcess` declares `fields = {"X_class", "E_class"}`.
  (It MAY be configured with a narrower subset; in practice DiGress
  uses both simultaneously at the same `t`.)
- `GaussianNoiseProcess` (new) declares
  `fields = {"E_feat"}` for pure edge-weight diffusion, or
  `fields = {"X_feat", "E_feat"}` for joint continuous diffusion.
- `ExactDensityNoiseProcess` (abstract intermediate) MUST be field-
  aware: the VLB estimators in `DiffusionModule.validation_step` MUST
  iterate over `fields` and accumulate per-field KL / reconstruction
  terms, then sum with configurable per-field weights.

Multiple noise processes compose trivially by applying each in
sequence; their `fields` sets MUST be disjoint. Composition checks
this invariant at Lightning-module `__init__` time and MUST raise a
`ValueError` naming the overlapping field(s). This makes a hybrid
experiment (categorical edge classes + Gaussian edge weights) a
matter of instantiating two noise processes and listing them.

**Shared timestep across composed processes.** When multiple noise
processes apply at the same training step, the Lightning module
samples one `t_int` per batch and passes it to every composed
process; each process returns its own `process_state_condition_vector(t)`
projection, and the per-process scalar vectors are concatenated in
a stable order (the order of the `noise_processes` list on the
module) to form the `t` passed to the architecture. The
architecture's `use_timestep` mechanism thus sees a single vector
whose length is the sum of per-process condition dimensions; for
all existing processes this is simply `len(noise_processes)` scalar
entries.

**Reverse-sampler composition.** The reverse sampler loop iterates
over timesteps `t → s = t − 1` as today. At each step the sampler
invokes every composed process's `posterior_sample` in list order;
because their field sets are disjoint, the order has no effect on
semantics for independent fields. When cross-field interactions are
introduced in future work, this rule will need revisiting; see the
Open Issues section.

Rationale for declaring `fields` as a set: the alternative — one
noise process per field — would force a DiGress-style run to compose
two categorical processes (one for `X_class`, one for `E_class`) and
then conditionally tie their `t` samples. Declaring a set keeps the
common case (DiGress: joint `X_class` + `E_class`) a single object
while preserving composability.

### Architecture contract (unchanged signature, stricter semantics)

`GraphModel.forward(data: GraphData, t: Tensor | None = None) -> GraphData`
is preserved as the single architecture contract.

New semantics (RFC 2119):

- The returned `GraphData` MUST populate the same `_class` / `_feat`
  fields that are populated in the input, unless explicitly configured
  otherwise via the architecture's output-dim config.
- Architectures that compute internally on a scalar adjacency view
  (spectral methods, GCN, etc.) MAY pull the scalar via a helper —
  but the helper MUST be explicit about whether it reads from
  `E_class` (via argmax over the class axis, or via
  `1 − P(no_edge)`) or from `E_feat` (direct scalar read). The
  single helper `data.to_edge_scalar(source: Literal["class",
  "feat"]) → (bs, n, n)` replaces the legacy `to_edge_state`.
  Architectures that emit categorical logits write to `E_class`
  directly; no separate `to_edge_logits` helper is introduced.
- The timestep `t` comes from the noise process via
  `process_state_condition_vector(t_int)` and is a normalised scalar
  per-sample. Architectures MAY concatenate `t` onto `data.y` if they
  choose to condition on it (`GraphTransformer` does so already;
  others SHOULD follow the same two-line pattern).
- Output widths are configuration. Every architecture MUST expose
  constructor parameters `output_dims_x_class`, `output_dims_x_feat`,
  `output_dims_e_class`, `output_dims_e_feat` (each may be `None` or
  a positive integer). `None` means the architecture does not predict
  that field and sets it to `None` on the returned `GraphData`.

### Evaluator contract

`graph_evaluator.py` currently derives a binary adjacency via
`to_binary_adjacency()` to feed into MMD / SBM / orbit metrics. The
new contract:

- If `E_class` is non-`None`, binary adjacency is derived by argmax
  over the class axis, then mapping non-zero classes to `1`.
- Else if `E_feat` is non-`None`, binary adjacency is derived by
  thresholding the scalar feature. Default threshold is `0.5`; the
  evaluator MUST accept a configurable `binarise_threshold` for
  experiments that care (score-based continuous diffusion often
  produces values outside `[0, 1]`).
- If both are non-`None`, the evaluator MUST use `E_class` and MAY
  emit a log warning when the two disagree on edge presence. The
  disagreement rate is computed per graph as

  ```
  disagreement = mean(
      (argmax(E_class, dim=-1) != 0) != (E_feat[..., 0] > binarise_threshold)
  )
  ```

  averaged over the batch. The evaluator MUST emit a single warning
  per validation pass when the mean disagreement exceeds
  `disagreement_warn_threshold` (default `0.05`).

## Key algorithms and data flow

### Training step

```
def training_step(batch: GraphData, ...):
    t_int = sample_timesteps(batch, noise_process)
    z_t = noise_process.forward_sample(batch, t_int)
    condition = noise_process.process_state_condition_vector(t_int)
    pred = model(z_t, t=condition)

    loss = 0.0
    for field in noise_process.fields:
        loss_type = noise_process.loss_for(field)  # "ce" or "mse"
        loss += loss_type(pred[field], batch[field], weight=lambda[field])
    return loss
```

`lambda[field]` is the per-field loss weight. Mirrors DiGress's
`lambda_E = 5.0` (the weight for `E_class`) generalised to every
field.

### Validation VLB

`DiffusionModule.validation_step` already uses analytic categorical
KL for the DiGress path (per Phase D of 2026-04-15). Under this spec
the VLB decomposes as a per-field sum:

```
NLL = -log_pN + sum_field kl_prior[field] + T * sum_field kl_diffusion[field] - sum_field reconstruction[field]
```

The `log_pN` term remains graph-level (node-count). Each `kl_*` is
computed on the per-field PMF (categorical) or per-field density
(Gaussian), with Gaussian KL in closed form given the schedule's
variance trajectory.

### Gaussian noise process (new)

```
class GaussianNoiseProcess(NoiseProcess):
    fields: frozenset[str]  # typically {"E_feat"} or {"X_feat", "E_feat"}

    def forward_sample(data, t):
        alpha_bar = self.schedule.get_alpha_bar(t)
        out = copy_references(data)
        for f in self.fields:
            x_0 = getattr(data, f)
            noise = torch.randn_like(x_0)
            setattr(out, f,
                    sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise)
        return out

    def posterior_sample(z_t, x0_param, t, s):
        # standard DDPM closed-form posterior mean + variance, per field.
        ...
```

This is not new research; it is the reference DDPM Gaussian
parametrisation, applied per-field. The implementation is
~100 lines.

## Migration plan — atomic full-repo rewrite

This is a single breaking-change commit cluster. Every consumer of
`GraphData` in the repository is rewritten in the same merge. The
old `X` / `E` fields and the `to_edge_state` / `from_edge_state`
helpers are deleted outright. No shim, no deprecation warnings, no
heuristic fallback path. Callsites that miss the migration fail
type-checking or test execution; none silently use the wrong field.

**This is viable because:**

- `GraphData` is the in-memory batch type. It is not serialised to
  disk anywhere. Lightning checkpoints save `state_dict` (keyed on
  `nn.Module` parameter paths) and `hparams` — neither references
  `GraphData` field names.
- Wandb / CSV log schemas name metrics, not batch fields.
- There are no external consumers of `GraphData`; it is entirely
  internal to this repository.
- Type-check (`basedpyright`) + the existing test suite provide a
  conservative completion signal: every field-access site that
  pyright accepts and every test that passes is correctly migrated.

**The single commit cluster, explicitly, modifies every consumer:**

1. **Dataclass.** `src/tmgg/data/datasets/graph_types.py` —
   `GraphData` rewritten with the new fields; `X` and `E` deleted;
   `from_binary_adjacency` / `from_edge_state` /
   `to_binary_adjacency` / `to_edge_state` replaced by
   `from_structure_only`, `from_edge_scalar`, `to_edge_scalar`,
   `replace` (dataclass-style with-copy).
2. **Datasets and collation.** `src/tmgg/data/data_modules/*.py` —
   every `GraphData(...)` constructor in
   `_collate_pyg_to_graphdata`, `GraphData.from_pyg_batch`, and
   every datamodule's setup path uses the new field names
   explicitly.
3. **Noise processes.** `src/tmgg/diffusion/noise_process.py` —
   `NoiseProcess` gains `fields: frozenset[str]`. Every method
   signature on every subclass is updated. `CategoricalNoiseProcess`
   reads/writes `X_class` / `E_class`; `GaussianNoiseProcess` (new)
   reads/writes `E_feat` (and optionally `X_feat`).
4. **Sampler.**
   `src/tmgg/diffusion/sampler.py`,
   `src/tmgg/diffusion/diffusion_sampling.py` — every `data.X` /
   `data.E` becomes explicit field access. Helpers that assumed a
   single tensor (`compute_posterior_distribution`,
   `compute_posterior_distribution_per_x0`, etc.) are generalised
   to operate on a named field tensor rather than a fixed
   position.
5. **Lightning modules.**
   `src/tmgg/training/lightning_modules/{diffusion_module.py,
   denoising_module.py, base_graph_module.py, train_loss_discrete.py,
   digress_checkpoint_compat.py}` — training / validation / test
   steps iterate the noise process's `fields` and accumulate loss
   per field with configurable per-field weights.
6. **Evaluator.** `src/tmgg/evaluation/graph_evaluator.py` —
   `to_binary_adjacency()` is rewritten to read `E_class` via
   argmax when present, `E_feat` with a configurable threshold
   otherwise, and error on ambiguity.
7. **Architectures.** `src/tmgg/models/**/*.py` — every `forward`
   method in every architecture family (GraphTransformer, GNN
   family, spectral denoiser family, baselines, hybrid, attention)
   is rewritten to read and write the new fields via the new
   helpers. Constructors gain per-field output-dim parameters.
8. **Tests.** `tests/**/*.py` — every fixture, every shape
   assertion, every mocked GraphData is rewritten. No test relies
   on the legacy interface.
9. **Configs.** `src/tmgg/experiments/exp_configs/models/**/*.yaml`
   — where architecture configs declare output dims, they now
   declare per-field output dims.

**Completion checks (all MUST pass before merge):**

- `rg "\\.X\\b|\\.E\\b" src tests` returns zero hits where the
  receiver is a `GraphData` (tolerated elsewhere — there is
  unrelated code with `.X` / `.E` on other types).
- `rg "to_edge_state|from_edge_state|from_binary_adjacency"` returns
  zero hits — the legacy helpers are entirely gone.
- `basedpyright` clean across the tree.
- `uv run pytest tests/ -x -m "not slow"` green.
- End-to-end Modal smoke run on SPECTRE SBM (the configuration we
  already know works) produces `val/loss` within `1e-5` absolute
  tolerance of a pre-refactor run with the same seed. The refactor
  is a read-path rewrite at training time: operator ordering is
  unchanged, so the tolerance accounts only for floating-point
  non-associativity introduced by any reordering of constructor
  argument evaluation. Numerical equivalence no longer holds once
  datasets stop emitting the degenerate `X_class` (because input
  shapes change); that transition is out of scope for the atomic
  refactor commit and handled by follow-up dataset edits.

**Branch strategy.** Dedicated branch
(`refactor/graph-data-split` or similar). No other unrelated
changes on the branch. Merge atomically via squash commit to keep
`main` history readable; the intra-branch commits MAY remain
granular for bisectability during review.

**Rollback.** `git revert` on the squash commit. Because the
refactor touches every consumer, partial rollback is impossible by
design — that is the point. If it breaks, it breaks everywhere and
is easy to diagnose; nothing drifts out of sync silently.

## Cross-cutting concerns

**Performance.** The split introduces no new tensor operations on the
hot path. Existing code becomes a pointer-chase through one extra
attribute; negligible compared to the forward pass. Gaussian noise
adds its own FLOPs only when configured.

**Memory.** In the common cases (pure DiGress or pure denoising), only
one of `_class` / `_feat` is non-`None` per node/edge axis. No memory
regression. The hybrid case (both set) doubles edge-tensor memory,
which is expected and opt-in.

**Testing.**
- `test_graph_data.py` gains tests for every valid combination of
  field occupancy and an invariant check for the "at least one `E_*`
  field non-None" rule.
- Architecture tests get parametrised over `(e_class, e_feat)`
  occupancy, asserting each family can run both modes.
- A cross-module test constructs a DiGress run and a continuous-edge
  run with *the same* architecture (e.g. `LinearPE`), confirming the
  outputs differ only in the field they populate.
- A failure-mode test asserts that composing two noise processes
  with overlapping `fields` raises `ValueError` at Lightning-module
  `__init__` time, with a message naming the offending field.

**Observability.** Metric names become per-field under the new
schema — `val/gen/e_class/degree_mmd`, `val/gen/e_feat/mse`, and so
on. Single-field runs (the overwhelming majority post-refactor)
produce metric keys with a predictable family-prefix; multi-field
runs yield one metric per field. Existing wandb dashboards that
reference the pre-refactor names (`val/gen/degree_mmd`) will need
one-time relabelling — there is no compatibility alias.

## Failure modes and error handling

Explicit contract for cases where construction, composition, or
loading go wrong:

| Failure | Where detected | Response |
|---|---|---|
| `GraphData` constructed with both `E_class=None` and `E_feat=None` | `GraphData.__post_init__` | Raise `ValueError` with message citing this spec's invariant. |
| Two composed noise processes share a field | `LightningModule.__init__` | Raise `ValueError`; message names the offending field. |
| Architecture configured with `output_dims_e_class=k` where `k ≠ dataset de_class` | First training-step forward pass | Pyright catches most; runtime shape mismatch surfaces as a torch shape-error. No dedicated spec-level guard. |
| Pre-refactor checkpoint loaded post-refactor | Lightning checkpoint load | Raise `RuntimeError` with message "pre-unified-schema checkpoint; load a post-refactor checkpoint or retrain." Loader is not expected to migrate old `state_dict` keys. |
| Dataset emits degenerate `X_class` duplicating `node_mask` | Dataset validation in tests only | Spec violation; test asserts dataset SHALL NOT re-encode node existence. No runtime check — too expensive per batch. |
| `E_class` and `E_feat` disagree above `disagreement_warn_threshold` | Evaluator at `on_validation_epoch_end` | Log `WARNING` once per validation pass with the measured rate. Does not fail the run. |

Checkpoint compatibility is deliberately narrow: the refactor breaks
`.ckpt` files written before it lands. This is acceptable because
checkpoints are research artefacts on a volume, not deliverables;
the atomic-rewrite stance implies retraining from scratch after the
split. No shim, no heuristic.

## Alternatives considered

### A. Keep single `E`, encode both via channel layout conventions

Reserve first `de_class` channels for class logits, remaining for
continuous features; document the split in comments and helper
functions.

- Strengths: minimal API churn; existing callers unaffected.
- Weaknesses: the split lives in documentation, not types. Slicing
  errors would surface as silent semantic bugs. Architectures
  emitting only the categorical part must know the split width,
  duplicating config. Gaussian noise applied to "the feature part"
  needs a channel range everywhere.
- **Rejection:** we would spend the implementation budget on shim
  code that maintains an invariant the type system refuses to
  enforce. The split is real and the data model should reflect it.

### B. Dict of tensors (`GraphData.features: dict[str, Tensor]`)

Replace named fields with a free-form dict keyed by string.

- Strengths: maximally flexible; adding a new feature type costs
  zero schema changes.
- Weaknesses: no type safety; typos become silent bugs; IDE
  autocompletion degrades; serialisation order becomes undefined;
  pyright cannot check field presence or dtype.
- **Rejection:** flexibility is not the bottleneck. Clarity is. Named
  fields with typed Optionals give the same expressive power with
  checkable guarantees.

### C. Subclass `GraphData` per task (`CategoricalGraphData`,
`ContinuousGraphData`, …)

Runtime type identifies which variant you have.

- Strengths: architectures can type-narrow on the input variant.
- Weaknesses: hybrid runs need a sum type or a subclass per combo;
  collation and batching multiplies by variant; the Lightning
  contract is single-batch-type which doesn't compose well with
  runtime variants.
- **Rejection:** task-specific subclasses mean the datamodule commits
  to a task at construction time, which breaks G2 (one architecture,
  multiple tasks).

### D. Tag `E` with a `semantic` string attribute, leave the tensor shape

Add `GraphData.e_semantic: Literal["class", "feat"]`.

- Strengths: small API change, cheap.
- Weaknesses: same runtime-check problem as A; cannot express "both"
  at all.
- **Rejection:** doesn't support the hybrid case, which is one of the
  explicit goals.

The proposed design (named Optional fields per family) is the only
option that expresses pure-class, pure-feat, both-fields, and
(trivially) neither-field graphs without runtime tagging.

## Open issues

- **Per-field loss weights.** DiGress hardcodes `lambda_E = 5.0`. The
  spec proposes a `lambda[field]` dict. Do we want lambdas declared
  on the `NoiseProcess` (closer to the theory) or on the
  `LightningModule` (closer to the training script)? Leaning noise
  process, for symmetry with other process-dependent scalars.
  *Owner:* Igor.

- **Evaluator disagreement threshold.** When `E_class` and `E_feat`
  are both populated and disagree on edge presence for many
  positions, do we warn, error, or pick one with a config? Proposed:
  warn + default to `E_class`. Needs confirmation after a real
  hybrid experiment runs. *Owner:* TBD.

- **Naming collision with existing `GraphData.y`.** No conflict, but
  worth asking whether `y` should be promoted to per-field
  `y_class` / `y_feat` while we are at it, for symmetry. Current
  recommendation: no. The global vector is small and DiGress's
  `y` has historically mixed time-embedding + learned scalars
  without trouble. Revisit if a real use case emerges.

- **Symmetry enforcement at architecture output time.** Spec says
  datamodules symmetrise at load. Should architectures also
  symmetrise before returning? Current practice is silently
  symmetric for all existing architectures; making it a contractual
  MUST would catch regressions but add cost. *Owner:* TBD.

## References

- DiGress (Vignac et al., ICLR 2023): https://github.com/cvignac/DiGress
- SPECTRE (Martinkus et al., NeurIPS 2022):
  https://github.com/KarolisMart/SPECTRE
- DDPM Gaussian parametrisation (Ho et al., NeurIPS 2020):
  https://arxiv.org/abs/2006.11239
- Local parity audit:
  `docs/reports/2026-04-15-upstream-digress-parity-audit.md`
- RFC 2119 requirements language:
  https://www.rfc-editor.org/rfc/rfc2119

---

*This document is a design proposal, not an implementation plan. The
implementation plan derived from this spec should land as a separate
markdown under `docs/plans/` once the open issues are resolved.*
