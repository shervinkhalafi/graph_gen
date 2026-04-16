# Implementation plan — unified graph-feature representation

**Spec:** `docs/specs/2026-04-15-unified-graph-features-spec.md`
(signed off 2026-04-15).
**Branch:** `refactor/graph-data-split` (dedicated; no other unrelated
work lands on it).
**Merge strategy:** squash commit into `main`. The intra-branch
commits listed below remain granular for bisectability during
review; the merged change on `main` is one atomic commit per the
spec's clean-break stance.

## Why this plan exists

The spec defines the target state — what `GraphData` looks like
after, how the noise-process interface composes, what the
architecture contract guarantees. It does not define an order of
operations that keeps the codebase in a compilable state as the
refactor progresses, nor the tests and commands that gate each
step. That's what this plan adds.

The plan respects two principles from the spec:

1. **The final change on `main` is atomic.** Squash merge. No
   partial-state landing on the trunk.
2. **No shim, no deprecation on the trunk.** Any "both X/E and the
   new fields coexist" state exists only temporarily on the
   refactor branch, and is squashed out before merge.

Inside the branch, a brief coexistence state (old fields remain on
`GraphData` while new fields are added alongside) is the only way
to keep each intra-branch commit compilable for review. This
coexistence is internal scaffolding, not a shim: it is deleted
before the branch is merged.

## Scope

In scope: everything named in the spec's "Migration plan" section
9-step list (dataclass, datasets, noise processes, sampler,
Lightning modules, evaluator, architectures, tests, configs) plus
the new `GaussianNoiseProcess` and per-field `t`-condition
concatenation in the reverse sampler.

Out of scope: net-new features beyond the spec (e.g. score-based
methods, flow matching, directed graphs). These are non-goals in
the spec and remain non-goals here.

## Commit sequence (intra-branch)

Each commit below is expected to leave the tree in a green state:
`basedpyright` clean on touched files, `uv run pytest tests/ -x -m
"not slow"` passes. Commits grouped by "wave" — waves may land in
parallel conceptually but commit to git sequentially.

### Wave 0 — preparation

**Commit 0.1** · Add the new helper module `graph_data_fields.py`
alongside `graph_types.py`.
- Define `FieldName = Literal["X_class", "X_feat", "E_class", "E_feat"]`.
- Define `FIELD_NAMES: frozenset[FieldName]`.
- Define `GRAPHDATA_LOSS_KIND: dict[FieldName, Literal["ce", "mse"]]`
  mapping `_class → "ce"`, `_feat → "mse"`.
- Pure new code. Adds nothing to existing call paths.

*Tests:* a trivial assert on the sets. No behaviour change.

### Wave 1 — additive dataclass

**Commit 1.1** · Extend `GraphData` with the four new optional
tensors (`X_class`, `X_feat`, `E_class`, `E_feat`). Keep `X` / `E`
as required fields during this wave. Add `GraphData.replace(**kwargs)`
that mirrors `dataclasses.replace` with typed kwargs.
- `src/tmgg/data/datasets/graph_types.py`
- Constructor accepts new fields; they default to `None`.
- `mask()` / symmetry checks extend to the new fields when present.
- `__post_init__` validates shapes and the "at least one `E_*`
  non-None" invariant against the new fields; falls back to the old
  `E` shape check when the new fields are both `None` (transition
  only).

*Tests:* `tests/data/test_graph_data.py` (new): every valid field-
occupancy combination constructs cleanly; invalid combinations
raise `ValueError` with the expected message.

**Commit 1.2** · Add the new helpers:
- `GraphData.from_structure_only(node_mask, edge_scalar=...)` —
  constructs a graph with only `E_feat` populated from a scalar
  adjacency.
- `GraphData.from_edge_scalar(edge_scalar, ..., target: Literal[
  "E_class", "E_feat"])` — constructs either-target.
- `GraphData.to_edge_scalar(source: Literal["class", "feat"]) →
  Tensor` — replaces `to_edge_state`; for the `"class"` source,
  defaults to `1 − P(no_edge)`.

Existing helpers (`from_edge_state`, `from_binary_adjacency`,
`to_edge_state`, `to_binary_adjacency`) stay in place during the
transition waves. They will be deleted in Wave 8.

*Tests:* per-helper round-trip: build a known adjacency, pass it
through each helper, assert shape and value equality.

### Wave 2 — noise processes

**Commit 2.1** · Add `fields: frozenset[FieldName]` class attribute
to `NoiseProcess` (abstract base).
- `CategoricalNoiseProcess.fields = frozenset({"X_class", "E_class"})`.
- `ContinuousNoiseProcess` is renamed to `GaussianNoiseProcess`
  in-place (same file, same class body); `fields` set based on
  constructor argument, default `frozenset({"E_feat"})`.
- Method bodies still read `data.X` / `data.E`; they will switch
  to `data.X_class` / `data.E_class` in 2.2. This commit is
  the attribute + rename only.

*Tests:* `tests/diffusion/test_noise_process.py` gains asserts that
each concrete subclass has non-empty `fields`.

**Commit 2.2** · Rewrite `CategoricalNoiseProcess` forward /
posterior / log-prob methods to read/write the declared fields
(`X_class`, `E_class`) instead of the legacy `X` / `E`.
- Inputs to these methods still arrive with both sets populated
  (datamodule still fills both — wave 3 changes that). The
  method just reads the new fields now.
- `forward_sample(data, t)` returns a `GraphData` where `X_class`
  / `E_class` are noised and `X` / `E` are updated for
  continuity (set to the same tensors as the new fields).

*Tests:* existing VLB + sampler tests pass unchanged; add a
parity test asserting that reading `data.X_class` gives the same
tensor as `data.X` after forward-noise.

**Commit 2.3** · Implement `GaussianNoiseProcess` fully:
`forward_sample`, `posterior_sample`, `posterior_log_prob`,
`prior_log_prob`, `forward_log_prob` per the spec's pseudocode.
- Uses DDPM closed-form posterior mean + variance.
- Writes only fields in `self.fields`; leaves others untouched.

*Tests:* `tests/diffusion/test_gaussian_noise_process.py` (new) —
unit tests over a small graph: forward-noise statistics approach
`N(0, 1)` at large `t`, posterior matches DDPM formula, composed
with `CategoricalNoiseProcess` on a hybrid graph runs to
completion without raising.

**Commit 2.4** · Noise-process composition: add
`CompositeNoiseProcess` (or equivalent) that wraps a sequence of
processes with disjoint `fields`, raises `ValueError` at
construction if any overlap.
- `process_state_condition_vector(t)` returns a concatenation of
  per-process condition vectors.
- `forward_sample` / `posterior_sample` invoke each sub-process
  in list order.

*Tests:* assert overlap detection raises; assert two disjoint
processes produce a graph where each set of fields was noised by
the correct sub-process.

### Wave 3 — datasets and collation

**Commit 3.1** · Rewrite `_collate_pyg_to_graphdata` and
`GraphData.from_pyg_batch` to emit the new fields (`E_class`
populated for categorical datasets; `E_feat` for continuous). The
legacy `X` / `E` fields are *also* populated with equivalent values
during this wave so downstream code that hasn't been migrated yet
keeps working.
- `src/tmgg/data/data_modules/multigraph_data_module.py`
- `src/tmgg/data/datasets/graph_types.py` (from_pyg_batch)

**Commit 3.2** (deferred) · Stop emitting the degenerate
`X_class` in `SpectreSBMDataModule` and
`SyntheticCategoricalDataModule`. Structural dependency: `GraphData`
still carries a required `X` field in Waves 3–8. This commit lands
as part of Wave 9 (see Commit 9.3) once `X` is removed from the
dataclass. Listed here only to mark the logical sibling; do not
write code in this wave.

### Wave 4 — sampler

**Commit 4.1** · `src/tmgg/diffusion/diffusion_sampling.py`:
the per-x0 helper and the direct posterior helper accept a
`field: FieldName` argument. Internal logic is field-neutral
(operates on a tensor the caller passes in); the caller names
which field is being sampled.

**Commit 4.2** · `src/tmgg/diffusion/sampler.py`: the reverse loop
dispatches to `noise_process.posterior_sample_marginalised` /
`posterior_sample` with per-field invocation. For
`CompositeNoiseProcess`, sample each sub-process in list order.

*Tests:* existing sampler tests pass; new test asserts reverse
loop on a hybrid noise process produces a `GraphData` with both
fields populated and correct one-hot structure.

### Wave 5 — Lightning modules

**Commit 5.1** · `DiffusionModule.training_step` and
`validation_step`: replace hardcoded `batch.X` / `batch.E` accesses
with iteration over `noise_process.fields`. Loss is per-field sum
with `lambda_[field]` weights from config.
- `val/gen/<field>/<metric>` naming convention wired through
  wandb logger and CSV.

**Commit 5.2** · `SingleStepDenoisingModule` same treatment: the
single-shot denoising loss is still field-keyed.

*Tests:* training-step + validation-step existing tests pass with
unchanged numerical output. Add a parametrised test covering
`(categorical, gaussian, composed)` configurations.

### Wave 6 — evaluator

**Commit 6.1** · `src/tmgg/evaluation/graph_evaluator.py`: rewrite
`to_binary_adjacency()` to read `E_class` via argmax (non-zero →
`1`) when present, else `E_feat` thresholded. Add
`disagreement_warn_threshold` config field; emit one warning per
validation pass when both fields present and disagree above the
threshold (formula per spec).

*Tests:* existing evaluator tests pass; new unit tests for the
single-field paths and the disagreement-warn path with known
input.

### Wave 7 — architectures

One commit per family, in this order:

**Commit 7.1** · `GraphTransformer` (DiGress). Already uses
`use_timestep` via `y`-concat; update only the input-read side
from `data.X`/`data.E` to `data.X_class`/`data.E_class`, and
output-write side to return `GraphData(..., X_class=pred_X,
E_class=pred_E, ...)`. Add `output_dims_*` constructor params;
default values match current behaviour.

**Commit 7.2** · GNN family (`GNN`, `GNNSymmetric`, `NodeVarGNN`).
- Input: `data.to_edge_scalar(source="class")` when working on
  a categorical-edge dataset, `source="feat"` otherwise. Select
  via config — new field `edge_source: Literal["class", "feat"]`
  with default `"feat"` (matches current denoising behaviour).
- Output: write to the corresponding field. Per-field
  `output_dims_e_*` config determines the final projection
  width.
- `t` concat: two lines at the top of `forward`, copied from
  `GraphTransformer`.

**Commit 7.3** · Spectral family (base `SpectralDenoiser` +
`LinearPE`, `GraphFilterBank`, `SelfAttentionDenoiser`,
`BilinearDenoiser`, `BilinearDenoiserWithMLP`,
`MultiLayerBilinearDenoiser`).
- Same treatment: `edge_source` parameter drives input read;
  `output_dims_e_*` drives output head width. The base
  `SpectralDenoiser.forward` stops calling `from_edge_state` and
  instead writes directly to the configured output field.

**Commit 7.4** · Baselines (`LinearBaseline`, `MLPBaseline`) +
`SequentialDenoisingModel` (hybrid) + `MultiLayerAttention`. Same
pattern; no new logic.

*Tests:* architecture-level tests parametrised over
`(edge_source, output_field)` — at least one representative per
family asserted runnable in both categorical and continuous modes
with the same class and config-only differences. This is the
concrete evidence for spec goal G2.

### Wave 8 — tests and configs

**Commit 8.1** · Update every test fixture that constructs
`GraphData` directly to use only the new fields. Surface:
`tests/**/*.py`.

**Commit 8.2** · Architecture config files:
- Each `models/<family>/<arch>.yaml` gains
  `output_dims_e_class`, `output_dims_e_feat`,
  `output_dims_x_class`, `output_dims_x_feat`, and `edge_source`
  as explicit fields, defaulting to values that reproduce current
  behaviour.
- New `models/discrete/<family>/<arch>.yaml` files (one per
  non-DiGress architecture) bind the DiGress-compatible values:
  `edge_source="class"`, `output_dims_e_class=2`. These become
  the inputs to the DiGress-architecture comparison panel
  described in earlier conversations; they are not wired into
  any launcher in this commit.

### Wave 9 — removal

**Commit 9.1** · Delete `X` and `E` from `GraphData`.
- `__post_init__` invariant now checks only the new fields.
- Every earlier commit that populated both sets is now touching
  code that only has one set of fields; pyright catches any
  stragglers.

**Commit 9.2** · Delete the legacy helpers:
- `GraphData.from_edge_state`
- `GraphData.from_binary_adjacency`
- `GraphData.to_edge_state`
- `GraphData.to_binary_adjacency`

**Commit 9.3** · Apply the deferred Commit 3.2 — datamodules stop
emitting `X_class` entirely. Existing datasets now produce
`X_class=None`; architectures that need a per-node feature
synthesise it per the spec ("architecture-internal concern").

### Wave 10 — verification + squash

**Commit 10.1** · Final housekeeping:
- Run the spec's completion-check `rg` patterns. Any residual hits
  are bugs.
- `basedpyright` clean on the whole tree (not just touched
  files).
- `uv run pytest tests/ -x -m "not slow"` green.

**Squash and merge** into `main` with the single commit message
`refactor: unify GraphData features — atomic schema rewrite (see
docs/specs/2026-04-15-unified-graph-features-spec.md)`. The intra-
branch commit history remains accessible via the branch for
bisection.

Post-merge, anyone who wants to run the DiGress SBM on Modal
re-launches from the usual wrappers (`run-upstream-digress-sbm-
modal-a100.zsh` etc.). Whether the new run produces the same
numbers as a pre-refactor one is an empirical follow-up, not a
merge gate.

## Completion gate

All of the following MUST be true before the squash merge lands on
`main`:

- [ ] `rg '\.X\b|\.E\b' src tests` returns zero hits where the
      receiver is a `GraphData`. (Approved exceptions documented in
      the PR description.)
- [ ] `rg 'to_edge_state|from_edge_state|from_binary_adjacency|
      to_binary_adjacency'` returns zero hits.
- [ ] `basedpyright` clean across the full tree.
- [ ] `uv run pytest tests/ -x -m "not slow"` green.
- [ ] `tests/diffusion/test_gaussian_noise_process.py` green.
- [ ] Architecture parity test (G2) green for at least one
      representative per family: GraphTransformer, GNN, spectral,
      baseline, hybrid.

No Modal smoke run is part of the gate. The refactor is a clean
break; numerical equivalence with pre-refactor runs is neither
expected nor required. Correctness is established by static checks
and the in-repo test suite alone.

## Rollback

`git revert` the squash commit on `main`. Because the refactor
touches every consumer, partial rollback is impossible by design.
If the squash lands and regresses, the revert is atomic; if the
revert reveals that some follow-up commit on `main` depends on
the new schema, resolve the merge conflict by reapplying the
refactor on top.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hidden downstream consumer of the old `X`/`E` fields (e.g. a notebook or script outside `src/`/`tests/`) | Completion-check rg sweeps `src tests`; also run the sweep on `scripts/` and `wandb-tools/` before merging. Anything outside those paths accepts migration-by-breakage. |
| Architecture family whose output head isn't cleanly parameterisable by `output_dims_e_*` | Spectral-family base class may need a small refactor of the final projection. Budget extra time for Commit 7.3. If an architecture resists the parameterisation, accept that family as not-yet-unified and document in a follow-up issue; do not block the refactor. |
| Gaussian noise process tests reveal numerical instability | The `GaussianNoiseProcess` is a reference DDPM implementation. If tests fail, it's most likely a schedule interpretation bug; cross-check against upstream DDPM test vectors. |

## Estimated wall time

- Waves 0–2: ~1 day (scaffolding + noise processes).
- Waves 3–6: ~1–2 days (datasets, sampler, Lightning modules,
  evaluator).
- Wave 7: ~1–2 days (architectures — the per-family work is small
  per-family but spread across ~12 files).
- Wave 8: ~half a day (tests + configs).
- Wave 9: ~few hours (removal is mechanical once all readers have
  migrated).
- Wave 10: ~half a day (verification and smoke).

Total: ~4–6 focused days, single author. Pairing or parallelism
within a wave (e.g. 7.2 and 7.3 concurrently by different authors)
could compress this, but conflict-resolution cost rises quickly —
the architecture files share base classes.

## Dependencies

No external dependencies. All changes are in-repo. Nothing needs
to be captured or measured before branching; the completion gate
is static analysis plus the in-repo test suite.

## After the plan lands

The spec's open issues (per-field loss weights, evaluator
disagreement threshold, symmetry enforcement at architecture
output) are not resolved by this plan. Each spawns its own small
follow-up — the refactor unblocks their investigation but does not
preempt their decisions.

The DiGress-architecture-comparison panel (sketched in previous
conversation but not yet a committed artefact) becomes
straightforward after Wave 8.2: flip the `edge_source="class"`
config flag on any non-DiGress architecture, point its
`output_dims_e_class=2`, and launch.
