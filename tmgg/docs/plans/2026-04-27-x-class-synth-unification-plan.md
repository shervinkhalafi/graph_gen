# Implementation plan — X_class synth unification

**Status:** ready to execute
**Spec:** `docs/specs/2026-04-27-x-class-synth-unification-spec.md` v5
**Owner:** igork
**Date:** 2026-04-27

## Approach

TDD where it pays: write the new parametrized tests **first** so the rest of the implementation has something to pass against. Then implement bottom-up in dependency order — `GraphData` classmethod first (leaf), noise process next (consumes the classmethod), diffusion module on top of that, model in parallel, evaluator + docs at the end. Each phase = one atomic conventional commit. ~9 commits total.

Why this ordering:
- The `GraphData` classmethod has no dependencies — any other order would leave us hand-rolling synthesis temporarily.
- `_read_categorical_x` strip-default forces every caller to be updated in the same commit, otherwise pyright fails. So that's a single commit.
- `_read_field` and `_synth_x_class` deletions depend on `_read_categorical_x` being correct.
- The init-time invariant comes after everything else lands so it can validate against the corrected code, not catch transitional state.
- Doc updates ride at the end so they reflect the final API.

## Pre-flight

Before starting:

```bash
# Verify clean working tree
git status --short
git rev-parse --abbrev-ref HEAD                  # should be main or igork/<branch>

# Confirm baseline tests pass
uv run pytest tests/diffusion/ tests/training/ tests/experiment_utils/test_diffusion_module.py tests/models/test_architecture_parity.py -q
```

Branch off `main` if not already on a feature branch:

```bash
git checkout -b igork/synth-unification main
```

## Phase 0 — TDD scaffolding (commit 1/9)

Write the new parametrized tests first. They will fail against current code; that's the point — they define the contract.

**Files to create:**

- `tests/diffusion/test_categorical_noise_process_x_classes.py`
- `tests/models/test_graph_transformer_structure_only.py`

**`tests/diffusion/test_categorical_noise_process_x_classes.py`** — parametrized over `c_x ∈ {1, 2}`, fixed `c_e = 2` (since C_e=1 isn't supported in our pipeline). Exercises:
- `forward_sample` (covers `_apply_noise`)
- `_posterior_probabilities` — confirmed bug site
- `_posterior_probabilities_marginalised` — latent
- `forward_pmf` — next-crash site
- `prior_pmf`
- `posterior_sample` (direct + marginalised)
- `_masked_graph_log_prob` (indirectly via VLB term)

For each path, assert `result.X_class.shape[-1] == c_x` and `result.E_class.shape[-1] == c_e`.

Plus three negative-path tests:
- `test_e_synthesis_raises_with_clear_message` — pass `E_class=None` to `_read_categorical_e`, expect `ValueError` matching the asymmetry message.
- `test_x_synthesis_raises_for_c_x_geq_3` — pass `X_class=None` with `x_classes=3`, expect `ValueError` from `synth_structure_only_x_class`.
- `test_e_classes_required_no_default` — call `_read_categorical_e(data)` without `e_classes`; expect `TypeError` (positional arg missing).

**`tests/models/test_graph_transformer_structure_only.py`** — exercises the model's input synth at `transformer_model.py:944`. Parametrized over `output_dims_x_class ∈ {1, 2}`:
- Build `GraphTransformer(input_dims={X: cx, E: 2, y: 0}, output_dims_x_class=cx, ...)`.
- Pass a structure-only batch (`X_class=None`, populated `E_class`, `node_mask`).
- Run `forward()`; assert no crash, output X_class width matches `cx`.
- Negative case: `output_dims_x_class=3` with `X_class=None` should raise via the helper.

**Verification before commit:**
```bash
# Tests should FAIL (they're the contract for upcoming work)
uv run pytest tests/diffusion/test_categorical_noise_process_x_classes.py tests/models/test_graph_transformer_structure_only.py -v
```
Expected: all parametrized cases fail (because helper doesn't exist yet, defaults still in place, model synth still hardcoded).

**Commit message:**
```
test(diffusion,models): scaffold C_x∈{1,2} parametrized tests (TDD)

Targets the X_class synth unification spec (v5). Tests fail against
current code — the failures are the contract that the spec's §5
implementation must satisfy.

- test_categorical_noise_process_x_classes.py — parametrized over
  c_x∈{1,2}, c_e=2; covers forward_sample, _posterior_probabilities
  (+marginalised), forward_pmf, prior_pmf, posterior_sample. Plus
  negative tests for E synth raising and C_x≥3 raising.
- test_graph_transformer_structure_only.py — parametrized over
  output_dims_x_class∈{1,2} for the model's synth at line 944.
```

## Phase 1 — `GraphData` classmethod (commit 2/9)

**File:** `src/tmgg/data/datasets/graph_types.py`

**Change:** add classmethod near other `GraphData` classmethods (probably near `from_pyg_batch` at line 502).

```python
@classmethod
def synth_structure_only_x_class(
    cls, node_mask: Tensor, c_x: int
) -> Tensor:
    """Derive a structure-only X_class tensor from node_mask alone.

    Synthesis convention (spec 2026-04-27-x-class-synth-unification):
    - C_x = 1: ones at valid nodes, zeros at padding. Canonical
      structure-only encoding; noise process is identity on X.
    - C_x = 2: [1 - node_ind, node_ind] one-hot (legacy [no-node,
      node]). Kept for backward compat with existing C_x=2 model
      presets.
    - C_x >= 3: raise — real categorical X must be populated by
      the dataset.

    E has no symmetric helper: edges are an adjacency property
    orthogonal to node_mask, so synthesis from node_mask alone is
    undefined for any C_e. ``_read_categorical_e`` raises when
    E_class is None.

    Parameters
    ----------
    node_mask
        Boolean mask of valid nodes, shape ``(B, N)``.
    c_x
        Categorical class width including any structural-filler
        slot. See spec §3 for the regime table.

    Returns
    -------
    Tensor
        Shape ``(B, N, c_x)``, dtype ``float32``, on the same
        device as ``node_mask``.
    """
    node_ind = node_mask.float()
    if c_x == 1:
        return node_ind.unsqueeze(-1)
    if c_x == 2:
        synth = torch.stack([1.0 - node_ind, node_ind], dim=-1)
        # Zero padding rows so loss predicates exclude them
        return synth * node_ind.unsqueeze(-1)
    raise ValueError(
        "GraphData.synth_structure_only_x_class: synthesis is only "
        f"defined for C_x in {{1, 2}}; got C_x={c_x}. C_x>=3 implies "
        "real categorical content; the dataset MUST populate "
        "X_class. See 2026-04-27-x-class-synth-unification-spec §3."
    )
```

**Verification:**
```bash
# Cheap unit test: instantiate, call with c_x=1 → shape (B, N, 1); c_x=2 → (B, N, 2); c_x=3 → raises
uv run pytest tests/models/test_graph_types.py -q     # existing tests must still pass
uv run python -c "
import torch
from tmgg.data.datasets.graph_types import GraphData
mask = torch.tensor([[True, True, False], [True, False, False]])
print(GraphData.synth_structure_only_x_class(mask, 1).shape)  # (2, 3, 1)
print(GraphData.synth_structure_only_x_class(mask, 2).shape)  # (2, 3, 2)
try:
    GraphData.synth_structure_only_x_class(mask, 3)
except ValueError as e:
    print(f'OK raised: {e}')
"
```

**Commit:**
```
feat(data): add GraphData.synth_structure_only_x_class classmethod

Single canonical helper for structure-only X synthesis per
2026-04-27 spec §5.1. Three regimes: C_x=1 (canonical, ones at
valid nodes), C_x=2 (legacy [no-node, node]), C_x>=3 raises.

Lives on GraphData to document intent via the data type whose
shape it determines; both noise process and model can call it
without depending on each other.

E has no symmetric helper — edges are an adjacency property
orthogonal to node_mask, synthesis is undefined.
```

## Phase 2 — `_read_categorical_x/e` strip default (commit 3/9)

**File:** `src/tmgg/diffusion/noise_process.py`

**Changes:**

1. `_read_categorical_x(data, x_classes)` — make `x_classes` required (no default), inline body delegates to `GraphData.synth_structure_only_x_class`.
2. `_read_categorical_e(data, e_classes)` — make `e_classes` required, raise on `E_class is None` with the asymmetry message.
3. Update **all 11 internal call sites** in `CategoricalNoiseProcess` to pass `x_classes=self.x_classes` / `e_classes=self.e_classes`:
   - line 1034 (`model_output_to_posterior_parameter`) — already correct in current code, keep
   - line 1179 (`_apply_noise`) — already correct, keep
   - lines 1243, 1245 (`_posterior_probabilities`) — add
   - line 1283 (`posterior_sample` direct) — add
   - lines 1334, 1336 (`_posterior_probabilities_marginalised`) — add
   - line 1403 (`posterior_sample_marginalised`) — add
   - line 1465 (`forward_pmf`) — add
   - line 1516 (`prior_log_prob` / similar) — add
4. `_masked_graph_log_prob` (lines 185-188) — add `x_classes=samples1.X_class.shape[-1]` defensively. Same for the `probs` argument.
5. Update docstring at line 272 to recast C_x=1 as canonical, C_x=2 as legacy (not "molecular default").
6. Delete the `getattr(self.noise_process, "x_classes", 2)` fallbacks at `diffusion_module.py:631, :858`. Use strict `self.noise_process.x_classes`.

**Verification:**
```bash
# Pyright catches every site that forgot to pass x_classes/e_classes
uv run basedpyright src/tmgg/diffusion/noise_process.py src/tmgg/training/lightning_modules/diffusion_module.py 2>&1 | rg "error"
# Expected: 0 errors

# All existing tests still pass (we haven't changed behavior for K=2)
uv run pytest tests/diffusion/ tests/training/ tests/experiment_utils/test_diffusion_module.py -q

# New tests — the noise-process portion now passes for c_x=1
uv run pytest tests/diffusion/test_categorical_noise_process_x_classes.py -v
# Expected: noise-process tests pass; model test still fails (Phase 4 hasn't landed)
```

**Commit:**
```
fix(diffusion): strip defaulted x_classes from _read_categorical_x

Per 2026-04-27 spec §5.2: every call site now passes its
authoritative C_x / C_e explicitly. No defaulted argument that
silently leaks the K=2 historical convention. Pyright catches
missing args.

- _read_categorical_x: x_classes now required, delegates synth
  to GraphData.synth_structure_only_x_class.
- _read_categorical_e: e_classes now required, raises on
  E_class=None with the X-vs-E asymmetry message.
- 9 internal call sites in CategoricalNoiseProcess threaded
  with self.x_classes / self.e_classes.
- _masked_graph_log_prob defensively passes data-derived widths.
- Removed getattr(..., "x_classes", 2) fallbacks from
  DiffusionModule that hid contract violations.

Closes the bug class that crashed the Vignac SBM repro on Modal:
_posterior_probabilities synthesised C_x=2 against a C_x=1
noise process.
```

## Phase 3 — DiffusionModule consolidation (commit 4/9)

**File:** `src/tmgg/training/lightning_modules/diffusion_module.py`

**Changes:**

1. `_read_field` (line 121) — drop `x_classes` parameter; the X_class branch now defers to `_read_categorical_x(data, x_classes=noise_process.x_classes)`. Update callers to pass `noise_process` instead of `x_classes`.
2. `_synth_x_class` closure inside `_categorical_reconstruction_log_prob` (line 233) — delete. Use `_read_categorical_x` / `_read_categorical_e` directly. Promote the surrounding function to take `x_classes` / `e_classes` arguments (callers `_compute_reconstruction*` have `self.noise_process` in scope).
3. Update docstring at line 130-136 — recast C_x=1 / C_x=2 regimes per spec §3.

**Verification:**
```bash
uv run basedpyright src/tmgg/training/lightning_modules/diffusion_module.py 2>&1 | rg "error"
uv run pytest tests/training/ tests/experiment_utils/test_diffusion_module.py -q
```

**Commit:**
```
refactor(training): consolidate X synth via single canonical helper

Per 2026-04-27 spec §5.3-5.4:
- _read_field defers X_class synthesis to noise process (which
  in turn calls GraphData.synth_structure_only_x_class).
- Deleted local _synth_x_class closure inside
  _categorical_reconstruction_log_prob; uses _read_categorical_x
  / _read_categorical_e directly.

One synth implementation in the codebase, not three.
```

## Phase 4 — Model synth fix (commit 5/9)

**File:** `src/tmgg/models/digress/transformer_model.py`

**Change at lines 943-946:**

```python
if data.X_class is not None:
    X = data.X_class
else:
    X = GraphData.synth_structure_only_x_class(
        node_mask, self.output_dims_x_class
    ).to(device=E.device, dtype=E.dtype)
```

Critical: uses `self.output_dims_x_class` (which is C_x), **not** `self.input_dims["X"]` (which is X_in = C_x + F_x — aggregate including extras).

Also tighten:
- Comment at lines 941-942 (currently reinforces "two-channel X" as universal — recast as "C_x is the model's authoritative X-class width").
- `input_dims` parameter docstrings at lines 518, 824, 850, 970 — clarify "X_in / E_in / y_in aggregates."

**Verification:**
```bash
uv run pytest tests/models/test_graph_transformer_structure_only.py -v
# Expected: passes for both c_x=1 and c_x=2; raises for c_x=3
uv run pytest tests/models/test_architecture_parity.py -q
# The existing test_structure_only_input_synthesises_node_features still passes
# (works for c_x=2; we'll parametrize it in Phase 7)
```

**Commit:**
```
fix(models): GraphTransformer X synth uses output_dims_x_class

Per 2026-04-27 spec §5.5: the inline torch.stack synth at
transformer_model.py:944 hardcoded C_x=2; replaced with the
canonical helper using self.output_dims_x_class as the
authoritative C_x source.

Critical distinction: output_dims_x_class is C_x; input_dims["X"]
is X_in (aggregate C_x + F_x + extras). Synth tensor is the
X_class slice; extras get concatenated downstream of this point.
Wrong-variable use was the root cause of the second crash mode
the spec called out.
```

## Phase 5 — Evaluator C_e=1 defensive fix (commit 6/9)

**File:** `src/tmgg/evaluation/graph_evaluator.py`

**Change at lines 597, 654:** branch on `E_class.shape[-1]` to handle C_e=1 explicitly.

```python
# Before:
edges = (data.E_class.argmax(dim=-1) != 0)

# After:
c_e = data.E_class.shape[-1]
if c_e == 1:
    raise ValueError(
        "GraphEvaluator: E_class with C_e=1 has no implicit "
        "no-edge class; adjacency cannot be inferred from "
        "argmax. Use C_e>=2 for evaluation, or override the "
        "evaluator's adjacency-recovery rule."
    )
# C_e >= 2: class 0 is no-edge by upstream convention
edges = (data.E_class.argmax(dim=-1) != 0)
```

**Verification:**
```bash
uv run pytest tests/experiment_utils/ -q -k "evaluator"
```

**Commit:**
```
fix(evaluation): raise loudly when GraphEvaluator sees C_e=1 E_class

Per 2026-04-27 spec §7.1 audit: argmax!=0 on a C_e=1 E_class
silently emits empty adjacencies (argmax always 0). Defensive
fix: branch on shape[-1] and raise with a clear message naming
the convention. C_e=1 isn't supported by this evaluator's
adjacency-recovery; the rule is documented inline.
```

## Phase 6 — Init-time invariant (commit 7/9)

**File:** `src/tmgg/training/lightning_modules/diffusion_module.py`

**Change in `setup()` (line 481):** after `noise_process.initialize_from_data(...)`, add the C_x / C_e triplet check:

```python
# Init-time invariant: C_x triplet (data, noise process, model output head)
# must agree. Catches Hydra-config drift before the first training step.
batch = next(iter(dm.train_dataloader()))
materialised_x = _read_categorical_x(batch, self.noise_process.x_classes)
materialised_e = _read_categorical_e(batch, self.noise_process.e_classes)

assert materialised_x.shape[-1] == self.noise_process.x_classes
assert materialised_e.shape[-1] == self.noise_process.e_classes

model_c_x = getattr(self.model, "output_dims_x_class", None)
if model_c_x is not None:
    assert model_c_x == self.noise_process.x_classes, (
        f"Model output_dims_x_class={model_c_x} disagrees with "
        f"noise_process.x_classes={self.noise_process.x_classes} "
        f"(canonical name C_x). Configure both via the same Hydra "
        f"source. See 2026-04-27 spec §5.6."
    )
model_c_e = getattr(self.model, "output_dims_e_class", None)
if model_c_e is not None:
    assert model_c_e == self.noise_process.e_classes, (
        f"Model output_dims_e_class={model_c_e} disagrees with "
        f"noise_process.e_classes={self.noise_process.e_classes} "
        f"(canonical name C_e)."
    )
```

X_in / E_in / y_in are deliberately NOT asserted — model self-validates internally via `extra_features.adjust_dims`.

**Verification:**
```bash
uv run pytest tests/training/ tests/experiment_utils/test_diffusion_module.py -q
# All tests still pass (existing fixtures already have agreeing C_x)
```

**Commit:**
```
feat(training): add C_x/C_e triplet invariant to DiffusionModule.setup

Per 2026-04-27 spec §5.6: assert that data.X_class.shape[-1],
noise_process.x_classes, and model.output_dims_x_class agree at
setup time. Same for C_e. Catches Hydra-config drift with a clear
error message before the first training step (replaces a CUDA
bmm shape error 5 layers deep).

X_in / E_in / y_in not asserted — model self-validates aggregate
dims internally via extra_features.adjust_dims.
```

## Phase 7 — Existing test parametrization (commit 8/9)

**Files:**
- `tests/experiments/test_discrete_diffusion_module.py`
- `tests/models/test_architecture_parity.py`

**Changes:**

1. `test_output_shapes` — parametrize on `c_x ∈ {1, 2}`; assert `pred.X_class.shape[-1] == c_x`.
2. `test_training_loss_matches_train_loss_discrete` — replace inline K=2 synth at lines 372-380 with `GraphData.synth_structure_only_x_class(batch.node_mask, c_x)`; parametrize on c_x.
3. `test_training_loss_masks_padding` — parametrize on c_x.
4. `test_structure_only_input_synthesises_node_features` — parametrize on `output_dims_x_class ∈ {1, 2}`; add c_x=3 case asserting helper raises.

**Verification:**
```bash
uv run pytest tests/experiments/test_discrete_diffusion_module.py tests/models/test_architecture_parity.py -v
# All parametrized cases pass
```

**Commit:**
```
test(diffusion,models): parametrize existing X synth tests on C_x∈{1,2}

Per 2026-04-27 spec §7.4 audit: 4 existing tests pinned C_x=2
shapes and would silently allow regressions. Parametrized over
c_x∈{1,2}; added c_x=3 negative case asserting the canonical
helper raises.

- test_output_shapes
- test_training_loss_matches_train_loss_discrete (also replaces
  the inline [1-node_ind, node_ind] synth with the canonical
  helper)
- test_training_loss_masks_padding
- test_structure_only_input_synthesises_node_features
```

## Phase 8 — Doc updates (commit 9/9)

**Files:**
- `docs/specs/2026-04-15-unified-graph-features-spec.md`
- `docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md`
- `src/tmgg/diffusion/__init__.py`, `src/tmgg/data/__init__.py`, `src/tmgg/models/__init__.py` — module-level pointers (1-line each)

**Changes:**

1. **`2026-04-15 §"Architectures without explicit node features" (lines 169-179)**: append the canonical-helper paragraph (exact text in audit findings §7.5):

> "The canonical derivation is `GraphData.synth_structure_only_x_class(node_mask, C_x)` (see `2026-04-27-x-class-synth-unification-spec.md`); each consumer (model, noise process, loss) calls it with its own authoritative C_x. C_x = 1 is the canonical structure-only encoding; C_x = 2 (`[no-node, node]`) is retained for backward compat with existing C_x = 2 model presets; C_x ≥ 3 is real categorical and MUST come from the dataset."

2. **`2026-04-15 §"Removed fields" failure-mode table (line 521)**: add row:

> "`_read_categorical_e` invoked with `E_class=None` → Raise `ValueError` (E is never synthesised; see 2026-04-27 spec)."

3. **`2026-04-15 §"Proposed representation" pseudo-schema (lines 125-131)**: add footnote mapping `dx_class → C_x`, `dx_feat → F_x`, `de_class → C_e`, `de_feat → F_e`.

4. **`docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md` §1** (lines 530-546): postscript:

> "**Superseded 2026-04-27**: the synthesis was buggy on C_x=1 configs (Modal SBM repro crashed in `_posterior_probabilities`). See `docs/specs/2026-04-27-x-class-synth-unification-spec.md` for the fix."

5. **Module `__init__.py`**: one-line pointer added to existing module docstrings, e.g.:

```python
"""<existing docstring>

X_class / E_class synthesis convention: see
``docs/specs/2026-04-27-x-class-synth-unification-spec.md``.
"""
```

**Verification:** docs only — no test impact. Visual review.

**Commit:**
```
docs(specs): cross-reference X synth unification across related docs

Per 2026-04-27 spec §7.5 audit:
- 2026-04-15 spec §"Architectures": append canonical-helper paragraph
  with C_x regime table reference.
- 2026-04-15 spec §"Removed fields": failure-mode row for
  _read_categorical_e raising on E_class=None (X-vs-E asymmetry).
- 2026-04-15 spec §"Proposed representation": footnote mapping
  dx_class → C_x, etc.
- 2026-04-21 divergence-triage §1: postscript invalidating the
  "NO-OP" verdict that the Modal SBM crash falsified.
- Module __init__ docstrings: one-line synthesis-contract pointers.
```

## Optional Phase 9 — SBM preset comments (commit 10/9, optional)

18 SBM/structure-only Hydra presets pin `noise_process.x_classes: 2`. Add one clarifying comment per preset on the `input_dims` line:

```yaml
input_dims: { X: 2, E: 2, y: 0 }   # X=2 is C_x=2; F_x extras adjusted at runtime via extra_features.adjust_dims
```

Cheap (~18 lines), purely defensive against future readers conflating `input_dims["X"]` with C_x. **Only land if appetite remains** after the substantive work.

**Commit:**
```
docs(configs): annotate input_dims.X with C_x/X_in distinction in SBM presets

Per 2026-04-27 spec §7.6 audit: 18 SBM presets at C_x=2 work
correctly today; the one-line annotation prevents future readers
from conflating input_dims["X"] (X_in aggregate) with C_x.
```

## Final verification (after all commits)

```bash
# Full test suite
uv run pytest tests/ --ignore=tests/modal -m "not slow" -q
# Expected: ~1450 pass, plus pre-existing failures we already documented

# Pre-commit hooks
git status --short                       # working tree clean
pre-commit run --all-files               # ruff, basedpyright, tach all green

# The "smoke test" — can the Vignac SBM repro launch and get past the previous crash sites?
DEPLOY_FIRST=1 DRY_RUN=1 ./run-discrete-sbm-vignac-repro-modal-a100.zsh
# Then for real: DEPLOY_FIRST=1 ./run-discrete-sbm-vignac-repro-modal-a100.zsh
# Watch the modal logs — should pass _posterior_probabilities, forward_pmf,
# prior_pmf, the model's input synth, and reach training step 1.
```

## Rollback plan

If any phase breaks unexpectedly:

```bash
# Find the last good commit
git log --oneline | head

# Soft-reset to before the broken phase, fix, re-apply
git reset --soft <hash-before-break>
```

Each commit is atomic and conventional, so a single revert undoes one phase cleanly.

## Estimated wall clock

- Phase 0 (TDD scaffolding): ~30 min
- Phase 1 (classmethod): ~10 min
- Phase 2 (noise process strip-default): ~30 min — most call sites
- Phase 3 (diffusion module consolidation): ~20 min
- Phase 4 (model synth fix): ~10 min
- Phase 5 (evaluator C_e=1): ~10 min
- Phase 6 (setup invariant): ~10 min
- Phase 7 (test parametrization): ~30 min
- Phase 8 (docs): ~20 min
- Phase 9 (optional preset comments): ~10 min

Total: ~3 hours of focused work. Phases 0-7 are ~2.5 hours; docs and optional preset comments are independent and can land later.
