# Spec — Unify structure-only X_class synthesis

**Status:** v5.1 — implementation complete; §1 / §5.5 corrected for asymmetric input/output C_x semantics
**Owner:** igork
**Date:** 2026-04-27
**Triggered by:** Vignac SBM repro crash on Modal — `_posterior_probabilities` synthesised C_x=2 against a C_x=1 noise process, CUDA bmm shape mismatch.

## 1. Definitions and naming convention

To prevent the kind of conflation that caused the bug, this spec uses canonical names throughout. The codebase variables that map to each concept are listed; existing variable names (e.g. `x_classes`, `output_dims_x_class`) are NOT renamed — only docstrings and spec text adopt the canonical names.

| Concept | Canonical name | Definition | Codebase variables (must agree where applicable) |
|---|---|---|---|
| Node categorical-class channel width | **C_x** | Cardinality of node-class alphabet at this field, *including* any structural-filler slot. `[no-node, node]` ⇒ C_x=2; bare `[node]` ⇒ C_x=1; `{C, N, O, F}` ⇒ C_x=4. | `noise_process.x_classes` · `data.X_class.shape[-1]` · `model.output_dims_x_class` |
| Edge categorical-class channel width | **C_e** | Same for E. `[no-edge, edge]` ⇒ C_e=2; bond types (with optional no-edge slot) ⇒ C_e ∈ {3..5}. | `noise_process.e_classes` · `data.E_class.shape[-1]` · `model.output_dims_e_class` |
| Graph categorical-class channel width | **C_y** | Graph-level class alphabet cardinality. **C_y = 0 when no graph-level class target is given** (SBM convention with `lambda_y=0`). | `noise_process.y_classes` · `data.y.shape[-1]` (when categorical) · `model.output_dims_y_class` |
| Node continuous-feature channel width | **F_x** | Width of continuous node-feature channel **including any input-side encoded features** (cycles, eigenvalues, learnable embeddings). At input projection F_x absorbs extras; at output F_x equals the raw feat dim only. | input side: `data.X_feat.shape[-1]` + extras via `extra_features.adjust_dims(...)` · output side: `model.output_dims_x_feat` |
| Edge continuous-feature channel width | **F_e** | Same definition for E. | `data.E_feat.shape[-1]` + extras · `model.output_dims_e_feat` |
| Graph continuous-feature channel width | **F_y** | Same for y, **plus the diffusion-timestep scalar appended when `use_timestep=True`**. | `data.y.shape[-1]` (raw, when continuous) + extras + timestep at input |
| **Input-side C_x** (model perspective) | **C_x_in** | The model's input-side categorical width as configured by the user. Equals C_x for symmetric categorical-diffusion tasks; can differ from output-side C_x for asymmetric denoising tasks (e.g. `input_dims.X=2`, `output_dims_x_class=0` — model consumes X but predicts only clean E). The model's structure-only synth path uses **C_x_in**, not the output-side value, because the synthesised tensor feeds the first projection. | `model.input_dims["X"]` (constructor arg, stored on `self`) |
| Aggregate model input (X side, post-extras) | **X_in** | `C_x_in + F_x + extras_X` — actual width consumed by the first projection after `extra_features.adjust_dims()` runs. Lives in the local `adjusted_input_dims` inside `GraphTransformer.__init__`; **not stored as an attribute**. Out of scope for this spec. | `adjusted_input_dims["X"]` (local in `__init__`) |
| Aggregate model input (E side, post-extras) | **E_in** | `C_e + F_e + extras_E`. Same locality as X_in. | `adjusted_input_dims["E"]` (local) |
| Aggregate model input (y side, post-extras) | **y_in** | `C_y + F_y + (1 if use_timestep else 0)`. | `adjusted_input_dims["y"]` (local) |

X_in / E_in / y_in are **derived aggregates**, not primary concepts. The fix touches C_x specifically; F_<field> and X_in / E_in / y_in are out of scope.

## 2. Problem

Structure-only datasets emit `GraphData.X_class = None` per `docs/specs/2026-04-15-unified-graph-features-spec.md §"Removed fields"`: "Datasets MUST NOT re-encode `node_mask` as a degenerate `X_class` just to feed architectures that currently assume one." The intent: data-layer stays clean, architectures (model, noise process) derive their own per-node feature when they need one.

The implementation honors that rule but has a latent defect: derivation happens in **four independent consumer sites**, each with its own copy of the synthesis logic, all hardcoding the historical C_x=2 `[no-node, node]` shape:

| Site | File:line | Used by |
|---|---|---|
| `_read_categorical_x` | `noise_process.py:272` | the noise process pipeline (8 internal call sites) |
| `_read_field` X_class branch | `diffusion_module.py:138` | the per-field loss + training-step diagnostics |
| `_synth_x_class` (closure inside `_categorical_reconstruction_log_prob`) | `diffusion_module.py:233` | reconstruction VLB term |
| Inline X synth | `transformer_model.py:944` | the model's input projection |

When a config sets `noise_process.x_classes = 1` (upstream-DiGress / GDPO SBM convention), the synthesis sites still emit C_x=2 unless every caller threads `x_classes` explicitly. The first crash was at `_posterior_probabilities`; deeper paths (`forward_pmf`, `prior_pmf`, marginalised posterior, the model's own input synth, the reconstruction log-prob) carry the same defect.

## 3. C_x semantic regimes

C_x is the cardinality of the X-class alphabet *including any structural-filler slot*. The regime determines whether synthesis is even meaningful:

| C_x | Regime | What X carries | Who provides it |
|---|---|---|---|
| **C_x = 1** | structure-only canonical | A single abstract class. X is shape-only; noise process is identity on X_class. | Synthesis from `node_mask` alone — every valid position is the only class. |
| C_x = 2 (`[no-node, node]`) | structure-only legacy | Re-encodes `node_mask` as a one-hot. Equivalent to C_x=1 + node_mask but with redundant width. Kept for backward compat with existing C_x=2 model presets. | Synthesis from `node_mask` (the historical default). |
| **C_x ≥ 3** | real categorical | Actual labels (atom types, etc.). Synthesis is **undefined** — there is no canonical "structure-only C_x=3" interpretation. | The dataset MUST populate; consumer reads `data.X_class` directly. Synthesis attempts MUST raise. |

The C_e analogue exists in principle but **E never synthesises**. C_e=1 has no implicit "absent" class slot to model non-edges; C_e=2 has the slot but synthesis still needs adjacency context (which lives in the dataset, not in `node_mask`); C_e≥3 implies real labels. In our pipeline `from_pyg_batch` always populates `E_class` from adjacency, so synthesis never fires for E. We codify "E never synthesises" as the rule with a clear error message naming the asymmetry: X can be synthesised because every valid node trivially exists as its only class; E cannot because edges are an adjacency property orthogonal to `node_mask`.

**Implication for synthesis:**

- **X helper** accepts C_x explicitly (no default) and dispatches: C_x=1 emit ones at valid nodes; C_x=2 emit `[1 - node_ind, node_ind]`; C_x≥3 raise.
- **E helper** does not exist. `_read_categorical_e` accepts `e_classes` explicitly (no default) for the init-time agreement check, but raises with a clear message if `E_class is None` for any C_e. The asymmetry is documented.

## 4. Root cause

Two structural smells make this bug class possible:

1. **Three independent C_x declarations.** Each synthesis site reads a different "what's C_x?" source: `self.x_classes` on the noise process, `self.input_dims['X']` on the model (which is X_in, not C_x — a separate conflation), the noise process via the LightningModule on the loss helper. Three views of one number, no system component enforcing they agree.
2. **Defaulted argument.** `_read_categorical_x(data, x_classes=2)` makes C_x=2 a silent default. Type checker can't see that callers sometimes need C_x=1; failures surface as CUDA scatter index errors several layers down the stack.

## 5. Proposed fix

Honor the existing 2026-04-15 spec rule (data layer never synthesises). Move synthesis to a single canonical helper. Each consumer (noise process, loss, model) calls that helper with **its own authoritative C_x** — no defaulted argument, no plumbing of C_x through helper functions, no duplicated synth code.

### 5.1 Single canonical helper, attached to `GraphData`

```python
class GraphData:
    @classmethod
    def synth_structure_only_x_class(
        cls, node_mask: Tensor, c_x: int
    ) -> Tensor:
        """Derive a structure-only X_class tensor from node_mask alone.

        Synthesis convention:
        - C_x = 1: ones at valid nodes, zeros at padding. Canonical
          structure-only encoding; noise process is identity on X.
        - C_x = 2: [1 - node_ind, node_ind] one-hot (legacy [no-node, node]).
          Kept for backward compat with existing C_x=2 model presets.
        - C_x >= 3: raise — real categorical X must be populated by the
          dataset.

        This is the single place X-synthesis logic lives. Consumers
        (noise process, loss helpers, model input projection) call it
        with their own authoritative C_x; the 2026-04-15 spec's
        data-layer-clean rule (datasets emit X_class=None for
        structure-only) stands.

        E has no symmetric helper: edges are an adjacency property
        orthogonal to node_mask, so synthesis from node_mask alone is
        undefined for any C_e. ``_read_categorical_e`` raises when
        E_class is None.
        """
```

Why a classmethod on `GraphData`: documents the convention via the data type whose shape it determines; both noise process and model can call without depending on each other.

### 5.2 `_read_categorical_x` — strip default, require explicit C_x

```python
def _read_categorical_x(data: GraphData, x_classes: int) -> Tensor:
    """Read X_class or synthesise via GraphData.synth_structure_only_x_class."""
    if data.X_class is not None:
        return data.X_class
    return GraphData.synth_structure_only_x_class(data.node_mask, x_classes)
```

No default. Every internal call site in `CategoricalNoiseProcess` switches to `_read_categorical_x(data, x_classes=self.x_classes)`. Module-level helpers (`_masked_graph_log_prob` lines 185, 187) only see populated data; pass `x_classes=samples1.X_class.shape[-1]` defensively.

For E: `_read_categorical_e` accepts explicit `e_classes` (no default). When `E_class is None`, raise with a clear message regardless of C_e:

```python
def _read_categorical_e(data: GraphData, e_classes: int) -> Tensor:
    if data.E_class is None:
        raise ValueError(
            "_read_categorical_e: E_class is None. E synthesis from "
            "node_mask alone is not supported — edges are an adjacency "
            "property orthogonal to node_mask. Populate E_class from "
            "the dataset's adjacency in from_pyg_batch (or analogous). "
            f"Configured e_classes={e_classes}: even with C_e=1, there "
            "is no implicit non-edge class to model absent edges."
        )
    return data.E_class
```

### 5.3 `_read_field` (loss helpers) — defer to noise process's C_x

Drop the X-synth branch from `_read_field`. Replace with `_read_categorical_x(data, x_classes=noise_process.x_classes)`. Same pattern for E.

### 5.4 `_synth_x_class` inside `_categorical_reconstruction_log_prob` — delete

Local copy of the synthesis logic. Delete it. Promote `_categorical_reconstruction_log_prob` to take `x_classes` / `e_classes` arguments, use `_read_categorical_x` / `_read_categorical_e` directly.

### 5.5 `transformer_model.py:944` — synthesise at C_x = `self.input_dims["X"]`

Replace the hardcoded `[1.0 - node_ind, node_ind]` synth with:

```python
if data.X_class is not None:
    X = data.X_class
else:
    X = GraphData.synth_structure_only_x_class(
        node_mask, self.input_dims["X"]
    ).to(device=E.device, dtype=E.dtype)
```

Important: uses `self.input_dims["X"]` (which is C_x_in — the configured pre-extras input class width). The post-extras X_in aggregate lives in the local `adjusted_input_dims` and is not stored as an attribute. Extras get concatenated to the synthesised tensor downstream of this point (line 953); the synth itself produces the C_x_in-wide X_class slice.

For symmetric categorical-diffusion configs `input_dims["X"] == output_dims_x_class` and either choice works. For asymmetric denoising configs (e.g. `digress_base.yaml`: `input_dims.X=2`, `output_dims_x_class=0` — model consumes X but predicts only clean E, no X_class output head), only `input_dims["X"]` matches what the first projection expects. Using `output_dims_x_class` would raise on `C_x=0` because synthesis from `node_mask` with zero classes is undefined.

(v5 → v5.1 correction: an earlier draft of this section directed using `output_dims_x_class` based on a misread of what `input_dims["X"]` represents — see commit `72dcd8fb` for the implementation correction.)

### 5.6 `DiffusionModule.setup()` — init-time invariant assert

After `noise_process.initialize_from_data(...)`, pull one batch and assert the C_x and C_e triplets agree:

```python
batch = next(iter(dm.train_dataloader()))
materialised_x = _read_categorical_x(batch, self.noise_process.x_classes)
materialised_e = _read_categorical_e(batch, self.noise_process.e_classes)

assert materialised_x.shape[-1] == self.noise_process.x_classes
assert materialised_e.shape[-1] == self.noise_process.e_classes

# Triplet check: data-side, noise-process, model-output-class-head must agree.
model_c_x = getattr(self.model, "output_dims_x_class", None)
if model_c_x is not None:
    assert model_c_x == self.noise_process.x_classes, (
        f"Model output_dims_x_class={model_c_x} disagrees with "
        f"noise_process.x_classes={self.noise_process.x_classes} "
        f"(canonical name C_x). Configure both via the same Hydra source."
    )
model_c_e = getattr(self.model, "output_dims_e_class", None)
if model_c_e is not None:
    assert model_c_e == self.noise_process.e_classes, (...)
```

`X_in` / `E_in` / `y_in` are **deliberately not checked** — they're aggregates that the model self-validates internally via `extra_features.adjust_dims`.

Per project convention (`CLAUDE.md` "fail loudly"), `assert` raises a clear `RuntimeError`.

### 5.7 Targeted unit tests

Parametrized test `tests/diffusion/test_categorical_noise_process_x_classes.py`:

```python
@pytest.mark.parametrize("c_x", [1, 2])
@pytest.mark.parametrize("c_e", [2])  # C_e=1 is config-allowed but synthesis still raises; cover separately
def test_categorical_noise_process_handles_structure_only_batch(c_x, c_e):
    process = CategoricalNoiseProcess(..., x_classes=c_x, e_classes=c_e)
    batch = make_structure_only_batch(...)  # X_class=None; E_class populated C_e=2 from_pyg

    z_t = process.forward_sample(batch, t=t).z_t
    posterior = process._posterior_probabilities(z_t, batch, t, t-1)
    marginalised = process._posterior_probabilities_marginalised(z_t, x0_param, t, t-1)
    forward = process.forward_pmf(batch, t)
    prior = process.prior_pmf(batch.node_mask)
    posterior_sample = process.posterior_sample(z_t, x0_param, t, t-1)

    for graphdata in [posterior, marginalised, forward, prior, posterior_sample]:
        assert graphdata.X_class.shape[-1] == c_x
        assert graphdata.E_class.shape[-1] == c_e


def test_e_synthesis_raises_with_clear_message():
    """E_class=None must raise with the asymmetry-explanation message."""
    ...


def test_x_synthesis_raises_for_c_x_geq_3():
    """C_x>=3 with X_class=None must raise."""
    ...
```

Plus `GraphTransformer.forward` test for C_x=1 with `output_dims_x_class=1`, asserting no shape errors.

## 6. Decisions resolved

1. **Helper home:** `@classmethod` on `GraphData` (`synth_structure_only_x_class`). ✓
2. **C_x semantics:** structure-only X convenience only. C_x=1 canonical, C_x=2 legacy, C_x≥3 raises. E never synthesises. ✓
3. **Model synth fix:** uses `self.output_dims_x_class` (which is C_x, not X_in). ✓
4. **Init-time invariant:** `assert` (RuntimeError) per fail-loudly convention. ✓
5. **E discipline:** explicit `e_classes` arg on `_read_categorical_e`; raises clearly when `E_class is None` for any C_e. No `synth_structure_only_e_class` helper. ✓
6. **Tests:** targeted parametrized unit tests. ✓
7. **Naming convention:** `C_x`, `C_e`, `C_y` for class widths (incl. structural-filler slot); `F_x`, `F_e`, `F_y` for continuous-feature widths (incl. extras at input); `X_in`, `E_in`, `y_in` for the aggregate model-input concatenations. Used in spec/docs; existing code variable names unchanged. ✓

## 7. Audit findings (3 parallel reviewers — code, tests/configs, specs/docs)

### 7.1 New high-severity sites (scope additions to §5/§6)

The audit confirmed the four sites identified in §2 and surfaced **two more** that must be in scope:

**`src/tmgg/evaluation/graph_evaluator.py:597, 654`** — uses `(data.E_class.argmax(dim=-1) != 0)` to detect edges, which returns all-False for C_e=1 (argmax always 0). With a C_e=1 noise process this would silently emit empty adjacencies during evaluation. Same naming-discipline bug class but on E. Defensive fix in this sweep: branch on `E_class.shape[-1]` (handle C_e=1 explicitly, raise or special-case) instead of assuming class 0 is "no-edge."

**`src/tmgg/diffusion/noise_process.py:185-188`** (`_masked_graph_log_prob`) — v3/v4 of this spec said the module-level helper "only sees populated data" so the K=2 default is safe. The audit re-checked and confirmed this: callers pass already-populated `GraphData`. However, defensive plumbing still wins: pass `x_classes=samples1.X_class.shape[-1]` explicitly. The function then gets the discipline upgrade for free without needing to track which call sites are "safe."

### 7.2 Sites confirmed deferred-to-implementation (§5 covers them)

The remaining 8 `_read_categorical_x` call sites in `noise_process.py` (lines 1243-1517 across `_posterior_probabilities`, `_posterior_probabilities_marginalised`, `posterior_sample`, `posterior_sample_marginalised`, `forward_pmf`, `prior_log_prob`) are precisely the "deeper paths carry the same defect" the spec named in §2. Confirmed by the code audit. All resolved by stripping the default + threading `self.x_classes` per §5.2.

### 7.3 Defaulted-K fallbacks added by previous partial fix — must be removed

`diffusion_module.py:631, :858` currently use `getattr(self.noise_process, "x_classes", 2)` as a defensive fallback when threading C_x through the diagnostic logging. The fallback hides a contract violation: if `x_classes` were ever missing, defaulting to 2 silently re-introduces the bug. v5 removes the `getattr(..., 2)` pattern and uses strict `self.noise_process.x_classes` access (the noise process always has it when categorical X is involved; if not, an `AttributeError` is the right failure).

### 7.4 Tests (audit-confirmed)

Existing tests requiring parametrization (not just new tests as v4 suggested):
- `tests/experiments/test_discrete_diffusion_module.py::test_output_shapes` — pins `pred.X_class.shape == (bs, n, 2)`; parametrize on `c_x ∈ {1, 2}`.
- `tests/experiments/test_discrete_diffusion_module.py::test_training_loss_matches_train_loss_discrete` (lines 372-380) — has an inline reproduction of the legacy K=2 synth; replace with the canonical helper, parametrize.
- `tests/experiments/test_discrete_diffusion_module.py::test_training_loss_masks_padding` — exercises `_compute_loss` on structure-only batch; will hit the new `_read_field` path.
- `tests/models/test_architecture_parity.py::test_structure_only_input_synthesises_node_features` — directly exercises `transformer_model.py:944` synth; explicit comment in the test cements the bug. Parametrize on `output_dims_x_class ∈ {1, 2}`; assert raise for ≥3.

Plus: `tests/diffusion/test_noise_process.py` has 40+ instances of `x_classes ∈ {2, 3}` and **zero coverage of x_classes=1**. The new parametrized test (`tests/diffusion/test_categorical_noise_process_x_classes.py`) closes this gap.

### 7.5 Doc updates (spec cross-references)

The audit confirmed that `docs/specs/2026-04-15-unified-graph-features-spec.md` has three doc-edit needs:

1. **§"Architectures without explicit node features" (lines 169-179)** — append the canonical-helper reference and the C_x regime table:

> "The canonical derivation is `GraphData.synth_structure_only_x_class(node_mask, C_x)` (see `2026-04-27-x-class-synth-unification-spec.md`); each consumer (model, noise process, loss) calls it with its own authoritative C_x. C_x = 1 is the canonical structure-only encoding; C_x = 2 (`[no-node, node]`) is retained for backward compat with existing C_x = 2 model presets; C_x ≥ 3 is real categorical and MUST come from the dataset."

2. **§"Removed fields" failure-mode table (line 521)** — add a row codifying the X-vs-E asymmetry:

> "`_read_categorical_e` invoked with `E_class=None` → Raise `ValueError` (E is never synthesised; see 2026-04-27 spec)."

3. **§"Proposed representation" pseudo-schema (lines 125-131)** — uses `dx_class`, `dx_feat`, `de_class`, `de_feat` for channel widths. Add a footnote mapping `dx_class → C_x`, `dx_feat → F_x`, etc., or update the schema names. This is purely vocabulary alignment.

Plus one historical-report postscript:
- **`docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md` §1** — declared "X_class=None synthesised as dX=2 vs upstream dX=1 — NO-OP". The Modal SBM crash invalidated this verdict. Add postscript: "Superseded 2026-04-27: synthesis was buggy on C_x=1 configs; see new spec."

### 7.6 Configs — confirmed out of scope

18 production SBM/structure-only Hydra presets at `noise_process.x_classes: 2` are the legacy structure-only convention. They work. Spec §10 keeps them out of scope. Only `experiment/discrete_sbm_vignac_repro.yaml` is at C_x=1 today. **Suggested low-cost addition:** one-line clarifying comment ("`X: 2` here means C_x = 2; F_x extras adjusted at runtime via `extra_features.adjust_dims`") on the `input_dims` line of each preset, to prevent future readers from conflating `input_dims["X"]` with C_x. Cheap, prevents the same bug recurring.

### 7.7 Low-severity nits

- `src/tmgg/diffusion/noise_process.py:284-291` and `src/tmgg/training/lightning_modules/diffusion_module.py:130-136` docstrings call C_x=2 the "default" / "molecular convention" — recast as legacy structure-only per §3.
- `src/tmgg/models/digress/extra_features.py:62` docstring has `input_dims["X"] = base_dx + extra_X` — this IS X_in. Annotate with the canonical name where the aggregation happens.
- `src/tmgg/models/digress/transformer_model.py:518, :824, :850, :970` `input_dims` parameter docstrings should clarify "X_in / E_in / y_in aggregates (C+F including extras)."
- Module-level `__init__.py` docstrings (`tmgg/diffusion/`, `tmgg/data/`, `tmgg/models/`) lack pointers to the synthesis contract; add when implementing.
- `tests/models/test_digress_gnn_projections.py:330, :362` has within-file inconsistency — line 330 treats `input_dims["X"]` as C_x; line 362 correctly uses X_in semantics. Pick one.
- 8 frozen JSON discrete_gen run configs at `configs/discrete_gen/*.json` encode legacy C_x=2 in record form; leave alone.

## 8. Files to modify (final, audit-aligned)

| File | Change |
|---|---|
| `src/tmgg/data/datasets/graph_types.py` | Add `@classmethod synth_structure_only_x_class` on `GraphData`. Tighten `from_pyg_batch` and `binarised_adjacency` docstrings to flag C_e=1 vs C_e=2 branches. |
| `src/tmgg/diffusion/noise_process.py` | `_read_categorical_x` strip default; **all 11** call sites (8 deep-path + 2 in `_masked_graph_log_prob` + 1 in `_apply_noise` already correct) pass `x_classes` explicitly. `_read_categorical_e` add explicit `e_classes` arg + raise on None with clear asymmetry message. Update the C_x=2 "default/molecular" docstring to reflect C_x regimes. |
| `src/tmgg/training/lightning_modules/diffusion_module.py` | `_read_field` defers to noise process; `_synth_x_class` deleted; `setup()` triplet invariant assert. Remove `getattr(..., 2)` fallbacks at lines 631/858. Tighten docstring at line 130-136. |
| `src/tmgg/models/digress/transformer_model.py` | Inline synth → `GraphData.synth_structure_only_x_class(node_mask, self.output_dims_x_class)`. Tighten `input_dims` parameter docstrings to clarify X_in semantics. |
| `src/tmgg/evaluation/graph_evaluator.py` | Defensive fix for C_e=1 in `argmax != 0` checks at lines 597, 654. Branch on `E_class.shape[-1]`. |
| `docs/specs/2026-04-15-unified-graph-features-spec.md` | Three insertions: §"Architectures without explicit node features" canonical-helper reference; failure-mode table row for E never synthesises; vocabulary footnote mapping `dx_class → C_x` etc. |
| `docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md` | Postscript on §1 superseding the "NO-OP" verdict. |
| `tests/diffusion/test_noise_process.py` | Existing — leave alone (already covers C_x ∈ {2, 3}); the new parametrized test below covers C_x=1. |
| `tests/diffusion/test_categorical_noise_process_x_classes.py` | NEW — parametrized over (C_x, C_e). |
| `tests/models/test_graph_transformer_structure_only.py` | NEW — C_x=1 model path. |
| `tests/experiments/test_discrete_diffusion_module.py` | Parametrize 3 existing tests (`test_output_shapes`, `test_training_loss_matches_train_loss_discrete`, `test_training_loss_masks_padding`) on `c_x ∈ {1, 2}`. Replace inline K=2 synth at lines 372-380 with the canonical helper. |
| `tests/models/test_architecture_parity.py` | Parametrize `test_structure_only_input_synthesises_node_features` on `output_dims_x_class ∈ {1, 2}`; add a C_x≥3 case asserting the helper raises. |
| `src/tmgg/experiments/exp_configs/models/discrete/*.yaml` (18 presets, optional but cheap) | One-line clarifying comment on the `input_dims` line: "`X: 2` here means C_x = 2; F_x extras adjusted at runtime via `extra_features.adjust_dims`." Prevents future C_x / X_in conflation by readers of these presets. |

## 9. Effort & risk (audit-revised)

| Layer | Effort | Risk |
|---|---|---|
| `GraphData` classmethod | ~25 lines | low |
| `_read_categorical_{x,e}` strip default + thread C_x/C_e (11 X sites incl. `_masked_graph_log_prob`, plus E sites) | ~30 lines | low — pyright catches missing args |
| `_read_field` + `_synth_x_class` consolidation, remove `getattr(..., 2)` fallbacks | ~20 lines (mostly deletions) | low |
| `transformer_model.py` synth fix | ~5 lines | low |
| `evaluation/graph_evaluator.py` C_e=1 defensive fix | ~10 lines | low |
| `setup()` triplet invariant | ~20 lines | low |
| Docstring/comment tightening (noise_process / diffusion_module / transformer_model / extra_features / module __init__.py) | ~25 lines | none |
| Spec doc edits (2026-04-15 §"Architectures" + failure-mode row + vocabulary footnote; 2026-04-21 triage postscript) | ~45 lines across 2 files | none |
| 18 SBM model preset comment additions (1 line each, optional) | ~18 lines | none |
| Existing test parametrization (4 tests) | ~30 lines | low |
| New parametrized tests (2 files) | ~80 lines | low |
| **Total** | **~310 lines, ~12 files** | low overall |

## 10. Out of scope

- Updating molecular configs to C_x=1 (each dataset's C_x is its own decision).
- Eliminating `X_class=None` from the data layer (would contradict 2026-04-15 spec; wider refactor).
- Bulk-fixing the 19 C_x=2 model presets — they work, no need to touch them.
- F_<field> and X_in / E_in / y_in concerns — separate continuous-features audit, not this spec.
- Renaming codebase variables (`x_classes` → `c_x`, etc.) — out of scope; the canonical names are doc-level only.
- Full integration test on Modal — relaunched training run serves as integration after unit tests pass.
