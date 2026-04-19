# 2026-04-18 — Plan: Improvement-Gap Surrogate + Eigenspectrum-Diversity Knob

> **Revision history**
> - 2026-04-18: initial draft.
> - 2026-04-19: revised after a reviewer-2 audit of the Phase 3 v1 run.
>   Revisions live in the text below, called out as
>   "(v2)" where they replace v1 text.

Follow-up plan for the two outstanding Shervin asks from `next_steps.md`.
Paper context: `_NeurIPS_2026__Understanding_Graph_Denoising.pdf`,
equation (18):

```
ℓ_lin − ℓ_f = tr(Cov(E[vec(B) | Λ̃_k]))                          (18)
            = E ‖E[B | Λ̃_k] − E[B]‖²_F                         (19)
```

with

- `A` — clean adjacency (signal),
- `Ã = A + noise` — noisy observation,
- `V̂_k, Λ̃_k` — top-k eigendecomposition of the noisy graph,
- `B = V̂_k^⊤ A V̂_k` — clean signal projected into the noisy eigenbasis,
- `ℓ_lin` — MSE of the best constant denoiser (no eigenvalue dependence),
- `ℓ_f` — MSE of the best eigenvalue-function denoiser.

The gap is the between-graph variance of the conditional mean `E[B | Λ̃_k]`.
If the dataset's eigenspectrum is homogeneous, the gap shrinks to zero; if
the spectrum varies across graphs, the gap grows and motivates spectral
denoisers over linear ones.

Sequencing is: build the measurement first, then the knob that's supposed
to move it, then validate. Shervin's ordering in `next_steps.md` matches.

## Phase 1 — Improvement-gap surrogate

### 1.1 Data

What the existing `eigenstructure_study` already has:

- `NoisedEigenstructureCollector` (`noised_collector.py:268`) produces, per
  clean graph `A_i` and noise level `ε`, the noisy top-k eigendecomposition
  `(V̂_{k,i}, Λ̃_{k,i})`.
- Storage layer under `storage.py` emits batch files keyed by noise level.

What's missing:

- The projection `B_i = V̂_{k,i}^⊤ A_i V̂_{k,i}` (shape `k × k`).
- An estimator for `E[B | Λ̃_k]`.

### 1.2 Frame convention (v2 — supersedes "sign canonicalisation")

`B_i = V̂_{k,i}^T A_i V̂_{k,i}` is not sign-invariant (flipping a column of
`V̂_{k,i}` flips the corresponding row/column of `B_i`) and is not
rotation-invariant within degenerate eigenspaces (the free orthogonal
rotation changes entries of `B_i` while preserving the subspace). Both
ambiguities must be resolved before any cross-graph averaging or
`E[B | Λ̃_k]` computation.

**v1 prescribed** a sign canonicalisation (largest-magnitude entry
positive per column). This fixes signs but not degenerate-block rotations.

**v2 implementation** uses orthogonal Procrustes alignment, which
subsumes sign flips (via SVD without det-constraint) *and* resolves the
degenerate-block rotation. The alignment reference is parameterised via
``frame_mode`` in ``NoisedEigenstructureCollector``:

- ``frame_mode = "frechet"`` (default): every noisy `V̂_{k,i}` is
  Procrustes-aligned to a single dataset-wide reference `V*_k`, the
  extrinsic Grassmannian mean of the clean top-*k* blocks (top-*k* left
  singular vectors of the `(n, N·k)` stack of clean blocks). This is the
  *common-frame* alignment required for `Cov(vec(B))` and
  `E[B | Λ̃_k]` to be well-defined population quantities in eq. (18).
- ``frame_mode = "per_graph"``: each `V̂_{k,i}` is aligned to that
  graph's own clean `V_{k,i}`. Retained as a diagnostic; **not** valid
  for the population claim on a heterogeneous dataset because different
  graphs end up in different graph-specific frames.

**Why the switch.** The reviewer-2 audit of the Phase 3 v1 run flagged
that per-graph alignment leaves every `B_i` in a graph-specific frame,
so the sample mean `μ_B` and `trace(Cov(B))` add matrices from
incommensurable frames. The Phase 3 v2 run (see
`docs/reports/2026-04-19-diversity-knob-validation/`) empirically
confirms this: at `diversity = 0` the per-graph ratio is ~0.56 while
the Fréchet ratio is ~0.21, reflecting the frame-incoherence
contamination.

Unit tests in ``tests/experiment_utils/test_improvement_gap.py``:

- ``test_frechet_mean_subspace_is_orthonormal``
- ``test_align_to_reference_is_closer_to_reference_than_raw``
- ``test_frechet_and_per_graph_produce_B_in_different_frames`` —
  confirms the two paths produce materially different `B` matrices but
  identical frame-invariant summaries to numerical precision.

### 1.3 Surrogate estimator

Add a new analyzer function, `compute_improvement_gap_surrogate` in
`src/tmgg/experiments/eigenstructure_study/analyzer.py`:

```python
def compute_improvement_gap_surrogate(
    noise_level: float,
    *,
    k: int,
    conditional_estimator: Literal["knn", "bin"] = "knn",
    knn_neighbours: int = 10,
) -> ImprovementGapResult: ...
```

Reference implementation, kNN variant:

1. Iterate paired batches (clean `A_i`, noisy `V̂_{k,i}`, `Λ̃_{k,i}`) at the
   given noise level. Compute and stack `B_i`.
2. `μ_B := (1/N) Σ_i B_i` — sample estimate of `E[B]`.
3. For each `i`, find its `m` nearest neighbours in `ℝ^k` under Λ̃-space
   (sorted eigenvalues, Euclidean). Average the neighbours' `B` to get
   `B̂_i ≈ E[B | Λ̃_{k,i}]`.
4. Surrogate: `ĝ = (1/N) Σ_i ‖B̂_i − μ_B‖²_F`.
5. Report `ĝ` alongside the total `tr(Cov(B))` so the fraction
   `ĝ / tr(Cov(B))` (conditional-variance share) is visible.

Binning variant (cheap check):

- Group graphs by quantiles of the spectral gap `λ_1 − λ_2`.
- Compute between-bin variance of `B`. This is the classical
  variance-decomposition surrogate and is fast.

**v2 clarification — estimator comparability.** kNN on the full top-*k*
eigenvalues and binning on the scalar spectral gap measure
*different* quantities: kNN on a `k`-dim feature and binning on a
1-dim feature are not comparable. The v2 runner therefore exposes a
third estimator variant, ``knn_1d``, which runs kNN on the same 1-D
spectral-gap feature the binning estimator uses. ``knn_1d`` and
``bin_1d`` are apples-to-apples and are expected to agree up to
estimator noise; disagreement between ``knn_top_k`` and ``knn_1d``
reflects the information gained by looking at the full eigenvalue
vector, not a calibration problem.

**v2 frame-invariant cross-check.** Add ``target="invariants"`` path:
replace `B_i` with `(tr(B_i), ||B_i||_F², sorted eigvals(B_i))` — a
`(k + 2)`-dim vector of orthogonal-conjugation invariants — and run
the standard kNN conditional-mean-variance decomposition on that.
The invariants target is exactly frame-independent by construction,
so comparing the ``frechet`` and ``per_graph`` frame modes on this
target gives a zero-cost sanity check: the two ratios must agree to
within float precision. In the Phase 3 v2 report they do.

### 1.4 Permutation null (v2 — new)

The reviewer-2 audit found that the v1 Phase 3 report claimed success
using raw ``ratio = ĝ / tr(Cov B)`` numbers without controlling for
finite-sample kNN bias. A correctly-calibrated estimator should return
``ratio ≈ 0`` when the conditioning features are decorrelated from the
targets; kNN with small `m` on compact feature clouds empirically gives
``ratio ≈ 1/m`` under that null.

``estimate_improvement_gap`` now accepts ``permute_features=True`` and a
``permutation_seed``; the runner emits a matched null row for every
real-features row. A cell is considered calibrated if the null ratio
stays below 0.30 across diversity levels. Report the null column
alongside the real column so readers can assess the real-minus-null
gap, not just the absolute ratio.

Unit test: ``test_permutation_null_ratio_is_small_on_cluster_input``.

### 1.4 Storage + reporting

- `ImprovementGapResult` dataclass mirrors the existing
  `DriftComparisonResult` pattern: flat JSON, mean/std/quantiles.
- Extend `storage.save_dataset_manifest` to include the surrogate per noise
  level.
- Add to the existing eigenstructure report template
  (`exp_configs/report/eigenstructure.yaml`) so plots include a
  gap-vs-noise-level curve.

### 1.5 Tests

- Synthetic sanity: `N` identical graphs + tiny noise → `ĝ ≈ 0`.
- Synthetic sanity: two well-separated spectrum clusters → `ĝ` close to
  the between-cluster variance of `B`.
- Property test: `ĝ ≤ tr(Cov(B))` (conditional variance cannot exceed
  total).

### 1.6 Expected effort

Half-day to a day for the analyzer + tests. Collector changes are small
because the projection is one matmul per graph.

## Phase 2 — Eigenspectrum-diversity knob

### 2.1 Target

Controlled variation of the SBM hyperparameters across graphs in one
synthetic dataset. At `diversity = 0` the existing behaviour is preserved
(all graphs share identical hyperparams). At `diversity = 1` each graph
draws its own hyperparams from a maximum-spread distribution.

### 2.2 Where

`src/tmgg/data/datasets/sbm.py` holds `generate_sbm_batch`; extend or wrap
it. The matching datamodule is `SyntheticCategoricalDataModule` (which
wraps `MultiGraphDataModule`). Extend the config schema at
`exp_configs/data/sbm_default.yaml` (and variants).

### 2.3 Which knobs to randomise

Pick a minimal set that has direct spectral interpretation:

- `num_blocks` — v1 drew from `{2, 3, 4, 5}`; **v2 freezes this** in the
  validation runner. The `(2, 5)` range has integer midpoint 3.5 but
  uniform-draw mean 3.5, which rounds to 4 at `diversity = 0` and gives a
  subtle asymmetry between the reference and the full-diversity
  distribution (the reviewer's M1 finding). The code still supports
  tuple-ranged `num_blocks`; the runner just opts not to vary it so the
  attribution of ĝ growth is unambiguous.
- `p_intra` — drawn from `[p_intra_min, p_intra_max]`. Controls
  intra-block density and therefore the leading eigenvalue magnitudes.
- `p_inter` — same pattern; controls block-separation strength and hence
  the eigengap.
- `block_size_vector` — drawn from a symmetric Dirichlet distribution
  with concentration `block_size_alpha`, scaled by `N`. Unequal blocks
  introduce additional spread in the top-k spectrum.

`diversity ∈ [0, 1]` interpolates: intervals collapse to their midpoint at
`0`, widen to the full configured range at `1`.

### 2.4 API sketch

```python
def generate_sbm_batch(
    num_graphs: int,
    num_nodes: int,
    *,
    num_blocks: int | tuple[int, int] = 2,
    p_intra: float | tuple[float, float] = 0.7,
    p_inter: float | tuple[float, float] = 0.1,
    block_size_alpha: float | None = None,  # None → equal blocks
    diversity: float = 0.0,                 # [0, 1]; scales any tuple range
    seed: int = 42,
) -> np.ndarray: ...
```

- Scalar arguments keep the current behaviour identical at `diversity = 0`.
- Tuple arguments enable per-graph sampling, scaled by `diversity`.

Add a matching config schema so Hydra overrides like
`+data.diversity=0.7 +data.num_blocks=[2,5]` work.

### 2.5 Tests

- `diversity = 0` with scalar args → bitwise-identical output to the current
  `generate_sbm_batch` (regression gate).
- `diversity > 0` with tuple args → histogram of per-graph eigengaps spans
  more than the fixed case (Kolmogorov-Smirnov against the degenerate
  distribution).
- Typing / error tests: passing scalar args with `diversity > 0` raises; the
  intent is that tuple args are the explicit mechanism.

## Phase 3 — Validate that the knob moves the surrogate

Once Phase 1 + Phase 2 land, run a short sweep:

- Four SBM datasets generated with `diversity ∈ {0, 0.33, 0.67, 1.0}`,
  same `num_graphs`, same `num_nodes`, same noise level.
- For each, run the eigenstructure collector, then
  `compute_improvement_gap_surrogate` across every
  (`frame_mode`, `estimator`, `target`, `permuted`) cell of interest.
- **v2 success criterion** — per-cell *mean ratio* across seeds is
  monotone non-decreasing in diversity. Absolute `ĝ` is reported but not
  the gate: it tracks `trace(Cov B)`, which grows mechanically with the
  knob, so absolute-`ĝ` monotonicity is a weaker claim than ratio
  monotonicity.
- **v2 calibration gate** — permutation-null ratio stays below 0.30 in
  every cell. A larger null signals finite-sample estimator bias that
  must be deducted before interpreting the real-feature ratio.
- **v2 cross-check gate** — ``frechet`` and ``per_graph`` frame modes
  give approximately equal ratios on the ``target = "invariants"`` path
  (by construction, since invariants are frame-independent to numerical
  precision). Disagreement there means the invariants implementation has
  a bug.
- **v2 seeding** — run ≥3 seeds; report mean±std per cell.

If the knob is flat, non-monotone, or needs the null-ratio gate, the
knob parameterisation is wrong — iterate on which hyperparameters the
knob perturbs.

Artefact: markdown note + flat CSV in `docs/reports/` with the full
(seed × diversity × frame_mode × estimator × permuted) cross-product
and explicit monotonicity / null-ratio verdicts. If the knob works,
proceed to Phase 4; if not, fix Phase 2 before running anything
expensive.

## Phase 4 — Final eigenvalue study across all datasets

Gated on Phase 3 passing. Once the surrogate is implemented and the knob
is validated, run it over:

- `spectre_sbm` (upstream fixture),
- the synthetic SBM sweeps (low / medium / high diversity),
- whatever TUDatasets Shervin lands (ENZYMES, PROTEINS already wired;
  more coming).

Report: gap vs. dataset, gap vs. noise level, gap vs. diversity. This is
the headline figure the paper's empirical section will argue from.

## File-by-file deliverables

| Path | Change |
|------|--------|
| `src/tmgg/experiments/eigenstructure_study/noised_collector.py` | Compute + store `B = V̂_k^⊤ A V̂_k` per graph per noise level; apply sign canonicalisation before storage. |
| `src/tmgg/experiments/eigenstructure_study/analyzer.py` | Add `ImprovementGapResult` dataclass + `compute_improvement_gap_surrogate` (kNN + binning estimators). |
| `src/tmgg/experiments/eigenstructure_study/storage.py` | Persist B tensors and surrogate results in the existing batch-file layout. |
| `src/tmgg/data/datasets/sbm.py` | Extend `generate_sbm_batch` with tuple-ranged hyperparameters + `diversity` scaling. |
| `src/tmgg/experiments/exp_configs/data/sbm_default.yaml` (+ variants) | Expose the new fields. |
| `src/tmgg/experiments/exp_configs/report/eigenstructure.yaml` | Surface gap-vs-noise and gap-vs-diversity plots. |
| `tests/experiments/eigenstructure_study/test_improvement_gap.py` | New — sanity, property, and estimator-agreement tests. |
| `tests/data/test_sbm_generation.py` | Extend — regression gate at `diversity=0` and KS test at `diversity>0`. |
| `docs/reports/2026-04-??-diversity-knob-validation.md` | Phase-3 artefact; written after the short sweep. |

## Resolved questions (v2)

Previous "open questions" status, with resolutions confirmed during the
v2 revision cycle (see commit messages on branch
``igork/improvement-gap-surrogate``):

1. **Conditional-mean estimator.** Resolved to kNN (m = 10) in
   Λ̃-space as primary; binning on 1-D spectral gap + kNN on 1-D
   spectral gap as comparable cross-checks. The Phase 3 v2 run reports
   all three alongside a frame-invariant kNN on ``invariants(B)``.
2. **Choice of `k`.** Resolved to a sweep ``k ∈ {4, 8, 16, 32}``
   reported per dataset and per noise level. The v2 report uses the
   sweep to show that lower `k` gives the strongest signal (top
   eigenvalues carry most community structure) while higher `k` gives
   diminishing returns.
3. **Frame convention.** Resolved to Procrustes alignment against the
   extrinsic Grassmannian mean of the clean top-*k* blocks
   (``frame_mode = "frechet"``). Confirmed with Shervin. The legacy
   per-graph alignment is retained as a diagnostic
   (``frame_mode = "per_graph"``) to surface the frame-choice effect in
   the report tables.

## Non-goals

- No refactor of the existing eigenstructure collector. Extend, don't
  rewrite.
- No new datamodule class. Reuse `SyntheticCategoricalDataModule` and
  `MultiGraphDataModule`.
- No performance tuning for the surrogate. It runs once per noise level
  per dataset and can be a few minutes long without hurting anyone.
