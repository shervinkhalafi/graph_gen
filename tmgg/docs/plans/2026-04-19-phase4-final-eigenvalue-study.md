# 2026-04-19 — Plan: Phase 4 Final Eigenvalue Study

Scope for the headline eigenvalue-study sweep that follows the Phase 3
v2 validation at
`docs/reports/2026-04-19-diversity-knob-validation/`. Phase 3 landed
with every (frame × estimator × k × seed) cell monotone in the
improvement-gap ratio and every permutation-null cell below 0.30, so
the surrogate is calibrated and the diversity knob actually moves it.
Phase 4 applies the same instrumentation across the real datasets the
paper's empirical section will argue from.

## Goal

Produce the headline figure(s) for the paper's empirical section:
surrogate improvement-gap vs. (dataset, noise level, k), with a
diversity axis for the synthetic SBM sweeps. The reader should be able
to conclude, per dataset, whether spectral denoising can meaningfully
beat linear denoising — that is, whether the improvement gap of eq.
(18) is materially positive after accounting for the kNN null.

Success criterion (deliberately stricter than Phase 3):

- For every real dataset, **calibrated ratio** = `ratio_real − ratio_null`
  is positive and separated from zero (≥ 0.10 margin) at the canonical
  `k` and noise level.
- The ordering of datasets by calibrated ratio is stable across `k` and
  across the three estimator variants (`knn_top_k`, `knn_1d`,
  `invariants_knn`). If they rank datasets differently, we have a
  methodology-choice sensitivity and the paper needs to pick one story.
- Synthetic SBM diversity sweep reproduces the v2 monotonicity with
  noise-level variation added — ratios monotone in diversity at every
  ε; the ε-axis slope tells us how the gap behaves as noise grows.

## Datasets

| Dataset | Source | n range | N (graphs) | Size-handling for Phase 4 |
|---|---|---|---|---|
| `spectre_sbm` | Existing fixture (`data.datasets.spectre_sbm`) | ~50 (fixed) | ~200 | No filtering needed — already uniform n. |
| `enzymes` | TU (`pyg_datasets.PyGDatasetWrapper`) | 2–126 | 600 | **Filter to 30 ≤ n ≤ 60** (uniform band → Fréchet mean well-defined without zero-pad contamination). Expected ≈ 300 graphs retained. |
| `proteins` | TU | 4–620 | 1113 | **Filter to 30 ≤ n ≤ 80**. Expected ≈ 500 graphs retained. |
| `synthetic_sbm_d{0,0.33,0.67,1.0}` | `data.datasets.sbm.generate_sbm_batch` | 50 (fixed) | 200 | Same parametrisation as Phase 3 v2. |

**Why size filtering, not zero-padding.** `B_k = V̂_k^T A V̂_k` requires
a well-defined eigenbasis. Zero-padding adds isolated virtual nodes,
injecting zero eigenvalues and shifting the top-k block. For a clean
cross-graph comparison within a dataset we need either (a) fixed n per
dataset or (b) invariants-only analysis. Filtering to a narrow band
gives us (a) at the cost of dropping ~30–50% of each TU dataset. The
dropped-data concern is a real limitation but acceptable for a first
pass; if the calibrated-ratio ordering depends on the filter band, we
revisit.

If more TU datasets land from Shervin (IMDB-BINARY, COLLAB, REDDIT-*),
add with the same filter policy.

## Parameters

Fixed across all datasets:

- `k_values = {4, 8, 16, 32}` (matches Phase 3).
- `noise_type = "gaussian"` primary; `digress` as a secondary column if
  Shervin wants it. Gaussian is the paper's canonical noise model for
  the eq. (18) derivation; digress adds discreteness that isn't
  necessary to make the surrogate claim.
- `noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]` (5 levels — matches the
  existing `eigenstructure.yaml` report config).
- `frame_mode = "frechet"` for the headline; `per_graph` reported as a
  diagnostic column (not headline).
- `estimators = {knn_top_k, knn_1d, bin_1d, invariants_knn}` all four,
  real + permutation-null pair per cell.
- `knn_neighbours = 10`, `num_bins = 4`.
- **Seeds: 5** (up from 3 in Phase 3) — real datasets have higher
  per-cell variance than controlled synthetics, and the paper needs
  tight error bars.

Per-dataset varied: only the `num_graphs` / filtered `N` and the
per-dataset `n` (fixed within dataset).

## Experimental matrix

Per dataset × noise level × frame mode × estimator × k × permuted × seed =
```
1 dataset
× 5 noise levels
× 2 frame modes
× 4 estimators
× 4 k values
× 2 (real + null)
× 5 seeds
= 1600 cells per dataset.
```

For 7 datasets (spectre_sbm + enzymes + proteins + 4 synthetic):
**~11,200 cells total**. Each cell is a single surrogate computation
(sub-second); the expensive step is the collector pass (one Phase 1 +
one noised collection per (dataset, seed, noise_level, frame_mode)).

## Compute budget

Rough per-cell cost:

- Phase 1 decomposition: `O(N · n³)` for a batch. At `N=200, n=50`
  this is ~0.5s; at `N=500, n=80` this is ~4s.
- Noised collection per noise level: same cost plus Procrustes per k.
  `L × (0.5 – 4)s` per seed per frame mode ≈ 5 – 40s per (dataset,
  seed, frame_mode).
- Surrogate analysis per cell: <10ms.

Estimate for 7 datasets × 5 seeds × 2 frame modes ≈ 70 collector runs.
At ~30s each → ~35 minutes of CPU. Plus surrogate analysis (11k cells
× 10ms ≈ 2 minutes). **Total: ~40 min, single-laptop local**. No Modal
needed. If PROTEINS filtering surfaces graphs with n=80 and is slower,
budget up to 90 minutes.

## Report structure

Deliverable: `docs/reports/2026-04-XX-phase4-eigenvalue-study/` with:

- `README.md` — narrative: goal recap, per-dataset calibrated-ratio
  ranking, noise-level sensitivity plots (in-text as inline SVG or
  referenced PNG), synthetic diversity axis.
- `raw_sweep.csv` — the full ~11k-row flat table matching the Phase 3
  v2 schema (`seed, dataset, diversity, frame_mode, estimator_label,
  permuted, k, noise_level, g_hat, trace_cov_B, ratio, num_graphs`).
- Headline figures:
  - **F1**: ratio vs. noise level, one panel per dataset, one line per
    k, for the Fréchet + knn_top_k cell — with null ribbon behind.
  - **F2**: calibrated ratio (`real − null`) vs. diversity for the
    synthetic sweep, at ε=0.1, across k.
  - **F3**: dataset-ranking bar chart at (k=8, ε=0.1, Fréchet,
    knn_top_k) with calibrated ratios and null error bars.

Monotonicity verdicts per dataset like the Phase 3 report; no "Phase 5"
gate since Phase 4 is the terminal study.

## Implementation sequence

1. **Extend the runner.** Refactor `scripts/run_diversity_sweep.py`
   into a generalised `scripts/run_phase4_eigenvalue_study.py` that
   takes a list of dataset configs (`synthetic_sbm(diversity=d, …)` +
   `pyg(name=enzymes, n_range=(30,60), …)` + `spectre(…)`) and runs
   the full matrix. Reuse `build_phase1_decompositions`. The only new
   code is per-dataset loading and size filtering — the surrogate
   plumbing is already in place.
2. **Add a size-filtering helper.** `filter_adjacencies_by_n_range(
   batch, n_min, n_max)` returns a subset with uniform n. Document the
   retention-rate expectation per dataset.
3. **Wire PyG loading.** For ENZYMES/PROTEINS, use the existing
   `PyGDatasetWrapper` to load then filter. For `spectre_sbm`, use its
   existing loader. Confirm both return `(N, n, n)` numpy arrays that
   can feed `build_phase1_decompositions`.
4. **Smoke-test the runner** on spectre_sbm only (single dataset, 1
   seed, 1 noise level) to confirm end-to-end plumbing.
5. **Run the full matrix** locally (no Modal).
6. **Write up the report** — narrative + table + figures.

## Open decisions before starting

1. **Size-band filter.** I've proposed `30 ≤ n ≤ 60` for ENZYMES and
   `30 ≤ n ≤ 80` for PROTEINS. These are judgement calls; tighter bands
   mean cleaner n but less data. Worth a quick look at each dataset's
   node-count histogram to pick defensible cutoffs.
2. **Noise type.** Gaussian only, or include DiGress-style discrete
   noise? I'd default to gaussian unless you want both reported (doubles
   the compute, costs ~40 more minutes).
3. **Seed budget.** 5 seeds is the minimum I'd defend for a paper
   figure. Going to 10 would give tighter error bars but doubles compute
   to ~80 min. Worth deciding up front.
4. **Other TU datasets.** Per `next_steps.md` Shervin is landing more
   TU datasets. Do we wait for them and do Phase 4 once, or run now
   with {ENZYMES, PROTEINS} and add the others as a quick rerun when
   they arrive?

## Risks

- **Filter distorts the dataset.** Dropping 50% of PROTEINS to get a
  fixed-n band might bias the surrogate result toward a non-
  representative slice. Mitigation: report retention rate; if the
  calibrated-ratio ordering changes materially when the filter band
  shifts by ±10 nodes, flag in the report.
- **Fréchet mean instability on small datasets.** With a uniform-band
  ENZYMES subset of ~300 graphs, the Grassmannian mean estimate is fine;
  at smaller filter yields (<100 graphs) the Fréchet mean gets noisier
  and the frame-convention argument weakens. Monitor per-dataset N.
- **kNN null floor at ~0.10.** The paper's argument only survives if
  the calibrated ratio `real − null` is materially above zero on real
  data. If real ENZYMES ratio comes in at ~0.12 and null at ~0.10,
  margin of 0.02 is within seed noise — we'd be reporting a null
  result. This is a real empirical risk and exactly what the study is
  supposed to surface; no code-fix if it happens, just a different
  story for the paper.

## Non-goals

- No rewrite of the collector, storage, or analyzer. All plumbing is
  in place from Phase 3 v2.
- No comparison to other surrogate candidates from the paper's
  appendix (e.g. Lemma 1 closed form). Those can slot into the
  analyzer as another `estimator` variant if Shervin wants them, but
  they're outside Phase 4 scope.
- No variable-n support across graphs. Fixed-band filtering is the
  compromise; variable-n infrastructure is a separate, larger piece of
  work best done once we know we need it.
