# Phase 4 — Paper Figures Spec

Target paper: `_NeurIPS_2026__Understanding_Graph_Denoising.pdf`. Three figures (F1–F3) summarise the Phase 4 improvement-gap surrogate sweep. All figures read the same CSV and share one style module so the house look stays consistent.

## Shared conventions

- **Data source**: `docs/reports/2026-04-19-phase4-eigenvalue-study/phase4_sweep.csv` (25 600 rows).
- **Headline cell** (unless stated otherwise): `frame_mode='frechet'`, `estimator_label='knn_top_k'`, `noise_type='gaussian'`. This is the cell whose margins the README headline table reports.
- **Script**: `scripts/plot_phase4_figures.py`, `uv`-style `# /// script` header with `matplotlib>=3.8`, `pandas>=2.0`, `numpy>=1.24`. No side effects outside the figures directory.
- **Output dir**: `docs/reports/2026-04-19-phase4-eigenvalue-study/figures/`. Emit **both** `.pdf` (for LaTeX inclusion) and `.png` at 300 dpi (for quick review and Slack).
- **Sizes**: NeurIPS single-column 3.25", double-column 6.75". F1 and F3 are double-column; F2 is single-column.
- **Style**:
  - `plt.rcParams['font.family'] = 'serif'`, `font.size=8`, `axes.labelsize=8`, `legend.fontsize=7`.
  - No title on the figure itself — caption lives in the `.tex`. Panel titles OK when there are multiple subplots.
  - Tight layout; `bbox_inches='tight'` on save.
- **Error semantics**: error bars and ribbons are ±1 SD across the five seeds `{42, 123, 2024, 7, 11}`. Points are seed means. See the *Metric definition* and *Estimator definition* sections below for the quantity on the y-axes.
- **Palette**:
  - Synthetic SBMs: viridis gradient over diversity `{0.0, 0.33, 0.67, 1.0}` so the reader sees monotonicity at a glance.
  - Real datasets: tab10 warm colours — `spectre_sbm` red, `enzymes` orange, `proteins` green, `collab` purple.
  - Null ribbon: neutral grey `#b0b0b0` at alpha=0.35.
  - Pass / fail bar colouring in F3: pass = `#3b7dd8`, fail = `#b0b0b0`.

## Metric definition — fraction of variance explained (FVE)

Let `B_i = V̂_{k,i}^T A_i V̂_{k,i} ∈ ℝ^{k×k}` be the clean adjacency of graph *i* re-expressed in the noisy top-*k* eigenbasis, and let `Λ̃_{k,i} ∈ ℝ^k` be the corresponding noisy eigenvalues (both in the dataset-wide Fréchet frame; see `figures.md` for why). The metric on all figure y-axes derives from

$$\mathrm{FVE}(k) \;=\; \frac{\operatorname{Var}\!\bigl(\,\mathbb{E}[B_i \mid \tilde\Lambda_{k,i}]\,\bigr)}{\operatorname{Var}(B_i)} \;\in\; [0,1].$$

Both variances are Frobenius-sum variances taken across the graph population. By the law of total variance this is exactly the **coefficient of determination (R²) of the Bayes-optimal predictor** of `B` from `Λ̃_k`: it measures the fraction of `B`'s Frobenius variance that **any** function of `Λ̃_k` could predict. `FVE = 0` means `Λ̃_k` carries no information about `B`; `FVE = 1` means `Λ̃_k` determines `B` entirely.

The **calibrated FVE margin** `FVE_real − FVE_null` is the quantity F2 and F3 plot directly, where `FVE_null` is FVE recomputed with `Λ̃_k` permuted across graphs (so feature–target pairing is destroyed while marginals are preserved).

### Why FVE rather than alternatives

| Alternative | Why we didn't pick it |
|---|---|
| **Raw ĝ_k** (unnormalised numerator) | Has units of Frobenius² and scales mechanically with `trace Cov(B)`, which itself grows with edge density and n. Cross-dataset and cross-k comparisons become meaningless. FVE normalises this away. |
| **Linear R²** (OLS regression of `vec(B)` on `Λ̃_k`) | Assumes a linear relationship; the improvement-gap theory (eq. 18) is explicitly about the **non-linear** Bayes gap `ℓ_lin − ℓ_f`. Using linear R² would collapse the quantity we care about to zero by construction. |
| **Mutual information `I(B ; Λ̃_k)`** | Correct in principle but very hard to estimate for matrix-valued `B` at our sample sizes (N ≈ 68–3700). Known estimators (kNN-MI of Kraskov, KSG) have larger and less-predictable finite-sample bias than the conditional-mean estimator, and no natural `[0,1]` scale. |
| **HSIC / dCor** (kernel / distance dependence measures) | Detect arbitrary dependence, but lose the variance-share interpretation. A HSIC value of 0.3 does not translate to "30% of the variance in `B` is recoverable from `Λ̃_k`" — it just says the two are statistically dependent. Reviewers will ask "how much is that", and we would have no answer. |
| **MSE of a fitted regressor** (train/test MLP, GP, kernel ridge) | Depends on model class choice and hyperparameter tuning. Each choice raises a "did you pick the best class?" objection. The Bayes-optimal benchmark underlying FVE sidesteps this entirely. |
| **Spearman / Pearson** on scalar summaries (e.g. spectral gap vs. trace(B)) | Too coarse: `B` is a `k×k` matrix, and projecting it to a scalar before computing a rank correlation discards the information the paper's surrogate explicitly keeps. |

## Estimator definition — kNN conditional mean

We estimate `E[B_i | Λ̃_{k,i}]` by nearest-neighbour averaging in `Λ̃_k`-space:

1. Compute the Euclidean pairwise distance matrix `D ∈ ℝ^{N×N}` with `D_{ij} = ‖Λ̃_{k,i} − Λ̃_{k,j}‖_2`.
2. For each graph *i*, let `𝒩_i` be the indices of the **m = 10** nearest neighbours (excluding *i* itself — the self-entry sits at distance 0).
3. Form the kNN conditional-mean estimate `B̂_i = (1/m) Σ_{j ∈ 𝒩_i} B_j`.
4. Compute `ĝ_k = (1/N) Σ_i ‖B̂_i − μ_B‖_F²` where `μ_B = (1/N) Σ_i B_i`.
5. Report `FVE = ĝ_k / (1/N) Σ_i ‖B_i − μ_B‖_F²`.

The CSV's `estimator_label='knn_top_k'` column corresponds to this choice with `Λ̃_k` as the full top-*k* eigenvalue vector. The other labels (`knn_1d`, `bin_1d`, `invariants_knn`) are sensitivity variants and do not feed the main paper figures.

### Why kNN with m=10

- **Nonparametric.** No distributional assumption on `(B, Λ̃_k)`. We don't know, a priori, the functional form of `E[B | Λ̃_k]` — the whole premise of the paper is that the relationship is nonlinear in a specific way — so a nonparametric regressor is the right default.
- **No bandwidth selection.** m is the only hyperparameter. Kernel regression would require choosing a kernel and a bandwidth (Silverman's rule, cross-validation, Lepski); each choice is another reviewer hook. Changing m in `{5, 10, 20}` does not change the qualitative picture in our sensitivity runs.
- **Predictable finite-sample bias.** The average over m independent neighbours introduces a known ≈`1/m` residual variance under H₀ (features independent of targets), which appears as a flat ~0.10 floor in the permutation null. This floor is what the calibrated margin subtracts out. No other estimator we considered has such a cleanly-measurable bias; MLPs and kernel regressors mix bias and variance in harder-to-isolate ways.
- **Consistent.** `ĝ_k → g_k` as `N → ∞`, `m → ∞`, `m/N → 0` (standard kNN-regression consistency). At our sample sizes (N ≥ 68) m=10 sits comfortably in the safe regime.
- **Computationally trivial.** One N×N distance matrix plus a single stack-and-average per graph. Every cell in the 25 600-row sweep runs in milliseconds.

### The permutation null as a bias diagnostic, not a significance test

Shuffling `Λ̃_k` across graphs preserves the marginal distribution of the features but breaks the pairing with `B`, so under the null the true conditional mean is constant: `E[B | Λ̃_k^{\mathrm{perm}}] ≡ μ_B`. Any non-zero `FVE_null` is therefore estimator bias, not genuine structure. We report it and subtract it; we do **not** report a p-value. This matches the conceptual framing in the paper — the surrogate is an effect-size measurement of the Bayes gap, not a hypothesis test.

### Pass criterion

A dataset cell passes when `FVE_real − FVE_null ≥ 0.10` **and** `FVE_null < 0.30`. The first clause is the signal requirement; the second clause guards against small-N regimes where the kNN bias inflates and leaves little headroom between the null and the real curve. The two thresholds were fixed before Phase 4 data collection and are the same thresholds the Phase 3 diversity-knob validation used.

## F1 — FVE vs ε per dataset (with permutation-null ribbon)

**Question**: does the FVE stay above the permutation null across ε, and does it behave similarly across k?

- **Layout**: 2 rows × 4 cols grid, 8 panels, shared y-axis per row. Landscape, ~6.75" × 3.5".
- **Panel order**: row-major top-left to bottom-right — `sbm_d0.00, sbm_d0.33, sbm_d0.67, sbm_d1.00` on top row; `spectre_sbm, enzymes, proteins, collab` on bottom row.
- **Per panel**:
  - x-axis: `noise_level ∈ {0.01, 0.05, 0.1, 0.15, 0.2}` on linear scale.
  - y-axis: `FVE` (fraction of variance explained, `ĝ_k / tr Cov(B)`). Range 0–0.8 (shared).
  - Foreground: four lines for `k ∈ {4, 8, 16, 32}` over `permuted=False`, coloured by k with a sequential cmap (`plasma`). Markers at each ε, line joining them, ±1 SD vertical error bars across seeds.
  - Background ribbon: for each k, fill-between ±1 SD of the `permuted=True` curve in the same k-colour at alpha=0.15. This keeps the k-specific null visible rather than collapsing it to a single band.
  - Dashed horizontal line at `y=0.10` (pass threshold for the calibrated FVE margin, after subtracting null).
  - Panel subtitle: dataset name + retained N (e.g. `sbm_d0.67 (N=200)`).
- **Legend**: single legend bottom-centre with k=4/8/16/32 swatches plus a grey ribbon swatch labelled "permutation null (±1 SD)".
- **Filename**: `F1_fve_vs_epsilon.pdf` / `.png`.

## F2 — calibrated FVE margin vs diversity (synthetic SBM only)

**Question**: does the FVE track community strength monotonically, and does the effect size grow with k?

- **Layout**: single panel, single-column, ~3.25" × 2.5".
- **Data**: subset to `dataset ∈ {sbm_d0.00, sbm_d0.33, sbm_d0.67, sbm_d1.00}`, `noise_level=0.1`, headline cell (Fréchet, knn_top_k, gaussian).
- **x-axis**: diversity ∈ {0.0, 0.33, 0.67, 1.0} on a linear scale, ticks at the four grid points.
- **y-axis**: **calibrated FVE margin** `= FVE_real − FVE_null` per seed, averaged across seeds (pairing at the seed level so error bars propagate seed-to-seed).
- **Lines**: one line per `k ∈ {4, 8, 16, 32}`, coloured with the same plasma cmap as F1. Markers at each diversity point; error bars = SD of the per-seed calibrated margin.
- **Dashed horizontal at y=0.10**: pass threshold.
- **Legend**: upper-left inside the axes, 4 entries.
- **Filename**: `F2_calibrated_margin_vs_diversity.pdf` / `.png`.

## F3 — dataset ranking at the headline cell

**Question**: which datasets carry a community signal the surrogate recovers, ordered by margin?

- **Layout**: single panel, double-column width (for readability), ~6.75" × 2.75".
- **Data**: subset to headline cell + `k=8`, `noise_level=0.1`, `noise_type='gaussian'`, `frame_mode='frechet'`, `estimator_label='knn_top_k'`. One row per dataset (seed-averaged), sorted by calibrated margin descending.
- **Chart**: horizontal bar chart.
  - Bar length = calibrated FVE margin (`FVE_real − FVE_null`).
  - Bar colour: blue if margin ≥ 0.10 **and** `FVE_null` < 0.30 (= pass per the README criterion), else grey.
  - Black horizontal error bar on each bar: √(Var(FVE_real)/5 + Var(FVE_null)/5) across seeds (standard-error of the difference of two means).
  - Annotate each bar on the right with the numeric margin to two decimals.
  - Vertical dashed line at x=0.10 (pass threshold).
- **Axes**: y = dataset label; x = calibrated FVE margin (0 to ~0.7).
- **Filename**: `F3_dataset_ranking.pdf` / `.png`.

## Supplementary (optional, same script, flag-gated)

Flag `--supplementary` also renders:

- **F1-digress**: identical to F1 but with `noise_type='digress'`. Filename `FS1_fve_vs_epsilon_digress.pdf`.
- **F3-per-graph-frame**: identical to F3 but with `frame_mode='per_graph'`. Demonstrates the Fréchet-frame contribution to the signal. Filename `FS2_dataset_ranking_per_graph.pdf`.

The main paper cites only F1–F3; supplementary goes to the appendix.

## Implementation notes

- Load CSV once, filter per figure. No reliance on row ordering — always group by the full `(dataset, noise_type, noise_level, frame_mode, estimator_label, permuted, k)` key before aggregating across seeds.
- Seed count assertion: for every cell used in the figures, assert `len(group) == 5`; fail loudly if not.
- Save each figure twice (PDF, PNG). The PDF vector output is what LaTeX will `\includegraphics`.
- No `plt.show()`. No interactive state leaking between figures — call `plt.close(fig)` after saving.
- The script prints a one-line manifest per figure: `wrote F1 (panels=8, series=4, points=40) → figures/F1_ratio_vs_epsilon.{pdf,png}`.

## Open questions for the user

Before implementation, please confirm:

1. **Subplot grid for F1**: 2×4 landscape as specified, or 4×2 portrait? I default to 2×4 for double-column inclusion.
2. **Pass-threshold line** on F1: should it sit at `y=0.10` (absolute ratio) or at `y = mean_null + 0.10` per panel? I defaulted to absolute 0.10 for consistency with the README headline criterion, but per-panel null-shifted might read better visually.
3. **F2 k-lines**: include all four k-values, or only `k=8` with bigger error bars? I default to all four lines; swap to k=8-only if the panel gets noisy.
4. **Supplementary figures**: render by default, or leave behind the `--supplementary` flag? Flag-gated by default.
5. **DiGress headline**: the README headline is Gaussian; confirm we keep DiGress as supplementary rather than promoting it to a side-by-side panel in F1.
