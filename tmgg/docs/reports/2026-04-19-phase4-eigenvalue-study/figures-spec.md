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
- **Error semantics**: error bars and ribbons are ±1 SD across the five seeds `{42, 123, 2024, 7, 11}`. Points are seed means.
- **Palette**:
  - Synthetic SBMs: viridis gradient over diversity `{0.0, 0.33, 0.67, 1.0}` so the reader sees monotonicity at a glance.
  - Real datasets: tab10 warm colours — `spectre_sbm` red, `enzymes` orange, `proteins` green, `collab` purple.
  - Null ribbon: neutral grey `#b0b0b0` at alpha=0.35.
  - Pass / fail bar colouring in F3: pass = `#3b7dd8`, fail = `#b0b0b0`.

## F1 — ratio vs ε per dataset (with permutation-null ribbon)

**Question**: does the surrogate stay above the permutation null across ε, and does it behave similarly across k?

- **Layout**: 2 rows × 4 cols grid, 8 panels, shared y-axis per row. Landscape, ~6.75" × 3.5".
- **Panel order**: row-major top-left to bottom-right — `sbm_d0.00, sbm_d0.33, sbm_d0.67, sbm_d1.00` on top row; `spectre_sbm, enzymes, proteins, collab` on bottom row.
- **Per panel**:
  - x-axis: `noise_level ∈ {0.01, 0.05, 0.1, 0.15, 0.2}` on linear scale.
  - y-axis: `ratio` (surrogate / `trace_cov_B`). Range 0–0.8 (shared).
  - Foreground: four lines for `k ∈ {4, 8, 16, 32}` over `permuted=False`, coloured by k with a sequential cmap (`plasma`). Markers at each ε, line joining them, ±1 SD vertical error bars across seeds.
  - Background ribbon: for each k, fill-between ±1 SD of the `permuted=True` curve in the same k-colour at alpha=0.15. This keeps the k-specific null visible rather than collapsing it to a single band.
  - Dashed horizontal line at `y=0.10` (pass threshold for the calibrated margin, after subtracting null).
  - Panel subtitle: dataset name + retained N (e.g. `sbm_d0.67 (N=200)`).
- **Legend**: single legend bottom-centre with k=4/8/16/32 swatches plus a grey ribbon swatch labelled "permutation null (±1 SD)".
- **Filename**: `F1_ratio_vs_epsilon.pdf` / `.png`.

## F2 — calibrated margin vs diversity (synthetic SBM only)

**Question**: does the surrogate track community strength monotonically, and does the effect size grow with k?

- **Layout**: single panel, single-column, ~3.25" × 2.5".
- **Data**: subset to `dataset ∈ {sbm_d0.00, sbm_d0.33, sbm_d0.67, sbm_d1.00}`, `noise_level=0.1`, headline cell (Fréchet, knn_top_k, gaussian).
- **x-axis**: diversity ∈ {0.0, 0.33, 0.67, 1.0} on a linear scale, ticks at the four grid points.
- **y-axis**: **calibrated margin** `= mean(ratio | permuted=False) − mean(ratio | permuted=True)`, both aggregated over seeds first (so error bars propagate seed-to-seed, not cell-to-cell).
- **Lines**: one line per `k ∈ {4, 8, 16, 32}`, coloured with the same plasma cmap as F1. Markers at each diversity point; error bars = SD of the per-seed calibrated margin.
- **Dashed horizontal at y=0.10**: pass threshold.
- **Legend**: upper-left inside the axes, 4 entries.
- **Filename**: `F2_calibrated_margin_vs_diversity.pdf` / `.png`.

## F3 — dataset ranking at the headline cell

**Question**: which datasets carry a community signal the surrogate recovers, ordered by margin?

- **Layout**: single panel, double-column width (for readability), ~6.75" × 2.75".
- **Data**: subset to headline cell + `k=8`, `noise_level=0.1`, `noise_type='gaussian'`, `frame_mode='frechet'`, `estimator_label='knn_top_k'`. One row per dataset (seed-averaged), sorted by calibrated margin descending.
- **Chart**: horizontal bar chart.
  - Bar length = calibrated margin (ratio_real − ratio_null).
  - Bar colour: blue if margin ≥ 0.10 **and** null ratio < 0.30 (= pass per the README criterion), else grey.
  - Black horizontal error bar on each bar: √(Var(ratio_real)/5 + Var(ratio_null)/5) across seeds (standard-error of the difference of two means).
  - Annotate each bar on the right with the numeric margin to two decimals.
  - Vertical dashed line at x=0.10 (pass threshold).
- **Axes**: y = dataset label; x = calibrated margin (0 to ~0.7).
- **Filename**: `F3_dataset_ranking.pdf` / `.png`.

## Supplementary (optional, same script, flag-gated)

Flag `--supplementary` also renders:

- **F1-digress**: identical to F1 but with `noise_type='digress'`. Filename `FS1_ratio_vs_epsilon_digress.pdf`.
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
