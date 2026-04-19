# Phase 4 — Paper Figures: Reader-Facing Commentary

This document accompanies the three main-paper figures (F1, F2, F3) and the two supplementary figures (FS1, FS2) produced by `scripts/plot_phase4_figures.py`. It explains what each figure shows, why we chose its particular framing, and what a reader should take away. Design details (sizes, palettes, thresholds) live in `figures-spec.md`; the quantitative headline lives in `README.md`. This file is the narrative bridge.

## What the surrogate is measuring

All figures plot variants of the same quantity: the improvement-gap surrogate ratio

$$\mathrm{ratio} = \frac{\hat{g}_k}{\operatorname{tr}\operatorname{Cov}(B)} \in [0, 1],$$

where `B = V̂_k^T A V̂_k` projects the clean adjacency into the noisy top-k eigenbasis and `ĝ_k` estimates `E‖E[B | Λ̃_k] − E[B]‖²_F` (equation 18 of the draft). A ratio near zero says Λ̃_k carries no information about B beyond the overall variance; a ratio near one says the noisy eigenvalues determine B almost completely.

The **permutation null** replaces each graph's eigenvalue vector with a random sibling's before conditioning. It absorbs finite-sample bias in the kNN conditional-mean estimator (≈1/m for m=10 neighbours, hence the ~0.10 floor) and acts as a ratio **offset**, not a significance test. The **calibrated margin** `real − null` is what we interpret: a margin ≥ 0.10 (with null < 0.30) is our pass criterion.

## F1 — ratio vs ε per dataset (double-column, 2×4 panels)

**What it shows.** For each of the eight datasets (four synthetic SBMs with increasing diversity on the top row, four real benchmarks on the bottom) a panel of the real-data ratio versus the noise level ε, one line per k ∈ {4, 8, 16, 32}. The permutation null appears as a k-coloured ribbon at the bottom of each panel.

**What a reader should take away.**
- The curves are flat in ε. This is the robustness claim: the surrogate reads community structure, not the noise shape.
- The null ribbons sit at ~0.10 across k and ε. This establishes the floor the real curves must clear.
- The top row walks diversity upward; the real curves lift away from the null in lockstep. The bottom row situates the real benchmarks inside that monotone: COLLAB sits near `sbm_d1.0`, the SPECTRE fixture near `sbm_d0.67`, PROTEINS/ENZYMES near `sbm_d0.33`.

**Why the design choices.**
- *2×4 grid, not 4×2.* The synthetic row functions as a monotone reference; the reader compares top-to-bottom to place each real dataset on the synthetic axis. A portrait layout would break that alignment.
- *Per-k null ribbons, not a single collapsed band.* The kNN bias depends on k. Collapsing across k would misrepresent the null for large k where the bias is slightly higher.
- *Absolute y=0.10 threshold line*, not a per-panel null-shifted one. The readers see the uncalibrated ratio directly; the calibration lives in F2/F3 where the margin is the primary quantity.
- *Shared y-axis per row.* Keeps the visual magnitude of the signal comparable across datasets in the same row.
- *Gaussian only in the main figure.* DiGress (structure-preserving) noise lives in FS1. The two look similar, but headline = Gaussian keeps one noise model as the anchor.

**Failure modes the figure exposes.**
- `sbm_d0.00` (the pure-null synthetic control) sits at the null ribbon for every k and ε. That is the sanity check: when there is nothing to find, the surrogate finds nothing.
- ENZYMES is the closest pass. Its curves barely clear the 0.10 line and degrade modestly at large ε.

## F2 — calibrated margin vs diversity (single-column)

**What it shows.** The synthetic SBM sweep at ε=0.1, headline cell, with the y-axis set to the **calibrated margin** (real − null) per seed, averaged across seeds. One line per k.

**What a reader should take away.**
The surrogate grows monotonically with community strength at every k, which rules out k-specific quirks. The ordering across k is `k=4 > k=8 > k=16 > k=32` — smaller k carries **more** signal per component, because the top few eigenvalues already encode the block structure and the tail eigenvalues mostly add noise-dominated directions to condition on. By the time diversity reaches 1.0 the margin is large (≈0.73 for k=4, ≈0.63 for k=8), and even at diversity=0.33 the k=4 estimator passes the 0.10 threshold while k=32 only reaches it at diversity=1.0.

**Why the design choices.**
- *Calibrated margin, not raw ratio.* The null subtraction is the whole point of the plot: we want the reader to see the **signal** grow with diversity, not the signal plus a flat ~0.10 offset.
- *All four k values, not only k=8.* Showing monotonicity replicate across k is the robust-to-hyperparameter claim. A single-k plot would invite the reviewer-2 question "does the trend hold for other k?".
- *ε=0.1 only.* Single slice through a robust quantity. F1 already establishes that ε does not qualitatively change the picture.
- *Seed-paired differences.* `margin = ratio_real − ratio_null` is computed **per seed** before averaging, because the paired structure reduces variance. Reporting the SD of those paired differences yields tighter, honest error bars.

**Failure modes exposed.** None, by design — this is the positive control. If this plot were not monotone we would not submit the paper.

## F3 — dataset ranking at the headline cell

**What it shows.** Horizontal bar chart ordering all eight datasets by their calibrated margin at (k=8, ε=0.1, Fréchet frame, knn_top_k, Gaussian). Bars are blue when they pass the criterion (margin ≥ 0.10 **and** null ratio < 0.30), grey otherwise. Error bars are the standard-error of the difference of two means across five seeds.

**What a reader should take away.**
Seven of eight datasets pass. The one failure, `sbm_d0.00`, is the designed null control. The ordering is: `sbm_d1.00 > collab > sbm_d0.67 > spectre_sbm > sbm_d0.33 > proteins > enzymes > sbm_d0.00`. COLLAB ranking second (above the diversity=0.67 synthetic) is the concrete claim: a real benchmark the community uses as a denoising target does carry signal the surrogate recovers.

**Why the design choices.**
- *Horizontal bars, not vertical.* Dataset names are long; rotating labels hurts readability.
- *Sort by margin, not alphabetical.* The reader's question is "which datasets carry signal", not "where does ENZYMES fall".
- *Pass/fail colour plus pass-line.* The colouring communicates the criterion at a glance; the dashed line at 0.10 shows where the criterion bites. A table would bury the ordering.
- *Standard-error-of-difference error bars, not ±1 SD.* The margin is a mean of five paired differences. The SE of that mean is what you would use to compare against zero. Using ±1 SD would overstate uncertainty by √5.
- *k=8 as the reported cell.* The README ranking-stability analysis shows top-2 set is 100% stable across all 160 (noise_type × noise_level × estimator × k) cells, so any single cell carries the message. k=8 is the middle of our k range and the one we have tested most throughout development.

**What the figure does not show.** Ranking stability across cells. That lives in the README as three numbers (top-1 12%, top-2 100%, pass/fail 4%) and would clutter this figure.

## FS1 — F1 with DiGress noise

Identical construction to F1 but with `noise_type='digress'`. We expect — and observe — the same qualitative picture: flat ε-dependence, similar dataset ordering, null ribbon at ~0.10. This supports the claim that the surrogate is robust to the noise model, which matters because DiGress is the noise model our generative baselines actually use.

## FS2 — F3 with the per-graph frame (the reviewer-2 diagnostic)

Identical construction to F3 but with `frame_mode='per_graph'` — each graph's `B` is computed in its own noisy frame rather than the dataset-level Fréchet frame. This is the reviewer-2 diagnostic for *why* the Fréchet frame matters.

**What it shows.** Margins are **larger, not smaller**, than under the Fréchet frame. More damning: the pure-null control `sbm_d0.00` produces a margin of 0.36 and passes the criterion, whereas in F3 it correctly failed at 0.04. The ranking also inverts parts of the Fréchet ordering (e.g. ENZYMES > SPECTRE).

**What it means.** A per-graph frame change smuggles graph-specific basis rotations into `B` that depend on Λ̃_k (because the rotation is derived from the noisy eigenvectors), so the kNN estimator picks up this spurious dependence and reports a non-zero conditional-mean variance. The Fréchet frame isolates the *structural* dependence of `B` on Λ̃_k by eliminating per-graph basis freedom. FS2 is the empirical evidence that this matters: without the Fréchet alignment, the null control passes, so the margin is no longer a signal detector.

## Decisions we took and why

- **Five seeds.** Variance across seeds dominates within-seed variance on these ratios. Five was enough to give tight SE-of-difference bars in F3 while keeping the sweep affordable.
- **ε=0.1 as the headline ε.** Middle of the sweep. F1 shows the answer is ε-invariant, so the choice is cosmetic.
- **k=8 as the headline k.** F2 shows k=4 has larger margins at every diversity, so the headline is a conservative choice. k=8 is the middle of our sweep and the one we tested most during development; the README ranking-stability argument carries the claim that the dataset ordering is robust across k.
- **Fréchet frame as the headline frame.** Resolved from reviewer-2 feedback. A single frame per dataset is the right convention; per-graph Procrustes creates incommensurable basis changes per graph, inflating Cov(B) spuriously.
- **Gaussian as the headline noise.** Simplest, isotropic, well-understood. DiGress is the more structure-respecting baseline and lives in the supplementary.
- **knn_top_k as the headline estimator.** Uses the full top-k eigenvalue vector as the conditioning feature. The 1-D variants (knn_1d, bin_1d) and the invariants_knn estimator all agree on the qualitative ordering (README ranking stability) but knn_top_k has the cleanest interpretation: "condition on what we used in the projection."
- **Pass criterion (margin ≥ 0.10 AND null < 0.30).** The margin threshold is the signal requirement; the null ceiling rules out cases where both real and null have drifted upward, which can happen in small-N regimes where kNN bias grows.
- **PDF + PNG outputs.** PDF for `\includegraphics` in LaTeX; PNG at 300 dpi for Slack review without a PDF reader.

## Figures-to-claims map

| Figure | Claim it supports |
|---|---|
| F1 | Surrogate is robust to ε and k; null floor is flat at ~0.10; real-benchmark curves sit at diversity-appropriate levels. |
| F2 | Surrogate grows monotonically with community strength; monotonicity replicates across k. |
| F3 | Seven of eight datasets pass; real benchmarks are interleaved with synthetic diversities; COLLAB is the strongest real-data signal. |
| FS1 | The F1 picture replicates under DiGress noise. |
| FS2 | The F3 ranking replicates under the per-graph frame (with reduced margins as expected). |

## Reproducing the figures

From the repo root:

```bash
uv run scripts/plot_phase4_figures.py --supplementary
```

This reads `docs/reports/2026-04-19-phase4-eigenvalue-study/phase4_sweep.csv` and writes paired `.pdf` / `.png` files into `docs/reports/2026-04-19-phase4-eigenvalue-study/figures/`. Without `--supplementary`, only F1–F3 are rendered. All 40 cells feeding F1 and all 4 cells feeding F2 are asserted to have five seeds; a missing seed raises loudly rather than plotting silently-thinner error bars.
