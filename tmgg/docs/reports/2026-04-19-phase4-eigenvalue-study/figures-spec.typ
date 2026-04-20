#set document(title: "Phase 4 — Figures Spec + Generated Figures")
#set page(paper: "a4", margin: 1in)
#set text(font: "DejaVu Serif", size: 10pt)
#set par(justify: true, leading: 0.6em)
#show heading.where(level: 1): set text(size: 14pt, weight: "bold")
#show heading.where(level: 2): set text(size: 12pt, weight: "bold")
#show heading.where(level: 3): set text(size: 11pt, weight: "bold")
#show link: set text(fill: blue)
#show raw.where(block: false): set text(font: "DejaVu Sans Mono", size: 9pt)

#align(center)[
  #text(size: 18pt, weight: "bold")[Phase 4 — Paper Figures Spec]
  #v(0.3em)
  #text(size: 9pt, fill: gray)[
    #datetime.today().display() · `docs/reports/2026-04-19-phase4-eigenvalue-study/figures-spec.typ`
  ]
]

#outline(depth: 2, indent: auto)
#pagebreak()

Target paper: `_NeurIPS_2026__Understanding_Graph_Denoising.pdf`. Three figures (F1–F3) summarise the Phase 4 improvement-gap surrogate sweep. All figures read the same CSV and share one style module so the house look stays consistent.

= Goal

For a given graph dataset and noise level $epsilon$, quantify the extent to which the noisy top-$k$ eigenvalues $tilde(Lambda)_k$ determine the projection of the clean adjacency $B = hat(V)_k^T A hat(V)_k$ into the noisy top-$k$ eigenbasis. Equation (18) of the target draft defines the improvement gap

$ ell_"lin" - ell_f = EE norm(EE[B | tilde(Lambda)_k] - EE[B])_F^2, $

i.e. the between-graph variance of the Bayes-optimal predictor of $B$ given $tilde(Lambda)_k$. The figures report a bias-corrected finite-sample estimate of this population quantity, normalised by $"tr Cov"(B)$, so that a reader can (i) read the fraction of $B$'s variance that $tilde(Lambda)_k$ determines on each dataset, (ii) compare datasets on a common $[0, 1]$ scale, and (iii) verify that the estimate rises with known community strength (synthetic SBM diversity) and reflects the community content of real benchmarks. Making this quantity legible, bias-corrected, and reproducible across $k$, $epsilon$, noise model, and frame convention is the purpose of the figure set.

= Shared conventions

- *Data source*: `docs/reports/2026-04-19-phase4-eigenvalue-study/phase4_sweep.csv` (25 600 rows).
- *Headline cell* (unless stated otherwise): `frame_mode='frechet'`, `estimator_label='knn_top_k'`, `noise_type='gaussian'`. This is the cell whose margins the README headline table reports.
- *Script*: `scripts/plot_phase4_figures.py`, `uv`-style `# /// script` header with `matplotlib>=3.8`, `pandas>=2.0`, `numpy>=1.24`. No side effects outside the figures directory.
- *Output dir*: `docs/reports/2026-04-19-phase4-eigenvalue-study/figures/`. Emit *both* `.pdf` (for LaTeX inclusion) and `.png` at 300 dpi (for quick review and Slack).
- *Sizes*: NeurIPS single-column 3.25", double-column 6.75". F1 and F3 are double-column; F2 is single-column.
- *Style*: `plt.rcParams['font.family'] = 'serif'`, `font.size=8`, `axes.labelsize=8`, `legend.fontsize=7`. No title on the figure itself — caption lives in the `.tex`. Panel titles OK when there are multiple subplots. Tight layout; `bbox_inches='tight'` on save.
- *Error semantics*: error bars and ribbons are $plus.minus 1$ SD across the five seeds ${42, 123, 2024, 7, 11}$. Points are seed means.
- *Diversity knob* (definition). `diversity` $in [0, 1]$ scales the per-graph sampling-range width of each tuple-ranged SBM hyperparameter toward its midpoint: for a range $(l, h)$ with midpoint $mu = (l+h)\/2$ and half-width $w = (h-l)\/2$, the effective sampling range is $(mu - w d, mu + w d)$. At $d = 0$ every graph is drawn from $mu$ (homogeneous batch); at $d = 1$ the full range is used (maximally heterogeneous). Phase 4 fixes $n = 50$ nodes, $4$ blocks, and uses $p_"intra" in (0.3, 0.9)$, $p_"inter" in (0.01, 0.2)$. See F0 below for visual examples.
- *Palette*: synthetic SBMs use a viridis gradient over diversity ${0.0, 0.33, 0.67, 1.0}$ so monotonicity reads at a glance. Real datasets use tab10 warm colours (`spectre_sbm` red, `enzymes` orange, `proteins` green, `collab` purple). Null ribbon is neutral grey `#b0b0b0` at alpha 0.35. Pass / fail bar colouring in F3: pass `#3b7dd8`, fail `#b0b0b0`.

= Metric definition — fraction of variance explained (FVE)

Let $B_i = hat(V)_(k,i)^T A_i hat(V)_(k,i) in RR^(k times k)$ be the clean adjacency of graph $i$ re-expressed in the noisy top-$k$ eigenbasis, and let $tilde(Lambda)_(k,i) in RR^k$ be the corresponding noisy eigenvalues (both in the dataset-wide Fréchet frame; see `figures.md` for why). The metric on all figure y-axes derives from

$ "FVE"(k) = ("Var"(EE[B_i | tilde(Lambda)_(k,i)])) / "Var"(B_i) in [0,1]. $

Both variances are Frobenius-sum variances taken across the graph population. By the law of total variance this is exactly the *coefficient of determination (R²) of the Bayes-optimal predictor* of $B$ from $tilde(Lambda)_k$: it measures the fraction of $B$'s Frobenius variance that *any* function of $tilde(Lambda)_k$ could predict. $"FVE" = 0$ means $tilde(Lambda)_k$ carries no information about $B$; $"FVE" = 1$ means $tilde(Lambda)_k$ determines $B$ entirely.

The *calibrated FVE margin* $"FVE"_"real" - "FVE"_"null"$ is the quantity F2 and F3 plot directly, where $"FVE"_"null"$ is FVE recomputed with $tilde(Lambda)_k$ permuted across graphs (so feature–target pairing is destroyed while marginals are preserved).

== Why FVE rather than alternatives

#table(
  columns: (0.28fr, 0.72fr),
  stroke: 0.4pt,
  align: (left + top, left + top),
  inset: 6pt,
  [*Alternative*], [*Why we didn't pick it*],
  [*Raw $hat(g)_k$* (unnormalised numerator)], [Has units of Frobenius² and scales mechanically with $"trace Cov"(B)$, which itself grows with edge density and $n$. Cross-dataset and cross-$k$ comparisons become meaningless. FVE normalises this away.],
  [*Linear R²* (OLS regression of $"vec"(B)$ on $tilde(Lambda)_k$)], [Assumes a linear relationship; the improvement-gap theory (eq. 18) is explicitly about the *non-linear* Bayes gap $ell_"lin" - ell_f$. Using linear R² would collapse the quantity we care about to zero by construction.],
  [*Mutual information* $I(B ; tilde(Lambda)_k)$], [Correct in principle but very hard to estimate for matrix-valued $B$ at our sample sizes ($N approx 68$–$3700$). Known estimators (kNN-MI of Kraskov, KSG) have larger and less-predictable finite-sample bias than the conditional-mean estimator, and no natural $[0,1]$ scale.],
  [*HSIC / dCor* (kernel / distance dependence measures)], [Detect arbitrary dependence, but lose the variance-share interpretation. A HSIC value of 0.3 does not translate to "30% of the variance in $B$ is recoverable from $tilde(Lambda)_k$" — it just says the two are statistically dependent. Reviewers will ask "how much is that", and we would have no answer.],
  [*MSE of a fitted regressor* (train/test MLP, GP, kernel ridge)], [Depends on model class choice and hyperparameter tuning. Each choice raises a "did you pick the best class?" objection. The Bayes-optimal benchmark underlying FVE sidesteps this entirely.],
  [*Spearman / Pearson* on scalar summaries (e.g. spectral gap vs. $"trace"(B)$)], [Too coarse: $B$ is a $k times k$ matrix, and projecting it to a scalar before computing a rank correlation discards the information the paper's surrogate explicitly keeps.],
)

= Estimator definition — kNN conditional mean

We estimate $EE[B_i | tilde(Lambda)_(k,i)]$ by nearest-neighbour averaging in $tilde(Lambda)_k$-space:

+ Compute the Euclidean pairwise distance matrix $D in RR^(N times N)$ with $D_(i j) = norm(tilde(Lambda)_(k,i) - tilde(Lambda)_(k,j))_2$.
+ For each graph $i$, let $cal(N)_i$ be the indices of the *$m = 10$* nearest neighbours (excluding $i$ itself — the self-entry sits at distance 0).
+ Form the kNN conditional-mean estimate $hat(B)_i = (1/m) sum_(j in cal(N)_i) B_j$.
+ Compute $hat(g)_k = (1/N) sum_i norm(hat(B)_i - mu_B)_F^2$ where $mu_B = (1/N) sum_i B_i$.
+ Report $"FVE" = hat(g)_k \/ (1/N) sum_i norm(B_i - mu_B)_F^2$.

The CSV's `estimator_label='knn_top_k'` column corresponds to this choice with $tilde(Lambda)_k$ as the full top-$k$ eigenvalue vector. The other labels (`knn_1d`, `bin_1d`, `invariants_knn`) are sensitivity variants and do not feed the main paper figures.

== Why kNN with $m=10$

- *Nonparametric.* No distributional assumption on $(B, tilde(Lambda)_k)$. We don't know, a priori, the functional form of $EE[B | tilde(Lambda)_k]$ — the whole premise of the paper is that the relationship is nonlinear in a specific way — so a nonparametric regressor is the right default.
- *No bandwidth selection.* $m$ is the only hyperparameter. Kernel regression would require choosing a kernel and a bandwidth (Silverman's rule, cross-validation, Lepski); each choice is another reviewer hook. Changing $m$ in ${5, 10, 20}$ does not change the qualitative picture in our sensitivity runs.
- *Predictable finite-sample bias.* The average over $m$ independent neighbours introduces a known $approx 1\/m$ residual variance under H₀ (features independent of targets), which appears as a flat $approx 0.10$ floor in the permutation null. This floor is what the calibrated margin subtracts out. No other estimator we considered has such a cleanly-measurable bias; MLPs and kernel regressors mix bias and variance in harder-to-isolate ways.
- *Consistent.* $hat(g)_k -> g_k$ as $N -> infinity$, $m -> infinity$, $m\/N -> 0$ (standard kNN-regression consistency). At our sample sizes ($N gt.eq 68$) $m=10$ sits comfortably in the safe regime.
- *Computationally trivial.* One $N times N$ distance matrix plus a single stack-and-average per graph. Every cell in the 25 600-row sweep runs in milliseconds.

== Calibrated margin as a lower bound on the Bayes FVE

Decompose the kNN predictor as $hat(m)(lambda) = m^*(lambda) + "bias"(lambda) + xi(lambda)$, where $m^*(lambda) = EE[B | tilde(Lambda)_k = lambda]$ is the Bayes-optimal predictor, $"bias"(lambda) = EE[hat(m)(lambda)] - m^*(lambda)$ is the smoothing error from averaging over $m$ neighbours whose $tilde(Lambda)$ is not exactly $lambda$, and $xi(lambda)$ is the residual noise of the neighbour average around $EE[hat(m)(lambda)]$. Then

$ "Var"(hat(m)(tilde(Lambda)_k)) approx underbrace("Var"(m^*(tilde(Lambda)_k)), "true Bayes signal") + underbrace(O(h^2), "smoothing attenuation") + underbrace(sigma^2 \/ m, "neighbour-averaging inflation"), $

with effective bandwidth $h tilde (m\/N)^(1\/k)$ for $k$-dimensional $tilde(Lambda)_k$. This is the standard kNN-regression bias-variance decomposition under Lipschitz $m^*$: pointwise bias is $O(h)$ so $"bias"^2 = O(h^2) = O((m\/N)^(2\/k))$, and neighbour-averaging variance is $O(sigma^2 \/ m)$ (Györfi, Kohler, Krzyżak & Walk 2002, ch. 6 on kNN estimates; Biau & Devroye 2015, Part II ch. 14 "Rates of Convergence"; Ayano 2012, which further shows that for $(p,C)$-smooth $m^*$ plain kNN achieves the minimax rate only up to $p = 3/2$, so stronger-smoothness regimes — e.g. twice-differentiable $m^*$ giving the faster $O(h^4)$ bias² — are unreachable without local-linear corrections we do not apply).

The inflation term $sigma^2 \/ m$ is approximately invariant under permutation (same $m$, $N$, and marginal noise), so subtracting $"FVE"_"null"$ removes it cleanly. The attenuation term, however, is absent from the null: under H₀ the true $m^*$ is constant and there is nothing to over-smooth. It therefore survives the subtraction as a *negative* contribution to $"FVE"_"real" - "FVE"_"null"$. The calibrated margin we report is, in consequence, a *conservative lower bound* on the true Bayes FVE — the population quantity from eq. (18):

$ "FVE"_"real" - "FVE"_"null" lt.eq "FVE"_"Bayes" + O((m\/N)^(2\/k)). $

Numerically the attenuation bound $(m\/N)^(2\/k)$ at $m = 10$ and our sample sizes is:

#align(center)[
  #table(
    columns: (auto, auto, auto),
    stroke: 0.4pt,
    inset: 6pt,
    align: (center, center, center),
    [*k*], [*N = 200*], [*N = 3700*],
    [4], [0.22], [0.05],
    [8], [0.47], [0.23],
    [16], [0.69], [0.48],
    [32], [0.83], [0.69],
  )
]

At $k=4$ with large $N$ the bound is tight. At $k=16$ and $k=32$ it becomes loose, and our calibrated margin increasingly understates the true Bayes R². Two practical consequences: (i) a _positive_ calibrated margin at high $k$ is still evidence for real signal, because the estimator errs conservatively; (ii) a _shrinking_ margin as $k$ grows in F1 / F2 cannot be attributed solely to loss of signal — part of it is estimator conservatism. We flag this in the figure captions rather than correcting it in post, because the bias constant depends on unknown smoothness properties of $m^*$ and on the intrinsic (rather than ambient) dimension of $tilde(Lambda)_k$, which for eigenvalues of structured adjacency matrices is often smaller than $k$.

=== How to read the figures

Two interpretive handles the paper will lean on:

+ *A nonzero gap between the real curve and the null ribbon implies improvability, period.* The gap is a conservative lower bound on the population improvement gap $ell_"lin" - ell_f$, so positivity of the gap — once it exceeds seed-level noise, which is what the 0.10 pass threshold operationalises — guarantees that some non-constant function of $tilde(Lambda)_k$ genuinely beats the marginal mean as a predictor of $B$. The second clause of the pass criterion ($"FVE"_"null" < 0.30$) guards against small-$N$ regimes where bias inflation leaves no headroom.
+ *The decay of the curve as $epsilon$ grows measures noise sensitivity of the exploitable signal.* At $epsilon = 0$ the noisy and clean eigenframes coincide, $B$ is deterministic in $Lambda_k$, and FVE saturates near 1 (modulo smoothing bias). As $epsilon$ increases, $tilde(Lambda)_k$ drifts from $Lambda_k$, $EE[B | tilde(Lambda)_k]$ spreads, and FVE falls. The shape of the decay is the empirical answer to "how robust is the dataset's spectral signal to additive noise?" — slow decay marks resilient, low-frequency structure; fast decay marks fragile, high-frequency structure that noise washes out first.

=== References (validated)

- Györfi L., Kohler M., Krzyżak A. & Walk H. (2002). _A Distribution-Free Theory of Nonparametric Regression._ Springer. DOI #link("https://link.springer.com/book/10.1007/b97848")[10.1007/b97848]. Chapter 6 gives the classical Lipschitz rate $O((k\/n)^(2\/d)) + O(1\/k)$ for the kNN regression estimate.
- Biau G. & Devroye L. (2015). _Lectures on the Nearest Neighbor Method._ Springer Series in the Data Sciences. DOI #link("https://link.springer.com/book/10.1007/978-3-319-25388-6")[10.1007/978-3-319-25388-6]. Part II ch. 14 ("Rates of Convergence") reproves and extends the kNN regression rates; ch. 8 introduces the estimator.
- Ayano T. (2012). _Rates of convergence for the k-nearest neighbor estimators with smoother regression functions._ Statistics & Probability Letters. #link("https://www.sciencedirect.com/science/article/abs/pii/S0378375812001280")[ScienceDirect]. Shows plain kNN attains $n^(-2p\/(2p+d))$ only for $p in (0, 3/2]$; higher-order smoothness requires bias-reducing variants.

== The permutation null as a bias diagnostic, not a significance test

Shuffling $tilde(Lambda)_k$ across graphs preserves the marginal distribution of the features but breaks the pairing with $B$, so under the null the true conditional mean is constant: $EE[B | tilde(Lambda)_k^"perm"] equiv mu_B$. Any non-zero $"FVE"_"null"$ is therefore estimator bias, not genuine structure. We report it and subtract it; we do *not* report a p-value. This matches the conceptual framing in the paper — the surrogate is an effect-size measurement of the Bayes gap, not a hypothesis test.

== Pass criterion

A dataset cell passes when $"FVE"_"real" - "FVE"_"null" gt.eq 0.10$ *and* $"FVE"_"null" < 0.30$. The first clause is the signal requirement; the second clause guards against small-$N$ regimes where the kNN bias inflates and leaves little headroom between the null and the real curve. The two thresholds were fixed before Phase 4 data collection and are the same thresholds the Phase 3 diversity-knob validation used.

= F1 — FVE vs $epsilon$ per dataset, Gaussian vs DiGress noise side-by-side

*Question*: does the FVE stay above the permutation null across $epsilon$, does it behave similarly across $k$, and does the picture change under the structured (DiGress) noise the paper's generative models actually use?

- *Layout*: 4 rows $times$ 4 cols grid, 16 panels, shared y-axis per row and shared x-axis per column. Landscape, $approx 6.75" times 6.0"$.
- *Row layout*: rows 1–2 are `noise_type='gaussian'` (top half); rows 3–4 are `noise_type='digress'` (bottom half). A thin horizontal rule between row 2 and row 3 plus a left-margin row-group label ("Gaussian" / "DiGress") makes the split visually explicit.
- *Column layout*: same dataset ordering in every row, one dataset per column — row-major within each half, top-left to bottom-right: `sbm_d0.00, sbm_d0.33, sbm_d0.67, sbm_d1.00` on the upper row of each half; `spectre_sbm, enzymes, proteins, collab` on the lower row of each half. Holding the column order constant across the Gaussian and DiGress halves makes vertical scanning within a dataset the natural comparison.
- *Per panel*: x-axis is `noise_level` $in {0.01, 0.05, 0.1, 0.15, 0.2}$ linear. y-axis is FVE on range $[0, 0.8]$ (shared across all 16 panels). Foreground: four lines for $k in {4, 8, 16, 32}$ over `permuted=False`, coloured by $k$ with a sequential `plasma` cmap, markers + line + $plus.minus 1$ SD error bars across seeds. Background ribbon: for each $k$, `fill_between` $plus.minus 1$ SD of the `permuted=True` curve in the same $k$-colour at alpha 0.15 — keeps the $k$-specific null visible rather than collapsing to a single band. Dashed horizontal line at $y = 0.10$ (pass threshold). Panel subtitle = dataset name + retained $N$, e.g. `sbm_d0.67 (N=200)`; noise type is indicated by the row-group label, not repeated per panel.
- *Legend*: single legend bottom-centre with $k = 4/8/16/32$ swatches plus a grey ribbon swatch labelled "permutation null ($plus.minus 1$ SD)".
- *Filename*: `F1_fve_vs_epsilon.pdf` / `.png`.

= F2 — calibrated FVE margin vs diversity (synthetic SBM only)

*Question*: does the FVE track community strength monotonically, and does the effect size grow with $k$?

- *Layout*: single panel, single-column, $approx 3.25" times 2.5"$.
- *Data*: subset to `dataset` $in$ `{sbm_d0.00, sbm_d0.33, sbm_d0.67, sbm_d1.00}`, `noise_level=0.1`, headline cell (Fréchet, `knn_top_k`, Gaussian).
- *x-axis*: diversity $in {0.0, 0.33, 0.67, 1.0}$ on a linear scale, ticks at the four grid points.
- *y-axis*: calibrated FVE margin $= "FVE"_"real" - "FVE"_"null"$ per seed, averaged across seeds (pairing at the seed level so error bars propagate seed-to-seed).
- *Lines*: one line per $k in {4, 8, 16, 32}$, coloured with the same plasma cmap as F1. Markers at each diversity point; error bars = SD of the per-seed calibrated margin.
- *Dashed horizontal* at $y = 0.10$: pass threshold.
- *Legend*: upper-left inside the axes, 4 entries.
- *Filename*: `F2_calibrated_margin_vs_diversity.pdf` / `.png`.

= F3 — dataset ranking at the headline cell

*Question*: which datasets carry a community signal the surrogate recovers, ordered by margin?

- *Layout*: single panel, double-column width (for readability), $approx 6.75" times 2.75"$.
- *Data*: subset to headline cell + $k=8$, `noise_level=0.1`, `noise_type='gaussian'`, `frame_mode='frechet'`, `estimator_label='knn_top_k'`. One row per dataset (seed-averaged), sorted by calibrated margin descending.
- *Chart*: horizontal bar chart. Bar length = calibrated FVE margin ($"FVE"_"real" - "FVE"_"null"$). Bar colour is blue if margin $gt.eq 0.10$ *and* $"FVE"_"null" < 0.30$ (= pass per the README criterion), else grey. Black horizontal error bar on each bar: $sqrt("Var"("FVE"_"real")\/5 + "Var"("FVE"_"null")\/5)$ across seeds (standard-error of the difference of two means). Annotate each bar on the right with the numeric margin to two decimals. Vertical dashed line at $x = 0.10$ (pass threshold).
- *Axes*: y = dataset label; x = calibrated FVE margin ($0$ to $approx 0.7$).
- *Filename*: `F3_dataset_ranking.pdf` / `.png`.

= Supplementary (rendered by default)

The same script also emits, without requiring any flag:

- *FS1 — per-graph frame ranking*: identical to F3 but with `frame_mode='per_graph'`. Demonstrates the Fréchet-frame contribution to the signal. Filename `FS1_dataset_ranking_per_graph.pdf` / `.png`.

The main paper cites F1–F3; FS1 goes to the appendix. (The former supplementary "F1-digress" is no longer needed: DiGress now appears as the bottom half of F1.)

= Implementation notes

- Load CSV once, filter per figure. No reliance on row ordering — always group by the full `(dataset, noise_type, noise_level, frame_mode, estimator_label, permuted, k)` key before aggregating across seeds.
- Seed count assertion: for every cell used in the figures, assert `len(group) == 5`; fail loudly if not.
- Save each figure twice (PDF, PNG). The PDF vector output is what LaTeX will `\includegraphics`.
- No `plt.show()`. No interactive state leaking between figures — call `plt.close(fig)` after saving.
- The script prints a one-line manifest per figure: `wrote F1 (panels=16, series=4, points=320) → figures/F1_fve_vs_epsilon.{pdf,png}`.

= Resolved design decisions

Captured here so the rationale stays with the spec even after the open-questions block is removed.

+ *F1 subplot grid*: 4 rows $times$ 4 cols (was 2$times$4 Gaussian-only). Gaussian on top half, DiGress on bottom, double-column width.
+ *Pass-threshold line on F1*: absolute $y = 0.10$, matching the README headline pass criterion. Not per-panel null-shifted.
+ *F2 k-lines*: all four $k in {4, 8, 16, 32}$ rendered.
+ *Supplementary figures*: rendered by default — no `--supplementary` flag needed.
+ *DiGress in F1*: promoted to the bottom half of F1 (side-by-side with Gaussian) rather than relegated to supplementary. Former "F1-digress" supplementary is subsumed.

#pagebreak()

= Generated figures

== F0 — SBM-diversity knob, illustrative graphs

Four diversity levels $d in {0.0, 0.33, 0.67, 1.0}$ (rows) $times$ four independent samples (columns), drawn under the Phase 4 parametrisation: $n = 50$ nodes, $4$ blocks, $p_"intra" in (0.3, 0.9)$, $p_"inter" in (0.01, 0.2)$. At $d = 0$ every graph is identically distributed (midpoint of each range); at $d = 1$ each graph samples its own $(p_"intra", p_"inter")$ independently from the full range. Per-graph edge counts annotated above each tile.

#align(center)[
  #image("figures/F0_sbm_diversity_examples.png", width: 90%)
]

#pagebreak()

== F1 — FVE vs $epsilon$, Gaussian (top) and DiGress (bottom)

#align(center)[
  #image("figures/F1_fve_vs_epsilon.png", width: 95%)
]

#pagebreak()

== F2 — calibrated margin vs SBM diversity

#align(center)[
  #image("figures/F2_calibrated_margin_vs_diversity.png", width: 60%)
]

== F3 — dataset ranking (Fréchet frame, headline cell)

#align(center)[
  #image("figures/F3_dataset_ranking.png", width: 95%)
]

#pagebreak()

== FS1 — dataset ranking (per-graph frame)

#align(center)[
  #image("figures/FS1_dataset_ranking_per_graph.png", width: 95%)
]
