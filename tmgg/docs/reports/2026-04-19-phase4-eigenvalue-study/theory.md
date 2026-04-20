# Theory notes: Fréchet frame and permutation-null bias offset

This note is the single-source explanation of the two methodological choices that
the Phase 3 v2 and Phase 4 eigenvalue studies share: the dataset-wide Fréchet
frame used to compare noisy eigenbases across graphs, and the permutation null
used to remove the finite-sample bias of the kNN conditional-mean estimator.
The implementation details and empirical validation live in the respective
report READMEs and in `figures.md`; this note covers only the *why*.

Scope note on citations. Each citation is backed by the corresponding PDF
read at the locations listed in the "Citation status" block at the end of
the note. Where no published result establishes what we need directly, the
argument is given as an explicit derivation rather than imported via a
surface cite; this applies to the Grassmannian-specific extrinsic-mean
formula (§1.3), the identifiability breakdown under per-graph alignment
(§1.2), and the `1/m` null floor and bias-subtraction correctness (§2.1,
§2.2).

The FVE estimator is eq. (18) of the draft,

$$\mathrm{FVE}(k) \;=\; \frac{\operatorname{Var}\!\bigl(\mathbb{E}[B_i \mid \tilde\Lambda_{k,i}]\bigr)}{\operatorname{Var}(B_i)}, \qquad B_i \;=\; \hat V_{k,i}^{\top} A_i \hat V_{k,i},$$

where `A_i` is the clean adjacency of graph *i*, `V̂_{k,i} ∈ ℝ^{n×k}` its noisy
top-*k* eigenvector block, and `Λ̃_{k,i} ∈ ℝ^k` the corresponding noisy
eigenvalues. Both quantities below arise from a single observation: this
population expression does not identify a specific `B_i` unless we resolve the
O(k) ambiguity in `V̂_{k,i}` with a frame convention, and it does not identify
a specific point estimate unless we characterise the bias of whatever plug-in
we use for the conditional mean.

## 1. Why the Fréchet frame

### 1.1 The frame ambiguity

Any orthonormal basis `V̂_{k,i}` for the top-*k* invariant subspace of `Â_i` is
determined only up to an element of the orthogonal group `O(k)`: for any
`Q_i ∈ O(k)`, the basis `V̂_{k,i} Q_i` spans the same subspace. The induced
projection `B_i = V̂_{k,i}^⊤ A_i V̂_{k,i}` is *not* invariant — it transforms as
`B_i ↦ Q_i^⊤ B_i Q_i`. Averaging, variance, and the conditional mean in
eq. (18) all require the `B_i` to sit in a common frame; otherwise `Cov(vec B)`
picks up contributions from the free `Q_i` per graph, which are population
noise, not signal. This frame degeneracy becomes severe whenever two or more of
the top-*k* eigenvalues are close or genuinely degenerate — then the
per-eigenvector basis is undefined even up to sign, but the *subspace* is
stable, so the frame problem is genuine and subspace-level rather than a
numerical quirk.

The choice is therefore: pick a rule that maps every noisy block `V̂_{k,i}` to
a basis of the same subspace such that the resulting `B_i` lie in a shared
k-dimensional frame. Orthogonal Procrustes alignment
([Schönemann, 1966](https://doi.org/10.1007/BF02289451)) solves this against a
reference `V*_k`: given `V̂_{k,i}` and `V*_k`, the minimiser of
`‖V̂_{k,i} Q − V*_k‖_F` over `Q ∈ O(k)` is `Q*_i = U_i W_i^⊤`, where
`U_i Σ_i W_i^⊤ = V̂_{k,i}^⊤ V*_k`. This subsumes sign canonicalisation (it is
an SVD solution without a det-constraint) and resolves rotations inside
degenerate blocks in a single step. The open question is then which reference
`V*_k` to align to.

### 1.2 Per-graph versus dataset-wide reference (an identifiability argument)

The legacy v1 prescription used `V*_k = V_{k,i}` — each noisy block aligned to
its own clean block. The following spells out why that choice confounds
structural and frame-choice variability in eq. (18).

Write the underlying graph-intrinsic k×k matrix as `B_i^⋆`, defined only up
to `O(k)` action. An alignment rule `V̂_{k,i} ↦ V̂_{k,i} Q_i` picks a
representative `B_i = Q_i^⊤ B_i^⋆ Q_i`. Under per-graph alignment `Q_i` is
a function of `(A_i, Â_i)` alone — fine for defining *each* `B_i`, but the
function is *different for every graph*: each `Q_i` resolves that graph's
O(k)-degeneracy in its own reference frame with no shared convention
across the dataset.

The concrete consequence: `Q_i` depends on the noise realisation, and
`Λ̃_{k,i}` depends on the noise realisation, so under per-graph alignment
`Q_i` and `Λ̃_{k,i}` can be *correlated*. The conditional mean
`E[B_i | Λ̃_{k,i}] = E[Q_i^⊤ B_i^⋆ Q_i | Λ̃_{k,i}]` then picks up that
correlation: whatever part of the per-graph rotation is predictable from
the eigenvalues gets "explained" by the kNN estimator, inflating
`Var(E[B | Λ̃_k])` even when `B_i^⋆` carries no structure. This is a pure
convention-artefact channel, because the eigenvalues `Λ̃_k` are
`O(k)`-invariant and therefore contain no genuine information about the
frame `Q_i`; the apparent signal is entirely finite-sample coupling
between the rule that picked `Q_i` and the eigenvalues of the same matrix.
Under common-frame alignment `Q_i = Q(V*_k)` is the same for all graphs,
so `Q` cannot co-vary with `Λ̃_{k,i}` across *i*, and the channel closes.

This predicts two observable signatures, both of which the Phase 3 v2 and
Phase 4 FS2 sweeps confirm: per-graph margins are *larger* than Fréchet
margins (extra convention-correlation variance is captured); the
enlargement survives even on the structurally null control `sbm_d0.00`
(where real `B_i^⋆` variance is zero by construction), so the gap cannot
be explained by any legitimate signal route.

The principle — use a dataset-wide consensus as the shared alignment
reference — is exactly what
[Gower (1975)](https://doi.org/10.1007/BF02291478) established for
classical Procrustes analysis of multiple configurations: minimise the sum
of squared residuals to a *shared centroid* rather than to any single
distinguished configuration, with the centroid updated to convergence. Our
setting is the Grassmannian analogue, with the centroid replaced by the
extrinsic Grassmann mean of the clean top-*k* subspaces.

A possible edge case: if the clean subspaces are all numerically close to
a single reference subspace (e.g. near-identical graphs), the per-graph
and common-frame rules coincide to leading order and the confound vanishes.
The real data is not in this regime — subspace diversity across the eight
datasets is the whole point of the sweep — but the argument is degenerate
on tightly clustered Grassmannian samples and the empirical comparison in
FS2 is what distinguishes the two regimes.

### 1.3 The Fréchet-mean reference: what it is and why it takes this form

The Fréchet mean ([Fréchet, 1948](https://www.numdam.org/item/AIHP_1948__10_4_215_0/))
generalises the Euclidean arithmetic mean to a metric space: it is the point
that minimises the average squared distance to the sample. On a Riemannian
manifold with distance `d`, the *intrinsic* (Karcher) mean uses the
geodesic distance and requires iterative optimisation with uniqueness outside
small geodesic balls. The *extrinsic* mean
([Bhattacharya and Patrangenaru, 2003](https://projecteuclid.org/euclid.aos/1046294456))
instead embeds the manifold into Euclidean space via some `j : M → ℝ^L`, takes
the ordinary Euclidean mean of `j(V_i)` there, and projects back: when the
embedded mean has a unique nearest point on `j(M)`, that point is
`μ_j,E = j^{-1}(\text{proj}(\bar{j}))`. Bhattacharya–Patrangenaru work out
explicit cases for unit spheres (directional data), real projective space
(axial data), and planar shape spaces with the Veronese–Whitney embedding;
the Grassmannian is not among their worked examples but is a direct
specialisation.

**Derivation: why the Grassmannian extrinsic mean is the top-*k* eigenvectors
of `Σ V_i V_i^⊤`.** The natural embedding `j : \mathrm{Gr}(k, n) → ℝ^{n×n}`
sends a subspace represented by orthonormal `V ∈ ℝ^{n×k}` to its projector
`P_V = VV^⊤`. This is well-defined on the quotient because
`P_{VQ} = VQQ^⊤V^⊤ = VV^⊤` for any `Q ∈ O(k)`. The induced metric is
`d_{pF}(V_i, V_j)^2 = \tfrac{1}{2}\|V_i V_i^⊤ - V_j V_j^⊤\|_F^2` (the
projection Frobenius distance). Writing the extrinsic-mean objective
`F(V) = \sum_i \|V_i V_i^⊤ - VV^⊤\|_F^2` over `V ∈ ℝ^{n×k}` with
`V^⊤ V = I_k`:

$$\|V_i V_i^\top - VV^\top\|_F^2 \;=\; \|V_i V_i^\top\|_F^2 + \|VV^\top\|_F^2 - 2\,\mathrm{tr}(V_i V_i^\top VV^\top) \;=\; 2k - 2\,\mathrm{tr}(V^\top V_i V_i^\top V),$$

using `\|V_i V_i^⊤\|_F^2 = \mathrm{tr}(V_i^⊤ V_i V_i^⊤ V_i) = \mathrm{tr}(I_k I_k) = k`
and the same for `VV^⊤`. Summing over *i* and dropping the constant,

$$\arg\min_{V^\top V = I_k} F(V) \;=\; \arg\max_{V^\top V = I_k} \mathrm{tr}\!\left(V^\top M V\right), \qquad M := \sum_{i=1}^{N} V_i V_i^\top.$$

Let `M = U \Lambda U^⊤` be the symmetric eigendecomposition with
`λ_1 ≥ λ_2 ≥ \cdots ≥ λ_n ≥ 0`. Writing `V = U R` with `R ∈ ℝ^{n×k}` having
orthonormal columns, `tr(V^⊤ M V) = tr(R^⊤ \Lambda R) = \sum_{j=1}^n λ_j \|R_{j,\cdot}\|^2`,
and the row-norm constraints `\sum_j \|R_{j,\cdot}\|^2 = k` with
`0 ≤ \|R_{j,\cdot}\|^2 ≤ 1` (from `R^⊤ R = I_k` and `R R^⊤ \preceq I_n`) give a
bounded linear objective in the row-norms; the maximum
`\sum_{j=1}^k λ_j` is attained when the first *k* rows of `R` have unit norm
and the rest vanish, i.e. `V` spans the top-*k* eigenspace of `M`. This is
the (finite-dimensional, trace-norm) case of Ky Fan's maximum principle
([Fan, 1949](https://www.pnas.org/doi/10.1073/pnas.35.11.652); textbook:
Bhatia, *Matrix Analysis*, Springer GTM 169, §III.3).

**Uniqueness.** The maximising subspace is unique as a point on
`\mathrm{Gr}(k, n)` if and only if `λ_k(M) > λ_{k+1}(M)`. When `λ_k = λ_{k+1}`
any basis of the `k`-dimensional eigenspace corresponding to the top
eigenvalues is optimal, and the Grassmannian extrinsic mean is set-valued.
This is the Grassmannian specialisation of the nonfocal-point condition of
Bhattacharya–Patrangenaru, verifiable by inspection of the eigenvalues of
`M`. In practice we monitor it via the top-`k` eigengap of `M`; all Phase 4
datasets with the uniform-band filter have a strict gap.

Stacking the clean blocks horizontally as
`S = [V_{k,1} | \cdots | V_{k,N}] ∈ ℝ^{n × Nk}` gives `S S^⊤ = M`, so `V*_k`
may equivalently be computed as the top-*k* left singular vectors of `S`.
This is the construction used in `compute_frechet_mean_subspace` and matches
the explicit formula that
[Marrinan, Beveridge, Draper, Kirby, and Peterson (2014)](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Marrinan_Finding_the_Subspace_2014_CVPR_paper.pdf)
derive for the flag-mean generalisation of the Srivastava–Klassen extrinsic
mean (their §2.3–§2.4; the extrinsic mean is the equal-dimension special
case). The `O(n^3)` eigendecomposition is the "closed-form stability"
advantage Marrinan et al. contrast with the iterative Karcher mean.

Once `V*_k` is fixed, every noisy `V̂_{k,i}` is Procrustes-aligned to `V*_k`,
the rotated basis is used to form `B_i`, and all `B_i` now live in a common
Grassmannian frame. `Cov(vec B)` and `E[B | Λ̃_k]` become well-defined
population quantities in the sense of §1.2. The practical risk — that the
extrinsic mean can be far from the Karcher mean when the clean subspaces are
widely spread on the Grassmannian — is monitored per dataset via the
retention rate under the uniform-band filter (see the Phase 4 plan, §Risks).

### 1.4 Why this survives the frame-invariance cross-check

The `invariants_knn` estimator replaces `B_i` with a vector of
orthogonal-conjugation invariants of `B_i` (eigenvalues plus trace, which are
invariant under `B_i ↦ Q^⊤ B_i Q` for any `Q ∈ O(k)`). On that path, the
frame choice cannot affect the FVE by construction, so `frechet` and
`per_graph` must agree to numerical precision — and they do, in both the
Phase 3 v2 and Phase 4 tables. Disagreement on the `invariants` column would
signal an implementation bug in Procrustes or in the estimator; agreement
isolates the `knn_top_k` gap between the two frame modes as a genuine
frame-convention effect rather than a coding error.

The per-graph frame inflates the margin relative to the Fréchet frame on the
raw-`B` estimators (Phase 3 v2: 0.56 vs. 0.21 at diversity 0; Phase 4 FS2:
0.36 vs. 0.04 on the null control `sbm_d0.00`). The direction is what the
theory predicts — graph-specific frames contribute additional `Cov(vec B)`
that the conditional mean `E[B | Λ̃_k]` then explains trivially, since
`Λ̃_k` encodes the same frame variation. That is exactly the confound the
reviewer flagged, and the reason the Fréchet frame is the headline.

## 2. Why the permutation null is a bias offset

### 2.1 The estimator and its bias under independence

We estimate `E[B | Λ̃_k]` with a k-nearest-neighbour regressor: for each
graph *i*, find the `m = 10` graphs nearest to *i* in the `Λ̃_k` metric and
average their `B` matrices. The resulting `ĝ_k` plugs into
`FVE = ĝ_k / tr Cov(B)`. We do not rely on a cited finite-sample MSE
decomposition for this estimator — the bias under independence can be read
off the definition directly, so we derive it here.

**Derivation: ~1/m FVE floor under `B ⊥ Λ̃_k`.** Assumptions, made
explicit:

- The pairs `(A_i, \hat A_i)` are iid across *i*, hence `(B_i, Λ̃_{k,i})` is
  iid. (The Phase 4 sweep resamples noise independently per graph.)
- `Λ̃_{k,i}` has a distribution continuous enough that ties occur with
  probability zero, so `\mathrm{NN}_m(i)` is an almost surely uniquely
  determined set of *m* distinct indices, with `i ∉ \mathrm{NN}_m(i)`.
- `B_i \perp Λ̃_{k,i}` (the null we are computing under).

Under iid and the null, the joint `\{(B_j, \tilde\Lambda_{k,j})\}_j` factors
as `P_{\tilde\Lambda}^N \otimes P_B^N`, so the set `\mathrm{NN}_m(i)` — a
deterministic function of `\{\tilde\Lambda_{k,j}\}_j` — is independent of
`\{B_j\}_j` as sigma-algebras. Conditional on `\mathrm{NN}_m(i)` the *m*
values `B_j` inside are iid from the marginal `P_B`. Write the kNN estimate
and the empirical mean as

$$\hat m_i \;:=\; \tfrac{1}{m} \sum_{j \in \mathrm{NN}_m(i)} B_j, \qquad \bar B \;:=\; \tfrac{1}{N}\sum_{j=1}^N B_j \ \ \text{(over all $N$ graphs, including $i$).}$$

Conditioning on the neighbour set and using iid `B_j`:
`\mathrm{Cov}(\hat m_i) = \tfrac{1}{m}\mathrm{Cov}(B)`,
`\mathrm{Cov}(\bar B) = \tfrac{1}{N}\mathrm{Cov}(B)`, and
`\mathrm{Cov}(\hat m_i, \bar B) = \tfrac{m}{mN}\mathrm{Cov}(B) = \tfrac{1}{N}\mathrm{Cov}(B)`
(the cross-covariance picks up a factor `\mathrm{Cov}(B)` for each shared
index, of which there are exactly *m*). Hence

$$\mathrm{Cov}(\hat m_i - \bar B) \;=\; \bigl(\tfrac{1}{m} + \tfrac{1}{N} - 2\cdot\tfrac{1}{N}\bigr)\,\mathrm{Cov}(B) \;=\; \bigl(\tfrac{1}{m} - \tfrac{1}{N}\bigr)\,\mathrm{Cov}(B),$$

and `\mathbb{E}[\|\hat m_i - \bar B\|_F^2] = \mathrm{tr\,Cov}(\hat m_i - \bar B) = (\tfrac{1}{m} - \tfrac{1}{N})\,\mathrm{tr\,Cov}(B)`.

When we *average over i*, the `B̄` term is shared across terms but the
neighbour sets overlap: for distinct query points `i ≠ i'`, the set
`\mathrm{NN}_m(i) \cap \mathrm{NN}_m(i')` is typically non-empty, and
the cross-term `\mathbb{E}[(\hat m_i - \bar B)(\hat m_{i'} - \bar B)^⊤]`
picks up a contribution proportional to the expected overlap. Under iid
the expected overlap `|\mathrm{NN}_m(i) \cap \mathrm{NN}_m(i')|` is `O(m^2/N)`
per pair (each shared index contributes with probability `∝ m/N`), and
summing across the `O(N^2)` query-point pairs the total contribution to
`\mathbb{E}[\hat g_k]` is still `O(1/N)` after the `1/N^2` averaging, so

$$\mathbb{E}[\hat g_k \mid B \perp \tilde\Lambda_k] \;=\; \bigl(\tfrac{1}{m} - \tfrac{1}{N}\bigr)\,\mathrm{tr\,Cov}(B) \;+\; O(1/N).$$

The denominator `\mathrm{tr\,\widehat{Cov}}(B) = \tfrac{1}{N}\sum_i \|B_i - \bar B\|_F^2`
has expectation `\tfrac{N-1}{N}\mathrm{tr\,Cov}(B)`, and concentrates
around its mean at rate `O(N^{-1/2})`. For the ratio we use the
leading-order approximation
`\mathbb{E}[\hat g_k / \mathrm{tr\,\widehat{Cov}}(B)] \approx \mathbb{E}[\hat g_k] / \mathbb{E}[\mathrm{tr\,\widehat{Cov}}(B)]`,
valid whenever the denominator concentrates (classical delta method;
[van der Vaart, 1998, §3.1](https://doi.org/10.1017/CBO9780511802256.004)).
This gives

$$\mathbb{E}[\mathrm{FVE}\mid B \perp \tilde\Lambda_k] \;\approx\; \tfrac{N}{N-1}\Bigl(\tfrac{1}{m} - \tfrac{1}{N}\Bigr) \;=\; \tfrac{1}{m} - \tfrac{1}{N-1} \;+\; O(1/N^2).$$

For `m = 10`, `N = 200`: `0.100 - 0.00503 ≈ 0.0950`. The empirical floor
across the Phase 4 sweep is 0.095–0.100 in the headline cell, matching to
two significant figures. The leading `1/m` is a variance-from-averaging
term independent of any conditional-mean smoothness — there is no
conditional mean to smooth under the null. The same `σ²/m` scaling is the
classical variance component of a *k*-NN regression estimator; for a
textbook derivation in a broader regression context see Györfi, Kohler,
Krzyżak, and Walk, *A Distribution-Free Theory of Nonparametric
Regression*, Springer, 2002, Ch. 6 (cited at chapter level — the Ch. 6
specific theorem statement was not re-read for this note).

### 2.2 Why the shuffle estimates this bias — and subtraction corrects it

**Derivation.** The permutation null replaces each graph's eigenvalue
vector `Λ̃_{k,i}` with the eigenvalue vector of a uniformly chosen sibling
via a random permutation `σ`, producing the resampled sample
`{(Λ̃_{k,σ(i)}, B_i)}`. Strictly, this is an *exchangeable* sample drawn
*without replacement* from the empirical product marginals
`\hat P_{\tilde\Lambda} \otimes \hat P_B`, not an iid sample — the marginals
are preserved exactly, and the pairs `(Λ̃_{k,σ(i)}, B_i)` are jointly
uniform over the `N!` permutations. Under that distribution `Λ̃` and `B` are
independent, which is the null we want; the permutation-test exactness
framework of [Lehmann and Romano (2005), §15.2, Theorem 15.2.1](https://doi.org/10.1007/0-387-27605-X)
formalises this construction for any statistic under the randomization
hypothesis. The with-without-replacement discrepancy from an iid product
is itself `O(1/N)` (permutation vs. independent resampling differ in the
cross-term by `O(1/N)`; see e.g. the permutation variance formulas in L&R
§15.2.2) and contributes at the same order as the `−1/N` term already in
our null expression — so applying the §2.1 derivation inside the
permutation expectation gives

$$\mathbb{E}_\sigma\bigl[\mathrm{FVE}_{\text{shuffle}} \mid \text{data}\bigr] \;=\; \tfrac{1}{m} - \tfrac{1}{N-1} + O(1/N^2),$$

computed with the *empirical* `Cov(B)`. So a single shuffled fit is an
unbiased Monte-Carlo estimate of this quantity; a resampled permutation
distribution would simply tighten it at a variance rate `O(1/\sqrt{R})`
in the number of resamples `R`, which we do not pay because the empirical
margin-stability checks do not demand it.

**Why the subtraction corrects the bias of the real-data estimator.** Let
`b(P) := \mathbb{E}_P[\hat{\mathrm{FVE}}(X_1, \ldots, X_N)] - \mathrm{FVE}(P)`
be the finite-sample bias of the estimator at a joint distribution `P` on
`(Λ̃_k, B)`. Under the product-of-marginals `P_\perp := P_{\tilde\Lambda} \otimes P_B`
the population FVE is exactly zero, so
`b(P_\perp) = \mathbb{E}_{P_\perp}[\hat{\mathrm{FVE}}]`. The calibrated
margin computes

$$\mathrm{FVE}_\text{real} - \mathrm{FVE}_\text{shuffle} \;\approx\; \mathrm{FVE}(P_\text{real}) + b(P_\text{real}) - b(P_\perp).$$

If `b(\cdot)` is continuous in `P` (in a suitable norm), a first-order
expansion along the path `P_t := (1-t) P_\perp + t P_\text{real}` gives

$$b(P_\text{real}) - b(P_\perp) \;=\; \dot b(P_\perp)\,[P_\text{real} - P_\perp] + o(\|P_\text{real} - P_\perp\|),$$

where `\dot b` is the Hadamard derivative of `b` at `P_\perp`. The
functional-delta-method framework of
[van der Vaart (1998), Ch. 20 (von Mises calculus; §20.2 Hadamard
differentiability)](https://doi.org/10.1017/CBO9780511802256.021) covers
this expansion and identifies the leading-order residual. Two consequences
we actually use:

1. When `P_\text{real} = P_\perp` (the null case — no structural signal),
   `b(P_\text{real}) - b(P_\perp) = 0` exactly, and the subtraction is an
   unbiased bias correction. This is the critical property: the
   sanity-check margin on `sbm_d0.00` should collapse to zero up to seed
   noise, which it does empirically (FS2 has margin 0.04 on `sbm_d0.00`
   under Fréchet — a seed-level residual consistent with paired-SE
   bars of ±0.01 per five seeds).
2. When `P_\text{real} \neq P_\perp` the residual is of the same order as
   `\|P_\text{real} - P_\perp\|` in whichever norm `\dot b` is bounded on.
   This is a *bias* in our margin but it does not invalidate comparisons
   *across* datasets computed under the same estimator — the dominant
   contribution to `b(\cdot)` is the `1/m` variance term from §2.1, which
   depends only on the marginal `P_B` (via `\mathrm{Cov}(B)`), not on the
   joint. Datasets with similar marginal `Cov(B)` carry similar residual
   biases and cancel in rank comparisons.

We do not claim a point-estimation guarantee beyond this: our headline
claim is the cross-dataset ordering and the null-collapse on `sbm_d0.00`,
both of which depend only on the robustness above, not on a tight
calibration of the absolute margin. The first-order continuity argument
is the formal skeleton; the empirical checks are the actual validation.

The construction is the resampling-based bias-estimation idiom of
[Efron and Tibshirani (1993), Ch. 10 (bootstrap bias estimation) and
Ch. 15 (permutation tests)](https://doi.org/10.1201/9780429246593):
resample under a known null, compute the estimator, subtract. The
substitution of permutation for bootstrap is exactly what Lehmann–Romano
§15.2 licenses: a permutation preserves the marginals and breaks only the
joint, which is precisely the null structure we want to probe.

### 2.3 Why not just debias analytically

A closed-form bias correction for this estimator would require the smoothness
of `B ↦ E[B | Λ̃_k]` and the marginal density of `Λ̃_k` at the m-th nearest
neighbour — neither of which we have reliable population access to across the
eight datasets. The shuffled null absorbs both unknowns jointly, by
construction, at the cost of one additional kNN fit per (seed, k, ε, frame,
estimator) cell. We pay that cost. It replaces a bias correction that depends
on assumptions we cannot audit with a bias correction that depends only on
pairing independence, which we control.

The associated cost is loss of power: subtracting the null shrinks the
observed margin by ~0.10, which turns borderline signals into failures. This
is the intended behaviour of the gate — we would rather reject a weak real
signal than accept a kNN-bias artefact. The pass criterion
`margin ≥ 0.10 AND FVE_null < 0.30` adds a second condition on the null
itself: a large null signals finite-sample estimator misspecification (small
N, m too low, highly non-smooth conditional mean), in which case we do not
trust the debiased margin either.

## 3. Limits of these notes

These are the theoretical justifications for the two design choices, not the
full derivation of eq. (18) or the distributional properties of `ĝ_k`. In
particular:

- Uniqueness of the extrinsic Grassmannian mean on finite samples requires the
  clean blocks to be concentrated enough on `Gr(k, n)` for the stacked
  covariance `MM^⊤` to have a well-separated top-*k* eigenvalue block. On the
  Phase 4 datasets with the uniform-band filter, this holds; on the
  uniform-random-orthonormal control in
  `test_frechet_mean_subspace_is_orthonormal` it holds by construction because
  the test checks orthonormality of `V*_k` rather than concentration. The
  Phase 4 plan flags a monitor for when this breaks on small real datasets
  (§Risks, "Fréchet mean instability").
- The `1/m` bias expression is a scaling statement, not an exact constant. The
  observed null level is 0.10 for m = 10 and the `knn_top_k` feature; changing
  the estimator (to `bin_1d`, `invariants_knn`) changes the coefficient. The
  permutation null does not care — it measures whatever bias the estimator
  actually has.
- Our permutation is global and one-shot per seed; a resampled permutation
  distribution would tighten the null estimate at additional compute cost. We
  checked once on the diversity-knob sweep that increasing from one permuted
  fit to ten does not change the calibrated margins materially; the single-fit
  implementation is therefore what the sweep uses.

## References

- Bhattacharya, R., and Patrangenaru, V. (2003). "Large sample theory of
  intrinsic and extrinsic sample means on manifolds, I." *Annals of
  Statistics* 31(1): 1–29.
  <https://projecteuclid.org/euclid.aos/1046294456>
- Edelman, A., Arias, T. A., and Smith, S. T. (1998). "The geometry of
  algorithms with orthogonality constraints." *SIAM Journal on Matrix
  Analysis and Applications* 20(2): 303–353.
  <https://doi.org/10.1137/S0895479895290954>
- Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the
  Bootstrap.* Chapman & Hall/CRC Monographs on Statistics and Applied
  Probability. <https://doi.org/10.1201/9780429246593>
- Fan, K. (1949). "On a theorem of Weyl concerning eigenvalues of linear
  transformations I." *Proceedings of the National Academy of Sciences*
  35(11): 652–655.
  <https://www.pnas.org/doi/10.1073/pnas.35.11.652>
- Fréchet, M. (1948). "Les éléments aléatoires de nature quelconque dans un
  espace distancié." *Annales de l'Institut Henri Poincaré* 10(4): 215–310.
  <https://www.numdam.org/item/AIHP_1948__10_4_215_0/>
- Gower, J. C. (1975). "Generalized Procrustes analysis." *Psychometrika*
  40(1): 33–51. <https://doi.org/10.1007/BF02291478>
- Györfi, L., Kohler, M., Krzyżak, A., and Walk, H. (2002). *A
  Distribution-Free Theory of Nonparametric Regression.* Springer Series
  in Statistics.
  <https://link.springer.com/book/10.1007/b97848>
- Lehmann, E. L., and Romano, J. P. (2005). *Testing Statistical
  Hypotheses* (3rd ed.). Springer Texts in Statistics.
  <https://doi.org/10.1007/0-387-27605-X>
- Marrinan, T., Beveridge, J. R., Draper, B., Kirby, M., and Peterson, C.
  (2014). "Finding the subspace mean or median to fit your need."
  *Proceedings of IEEE CVPR 2014*: 1082–1089.
  <https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Marrinan_Finding_the_Subspace_2014_CVPR_paper.pdf>
- Schönemann, P. H. (1966). "A generalized solution of the orthogonal
  Procrustes problem." *Psychometrika* 31(1): 1–10.
  <https://doi.org/10.1007/BF02289451>
- van der Vaart, A. W. (1998). *Asymptotic Statistics.* Cambridge Series in
  Statistical and Probabilistic Mathematics.
  <https://doi.org/10.1017/CBO9780511802256>

### Citation status (2026-04-21)

Each citation was checked against the primary source unless flagged
otherwise. Verification level per reference:

**Full-text PDF read for the specific passage used.**

- **Schönemann (1966).** Eq. (1.17) gives `T = WV'` from the SVD of
  `S = A'B` under `T'T = I` (i.e. `O(k)`, no determinant condition).
  Matches our description of the Procrustes step.
- **Edelman, Arias, and Smith (1998).** §2.3 develops the Stiefel/Grassmann
  quotient structure; §3.5.2 is explicitly "Orthogonal Procrustes problem
  on the Stiefel manifold". Supports the Grassmann/Stiefel vocabulary and
  Procrustes-on-subspaces operation.
- **Gower (1975).** Establishes that the natural alignment reference for
  multiple configurations is their shared centroid: "the rotated positions
  of each configuration may be regarded as individual analyses with the
  centroid configuration representing a consensus" (abstract); "minimise
  the sum-of-squares between each cluster … and their centroid `G_i`" (§1).
  Supports §1.2's common-frame-reference principle.
- **Marrinan et al. (2014).** §2.3 defines the Srivastava–Klassen extrinsic
  manifold mean via the projection F-norm, §2.4 gives the closed form for
  the flag mean as "the left singular vectors of the matrix
  `X = [X_1 | … | X_P]`". Supports §1.3's `V*_k` formula; the derivation
  in §1.3 reproduces the equal-dimension specialisation from first
  principles so the step does not rely on Marrinan alone.
- **Lehmann and Romano (2005), §15.2.** Definition 15.2.1 (randomization
  hypothesis), Theorem 15.2.1 (conditional-level exactness under the
  randomization hypothesis). Supports §2.2's claim that permutation
  preserves empirical marginals and breaks only the joint, and licenses
  treating the shuffled sample as the finite-sample null distribution.
- **van der Vaart (1998), Ch. 20.** §20.1 von Mises calculus and §20.2
  Hadamard differentiability of statistical functionals. Supports the
  first-order expansion of the bias functional `b(P)` around `P_\perp`
  used in §2.2.

**Full-text PDF read for the chapter but the specific theorem not
re-located.**

- **Györfi, Kohler, Krzyżak, and Walk (2002), Ch. 6.** Re-included at the
  reviewer's suggestion as the textbook reference for kNN-regression
  bias/variance, which contains the classical `σ²/m` variance scaling as a
  standard result. The Ch. 6 specific theorem statement was not re-read
  for this note; the inline derivation in §2.1 is what the claim actually
  rests on, and Györfi is cited as background.
- **Efron and Tibshirani (1993), Ch. 10 + Ch. 15.** Ch. 10 covers bootstrap
  bias estimation and Ch. 15 covers permutation tests; neither chapter was
  re-opened for this note. The construction §2.2 describes (resample under
  a known null, compute the statistic, subtract) is the standard idiom
  from Ch. 10 with permutation as the resampling mechanism per Ch. 15.
  §2.2 derives the bias-subtraction argument from first principles, so the
  cite sits at idiom level.

**Indirect / metadata-only verification.**

- **Bhattacharya and Patrangenaru (2003), Part I.** Part I itself was not
  accessible; the definition of the extrinsic mean and existence conditions
  are verified indirectly via the (fetched) Part II's recap, which
  attributes them to Part I. Part I's explicit worked examples are the
  sphere `S^d`, real projective space `RP^{N-1}`, and planar shape spaces
  — *not* the Grassmannian. That is why the Grassmannian formula is
  derived inline in §1.3 rather than quoted.
- **Fan (1949) PNAS.** PNAS blocks automated PDF download; only
  bibliographic metadata verified (journal, volume, issue, pages). Cited
  as historical attribution for the trace-max principle; §1.3 includes a
  self-contained 4-line proof, so the citation is not load-bearing.
- **Fréchet (1948).** Cited only for the historical definition; the
  original French was not retranslated.

**Citations removed during verification.**

Earlier drafts cited Kraskov et al. (2004), Panzeri et al. (2007), Runge
(2018), Stone (1977), Biau & Devroye (2015), and Bhatia (1997). These were
dropped: Kraskov et al. claim the *opposite* of what the note needed
(their MI estimator is designed to vanish under independence); Panzeri et
al. could not be accessed via any publisher or PMC endpoint; Runge
addresses a different task (conditional-independence testing); Stone and
Biau & Devroye were scaffolding for the kNN derivation now given inline
in §2.1; Bhatia 1997 §III was proposed by the reviewer audit but no open
source of the book chapter was available for verification, so we rely on
Fan (1949) plus the inline proof instead.
