# Section 13 — Evaluation metrics (sampling time): spec vs. our implementation

## Summary verdict

| Check | Status | Notes |
|---|---|---|
| SBM block-count filter (2–5) | PASS | Identical threshold |
| SBM block-size filter (20–40) | PASS | Identical threshold |
| SBM p > 0.9 criterion | PASS | Identical |
| `refinement_steps=100` (live call site) | **GAP** | Our `evaluate()` passes no `refinement_steps`; `compute_sbm_accuracy` defaults to 1000, not 100 |
| `multiflip_mcmc_sweep(beta=inf)` semantics | PASS | Identical call, `niter=10` per sweep |
| ProcessPoolExecutor vs ThreadPoolExecutor | NOTED | Deliberate deviation for graph-tool thread safety; see evaluator docstring |
| MMD kernel: gaussian_tv formula | PASS | Identical TV = 0.5 * L1, Gaussian wrapper |
| `clustering_sigma=0.1` | PASS | Wired via `clustering_sigma` constructor param |
| `sigma=1.0` default for degree + spectral | PASS | Matches upstream call sites |
| Spectral histogram: 200 bins, range (-1e-5, 2) | PASS | Matches upstream |
| Clustering histogram: 100 bins | PASS | Matches upstream |
| MMD biased V-statistic (i==j included) | PASS | Correct, documented |
| Reference graphs from val dataloader | PASS | `get_reference_graphs("val", n)` iterates val dataloader |
| No synthesis from p_intra/p_inter | PASS | `p_intra`/`p_inter` only used by SBM Wald test |
| Uniqueness: `faster_could_be_isomorphic` + `is_isomorphic` | PASS | Stricter than upstream default but matches SBM metrics path |
| Sample decoding: `E_class.argmax(-1) != 0` | PASS | Class 0 = no-edge convention correct |
| No caching across `evaluate()` calls | PASS | Stateless except config + train_graphs |

**One gap.** `GraphEvaluator.evaluate()` calls `compute_sbm_accuracy` without a `refinement_steps` argument, so the MCMC budget is 1000 instead of the upstream-live 100. All other checks pass.

---

## Spec summary (§13)

Upstream evaluation for SBM and SPECTRE datasets lives in
`src/analysis/spectre_utils.py`. `SBMSamplingMetrics` uses
`metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'sbm']` and
`compute_emd=False`.

`SpectreSamplingMetrics.forward` calls `eval_acc_sbm_graph(networkx_graphs,
refinement_steps=100, strict=True)` — the live `refinement_steps` is 100, not the
`is_sbm_graph` default of 1000. `is_sbm_graph` fits `minimize_blockmodel_dl`,
refines with `refinement_steps` calls of `multiflip_mcmc_sweep(beta=inf, niter=10)`,
then applies strict filters: `n_blocks in [2, 5]`, per-block sizes in `[20, 40]`.
It computes a Wald statistic for intra/inter probabilities referenced at
`p_intra=0.3`, `p_inter=0.005`, converts via chi^2_1, takes the mean p-value,
and returns `p > 0.9`.

MMD kernels: `gaussian_tv` for degree/spectral/clustering (sigma 1.0/1.0/0.1);
`gaussian_tv, sigma=30, is_hist=False` for orbit. Spectral uses 200 bins over
(-1e-5, 2); clustering uses 100 bins. MMD is the biased V-statistic (all n^2
ordered pairs, including i==j). Histograms are re-normalised to PMF before
kernel evaluation.

Reference graphs: the val dataloader's graphs (`self.val_graphs` from
`loader_to_nx`), not synthesised from `p_intra`/`p_inter`.

Uniqueness: `eval_fraction_unique_non_isomorphic_valid` uses `nx.is_isomorphic`
unconditionally (precise test). `eval_fraction_unique(precise=False)` exists as
a separate function but is not called by `SBMSamplingMetrics`.

---

## Our implementation

**`graph_evaluator.py`**

`GraphEvaluator.evaluate()` calls `compute_sbm_accuracy(generated, p_intra=...,
p_inter=...)` with no `refinement_steps` keyword (line 766–770). `compute_sbm_accuracy`
defaults to `refinement_steps=1000`. This is a 10x MCMC over-budget relative to
upstream's live call (`refinement_steps=100`).

`_is_sbm_graph` structure matches upstream exactly: same `minimize_blockmodel_dl`,
same loop `for _ in range(refinement_steps): state.multiflip_mcmc_sweep(beta=np.inf,
niter=10)`, same strict filter thresholds (20/40/2/5), same Wald stats, same
`chi2.cdf`, same `mean > 0.9` criterion. Returns `float` (0.0/1.0) vs. upstream's
`bool` — semantically identical when averaged.

`compute_sbm_accuracy` uses `ProcessPoolExecutor(mp_context=spawn)` instead of
upstream's `ThreadPoolExecutor`. This is a deliberate and documented deviation to
avoid graph-tool thread-safety crashes (`SIGABRT` in C++ heap, documented in
`docs/reports/2026-04-15-bug-modal-sigabrt.md`). Sequential semantics are unchanged;
pass/fail results are identical.

`compute_uniqueness` uses `faster_could_be_isomorphic` + `is_isomorphic` (precise),
matching the upstream path actually wired into `SBMSamplingMetrics` via
`eval_fraction_unique_non_isomorphic_valid`. Our function computes only "uniqueness"
(fraction distinct among generated), not the joint unique+novel+valid triple that
upstream computes in a single pass. The novelty component is handled separately by
`compute_novelty`, and "valid" is not reported as a combined metric. This is a
structural divergence from upstream's combined metric, but all three quantities are
available separately in `EvaluationResults`.

**`mmd_metrics.py`**

`gaussian_tv_kernel`: `tv = 0.5 * sum(abs(x - y))`, `exp(-tv^2 / (2 * sigma^2))`.
Identical to upstream `dist_helper.gaussian_tv`.

`compute_mmd`: V-statistic. Enumerates all `n*n` ordered pairs for within-set terms
(including `i==j`), all `n*m` pairs for cross-term. Matches upstream `disc(s,s) + disc(s2,s2) - 2*disc(s,s2)` where `disc` averages over `len(s1)*len(s2)` pairs.

Upstream `compute_mmd` with `is_hist=True` (default) normalises via
`s / (sum(s) + 1e-6)` before kernel calls. Our `compute_mmd` does not re-normalise
at the MMD level — instead each per-graph histogram is already a PMF (sum-to-one) by
construction in `compute_degree_histogram`, `compute_clustering_histogram`, and
`compute_spectral_histogram`. Both approaches deliver normalised PMFs to the kernel;
the difference is where normalisation is applied. The `1e-6` stabiliser present in
upstream's normalisation is absent from ours, but the histograms are already
non-degenerate at that point so the difference is numerically negligible.

Histogram parameters:
- Spectral: 200 bins, range (-1e-5, 2.0) — matches `spectre_utils.py:91`.
- Clustering: 100 bins over [0, 1] — matches `spectre_utils.py:285`.
- Degree: integer bins up to `max(degrees) + 1` — matches `nx.degree_histogram` semantics.

Bandwidth wiring: `clustering_sigma=0.1` flows through the constructor and
`compute_mmd_metrics`; degree and spectral fall back to `sigma=1.0` unless
overridden. This matches the upstream call sites exactly.

**`reference_graphs.py`**

`generate_reference_graphs` synthesises adjacency tensors from SBM/ER/etc. and is
only used by the offline CLI eval path (`evaluate_cli.py`). It is not on the
validation path. The validation path uses `datamodule.get_reference_graphs("val", n)`,
which iterates the val dataloader. This matches upstream's `self.val_graphs` from
`loader_to_nx`. The `p_intra`/`p_inter` attributes on `GraphEvaluator` are consumed
only by `compute_sbm_accuracy`; they do not affect reference graph construction.

---

## Per-check verdicts

**SBM Wald test structure.** All structural details match: block-count filter 2–5,
block-size filter 20–40, `beta=inf` MCMC sweeps with `niter=10` each, Wald statistic
formula, chi^2 conversion, `p > 0.9` threshold. PASS.

**`refinement_steps=100`.** Upstream `SpectreSamplingMetrics.forward` explicitly
passes `refinement_steps=100` to `eval_acc_sbm_graph` (line 830). Our `evaluate()`
omits this argument, so `compute_sbm_accuracy` receives its default of 1000. The
`GraphEvaluator.__init__` has no `refinement_steps` parameter, so there is no config
knob to fix this without a code change. GAP — 10x budget divergence.

**ProcessPoolExecutor vs ThreadPoolExecutor.** Upstream uses `ThreadPoolExecutor`.
Ours uses `ProcessPoolExecutor` with `spawn` context. The reason is documented and
reproducible: graph-tool's C++ internals are not thread-safe when called concurrently
from Python threads. The process-based approach avoids `SIGABRT` crashes at the cost
of spawn overhead. This is a behavioural difference in execution strategy, not in
numerical outcome. NOTED.

**MMD Gaussian TV kernel.** Formula, sigma defaults, and per-metric overrides are
all parity-correct. PASS.

**Histogram bins.** 200 bins spectral, 100 bins clustering — both match. PASS.

**Biased V-statistic.** Both upstream and ours include the diagonal terms in the
within-set sums. PASS.

**Reference graphs from val dataloader.** Confirmed from `diffusion_module.py` →
`get_reference_graphs("val", n)`. Not synthesised. PASS.

**Uniqueness.** Uses `faster_could_be_isomorphic` + `is_isomorphic`. This is the
precise test, matching the path actually wired into `SBMSamplingMetrics` via
`eval_fraction_unique_non_isomorphic_valid`. PASS.

**Sample decoding.** `E_class.argmax(dim=-1) != 0` maps class-0 to no-edge, any
other class to edge — correct for DiGress's `(no_edge, edge)` convention. PASS.

**No caching.** `GraphEvaluator.__init__` stores only config and `train_graphs`
(passed once for novelty). `evaluate()` is a pure computation over its arguments.
PASS.

---

## Remaining gaps

**`refinement_steps` hardwired to 1000** (`_is_sbm_graph` default) instead of
upstream's live 100. The effect is 10x more MCMC per graph, making SBM evaluation
significantly slower. The accept/reject outcome for clearly degenerate graphs is
unlikely to differ, but for borderline graphs the extra refinement may shift block
assignments and therefore the Wald p-value. For parity with reported upstream SBM
accuracy numbers the default in `GraphEvaluator` should be 100. Fix: add
`sbm_refinement_steps: int = 100` to `GraphEvaluator.__init__` and thread it through
to `compute_sbm_accuracy`.
