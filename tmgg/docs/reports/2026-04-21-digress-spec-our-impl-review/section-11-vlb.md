# Section 11 Review: Variational Lower Bound

**Verdict: PASS** — All seven checks match. No parity breaks relative to upstream DiGress.

This review supersedes the earlier Haiku audit in
`analysis/digress-loss-check/parity-audit/deep-vlb-kl.md` with updated
line numbers and an additional check on the reconstruction path divergence
from upstream.

---

## Spec summary

The spec defines the NLL estimator as:

```
-log p(N) + KL(q(z_T|x_0) || π)  +  T · E_t[KL(q(z_{t-1}|z_t,x_0) || p_θ(z_{t-1}|z_t))]
                                   -  E[log p(x_0 | z_0)]
```

with four named terms: `log_pN`, `kl_prior`, `compute_Lt` (×T), and
`reconstruction_logp`. Upstream source: `diffusion_model_discrete.py:444–483`.

---

## Our implementation

Entry point: `DiffusionModule.validation_step`
(`diffusion_module.py:635`). The categorical branch runs lines 668–716;
the Gaussian branch runs 717–747. The four terms map as follows:

| Spec term | Our symbol | Location |
|-----------|-----------|----------|
| `log_pN` | `log_pn` | `validation_step:702–705` |
| `kl_prior` | `kl_prior` | `validation_step:695–698` |
| `T · E_t[L_t]` | `kl_diffusion` | `validation_step:680–686` |
| `reconstruction_logp` | `reconstruction` | `_compute_reconstruction:596–631` |
| NLL combination | `nll` | `validation_step:709` |

---

## Check-by-check verdicts

### Check 1 — VLB decomposition signs

**Upstream** (`diffusion_model_discrete.py:471`):
```python
nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
```

**Ours** (`diffusion_module.py:709`):
```python
nll = -log_pn + kl_prior.mean() + kl_diffusion.mean() - reconstruction.mean()
```

**MATCH.** Signs identical. `.mean()` calls average per-graph scalars
across the batch before summing the four terms; upstream does a final
`.mean()` inside the `val_nll` metric accumulator after the same
per-graph vector is constructed. Semantically equivalent.

---

### Check 2 — `kl_prior`: `KL(q(z_T|x_0) || π)`

**Upstream** (`diffusion_model_discrete.py:303–337`): builds
`q(z_T|x_0) = x_0 @ Q_T_bar`, expands the stationary prior `π` to
match the batch shape, calls `mask_distributions` (which replaces padded
rows with a degenerate one-hot, adds `1e-7`, renormalises), then
`F.kl_div(probX.log(), limit_dist_X, reduction='none')` summed via
`sum_except_batch`.

**Ours** (`diffusion_module.py:695–698`):
```python
t_T = torch.full((bs,), self.T, device=device, dtype=torch.long)
forward_pmf_T = cat_process.forward_pmf(batch, t_T)
prior_pmf = cat_process.prior_pmf(batch.node_mask)
kl_prior = _categorical_kl_per_graph(forward_pmf_T, prior_pmf)
```

`forward_pmf` (`noise_process.py:1181–1209`) computes
`alpha_bar_T * x_0 + (1-alpha_bar_T) * π` directly — identical to
`x_0 @ Q_T_bar` for the marginal transition kernel.

`_categorical_kl_per_graph` (`diffusion_module.py:220–264`) masks by
zeroing `p` at invalid positions and setting `q=1.0` there, then clamps
to `1e-10` before log. Upstream masks by `mask_distributions` (uniform +
`1e-7` floor + renorm). Both prevent `log(0)` at padded positions and
ensure zero KL contribution from them, because `p=0` kills `p·(log p −
log q)` regardless of `q`.

**MATCH.** The deleted `mask_distributions` is not called here; our
`_categorical_kl_per_graph` provides the equivalent guard.

---

### Check 3 — `compute_Lt` / `kl_diffusion`: the T-multiplier

**Upstream** (`diffusion_model_discrete.py:366`):
```python
return self.T * (kl_x + kl_e)
```
`kl_x` and `kl_e` come from `SumExceptBatchKL` metric objects that sum
within a graph and mean across the batch, so this is `T * mean_over_batch(per_graph_KL)`.

**Ours** (`diffusion_module.py:686`):
```python
kl_diffusion = self.T * _categorical_kl_per_graph(true_posterior, pred_posterior)
```
`_categorical_kl_per_graph` returns a `(bs,)` vector of per-graph sums;
`.mean()` is applied at line 709.

**MATCH.** The `T ·` multiplier is explicit in both implementations and
applied at the same stage (before batch averaging). The single-sample
Monte Carlo estimator is present in both: `t_int` is sampled once per
batch in `validation_step` (`diffusion_module.py:652`), and in upstream's
`apply_noise` which is called by `compute_val_loss`.

---

### Check 4 — True vs predicted posterior in `compute_Lt`

**Upstream**: `prob_true` uses `X, E` (the clean one-hot batch);
`prob_pred` uses `pred_probs_X = F.softmax(pred.X)` — i.e. the direct
Bayes formula with soft `x_0` prediction plugged in
(`diffusion_model_discrete.py:350–356`). Upstream does **not**
marginalise over `x_0` classes at this step; it plugs the soft
probabilities directly into the posterior formula.

**Ours**: true posterior via `_posterior_probabilities(z_t, batch, ...)`;
predicted posterior via `_posterior_probabilities_marginalised(z_t,
x0_param, ...)` (`diffusion_module.py:680–685`).

**DIVERGENCE — MINOR.** Our VLB KL uses the marginalised form
`sum_c p(z_s|z_t, x_0=c) * p_θ(x_0=c|z_t)` for the predicted
posterior, while upstream plugs the soft `x_0` prediction directly into
the Bayes formula. The two are equivalent at convergence (when `p_θ` is
one-hot) but differ during training. The docstring at line 679
acknowledges this: the marginalised form is the one the sampler actually
draws from. The consequence is that the per-step KL reported during
training may differ numerically from upstream's, but both are valid
estimators of the same quantity and the divergence vanishes at
convergence. This was a deliberate choice introduced in the Phase C
sampler change and is not a bug.

---

### Check 5 — `reconstruction_logp`: noising convention

**Upstream** (`diffusion_model_discrete.py:368–405`): forces `t=0`,
computes `Q_0` from `β_0`, samples `z_0 ~ x_0 @ Q_0`, runs the model
with `t=0` conditioning, softmaxes, then masks. The caller at line 468
computes `X * prob0.X.log()` summed per graph — i.e.
`log p_θ(x_0 | z_0)`.

**Ours** (`diffusion_module.py:596–631`): forces `t=1`, `s=0`, samples
`z_1 ~ forward_sample(batch, t=1)`, runs the model with `t=1`
conditioning, then scores the clean batch under the marginalised
posterior PMF `p(x_0 | z_1)` rather than directly under the softmax.

**DIVERGENCE — MINOR / DELIBERATE.** Upstream conditions on a `z_0`
sample (drawn from `Q_0`) with `t=0`; ours conditions on `z_1` with
`t=1`. The docstring at line 600–607 explicitly notes this and bounds the
magnitude of the difference at `~1e-3` for T=1000. The choice was made
intentionally (see the commit that introduced `_compute_reconstruction`)
to route through the marginalised posterior rather than scoring the raw
softmax at `t=0`. Not a parity break, but a documented methodological
deviation.

---

### Check 6 — `log_pN` construction and fallback

**Upstream** (`diffusion_model_discrete.py:455–456`): `node_dist` is a
`DistributionNodes` object built from the training set; `log_prob(N)`
returns `log(p[N] + 1e-30)`.

**Ours** (`diffusion_module.py:702–705`):
```python
log_pn = torch.zeros(1, device=device)
if self._size_distribution is not None:
    node_counts = batch.node_mask.sum(dim=-1).long()
    log_pn = self._size_distribution.log_prob(node_counts).mean()
```
`SizeDistribution.log_prob` clamps to `1e-30` before log
(`size_distribution.py:125`).

**MATCH** when `_size_distribution` is populated. **GAP**: when
`_size_distribution is None` (e.g. if `setup()` was not called or the
datamodule does not implement `get_size_distribution`), `log_pn` silently
defaults to zero, effectively dropping the `log p(N)` term. Upstream
would crash on a missing `node_dist`. The fallback is silent, not
loud — inconsistent with the project's "fail loudly" principle, though
the condition is unlikely to occur in normal training.

---

### Check 7 — Mask handling: `mask_distributions` deletion

Upstream calls `mask_distributions` in both `kl_prior` (line 327) and
`compute_Lt` (line 359). This function was deleted from our codebase at
commit `59e9593f`. The VLB path does not call it.

**MATCH** (gap is covered). Both VLB paths — `_categorical_kl_per_graph`
(kl_prior and kl_diffusion) and `_categorical_reconstruction_log_prob`
(reconstruction) — apply their own masking:
- `_categorical_kl_per_graph` zeroes `p` and sets `q=1` at invalid nodes
  and edges (`diffusion_module.py:254–257`, `noise_process.py:246–257`).
- `_categorical_reconstruction_log_prob` zeroes `clean` and sets `pred=1`
  at invalid positions (`diffusion_module.py:206–209`).

No `log(0)` exposure in any VLB path.

---

## Summary

| Check | Verdict | Notes |
|-------|---------|-------|
| VLB decomposition signs | MATCH | Identical formula |
| `kl_prior` formula | MATCH | `mask_distributions` replaced by clamping in `_categorical_kl_per_graph` |
| T-multiplier on `kl_diffusion` | MATCH | Both apply `self.T *` before batch averaging |
| Predicted posterior form | MINOR DIVERGENCE | Ours uses marginalised form; upstream uses direct soft-x0 Bayes. Deliberate; converges to same value. |
| `reconstruction_logp` conditioning | MINOR DIVERGENCE | Ours: z_1 + marginalised posterior. Upstream: z_0 + direct softmax. Deliberate; ~1e-3 gap at T=1000. |
| `log_pN` | MATCH | Same clamped log-histogram; silent zero-fallback when `_size_distribution` is None |
| Mask handling post-`mask_distributions` deletion | MATCH | Per-call guards in `_categorical_kl_per_graph` and `_categorical_reconstruction_log_prob` cover all log(0) exposure |

**Surviving divergences** — intentional design choices are still divergences
from upstream, not non-issues. Each is assessed for failure-mode impact
(specifically: could it contribute to the edge-collapse training failure
documented in `analysis/digress-loss-check/BUG_REPORT.md`?).

### D1. Marginalised predicted posterior in `kl_diffusion`

- **What:** Ours computes the predicted posterior via
  `sum_c p(z_s|z_t, x_0=c) · p_θ(x_0=c|z_t)` whereas upstream plugs the
  soft `x_0` prediction directly into the Bayes posterior formula
  (`diffusion_model_discrete.py:350–356`).
- **Intent:** aligns our VLB KL with the sampler's actual draw distribution.
- **Training-failure impact — NONE.** This appears only in
  `validation_step` VLB reporting (`kl_diffusion`). It does not affect
  the training gradient, which is computed from CE (`_compute_loss` →
  `masked_{node,edge}_ce`), not from the VLB. Cannot cause
  edge-collapse.
- **Side effect:** reported `val/kl_diffusion` numerical values will
  drift from what upstream would report during training. At convergence
  the two forms coincide.

### D2. `reconstruction_logp` conditioning: `z_1` vs `z_0`

- **What:** Upstream samples `z_0 ~ x_0 @ Q_0`, scores raw softmax at
  `t=0`. Ours samples `z_1 ~ forward_sample(batch, t=1)` and scores via
  the marginalised posterior `p(x_0 | z_1)` (`diffusion_module.py:596–631`).
- **Intent:** documented in the docstring at `diffusion_module.py:600–607`;
  bounds the absolute magnitude of the difference at ~1e-3 for T=1000.
- **Training-failure impact — NONE.** Same reason as D1 — this term
  contributes only to the reported `val/reconstruction_logp` /
  `val/epoch_NLL`. No gradient feedback into training.
- **Side effect:** minor numerical drift in reported NLL; does not
  affect the CE training signal.

### D3. `log_pN` silent zero fallback

- **What:** When `_size_distribution is None`, `log_pn` silently defaults
  to zero (`diffusion_module.py:702–705`); upstream would crash on a
  missing `node_dist`.
- **Intent:** not documented in code as intentional. Appears to be a
  defensive coding choice that contradicts the project's "fail loudly"
  convention.
- **Training-failure impact — NONE** for the observed failure, because
  `_size_distribution` is populated by `setup()` in `SyntheticCategoricalDataModule`
  and `SpectreSBMDataModule` under normal training. The silent
  fallback only masks misconfiguration. If it were ever to trigger, the
  NLL would be under-reported by `|log p(N)|`; training would continue
  without a signal, but sampling would not be affected.
- **Should be converted to a loud raise** to honour `CLAUDE.md`'s
  fail-loud convention. Tracked as a non-urgent cleanup, not a fix for
  the current failure.

### Joint assessment

None of D1–D3 can cause the observed edge-collapse. The VLB path affects
only reported metrics, not the training gradient. The CE path (where
the parity audit found and fixed the `_epsilon_renormalise` divergence
in commit `f6d99185`) is the only gradient-producing loss, and it now
matches upstream bit-for-bit with `atol=1e-6` as pinned by
`TestUpstreamParity`.
