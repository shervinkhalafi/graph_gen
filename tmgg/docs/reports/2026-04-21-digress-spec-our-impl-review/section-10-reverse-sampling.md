# Section 10: Reverse Sampling Chain — Implementation Review

**Spec source:** `docs/reports/2026-04-21-digress-upstream-spec.md §10`
**Upstream ref:** `digress-upstream-readonly/src/diffusion_model_discrete.py:491-655`,
`src/diffusion/diffusion_utils.py:233-394`
**Our code:** `src/tmgg/diffusion/sampler.py`, `diffusion_sampling.py`, `noise_process.py`

---

## Summary verdict

Six of seven checks pass with exact parity. One genuine behavioural divergence
(missing post-contraction renormalisation) is confirmed benign under
`torch.multinomial`. One upstream invariant (`assert` after every reverse step)
is absent on our side — a modest defensive gap, not a correctness bug.

| # | Check | Verdict |
|---|-------|---------|
| 1 | `z_T ~ π` per-sample per-position via `multinomial(1)` | MATCH |
| 2 | Reverse step draws from posterior categorical (no argmax) | MATCH |
| 3 | Q-index: `Qt` at t, `Qsb = Qt_bar[s]`, `Qtb = Qt_bar[t]` | MATCH |
| 4 | Posterior mixing formula shape and axis contractions | MATCH, minor divergence on final normalisation |
| 5 | Edge symmetry enforcement on each reverse sample | PARTIAL — prior draw asserts, per-step does not |
| 6 | No module-scope RNG / `torch.manual_seed` in sampler path | MATCH |
| 7 | `@torch.no_grad` wrapping on the full reverse chain | MATCH |

---

## Per-check detail

### 1. Initial `z_T ~ π`

Upstream `sample_discrete_feature_noise` (`diffusion_utils.py:366-394`) tiles
the 1-D limit PMF to `(bs, n, d)` / `(bs, n, n, d)` via stride-0 `expand`, then
calls `.flatten(end_dim=-2).multinomial(1)`, giving `bs*n` and `bs*n*n`
independent draws. Upper triangle of the edge tensor is kept; the lower is
overwritten by the transpose, then a hard symmetry assertion fires.

Our `sample_discrete_feature_noise` (`diffusion_sampling.py:187-243`) is
line-for-line identical in structure: same `expand` recipe, same `flatten →
multinomial(1) → reshape`, same `triu_indices` mask + transpose. An
`AssertionError` is raised (not an `assert` statement) on
`diffusion_sampling.py:235-236` when the result is non-symmetric. `sample_prior`
on `CategoricalNoiseProcess` (`noise_process.py:869-872`) calls this after
pulling `(_limit_x, _limit_e, _limit_y)` from registered buffers. The sampler
constructs `node_mask` as `arange < n_nodes.unsqueeze(1)` giving shape `(bs,
n_max)` — not `(1, n_max)` — before calling `sample_prior`
(`sampler.py:295-301`).

**Verdict: exact match.** Prior audits (01-sampler check 1, review-01 claim 1)
confirmed this in detail.

---

### 2. Per-step reverse: `multinomial(1)`, no argmax

Upstream `sample_p_zs_given_zt` (`diffusion_model_discrete.py:644-647`) calls
`sample_discrete_features` which runs `.multinomial(1)` on a `(bs*n, dx)` row
view for nodes and a `(bs*n*n, de)` row view for edges
(`diffusion_utils.py:248,262`). The final step at `s == 0` goes through the
same path — there is no argmax branch.

Our `posterior_sample_from_model_output` on `CategoricalNoiseProcess`
(`noise_process.py:1153-1169`) routes to `posterior_sample_marginalised`
(`1119-1151`), which calls `sample_discrete_features`
(`diffusion_sampling.py:13-68`): `.multinomial(1)` on `(bs*n, dx)` and
`(bs*n*n, de)` views, followed by `F.one_hot`. No argmax, no `s == 0` special
case. Confirmed by grep: zero matches for `argmax|\.mode(|top_?k` across
`src/tmgg/diffusion/`.

**Verdict: exact match.** Prior audits (01-sampler check 2, review-01 claim 2)
confirmed.

---

### 3. Q-index for `Qt` / `Qsb` / `Qtb` at step t → t-1

Upstream (`diffusion_model_discrete.py:601-608`):
```python
beta_t  = noise_schedule(t_normalized=t)
Qsb     = transition_model.get_Qt_bar(alpha_s_bar, ...)   # s = t - 1
Qtb     = transition_model.get_Qt_bar(alpha_t_bar, ...)   # t
Qt      = transition_model.get_Qt(beta_t, ...)            # per-step at t
```
`Qsb` is the cumulative kernel at `s`, not at `t`; the spec names this correctly.

Our `_posterior_probabilities_marginalised` (`noise_process.py:1075-1085`):
```python
beta_t       = noise_schedule(t_int=t)
alpha_bar_s  = noise_schedule.get_alpha_bar(t_int=s)      # s
alpha_bar_t  = noise_schedule.get_alpha_bar(t_int=t)      # t
q_x          = _categorical_kernel(1.0 - beta_t, x_limit) # Qt
qsb_x        = _categorical_kernel(alpha_bar_s, x_limit)  # Qsb = Qt_bar[s]
qtb_x        = _categorical_kernel(alpha_bar_t, x_limit)  # Qtb = Qt_bar[t]
```
Indices are aligned: `Qsb` uses `alpha_bar` at `s` (the target step), `Qtb`
uses it at `t` (the source step), and `Qt` is the single-step kernel at `t`.
Both the direct form (`_posterior_probabilities`, `noise_process.py:990-999`)
and the marginalised form use the same three lookups with the same semantics.
The sampler always passes `s = t - 1` (`sampler.py:305-308`).

**Verdict: exact match.** Confirmed by candidate-5-posterior-mixing.md, which
found no index-level divergence.

---

### 4. Posterior mixing formula — dim contractions and transposes

Upstream `compute_batched_over0_posterior_distribution`
(`diffusion_utils.py:293-321`) for the marginalised form:
```python
X_t  → flatten (bs, N, dt)
left  = (X_t @ Qt.T).unsqueeze(2)       # (bs, N, 1, d_{t-1})
right = Qsb.unsqueeze(1)                # (bs, 1, d0, d_{t-1})
num   = left * right                    # (bs, N, d0, d_{t-1})
denom = (Qtb @ X_t.T).T.unsqueeze(-1)  # (bs, N, d0, 1);  [denominator==0]=1e-6
out   = num / denom                     # (bs, N, d0, d_{t-1})
```
Contraction over `x_0` in `sample_p_zs_given_zt` (`lines 629-638`):
```python
weighted = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # (bs, n, d0, d_{t-1})
unnorm   = weighted.sum(dim=2)                          # (bs, n, d_{t-1})
unnorm[sum(...)==0] = 1e-5
prob_X   = unnorm / sum(unnorm, dim=-1, keepdim=True)   # normalised
```

Our `compute_posterior_distribution_per_x0`
(`diffusion_sampling.py:71-130`): identical dim contractions — same flattening,
same `Qt.T` matmul, same `unsqueeze(2)` / `unsqueeze(1)` for the outer product,
same `(Qtb @ M_t.T).T.unsqueeze(-1)` denominator with `[==0]=1e-6`.

Contraction in `_posterior_probabilities_marginalised`
(`noise_process.py:1102-1106`):
```python
prob_X = (per_x0_X * pred_X_flat.unsqueeze(-1)).sum(dim=2)
```
Same axis (`dim=2` = the `d0` axis), same broadcast.

**Divergence:** upstream follows the contraction with an explicit renormalisation
(`unnorm / sum(dim=-1)`) and a `sum==0 → 1e-5` zero-guard; we do neither. As
established in prior audits (01-sampler check 3, review-01 claim 4),
`torch.multinomial` is scale-invariant per row, so the sampling distribution
is unchanged. The practical difference is that a genuinely all-zero posterior
row triggers an uninformative `multinomial` error on our side rather than
upstream's graceful `1e-5` fallback. There is also no row-sum assertion
(upstream asserts `(prob_X.sum(dim=-1) - 1).abs() < 1e-4` at
`diffusion_model_discrete.py:641-642`).

**Verdict: match on tensor mechanics; benign divergence on post-contraction
normalisation and zero-guard.** Confirmed across all three prior audits.

---

### 5. Edge symmetry enforcement on each reverse sample

Upstream enforces symmetry at two points:
1. After the prior draw: hard `assert (U_E == U_E.transpose(1,2)).all()` in
   `sample_discrete_feature_noise` (`diffusion_utils.py:392`).
2. After every reverse step: hard `assert (E_s == E_s.transpose(1,2)).all()`
   in `sample_p_zs_given_zt` (`diffusion_model_discrete.py:649`).

And additionally, `PlaceHolder.mask(collapse=False)` asserts
`torch.allclose(self.E, E.transpose(1,2))` (`src/utils.py:130`) on every
masking call, which fires implicitly after each step.

Our prior draw: `sample_discrete_feature_noise` raises an `AssertionError`
(not a bare `assert`) at `diffusion_sampling.py:235-236` when non-symmetric.
This is equivalent in behaviour to upstream's `assert` at that stage.

Our per-step: `posterior_sample_marginalised` (`noise_process.py:1119-1151`)
calls `sample_discrete_features` → `triu + transpose` to enforce symmetry
structurally, then calls `z_t.replace(...).mask()`. `GraphData.mask()` does not
assert symmetry — there is no equivalent of upstream's
`assert (E_s == E_s.transpose(1,2))` after each reverse step or of the
`PlaceHolder.mask(collapse=False)` check.

**Verdict: partial match.** Structural symmetry is guaranteed by construction
(triu + transpose), but the runtime assertion after each reverse step is absent.
This is a defensive-gap difference, not a correctness bug — the symmetry is
enforced algebraically. Prior audits did not flag this; it is new to this review.

---

### 6. No module-scope RNG / `torch.manual_seed` in sampler path

Upstream: no `torch.manual_seed` or `torch.Generator` anywhere under
`digress-upstream-readonly/src/diffusion*` or `diffusion_model_discrete.py`.

Ours: grep across `src/tmgg/diffusion/` returns zero matches for
`manual_seed|torch.Generator|torch.random`. `Sampler` has no `__init__` and no
instance tensor attributes; `z_t` is a local rebind in `sample`.
`_BufferingCollector` stores only `dict[str, float]`. The one-time global
`manual_seed` in `training/orchestration/run_experiment.py:32-34` is equivalent
to upstream's Lightning `seed_everything` and is not re-invoked between
`Sampler.sample` calls.

**Verdict: exact match.** Prior audits (01-sampler check 5, review-01 claim 3)
confirmed.

---

### 7. `@torch.no_grad` wrapping

Upstream `sample_batch` is decorated `@torch.no_grad()` at
`diffusion_model_discrete.py:491`.

Our `Sampler.sample` is decorated `@torch.no_grad()` at `sampler.py:236`.
The decorator covers the full reverse loop including model forward passes,
posterior sampling, and `finalize_sample`.

**Verdict: exact match.**

---

## Remaining gaps

**Benign divergence — missing post-contraction renorm and zero-guard
(`_posterior_probabilities_marginalised`, `noise_process.py:1104-1106`).**
Behaviourally equivalent for `multinomial`; loses the `sum==0 → 1e-5` graceful
fallback and the upstream row-sum assertion. A zero-row edge case would produce
a hard `multinomial` error rather than a silent fallback to near-uniform.

**Defensive gap — no per-step symmetry assertion after reverse steps.** Upstream
fires `assert (E_s == E_s.transpose(1,2)).all()` inside `sample_p_zs_given_zt`
after every step (`diffusion_model_discrete.py:649`). We rely on the structural
triu+transpose guarantee in `sample_discrete_features` without a runtime check.
`GraphData.mask()` does not assert symmetry. No equivalent of
`PlaceHolder.mask(collapse=False)`'s `allclose` check exists on the reverse
path.
