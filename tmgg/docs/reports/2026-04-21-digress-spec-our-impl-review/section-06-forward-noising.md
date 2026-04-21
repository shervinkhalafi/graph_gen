# Section 6 review: Forward noising `apply_noise`

Upstream reference: `digress-upstream-readonly/src/diffusion_model_discrete.py:407-442`
and `src/diffusion/diffusion_utils.py:233-266`.

Our implementation: `src/tmgg/diffusion/noise_process.py:CategoricalNoiseProcess`,
`src/tmgg/diffusion/diffusion_sampling.py:sample_discrete_features`,
`src/tmgg/training/lightning_modules/diffusion_module.py:training_step`.

---

## Spec summary (§6)

1. Sample `t_int ~ Uniform{lowest_t, ..., T}` per batch element as shape `(B, 1)`,
   where `lowest_t = 0` in training and `1` in eval. Compute `s_int = t_int - 1`.
2. Normalise, look up `beta_t`, `alpha_s_bar`, `alpha_t_bar`.
3. Compute forward marginal: `probX = X @ Qtb.X`, `probE = E @ Qtb.E.unsqueeze(1)`.
4. Sample one-hot via `sample_discrete_features` (multinomial + symmetrise E via
   `triu(diagonal=1) + transpose`).
5. Return a dict with keys `{t_int, t, beta_t, alpha_s_bar, alpha_t_bar, X_t, E_t, y_t, node_mask}`,
   all schedule tensors at shape `(B, 1)`.

---

## Our implementation path

**`training_step`** (`diffusion_module.py:497`):

```python
t_int = torch.randint(1, self.T + 1, (bs,), device=device)
z_t = self.noise_process.forward_sample(batch, t_int)
```

**`CategoricalNoiseProcess.forward_sample`** (`noise_process.py:874-877`):
delegates immediately to `_apply_noise(x_0, alpha_bar_t)` where
`alpha_bar_t = schedule.get_alpha_bar(t_int=t)`.

**`_apply_noise`** (`noise_process.py:939-957`):

```python
prob_X = _mix_with_limit(x_class, noise_level, x_limit)  # alpha_bar * x + (1-alpha_bar) * pi
prob_E = _mix_with_limit(e_class, noise_level, e_limit)
x_idx, e_idx = sample_discrete_features(prob_X, prob_E, data.node_mask)
x_one_hot = F.one_hot(x_idx, num_classes=self.x_classes).float()
e_one_hot = F.one_hot(e_idx, num_classes=self.e_classes).float()
return data.replace(X_class=x_one_hot, E_class=e_one_hot).mask()
```

**`sample_discrete_features`** (`diffusion_sampling.py:13-68`): clones, sets
masked rows to uniform, flattens to `(bs*n, dx)`, calls `multinomial(1)`, then
applies `triu(diagonal=1) + transpose` for E symmetry.

---

## Per-check verdicts

### Check 1: `lowest_t` and t=0 training rule

**Upstream:** `lowest_t = 0 if self.training else 1`, so training can sample `t=0`
and the `t=0` term is included in the cross-entropy loss. Eval uses `lowest_t=1`
exclusively; the `t=0` contribution is computed separately via `reconstruction_logp`.

**Ours:** `training_step` hard-codes `torch.randint(1, self.T + 1, ...)`, so
`t=0` is **never sampled**, not even in training. There is no `reconstruction_logp`
analogue in `DiffusionLightningModule`.

**Verdict: divergence.** Training always skips `t=0`. For the CE training loss this
is a minor omission — the `t=0` term contributes roughly `1/T` of the expectation —
but it means our training objective is not identical to upstream's. During validation
the same `randint(1, T+1)` is used; the VLB path in `compute_val_loss` in the legacy
`DiGressDiffusionModule` (if used) would also need to handle the `reconstruction_logp`
term separately, which it does not appear to do via this training-step path.

### Check 2: t_int shape `(B, 1)` vs `(B,)`

**Upstream:** `t_int` is shape `(B, 1)` (explicit `size=(X.size(0), 1)`). Schedule
lookups use `t_normalized = t_int / T` and `get_alpha_bar(t_normalized)`, where the
schedule rounds `t_normalized * timesteps`.

**Ours:** `t_int` is shape `(B,)`. `NoiseSchedule.get_alpha_bar(t_int=t)` accepts a
1-D tensor and its `_resolve_index` handles the shape via `.reshape(-1)` and
`.long()` (verified in `schedule.py`). `_mix_with_limit` broadcasts alpha via
`alpha.view(-1, *([1] * (features.dim() - 1)))`, so shape `(B,)` is handled
correctly for 3-D node tensors `(B, n, dx)` and 4-D edge tensors `(B, n, n, de)`.

**Verdict: no numerical impact.** The shape difference `(B, 1)` vs `(B,)` does not
affect per-sample independence or the schedule lookup values. Both produce one
independent draw per batch element.

### Check 3: Forward marginal computation — `x @ Qtb` vs `_mix_with_limit`

**Upstream:** explicitly forms `Qtb = get_Qt_bar(alpha_t_bar)` as a `(B, dx, dx)`
matrix, then computes `probX = X @ Qtb.X` (matrix multiply). For the marginal
transition `Qtb = alpha_bar * I + (1 - alpha_bar) * 1*pi^T`, so
`X @ Qtb = alpha_bar * X + (1 - alpha_bar) * pi` (since `X @ 1 = 1` for one-hot X).

**Ours:** `_mix_with_limit(x_class, alpha_bar, x_limit)` computes
`alpha_bar * x + (1 - alpha_bar) * pi` directly without materialising `Qtb`.

**Verdict: mathematically equivalent.** The `x @ Qtb` path and `_mix_with_limit`
produce the same values for both uniform and marginal stationary distributions, given
valid one-hot inputs. No divergence.

### Check 4: E symmetry enforcement

**Upstream** `sample_discrete_features` (`diffusion_utils.py:263-264`):
```python
E_t = torch.triu(E_t, diagonal=1)
E_t = (E_t + torch.transpose(E_t, 1, 2))
```
Samples all `n×n` entries, zeroes the lower triangle and diagonal, then mirrors.

**Ours** (`diffusion_sampling.py:65-66`): identical pattern — `triu(diagonal=1)` then
`+ transpose`. The masks for invalid positions (off-`node_mask` and diagonal) are set
to uniform before the multinomial draw, and the `triu + transpose` step follows
immediately after.

**Verdict: match.** Both enforce symmetry via the same upper-triangular mirror.
`sample_discrete_feature_noise` (prior sampling) uses a slightly different but
equivalent approach — explicit `triu_indices` mask on the one-hot tensor — and raises
`AssertionError` if the result is not symmetric.

### Check 5: Return structure

**Upstream** returns a plain dict:
```python
{'t_int', 't', 'beta_t', 'alpha_s_bar', 'alpha_t_bar', 'X_t', 'E_t', 'y_t', 'node_mask'}
```
all schedule scalars at shape `(B, 1)`.

**Ours:** `forward_sample` returns a `GraphData` object with `X_class`, `E_class`,
`node_mask`, and `y` set. The schedule values (`beta_t`, `alpha_s_bar`,
`alpha_t_bar`) are **not** embedded in the return value. The training step gets them
by calling `noise_process.process_state_condition_vector(t_int)` to obtain the time
conditioning, and the loss does not consume `beta_t` or `alpha_s_bar` (those are only
needed for VLB computation in `compute_val_loss`, not for CE training loss).

**Verdict: structural divergence, not a bug for training.** The CE training loss
does not require `beta_t` / `alpha_s_bar` from `forward_sample`, so the missing
schedule keys do not affect training correctness. The VLB path (validation) would
need these, and `DiffusionLightningModule` handles that separately via its own
`compute_val_loss` call which re-looks up the schedule given the returned `t_int`.
That path needs auditing separately (§11 review).

---

## Summary

| Check | Verdict |
|-------|---------|
| `lowest_t` / t=0 training rule | **divergence** — we never sample `t=0`, upstream does in training |
| `t_int` shape `(B,)` vs `(B, 1)` | no impact — broadcast handles it |
| forward marginal `X @ Qtb` vs `_mix_with_limit` | equivalent |
| E symmetry via `triu + transpose` | match |
| return structure (dict vs `GraphData`) | structural difference, not a bug for CE loss |

---

## Remaining gaps

1. **`t=0` training term.** Our `randint(1, T+1)` permanently excludes `t=0`.
   Upstream's CE loss includes it (the model learns to denoise nearly-clean graphs);
   we skip that `1/T` fraction. This is a minor but real deviation from the DiGress
   training objective. Whether to add it requires a design decision: it requires
   either special-casing `t=0` in `forward_sample` (no noise applied) or accepting
   the `alpha_bar[0] ≈ 0.99996` near-identity case as "close enough."

2. **Validation `lowest_t`.** Upstream sets `lowest_t=1` for eval and handles `t=0`
   via `reconstruction_logp` in `compute_val_loss`. Our validation step also uses
   `randint(1, T+1)`, which aligns with `lowest_t=1`, but we have no
   `reconstruction_logp` term. This means our VLB estimator is missing the `L_0`
   term if/when `compute_val_loss` is used (see §11 review).

3. **`beta_t` / `alpha_s_bar` not returned.** These are absent from `GraphData`.
   Any caller that needs them (VLB path) must re-derive them from `t_int`, which
   requires keeping `t_int` available. The current architecture does this, but it
   introduces a silent coupling between the training step and the loss module.
