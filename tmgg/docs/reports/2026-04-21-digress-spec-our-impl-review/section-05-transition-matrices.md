# Section 5 review: transition matrices Q_t and Q_t_bar

Paths used below:
- `UP/` = `digress-upstream-readonly/src/diffusion/noise_schedule.py`
- `OUR/` = `src/tmgg/diffusion/noise_process.py`
- `OUR/sampling` = `src/tmgg/diffusion/diffusion_sampling.py`

---

## Spec summary (§5)

For the **marginal** variant (`MarginalUniformTransition`):

```
Q_t     = (1 - β_t) I + β_t · 1 π^T          (single-step, rows sum to 1)
Q_t_bar = ᾱ_t I + (1 - ᾱ_t) · 1 π^T          (cumulative, closed form)
```

`u_x` / `u_e` store the `(1, K, K)` PMF-row matrix; `get_Qt` takes `β_t`
shape `(B,)`, `get_Qt_bar` takes `ᾱ_t` shape `(B,)`, both return a
`PlaceHolder` with `X: (B, d_X, d_X)`, `E: (B, d_E, d_E)`, `y: (B, d_y, d_y)`.

Forward noising computes `prob = x_0 @ Q_t_bar` (shape `(B, n, d)`) and
samples from it; no matrix is materialised when that shortcut is available.

---

## Our implementation

### `_categorical_kernel` — the workhorse

`OUR/noise_process.py:209-223`:

```python
def _categorical_kernel(identity_weight: Tensor, limit: Tensor) -> Tensor:
    identity_weight = identity_weight.reshape(-1, 1, 1)     # (B, 1, 1)
    classes = limit.numel()
    eye = torch.eye(classes, ...).unsqueeze(0)              # (1, K, K)
    stationary_rows = (
        limit.view(1, 1, classes).expand(B, classes, classes)
    )                                                       # (B, K, K), each row = π
    return identity_weight * eye + (1.0 - identity_weight) * stationary_rows
```

Called with:
- `_categorical_kernel(1.0 - beta_t, limit)` → single-step `Q_t`
- `_categorical_kernel(alpha_bar_s, limit)` → cumulative `Q_s_bar`
- `_categorical_kernel(alpha_bar_t, limit)` → cumulative `Q_t_bar`

### `_mix_with_limit` — forward-noising short-circuit

`OUR/noise_process.py:198-206`:

```python
def _mix_with_limit(features, alpha, limit):
    alpha = alpha.view(-1, *([1] * (features.dim() - 1)))  # (B, 1, 1, …)
    limit = limit.view(*([1] * (features.dim() - 1)), limit.numel())
    return alpha * features.float() + (1.0 - alpha) * limit
```

Used in `_apply_noise` (forward sampling) and `forward_pmf` (analytic PMF):

```python
prob_X = _mix_with_limit(x_class, alpha_bar, x_limit)   # x_0 @ Q_t_bar inline
prob_E = _mix_with_limit(e_class, alpha_bar, e_limit)
```

This is the closed-form `x_0 @ Q_t_bar = ᾱ_t x_0 + (1 - ᾱ_t) π` without
materialising the `(B, K, K)` matrix.

---

## Per-check verdicts

### 1. Single-step Q_t formula

**Spec**: `(1 - β_t) I + β_t · 1 π^T`

**Upstream** (`UP:164-166`): `beta_t * u_x + (1 - beta_t) * eye`

**Ours**: `_categorical_kernel(1.0 - beta_t, limit)` expands to
`(1 - beta_t) * eye + beta_t * stationary_rows`

**Verdict: match.** Identical algebra; argument order in the sum is reversed
but the result is the same.

---

### 2. Cumulative Q_t_bar — closed form vs. iterative

**Spec**: closed form `ᾱ_t I + (1 - ᾱ_t) · 1 π^T`; iterative product is wrong.

**Upstream** (`UP:183-185`): `alpha_bar_t * eye + (1 - alpha_bar_t) * u_x`
— direct closed form, `alpha_bar_t` read from precomputed cumsum buffer.

**Ours**: `_categorical_kernel(alpha_bar_t, limit)` — same closed form,
same precomputed buffer lookup via `noise_schedule.get_alpha_bar(t_int=t)`.
No `Q_1 · Q_2 · ...` product anywhere.

**Verdict: match.** Both use the closed form; no drift risk from iterative
multiplication.

---

### 3. Forward noising: materialised matrix vs. `_mix_with_limit`

**Upstream** (`apply_noise`, `diffusion_model_discrete.py:432-435`):
```python
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=X.device)
probX = X @ Qtb.X          # (B, n, d_X)  — matrix IS materialised
probE = E @ Qtb.E.unsqueeze(1)  # (B, n, n, d_E)
```
The `(B, K, K)` matrix is built explicitly; the product `x_0 @ Q_t_bar` is
a literal batched matmul.

**Ours** (`_apply_noise`, `forward_pmf`):
```python
prob_X = _mix_with_limit(x_class, alpha_bar, x_limit)
prob_E = _mix_with_limit(e_class, alpha_bar, e_limit)
```
No matrix materialisation; uses the algebraic identity
`x_0 @ (ᾱ I + (1-ᾱ) 1π^T) = ᾱ x_0 + (1-ᾱ) π` directly on the
per-position PMF vectors. This is mathematically equivalent and more
memory-efficient.

**Verdict: equivalent, approach differs.** Both compute the correct
forward marginal. Our inline form avoids the `(B, K, K)` allocation.

---

### 4. Tensor shapes and batching

**Upstream**: `beta_t.unsqueeze(1)` promotes `(B,)` → `(B, 1)` before
the broadcast, yielding `Q_t: (B, K, K)`. `Q_t_bar` is the same shape.

**Ours** (`_categorical_kernel`): `identity_weight.reshape(-1, 1, 1)`
promotes to `(B, 1, 1)`, yielding `(B, K, K)`. The `_mix_with_limit`
path broadcasts differently but produces the same per-sample `(B, n, d)`
probability tensor.

**Verdict: match.** Both produce per-sample `(B, K, K)` matrices when the
matrix is materialised; `_mix_with_limit` avoids the matrix entirely and
works directly in `(B, …, K)` probability-vector space.

---

### 5. Axis handling: `x @ Q_t.T` vs. `x @ Q_t_bar`

**Spec/upstream posterior** (`diffusion_utils.py:293-305`):
```python
Qt_T = Qt.transpose(-1, -2)    # (B, d, d)
left_term = X_t @ Qt_T         # (B, N, d) — z_t row-multiplies Q_t^T
right_term = Qsb.unsqueeze(1)  # used for x_0 @ Q_s_bar
```
Forward direction: `x_0 @ Q_t_bar` (row-vector × matrix, last axis of
`x_0` against second-to-last axis of `Q_t_bar`). Posterior left term:
`z_t @ Q_t^T` (transpose is along the two last dims of the kernel,
`(-1, -2)`).

**Ours** (`OUR/sampling:172-175`):
```python
Qt_M_T = torch.transpose(Qt_M, -2, -1)  # identical transpose axes
left_term = M_t @ Qt_M_T
right_term = M @ Qsb_M
```
`_mix_with_limit` computes the forward direction as a weighted sum, which
is algebraically `x @ Q_t_bar` without the explicit matmul; the limit
vector broadcasts onto the last axis, i.e. the same axis that `Q_t_bar`
would be applied to.

**Verdict: match.** No axis inversion anywhere. Transpose axes `(-1, -2)`
are correct and consistent on both sides.

---

### 6. Row-sum invariant

Spec notes every row of `Q_t` and `Q_t_bar` must sum to 1; upstream
`apply_noise` asserts this.

In ours: `_categorical_kernel` constructs rows as convex combinations of
a row of the identity (sums to 1) and a row of the PMF matrix (sums to 1
by construction), so row sums are preserved algebraically. `_mix_with_limit`
constructs convex combinations of `x_0` (already a PMF row) and `π` (a
PMF), so the forward-process probabilities also sum to 1. No explicit
assertion exists in our code, but the algebraic invariant holds.

**Verdict: match.** Row-sum property preserved algebraically; missing
assertion is a minor hardening gap, not a correctness defect.

---

## Summary

| Check | Verdict |
|---|---|
| `Q_t` formula `(1-β)I + β·1π^T` | match |
| `Q_t_bar` closed form (no iterative product) | match |
| Forward noising: `x @ Q_t_bar` | equivalent (our path avoids materialising the matrix) |
| Tensor shapes `(B, K, K)` | match |
| Axis ordering: `x @ Q_t.T` in posterior, `x @ Q_t_bar` in forward | match |
| Row-sum invariant | algebraically preserved; no runtime assert |

No functional divergence found between our implementation and the spec for
the marginal-transition case. The only structural difference is the
`_mix_with_limit` optimisation that bypasses the `(B, K, K)` matrix
construction during forward noising — this is mathematically equivalent
and is consistent with the identity `x_0 @ (ᾱ I + (1-ᾱ) 1π^T) = ᾱ x_0 + (1-ᾱ) π`.

The `denom.clamp(min=1e-6)` in `OUR/sampling:180` (posterior computation)
remains the only numerical difference relative to upstream; the prior audit
(`02-noise-process.md §5`) already classified this as low-risk and inert
for healthy posteriors.
