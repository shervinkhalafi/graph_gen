# Section 2 review: discrete diffusion process — formal definition

## 1. Spec summary

Section 2 defines the forward Markov chain on a single categorical variable:
single-step kernel `q(z_t | z_{t-1}) = Cat(z_t; z_{t-1} Q_t)` and closed-form
cumulative kernel `q(z_t | x_0) = Cat(z_t; x_0 Q̄_t)` with
`Q̄_t = Q_1 Q_2 ⋯ Q_t`. DiGress runs one independent chain per node entry of
`X` and one per off-diagonal entry of `E`; `y` is unchanged. Time convention:
`t = 0` is clean data, `t = T` is the prior. Normalized time is `t/T ∈ [0,1]`.

---

## 2. Our implementation — file:line anchors

### Class hierarchy

We have no separate `DiscreteUniformTransition` / `MarginalUniformTransition`
classes. All transition logic lives inside `CategoricalNoiseProcess`
(`noise_process.py:719-1258`), which is a concrete `ExactDensityNoiseProcess
(nn.Module)`.

Upstream uses a two-object design: `PredefinedNoiseScheduleDiscrete` for `β_t` /
`ᾱ_t` lookup, and a `*Transition` class for `Q_t` / `Q̄_t`. We collapsed these
into `CategoricalNoiseProcess` + `NoiseSchedule`.

### Cumulative kernel `Q̄_t` — forward PMF

Upstream builds an explicit `(B, K, K)` matrix via `get_Qt_bar(alpha_bar_t)`.
Our implementation inlines the closed-form marginal directly:

```python
# noise_process.py:198-206  _mix_with_limit
def _mix_with_limit(features, alpha, limit):
    alpha = alpha.view(-1, *([1] * (features.dim() - 1)))
    limit = limit.view(*([1] * (features.dim() - 1)), limit.numel())
    return alpha * features.float() + (1.0 - alpha) * limit
```

This computes `x_0 Q̄_t` in PMF space without materialising the matrix:
`prob_X = alpha_bar_t * x_0 + (1 - alpha_bar_t) * π`.  The math is equivalent
to upstream `get_Qt_bar` under both uniform and marginal limit distributions.

### Single-step kernel `Q_t` — `_categorical_kernel`

```python
# noise_process.py:209-223  _categorical_kernel
def _categorical_kernel(identity_weight, limit):
    eye = torch.eye(classes, ...).unsqueeze(0)
    stationary_rows = limit.view(1,1,classes).expand(...)
    return identity_weight * eye + (1.0 - identity_weight) * stationary_rows
```

Called with `identity_weight = 1 - beta_t`. Matches upstream
`get_Qt(beta_t)` = `(1-β_t) I + β_t π^T`. The materialised `(B, K, K)` matrix
is used only in the posterior; it is not used in the forward sample.

### Forward sample `q(z_t | x_0)`

`CategoricalNoiseProcess.forward_sample` (`noise_process.py:874-877`):

```python
def forward_sample(self, x_0, t):
    alpha_bar = self._schedule_to_level(t)   # get_alpha_bar(t_int=t)
    return self._apply_noise(x_0, alpha_bar)
```

`_apply_noise` (`noise_process.py:939-957`) calls `_mix_with_limit` to get the
per-position PMF and then draws via `sample_discrete_features`. Upstream
`apply_noise` (`diffusion_model_discrete.py:407-442`) does the same via
`probX = X @ Qtb.X` (explicit matrix multiply). Both are equivalent when `X` is
one-hot and `Qtb.X[i,j] = alpha_bar * delta_ij + (1-alpha_bar) * pi_j`.

### Noise schedule — `NoiseSchedule`

`schedule.py:33-324`. Precomputes `T+1` entries for `betas`, `alphas`,
`alpha_bar` as registered buffers, indexed `t ∈ {0, …, T}`. Buffer construction
for `cosine_iddpm`:

```python
# schedule.py:112-120
betas_np = np.clip(betas_np, 0.0, 0.9999)
alphas_np = 1.0 - betas_np
alpha_bar_np = np.exp(np.cumsum(np.log(alphas_np)))
```

Upstream `PredefinedNoiseScheduleDiscrete` (`noise_schedule.py:44-79`)
uses `torch.cumsum(log_alpha)`. The numpy-side equivalent produces identical
values for the same beta sequence.

One distinction: upstream's buffer has length `T+1` from betas returned by
`cosine_beta_schedule_discrete(T)` (which produces `T+1` entries via
`linspace(0, T+2, T+2)` + length-1 shift). Our `NoiseSchedule` validates
`betas.shape[0] == T+1` explicitly (`schedule.py:105-110`). Shapes match.

### Time indexing

`NoiseSchedule.get_alpha_bar` (`schedule.py:226-251`) enforces `t ∈ [0, T]`
and fails loudly on out-of-range indices. At `t=0` our `alpha_bar[0] ≈ 0.9999`
(not 1.0; same as upstream). At `t=T` it is `≈ 2.4×10⁻¹⁰` for `T=1000`.
The convention `t=0 = clean, t=T = prior` is identical to the spec.

The process-state conditioning vector is `t / T` (`noise_process.py:791`),
matching upstream's `t_float = t_int / self.T`.

---

## 3. Behavioural match verdict

**Match.** All four elements of Section 2 are correctly reproduced:

| Element | Verdict |
|---|---|
| Forward single-step kernel `q(z_t \| z_{t-1})` | Match — `_categorical_kernel(1-β_t, π)` |
| Cumulative kernel `q(z_t \| x_0)` via `Q̄_t` | Match — `_mix_with_limit` inline closed form |
| Time convention `t=0` clean, `t=T` prior | Match |
| Schedule buffer size `T+1`, index range `[0,T]` | Match, validated at construction |

The collapsed (no separate `*Transition` object) architecture is a deliberate
refactor choice and does not change the mathematical object.

---

## 4. Already fixed on main

The parity commits cited (`f6d99185`, `59e9593f`, `82bcec26`) targeted the
training-loss / data-layer side. The diffusion formalism code in
`noise_process.py` and `schedule.py` was not modified in those commits and was
correct before them. No Section-2-specific fixes are in the round.

---

## 5. Remaining gaps

One minor numerical difference from the upstream is worth recording, though it
has no behavioural impact:

- **`custom_vignac` schedule `p` constant.** Upstream hardcodes `p = 4/5`
  (`diffusion_utils.py:89`), which corresponds to 5 edge classes. Our
  `custom_beta_schedule_discrete` (`diffusion_math.py:110`) computes
  `p = 1 - 1/num_edge_classes` and accepts `num_edge_classes` as a parameter.
  For binary graphs (`num_edge_classes=2`) our `p = 1/2` while upstream's `p =
  4/5`, giving a different `beta_first` floor. This schedule is not selected by
  any shipped config (the SBM experiment uses `cosine_iddpm`), so it is a
  latent numerical difference, not an active bug. It is already noted in the
  existing parity audit.

No other gaps relative to Section 2 were found.
