# Section 3 Review: Noise Schedule

**Config under review:** `schedule_type=cosine_iddpm`, `timesteps=1000`
**Files:** `src/tmgg/diffusion/schedule.py`, `src/tmgg/diffusion/diffusion_math.py`
**Reference:** `digress-upstream-readonly/src/diffusion/noise_schedule.py` (class `PredefinedNoiseScheduleDiscrete`) and `src/diffusion/diffusion_utils.py`

---

## Summary verdict

**Match.** For the `cosine_iddpm` / `T=1000` path, the numerical output of our implementation is bit-for-bit identical to upstream at every spot-checked timestep. All five behavioural properties tested below pass. The 1e-5 alpha_bar clamp flagged by the earlier Haiku audit is **not active** on the cosine path; it is confined to the `linear_ddpm` branch and is irrelevant to SBM runs.

One structural divergence exists (upstream's `alphas` / `alphas_bar` are plain module attributes, not buffers), but it has no runtime effect. All other divergences are stricter or more explicit on our side.

---

## 1. Spec summary (Section 3)

The spec requires:

- Closed-form cosine schedule: `alpha_bar[t] = cos²(π/2 · (t/T_ext + s)/(1+s)) / cos²(π/2 · s/(1+s))` with `T_ext = T+2`, `s = 0.008`.
- `beta[t] = 1 - alpha_bar[t]/alpha_bar[t-1]`.
- `T+1` precomputed entries (indices `0..T`).
- Three precomputed buffers: `betas`, `alphas`, `alpha_bar` (upstream stores only `betas` as a buffer; the other two as attributes).
- Lookups via `get_alpha_bar(t_normalized)` and `forward(t_normalized)`, converting normalized time via `round(t_normalized * T)`.

---

## 2. Our implementation

**Formula** (`diffusion_math.py:60–71`):

```python
def cosine_beta_schedule_discrete(timesteps: int, s: float = 0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()   # shape (T+1,)
```

**Buffer registration** (`schedule.py:112–120`):

```python
def _register_schedule_from_betas(self, betas_np):
    betas_np = np.clip(betas_np, 0.0, 0.9999)
    alphas_np = 1.0 - betas_np
    alpha_bar_np = np.exp(np.cumsum(np.log(alphas_np)))
    self.register_buffer("betas", ...)
    self.register_buffer("alphas", ...)
    self.register_buffer("alpha_bar", ...)
```

**Shape invariant** (`schedule.py:105–110`): asserts `betas.shape[0] == T+1`; raises `ValueError` if violated.

**Lookup** (`schedule.py:253–291`, `_resolve_t` + `_resolve_index`):

```python
t_int = torch.round(t_normalized * self._timesteps)
# bounds-check: lo >= 0 and hi <= T
return self.betas[flat].reshape(shape)
```

---

## 3. Behavioural match per check

### 3.1 Cosine formula, s=0.008

**Match.** Our `diffusion_math.py:65–70` is character-for-character identical to upstream `diffusion_utils.py:67–73`. Both use `T_ext = T+2`, normalise by `alphas_cumprod[0]`, compute per-step alphas by ratios, and return `1 - alphas`.

### 3.2 Numerical spot-check at T=1000

Verified by running both paths on the same numpy arrays:

| t | alpha_bar (upstream) | alpha_bar (ours) | diff |
|---|----------------------|------------------|------|
| 0 | 0.9999587594 | 0.9999587594 | 0 |
| 500 | 0.4930651514 | 0.4930651514 | 0 |
| 1000 | 2.42e-10 | 2.42e-10 | 0 |

Beta values at the same points: `beta[0] = 4.12e-5`, `beta[500] = 3.15e-3`, `beta[1000] = 1.0` (before clamping).

### 3.3 Buffer registration and precomputation

**Structural divergence (benign).** Upstream registers only `betas` as a buffer; `alphas` and `alphas_bar` are plain `Tensor` attributes recomputed at `__init__`. We register all three as buffers. Both are device-safe and checkpoint-safe: upstream's non-buffer tensors are computed from the buffer at init and follow the module on `.to(device)` calls because they are already on the correct device at that point — though if someone calls `.to()` after construction they would not move. Our approach is strictly safer for late device moves.

### 3.4 Index range and bounds behaviour

**Match (ours stricter).** Both produce `T+1 = 1001` buffer entries for `T=1000`, indexed `[0, T]`. Upstream performs no bounds checking and would raise a bare `IndexError` for out-of-range indices. We check explicitly in `_resolve_index` (`schedule.py:191–196`) and raise an informative `ValueError`. In `_resolve_t` we also validate `t_normalized ∈ [0, 1]` before rounding. No functional divergence in the normal case; we fail faster and more clearly.

### 3.5 Numerical safeguards — the 1e-5 clamp

**The Haiku audit's concern does not apply to the cosine path.** The `1e-5` floor on `alpha_bar` appears only at `schedule.py:146`, inside `_register_schedule_from_alpha_bar`. That method is only called for `schedule_type="linear_ddpm"` (`schedule.py:96–102`). The cosine path calls `_register_schedule_from_betas` (`schedule.py:87`), which clips betas at `[0, 0.9999]` via `np.clip`, exactly as upstream does via `torch.clamp(self.betas, min=0, max=0.9999)`.

One detail: the last beta (`t=T=1000`) from the cosine schedule is exactly `1.0`, which the `0.9999` clamp reduces to `0.9999`. This makes `alpha_bar[T] ≈ 2.4e-10` rather than zero. Both upstream and ours apply the same `0.9999` cap, so the outputs match.

### 3.6 Float vs integer t

**Match.** Both accept either normalised float (`t_normalized ∈ [0, 1]`) or integer (`t_int`). Both convert via `round(t_normalized * T)`. We expose both via explicit keyword arguments with an XOR-validity check; upstream uses a positional `assert`. Rounding semantics are identical.

---

## 4. Remaining gaps

1. **`alphas` not a registered buffer upstream.** Upstream's `self.alphas` is a plain attribute (`noise_schedule.py:62`). Ours is a buffer. No numerical impact; ours is safer for late `.to(device)` calls.

2. **`custom_vignac` num_edge_classes parametrisation.** Upstream's `custom_beta_schedule_discrete` hardcodes `p = 4/5` (`diffusion_utils.py:89`), corresponding to `K=5` edge classes. Our version takes `num_edge_classes` as a parameter and computes `p = 1 - 1/K` (`diffusion_math.py:110`). For binary graphs (`K=2`, the SBM case) ours uses `p=0.5`, upstream uses `p=0.8`. This is a real divergence for the `custom_vignac` schedule but **does not affect SBM runs**, which use `cosine_iddpm`.

3. **`linear_ddpm` 1e-5 alpha_bar floor.** Present in our code at `schedule.py:146` for the linear schedule only. No upstream equivalent exists. Not relevant to any active experiment but documented for completeness.
