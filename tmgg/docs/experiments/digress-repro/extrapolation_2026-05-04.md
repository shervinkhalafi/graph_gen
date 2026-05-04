# Trajectory extrapolation to upstream step budget — 2026-05-04

Fit `metric = a + b·ln(step)` to the observed eval-all trajectories
and extrapolate to upstream-equivalent total step counts (96k QM9,
1.85M MOSES, 39.7M GuacaMol). Log-linear is the simplest defensible
fit shape; it's what the residual structure looks like on the
existing samples (R² > 0.92 in all three cases). The fit
extrapolates **outside the sampled range by 18× (MOSES) and 158×
(GuacaMol)**, so the predictions below are upper-bound speculation,
not predictions in any rigorous sense.

## Numbers

| metric | range fit on | log-linear fit | predicted @ upstream-equivalent | paper anchor | gap remaining |
|---|---|---|---|---|---|
| MOSES FCD ↓ | 5k–100k (10 ckpts) | y = 66.95 − 4.11·ln(s); R²=0.926 | **7.65** @ 1.85M | 1.19 | **6.4× still off** |
| GuacaMol FCD-ChEMBL score ↑ (direct fit) | 10k–250k (6 ckpts) | y = −0.205 + 0.022·ln(s); R²=0.926 | **0.181** @ 39.7M | 0.68 | **3.7× still off** |
| GuacaMol KL_div ↑ | 10k–250k (6 ckpts) | y = 0.069 + 0.060·ln(s); R²=0.922 | 1.12 (clamps at 1.0) @ 39.7M | 0.929 | **would match or exceed** |

(Plot: `extrapolation_2026-05-04.png`. The fit on GuacaMol raw FCD
goes negative at 39.7M steps because raw FCD must be ≥0; the linear
fit fails at the tail. The "score" column above is the
direct-on-score fit, which stays in [0,1] but also asymptotes
sub-linearly in reality.)

## Interpretation

1. **MOSES would close most but not all of the paper gap.** From
   17.5× off (FCD 20.8 vs 1.19) at our 100k-step endpoint, an
   additional 18× more training drops the predicted FCD to ~7.65,
   which is **6.4× still off**. The remaining gap is most plausibly
   attributed to the architecture drift (n_layers 8 vs 12,
   dim_ffE 64 vs 128, dy 64 vs 128) and to the missing
   `extra_features='all'` block. Step budget alone is not the whole
   story.

2. **GuacaMol score would only get to ~0.18 (vs paper 0.68).** The
   direct-on-score fit predicts a 2-3× improvement from 0.076 to
   ~0.18 at 39.7M steps — closing only one third of the way to
   paper. GuacaMol is currently the most architecturally
   under-resourced run (n_layers 8 vs 12, ¼ batch, no
   extra_features). At 39.7M steps the residual gap is dominated
   by everything except step count.

3. **GuacaMol KL_div is the one metric that would plausibly match
   or exceed paper** at the extrapolated endpoint. The trajectory
   is already at 0.83 (paper 0.929) and the slope hasn't visibly
   plateaued yet. Note KL_div in DiGress is normalised to [0,1] so
   a fit predicting 1.12 just means the curve saturates near 1.0
   long before then.

## Caveats

- **Log-linear is optimistic for steep, early-trajectory data.** Deep-net
  losses commonly flatten faster than logarithmic at large step
  counts; the FCD predictions above are very likely an
  *upper bound* on improvement (i.e. the actual endpoint at
  upstream budget would be *worse* than 7.65 / 0.18).
- **Extrapolating 18-158× past the sampled range is unreliable.**
  These should be read as "directionally what step-budget alone
  buys", not "what the model would actually score".
- **Fit ignores all confounders**: the batch size mismatch
  (¼ upstream on QM9/MOSES), the LR override (5× upstream), and
  the missing extra_features block all matter for absolute
  endpoint prediction but are baked into the trajectory we fit.

## Bottom line

Just **matching upstream's step budget would not close the paper
gap** for FCD on either MOSES or GuacaMol. To reach paper-anchor
performance, we need *both*:
- the upstream step budget (or close to it), AND
- the upstream architecture (n_layers, hidden_dims), AND
- `extra_features='all'`, AND
- the upstream optimizer (lr=2e-4, not the 1e-3 currently used).

The yaml fix (next commit) addresses items 2–4 so the gap to paper
becomes *only* about training duration, not about training duration
*plus* model-class drift.
