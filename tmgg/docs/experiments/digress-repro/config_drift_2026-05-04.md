# Config drift: `discrete_{qm9,moses,guacamol}_digress_repro` vs upstream cvignac/DiGress

Cloned `cvignac/DiGress@main` to `/tmp/digress_upstream/` 2026-05-04
and diffed `configs/experiment/{qm9_no_h,moses,guacamol}.yaml` plus
the resolved upstream defaults (`train/train_default.yaml`,
`model/discrete.yaml`, `general/general_default.yaml`,
`dataset/{qm9,moses,guacamol}.yaml`) against our
`src/tmgg/experiments/exp_configs/experiment/discrete_*_digress_repro.yaml`
and the `discrete_sbm_official` model preset they inherit.

## TL;DR

The pickup's hypothesis "configs ~4-8× lighter than upstream" is
right in spirit but **understated for MOSES and dramatically
understated for GuacaMol** when measured by total training-step
budget. Architecture is mildly under-spec'd (n_layers, dim_ffE,
dim_ffy on the MOSES/GuacaMol side); diffusion timesteps are
*over*-spec'd (1000 vs 500); `extra_features="all"` (a load-bearing
upstream knob) is **not configured** in any of our repro yamls; and
the wandb-run-name field `lr1e-3_wd1e-4` does not match the YAML
(`lr=2e-4, wd=1e-12`), so a launch-time CLI override drifts both
optimizer hyperparameters off the YAML record.

## Field-by-field comparison

### Architecture

| field | upstream QM9 | upstream MOSES | upstream GuacaMol | ours (all 3) | drift |
|---|---|---|---|---|---|
| `n_layers` | **9** | **12** | **12** | **8** | -1 / -4 / -4 |
| `hidden_dims.dx` | 256 | 256 | 256 | 256 | match |
| `hidden_dims.de` | 64 | 64 | 64 | 64 | match |
| `hidden_dims.dy` | 64 | **128** | **128** | **64** | match / -64 / -64 |
| `hidden_dims.n_head` | 8 | 8 | 8 | 8 | match |
| `hidden_dims.dim_ffX` | 256 | 256 | 256 | 256 | match |
| `hidden_dims.dim_ffE` | 128 | **128** | **128** | **64** | -64 across the board |
| `hidden_dims.dim_ffy` | 128 | **256** | **256** | **256** | +128 / match / match |
| `hidden_mlp_dims.X` | 256 | 256 | 256 | **128** | -128 across the board |
| `hidden_mlp_dims.E` | 128 | 128 | 128 | **64** | -64 across the board |
| `hidden_mlp_dims.y` | 128 | 256 | 256 | **128** | match / -128 / -128 |

Net: ours is meaningfully smaller on MOSES/GuacaMol (lighter MLP
heads, narrower edge feed-forward, four fewer transformer layers),
roughly comparable on QM9.

### Diffusion + features

| field | upstream | ours | drift |
|---|---|---|---|
| `diffusion_steps` | 500 (all 3) | **1000** (`discrete_sbm_official::noise_schedule.timesteps`) | **2× more steps** |
| `diffusion_noise_schedule` | cosine | cosine_iddpm | nominally match (iddpm-flavoured) |
| `extra_features` | `'all'` (all 3) | **NOT SET** in repro yamls | **load-bearing knob missing** |
| `transition` | marginal | empirical_marginal | match |
| `lambda_train` (=`lambda_E`) | `[5, 0]` → 5.0 | `lambda_E: 5.0` | match |

`extra_features='all'` adds cycle counts (3, 4, 5, 6) +
eigenvalue/eigenvector features as auxiliary node/edge features in
upstream. Our `discrete_sbm_vignac_repro.yaml` and
`discrete_planar_digress_repro.yaml` activate it explicitly via
`model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures`
+ `extra_features_type=all`. The three molecular repro yamls
**inherit nothing equivalent** — neither the experiment yaml nor
the `discrete_sbm_official` preset configures any extra_features
block. Our molecular runs train *without* the upstream
spectral/cycle features.

### Optimizer

| field | upstream `train_default.yaml` | upstream MOSES override | ours `discrete_sbm_official.yaml` | drift |
|---|---|---|---|---|
| `optimizer` | adamw | adamw | adamw | match |
| `lr` | 2e-4 | 2e-4 | 2e-4 | YAML matches |
| `weight_decay` | 1e-12 | (inherits) | 1e-12 | YAML matches |
| `clip_grad` / `gradient_clip_val` | null | (inherits) | null | match |
| `ema_decay` | 0 (off) | (inherits) | not configured | match in effect |
| `amsgrad` | (not set; default false) | — | **true** | drift, ours uses AMSGrad |

**Run-name mismatch.** All three actual training-run W&B records
have names containing `..._lr1e-3_wd1e-4_...`. YAML+preset specify
`lr=2e-4, wd=1e-12`. So the runs were launched with a CLI override
of `model.learning_rate=1e-3 model.weight_decay=1e-4`. Since the
W&B run name is the only persisted record of the actual launch
hyperparameters (the YAMLs check in the wrong values), this is
a silent-drift hazard. The run-name lr is **5×** the upstream lr
and **5× the YAML lr**; the run-name wd is **8 orders of magnitude
larger** than the YAML wd.

### Schedule (n_epochs vs max_steps)

Upstream uses `n_epochs` (epoch-based), we use `max_steps`. To
compare apples-to-apples, total step budgets:

| dataset | upstream n_epochs × steps/epoch = total steps | our `max_steps` | ratio |
|---|---|---|---|
| QM9 (≈98k train no-H, batch 1024 upstream / 256 ours) | 1000 × ⌈98k/1024⌉ ≈ **96,000** | **50,000** | **52%** |
| MOSES (≈1.58M train, batch 256 upstream / 64 ours) | 300 × ⌈1.58M/256⌉ ≈ **1,853,000** | **200,000** | **11%** |
| GuacaMol (≈1.27M train, batch 32) | 1000 × ⌈1.27M/32⌉ ≈ **39,720,000** | **500,000** | **1.3%** |

By **total optimizer-step count** the QM9 config is half upstream;
MOSES is one-ninth; GuacaMol is roughly **80× shorter**.

By **per-sample exposure** (steps × batch_size / dataset_size =
"effective epochs"):

| dataset | upstream effective epochs | ours effective epochs | ratio |
|---|---|---|---|
| QM9 | 1000 | 50000 × 256 / 98000 ≈ **131** | 13% |
| MOSES | 300 | 200000 × 64 / 1.58M ≈ **8.1** | 2.7% |
| GuacaMol | 1000 | 500000 × 32 / 1.27M ≈ **12.6** | 1.3% |

So ours sees only **2.7% of upstream's data exposure on MOSES** and
**1.3% on GuacaMol**. The pickup's "4-8×" estimate matched QM9 but
under-stated the molecular-side gap by an order of magnitude.

### Batch size

| dataset | upstream | ours | drift |
|---|---|---|---|
| QM9 | 1024 | **256** | -4× |
| MOSES | 256 | **64** | -4× |
| GuacaMol | 32 | 32 | match |

Smaller batch size increases gradient-update count for the same
data exposure and changes the optimizer's effective LR sensitivity.
Combined with the run-name `lr=1e-3` override (5× upstream lr),
optimizer dynamics are a separate axis of drift from the schedule.

### Validation cadence + sample count

| field | upstream MOSES | upstream GuacaMol | ours MOSES | ours GuacaMol |
|---|---|---|---|---|
| `samples_to_generate` (per val) | 256 | 500 | preset default 40 | preset default 40 |
| `val_check_interval` (steps) | null (epoch-based) | null | 5000 | 10000 |
| `check_val_every_n_epoch` | 1 | 2 | null | null |

We sample fewer graphs per validation pass (40 vs 256/500), which
costs us statistical resolution on MMD/FCD trajectories during
training but doesn't directly affect optimizer convergence.

## Why the 50%-cutoff happens (post-investigation)

The pickup hypothesis #4 "Modal timeout cut training" is **refuted**:
`DEFAULT_TIMEOUTS["fast"] = 86400` (24 h) was already the Modal max
during these runs (well above the 76-min QM9 / 6h28m MOSES / 10h17m
GuacaMol observed runtimes). The follow-up hypothesis "max_epochs
hit before max_steps" is also **refuted**: the base trainer config
(`exp_configs/base/trainer/default.yaml:8`) hard-sets
`max_epochs: -1`, which Lightning interprets as unlimited.

Three independent runs cut at *exactly* 50% of `max_steps` is too
coincidental to be a runtime accident; it points at a launch-side
or callback-side mechanism. Likely candidates:
- A `CheckpointResumer` / two-stage launch script that runs only
  the first half and expects a manual respawn.
- An EarlyStopping callback wired with patience/threshold that
  happens to fire near the midpoint (very unlikely to fire at 50%
  in three different datasets).
- A separate `--max-steps` CLI override at launch that halved the
  YAML value.

This needs W&B run-summary inspection to resolve (separate task).

## Drift impact ranking (largest → smallest)

1. **Step budget on GuacaMol: 1.3% of upstream** (and we hit 50%
   of that → effective 0.6% of upstream training). This *alone*
   explains FCD-ChEMBL score 0.076 vs 0.68 (9× gap).
2. **Step budget on MOSES: 11% of upstream** (and we hit 50% →
   5.5%). Explains FCD 20.80 vs 1.19 (17.5× gap) being mostly a
   training-volume gap rather than a model-class gap.
3. **`extra_features='all'` not configured.** Upstream relies on
   spectral + cycle features; we train without them. This is a
   structural model-class drift independent of training duration.
4. **`lr=1e-3` CLI override** (5× upstream's 2e-4). Possibly
   load-bearing in our short-training regime — large LR + few
   steps could bias toward a different optimum. But since the
   YAMLs say 2e-4, this is also a reproducibility hazard.
5. **Architecture: n_layers 8 vs 12 (MOSES/GuacaMol).** ~33% fewer
   transformer layers; meaningful but secondary to the step budget.
6. **`diffusion_steps=1000` vs upstream 500.** Doubles the reverse
   sampling cost during eval, which shows up as wall-clock during
   eval-all but does not affect training quality.
7. **Batch size: ¼ of upstream on QM9 + MOSES.** Combined with
   higher LR, an axis of drift but not obviously bad.

## Recommended next launch

Either (a) **match upstream upstream-equivalently** so the repro
is a true reproduction:
- Set `extra_features._target_=ExtraFeatures, extra_features_type=all`
  in all 3 repro yamls (cf. `discrete_sbm_vignac_repro.yaml:113`).
- Bump `n_layers` 8 → 9 (QM9), 8 → 12 (MOSES, GuacaMol).
- Pin `learning_rate=2e-4, weight_decay=1e-12` in the YAML and
  drop the launch-time `lr=1e-3 wd=1e-4` override.
- Set `max_steps` to upstream-equivalent: 96k (QM9), 1.85M
  (MOSES), 39.7M (GuacaMol). The latter two are infeasible at
  current per-step throughput on a single A100 — would need
  multi-GPU or a much larger budget.
- Bump `noise_schedule.timesteps` 1000 → 500 to match upstream.

…or (b) **declare it not a "repro" but an own-config baseline**
and drop the `_digress_repro` suffix. The YAMLs as they stand are
not faithful enough to upstream to call the resulting numbers a
DiGress reproduction.

## Files

- `eval_all_assessment_2026-05-04.md` — sibling assessment of the
  *outputs* of these runs against paper anchors.
- `eval_all_aggregate_2026-05-04.csv` — flat per-(dataset, step)
  metric DataFrame.
