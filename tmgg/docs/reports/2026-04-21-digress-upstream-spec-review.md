# DiGress Upstream Spec Review

Review of `docs/reports/2026-04-21-digress-upstream-spec.md` against the
read-only upstream checkout at
`../digress-upstream-readonly/` (HEAD `780242b`).

## Summary verdict

The spec is highly accurate. Every mathematical formula, code snippet,
tensor shape, and hyperparameter I checked matches the upstream
code. Anchors (`file:line` pairs) are mostly exact; a small handful
are off by 1–2 lines or name a range that covers both a helper and
its caller. Two substantive drifts: §13's "1000 `multiflip_mcmc_sweep`
steps" claim contradicts the actual call site (which passes
`refinement_steps=100`), and a couple of anchor ranges in §1 and §11
point slightly wider than intended. No mathematical errors, no
incorrect shapes, no misidentified transitions. The flagged upstream
oddities (`u_e` typo in `AbsorbingStateTransition`, hardcoded AdamW,
missing `utils.EMA`) are all real and correctly cited.

## Per-section findings

| § | Status | Notes |
|---|--------|-------|
| 1 | MINOR DRIFT | `to_dense` is 53-62 not 53-75; `encode_no_edge` is 65-75 (spec §1 labels the combined range 53-75 but quotes `to_dense` alone — cosmetic). |
| 2 | CORRECT | Time convention, chain structure, and `T` default all match. |
| 3 | CORRECT | Cosine schedule formula, length-(T+1) array, `torch.round(t*T)` lookup all verified. |
| 4 | CORRECT | Empirical marginals, PyG double-counting, and SPECTRE `node_types = [1]` all match. |
| 5 | CORRECT | Transition matrices verified; `AbsorbingStateTransition` typo verified. |
| 6 | CORRECT | `apply_noise` logic, `lowest_t = 0 if training else 1`, upper-triangular sampling, all verified. |
| 7 | CORRECT | Extra-features shapes, cycle formulas, eigenfeature logic all match; arithmetic `6/11/0` is right. |
| 8 | CORRECT | Transformer structure, FiLM ordering, symmetrisation all verified. |
| 9 | CORRECT | Cross-entropy loss, mask predicate, `lambda_train = [lambda_E, lambda_y]` convention verified. |
| 10 | CORRECT | Reverse sampling, `reversed(range(0, T))`, posterior formula, `number_chain_steps < T` assert all verified. |
| 11 | CORRECT | VLB decomposition, `T` factor inside `compute_Lt`, `reconstruction_logp` at t=0 all match. Note on range: spec says `compute_Lt` is at 339-366; actual is 339-366 — correct. |
| 12 | CORRECT | All symmetry/diagonal anchors match. |
| 13 | MINOR DRIFT | Spec says SBM check "refines with 1000 `multiflip_mcmc_sweep(beta=inf)` steps"; the default on `is_sbm_graph` is 1000 but the call site in `SpectreSamplingMetrics.forward` at line 830 passes `refinement_steps=100`. See details below. |
| 14 | CORRECT | Trainer args, checkpointing, EMA gate, step definitions all verified. |
| 15 | CORRECT | Hyperparameter table audit below; every spot-check matches. |

## Detailed findings on flagged items

### §1 — `to_dense` line range

Spec opening claims `src/utils.py:53-75` for `to_dense`. Reading the
file, `to_dense` is at lines 53-62 and `encode_no_edge` is a separate
function at 65-75. The spec clearly intends to cover both and later
correctly cites `encode_no_edge` at 65-75 on its own, so this is
cosmetic labelling. No correction needed unless readers interpret the
top range as `to_dense` alone.

### §11 — `abstract_dataset.py` range

Spec cites `AbstractDatasetInfos.complete_infos` at
`src/datasets/abstract_dataset.py:94-100`. The class begins at line 94
(`class AbstractDatasetInfos:`) but `complete_infos` itself is at lines
95-100. Cosmetic.

### §13 — SBM MCMC refinement step count

Spec quotes:

> fits a stochastic block model with `graph_tool.minimize_blockmodel_dl`,
> refines with 1000 `multiflip_mcmc_sweep(beta=inf)` steps

`is_sbm_graph` at `src/analysis/spectre_utils.py:608` declares
`refinement_steps=1000` as a *default*:

```python
def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
```

but at the actual call site used during sampling metrics
(`src/analysis/spectre_utils.py:830`) we have:

```python
acc = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
```

`eval_acc_sbm_graph` at line 519 fans that argument out to
`is_sbm_graph`:

```python
def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    ...
    executor.map(is_sbm_graph, ..., [refinement_steps for i in range(len(G_list))])
```

So the value used in practice when the SBM metric is evaluated is
100, not 1000. The spec should say "100 refinement steps" (or note
the signature/caller discrepancy).

### §13 — Wald test formula

Spec gives

$$W = \frac{(\hat{p} - p)^2}{\hat{p}(1 - \hat{p}) + \varepsilon}.$$

Code at lines 647-648:

```python
W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)
```

Matches. The p-value conversion `p = 1 - chi2.cdf(abs(W), 1)` at line
652 and `p > 0.9` threshold at line 655 match the spec's prose.

## Hyperparameter table audit

Every row I spot-checked matches upstream. Below is an explicit
per-row confirmation.

| Knob | Spec value | Actual value | File:line | Verdict |
|------|-----------|--------------|-----------|---------|
| `general.name` | `graph-tf-model` / SBM `sbm` | same | `general_default.yaml:2`, `sbm.yaml:3` | CORRECT |
| `general.wandb` | `online` | `online` | `general_default.yaml:4` | CORRECT |
| `general.gpus` | 1 | 1 | `general_default.yaml:5` | CORRECT |
| `general.check_val_every_n_epochs` | 5 / SBM 100 | 5 / 100 | `general_default.yaml:11`, `sbm.yaml:8` | CORRECT |
| `general.sample_every_val` | 4 | 4 | `general_default.yaml:12` | CORRECT |
| `general.samples_to_generate` | 512 / SBM 40 | 512 / 40 | `general_default.yaml:14`, `sbm.yaml:10` | CORRECT |
| `general.samples_to_save` | 20 / SBM 9 | 20 / 9 | `general_default.yaml:15`, `sbm.yaml:11` | CORRECT |
| `general.chains_to_save` | 1 | 1 | `general_default.yaml:16` | CORRECT |
| `general.log_every_steps` | 50 | 50 | `general_default.yaml:17` | CORRECT |
| `general.number_chain_steps` | 50 | 50 | `general_default.yaml:18` | CORRECT |
| `general.final_model_samples_to_generate` | 10000 / SBM 40 | 10000 / 40 | `general_default.yaml:20`, `sbm.yaml:13` | CORRECT |
| `general.final_model_samples_to_save` | 30 | 30 | `general_default.yaml:21` | CORRECT |
| `general.final_model_chains_to_save` | 20 | 20 | `general_default.yaml:22` | CORRECT |
| `general.evaluate_all_checkpoints` | False | False | `general_default.yaml:27` | CORRECT |
| `model.type` | `discrete` | `discrete` | `discrete.yaml:2` | CORRECT |
| `model.transition` | `marginal` | `marginal` | `discrete.yaml:3` | CORRECT |
| `model.model` | `graph_tf` | `graph_tf` | `discrete.yaml:4` | CORRECT |
| `model.diffusion_steps` | 500 / SBM 1000 | 500 / 1000 | `discrete.yaml:5`, `sbm.yaml:21` | CORRECT |
| `model.diffusion_noise_schedule` | `cosine` | `cosine` | `discrete.yaml:6` | CORRECT |
| `model.n_layers` | 5 / SBM 8 | 5 / 8 | `discrete.yaml:7`, `sbm.yaml:22` | CORRECT |
| `model.extra_features` | `all` | `all` | `discrete.yaml:10` | CORRECT |
| `model.hidden_mlp_dims` | X/E/y = 256/128/128 default; SBM 128/64/128 | matches | `discrete.yaml:14`, `sbm.yaml:28` | CORRECT |
| `model.hidden_dims.dx` | 256 | 256 | `discrete.yaml:17` | CORRECT |
| `model.hidden_dims.de` | 64 | 64 | same | CORRECT |
| `model.hidden_dims.dy` | 64 | 64 | same | CORRECT |
| `model.hidden_dims.n_head` | 8 | 8 | same | CORRECT |
| `model.hidden_dims.dim_ffX` | 256 | 256 | same | CORRECT |
| `model.hidden_dims.dim_ffE` | 128 default / SBM 64 | 128 / 64 | `discrete.yaml:17`, `sbm.yaml:31` | CORRECT |
| `model.hidden_dims.dim_ffy` | 128 default / SBM 256 | 128 / 256 | `discrete.yaml:17`, `sbm.yaml:31` | CORRECT |
| `model.lambda_train` | `[5, 0]` | `[5, 0]` | `discrete.yaml:19` | CORRECT |
| `train.n_epochs` | 1000 / SBM 50000 | 1000 / 50000 | `train_default.yaml:2`, `sbm.yaml:17` | CORRECT |
| `train.batch_size` | 512 / SBM 12 | 512 / 12 | `train_default.yaml:3`, `sbm.yaml:18` | CORRECT |
| `train.lr` | `2e-4` | `0.0002` (= `2e-4`) | `train_default.yaml:4` | CORRECT |
| `train.clip_grad` | `null` | `null` | `train_default.yaml:5` | CORRECT |
| `train.save_model` | `True` | `True` | `train_default.yaml:6` | CORRECT |
| `train.num_workers` | 0 | 0 | `train_default.yaml:7` | CORRECT |
| `train.ema_decay` | 0 | 0 | `train_default.yaml:8` | CORRECT |
| `train.weight_decay` | `1e-12` | `1e-12` | `train_default.yaml:10` | CORRECT |
| `train.optimizer` | `adamw` (unused) | `adamw` (unused) | `train_default.yaml:11` | CORRECT |
| `train.seed` | 0 | 0 | `train_default.yaml:12` | CORRECT |

Planar experiment claim (spec bottom of §15): `n_layers=10`,
`batch_size=64`, `n_epochs=100000` at
`configs/experiment/planar.yaml:17-22`. Verified: `planar.yaml:17`
`n_epochs: 100000`, `planar.yaml:18` `batch_size: 64`, `planar.yaml:22`
`n_layers: 10`. CORRECT.

## Verdicts on flagged upstream oddities

### `AbsorbingStateTransition` `u_e` vs `u_y` typo

Spec claim: "`self.u_e[:, :, abs_state] = 1` instead of `self.u_y`
(`src/diffusion/noise_schedule.py:202-203`)".

Actual at `src/diffusion/noise_schedule.py:199-203`:

```python
self.u_e = torch.zeros(1, self.E_classes, self.E_classes)
self.u_e[:, :, abs_state] = 1

self.u_y = torch.zeros(1, self.y_classes, self.y_classes)
self.u_e[:, :, abs_state] = 1       # <-- line 203: should be self.u_y
```

**Real bug, verified.** `u_y` is allocated but never filled with the
absorbing-state indicator; `u_e` is written twice. The class is not
instantiated by any shipped YAML (`discrete.yaml:3` sets
`transition: 'marginal'`), so it never fires.

### Hardcoded AdamW despite advertised `optimizer` config choices

Spec claim: the `optimizer` key in `configs/train/train_default.yaml:11`
advertises "adamw,nadamw,nadam" but the code only ever uses `AdamW`.

Verified. `train_default.yaml:11`:

```
optimizer: adamw # adamw,nadamw,nadam => nadamw for large batches, see ...
```

`src/diffusion_model_discrete.py:122-124`:

```python
def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                             weight_decay=self.cfg.train.weight_decay)
```

No branching on `cfg.train.optimizer`. **Real.**

### Missing `utils.EMA`

Spec claim: `src/main.py:181-183` calls `utils.EMA(decay=...)` but
`utils.EMA` is not defined in the repo.

Verified. `src/main.py:181-183`:

```python
if cfg.train.ema_decay > 0:
    ema_callback = utils.EMA(decay=cfg.train.ema_decay)
    callbacks.append(ema_callback)
```

`src/utils.py` (read in full) contains `create_folders`, `normalize`,
`unnormalize`, `to_dense`, `encode_no_edge`, `update_config_with_new_keys`,
`PlaceHolder`, `setup_wandb` — no `EMA`. Since `ema_decay: 0` by
default in `train_default.yaml:8`, the branch is never entered and
the missing symbol never raises. **Real** — enabling EMA via config
would crash.

## Mathematical spot-checks

A non-exhaustive sample of formulas I verified line-by-line against
upstream:

- **Cumulative cosine `\bar\alpha_t`**: spec formula matches
  `src/diffusion/diffusion_utils.py:65-74` after accounting for the
  ratio `alphas_cumprod / alphas_cumprod[0]`.
- **Marginal `Q_t`, `\bar Q_t`**: matches `get_Qt` /
  `get_Qt_bar` at `src/diffusion/noise_schedule.py:152-187`; each row
  of `u_x` / `u_e` is indeed the marginal (broadcast via
  `x_marginals.unsqueeze(0).expand(K, -1)`).
- **`q(z_s | z_t, x_0)` posterior** in §10 (`compute_batched_over0_posterior_distribution`):
  the proportionality `(z_t Q_t^T)_k · (x_0 \bar Q_s)_k /
  (x_0 \bar Q_t z_t^T)` matches `src/diffusion/diffusion_utils.py:293-321`
  exactly, including the `1e-6` zero-denominator guard.
- **`Lt` factor of T**: confirmed at `src/diffusion_model_discrete.py:366`
  (`return self.T * (kl_x + kl_e)`).
- **`lowest_t = 0 if training else 1`**: confirmed at
  `src/diffusion_model_discrete.py:412`.
- **Upper-triangular-only edge sampling**: confirmed at
  `src/diffusion/diffusion_utils.py:263-264` (`sample_discrete_features`)
  and `385-390` (`sample_discrete_feature_noise`).
- **`reconstruction_logp` replaces masked and diagonal rows with
  ones** so that `target * log(1) = 0` contributes nothing: confirmed
  at `src/diffusion_model_discrete.py:398-403`.
- **Cross-entropy mask via `(true != 0).any(-1)`**: confirmed at
  `src/metrics/train_metrics.py:86-87`.
- **`F.cross_entropy(preds, argmax(target), reduction='sum')` inside
  `CrossEntropyMetric`**: confirmed at
  `src/metrics/abstract_metrics.py:99-102`.
- **`val_X_kl.compute() * self.T` separate logging factor**:
  confirmed at `src/diffusion_model_discrete.py:169`.

No formula errors found.

## What I could not verify

Nothing in the spec required reading forbidden paths. All 15 sections
were verifiable from the upstream checkout alone.
