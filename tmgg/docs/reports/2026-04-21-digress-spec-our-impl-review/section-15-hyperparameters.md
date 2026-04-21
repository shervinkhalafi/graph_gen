# Section 15 hyperparameter review: upstream spec vs. our implementation

**Scope.** Every row from the spec's §15 table, compared against our
Hydra config tree (`src/tmgg/experiments/exp_configs/`). For the
"upstream-matching SBM run" the primary sources are:

- `models/discrete/discrete_sbm_official.yaml` — our SBM model config
- `data/spectre_sbm.yaml` — our SBM data config
- `_base_infra.yaml` + `base/trainer/default.yaml` — shared defaults
- `analysis/digress-loss-check/upstream_s1_lr2e-4/config.yaml` — the
  resolved config from the failing run (labelled **LIVE** in the notes
  below when it contradicts a static YAML default)

Upstream configs read: `configs/general/general_default.yaml`,
`configs/train/train_default.yaml`, `configs/model/discrete.yaml`,
`configs/experiment/sbm.yaml`.

Columns: **Spec name** | **Upstream SBM default** (per spec §15, verified
by prior reviewer) | **Our config value** | **Verdict**.

"Our config value" refers to the value a run using
`discrete_sbm_official + spectre_sbm` would resolve to.  When the LIVE
resolved config disagrees, that is noted separately.

---

## Full table

| Spec name | Upstream SBM default | Our config value | Verdict |
|---|---|---|---|
| `general.name` | `sbm` | not used (we use `experiment_name`) | NOT EXPOSED — different naming scheme |
| `general.wandb` | `online` | `allow_no_wandb: true` + logger list | NOT EXPOSED — our W&B plumbing differs; functionally equivalent |
| `general.gpus` | 1 | `devices: auto` (trainer default) | DIFFERENT — upstream hardcodes 1 GPU; we use Lightning auto-detect |
| `general.check_val_every_n_epochs` | 100 | `check_val_every_n_epoch: null` + `val_check_interval: 1000` steps | DIFFERENT — upstream uses epoch-cadence; we use step-cadence throughout |
| `general.sample_every_val` | 4 | not exposed | NOT EXPOSED — upstream gates generation every 4th val call; our `eval_every_n_steps: 1100` (LIVE: 1100) controls this directly in steps |
| `general.samples_to_generate` | 40 | `eval_num_samples: 40` (discrete_sbm_official) | MATCH — value matches; key name differs |
| `general.samples_to_save` | 9 | not exposed | NOT EXPOSED — we save all generated samples; no save-count knob |
| `general.chains_to_save` | 1 | not exposed | NOT EXPOSED — chain visualisation not implemented |
| `general.log_every_steps` | 50 | `log_every_n_steps: 50` (trainer default) | MATCH |
| `general.number_chain_steps` | 50 | not exposed | NOT EXPOSED — chain visualisation not implemented |
| `general.final_model_samples_to_generate` | 40 | not exposed | NOT EXPOSED — no separate test-time generation budget; test runs re-use eval_num_samples |
| `general.final_model_samples_to_save` | 30 | not exposed | NOT EXPOSED |
| `general.final_model_chains_to_save` | 20 | not exposed | NOT EXPOSED |
| `general.evaluate_all_checkpoints` | `False` | not exposed | NOT EXPOSED |
| `model.type` | `discrete` | implicit (`DiffusionModule` + `CategoricalNoiseProcess`) | NOT EXPOSED — hardcoded by class choice, not a config knob |
| `model.transition` | `marginal` | `limit_distribution: empirical_marginal` | MATCH — different key name, same semantics |
| `model.model` | `graph_tf` | `model_name: graph_transformer` | MATCH — different key name, same semantics |
| `model.diffusion_steps` ($T$) | 1000 (SBM) | `timesteps: 500` (discrete_sbm_official static YAML); `timesteps: 1000` (LIVE resolved config, overridden at CLI) | DIFFERENT — static YAML default is **500**, not 1000; the failing run was launched with `timesteps=1000` as a CLI override |
| `model.diffusion_noise_schedule` | `cosine` | `schedule_type: cosine_iddpm` | MATCH — `cosine_iddpm` implements the identical IDDPM/DiGress cosine formula (`cosine_beta_schedule_discrete`) |
| `model.n_layers` | 8 | 8 (`discrete_sbm_official`) | MATCH |
| `model.extra_features` | `all` | `extra_features_type: all` (LIVE; set in `model.model.extra_features._target_` block) | MATCH — value matches; baked into the resolved config as an `ExtraFeatures` instantiation |
| `model.hidden_mlp_dims.X` | 128 | 128 | MATCH |
| `model.hidden_mlp_dims.E` | 64 | 64 | MATCH |
| `model.hidden_mlp_dims.y` | 128 | 128 | MATCH |
| `model.hidden_dims.dx` | 256 | 256 | MATCH |
| `model.hidden_dims.de` | 64 | 64 | MATCH |
| `model.hidden_dims.dy` | 64 | 64 | MATCH |
| `model.hidden_dims.n_head` | 8 | 8 | MATCH |
| `model.hidden_dims.dim_ffX` | 256 | 256 | MATCH |
| `model.hidden_dims.dim_ffE` | 64 | 64 | MATCH |
| `model.hidden_dims.dim_ffy` | 256 | 256 | MATCH |
| `model.lambda_train` (`[lambda_E, lambda_y]`) | `[5, 0]` | `lambda_E: 5.0`; no `lambda_y` | MATCH for lambda_E; DIFFERENT structurally — upstream has a 2-tuple `[lambda_E, lambda_y]`; we expose only `lambda_E`, implying `lambda_y = 0` is hardcoded |
| `train.n_epochs` | 50000 | `max_steps: 550000` (LIVE) / `max_steps: 10000` (trainer default) | DIFFERENT — we train in steps, not epochs; no epoch cap is configured |
| `train.batch_size` | 12 | 12 (`spectre_sbm`) | MATCH |
| `train.lr` | `2e-4` | `learning_rate: 0.0002` | MATCH |
| `train.clip_grad` | `null` | `gradient_clip_val: 1.0` (trainer default and LIVE) | **DIFFERENT** — upstream disables gradient clipping; we clip at 1.0 by default |
| `train.save_model` | `True` | `enable_checkpointing: true` | MATCH |
| `train.num_workers` | 0 | 4 (`_base_infra`; LIVE: 4) | **DIFFERENT** — upstream default is 0 workers; we use 4 |
| `train.ema_decay` | 0 (disabled) | not exposed | NOT EXPOSED — no EMA support in our pipeline |
| `train.weight_decay` | `1e-12` | `weight_decay: 1e-12` (`discrete_sbm_official`) | MATCH |
| `train.optimizer` | `adamw` | `optimizer_type: adamw` | MATCH |
| `train.seed` | 0 | `seed: 1` (LIVE) / `seed: 42` (`_base_infra`) | DIFFERENT — upstream default seed is 0; our base default is 42; the failing run used seed 1 |
| `dataset.name` | `sbm` (via experiment) | implicit (`SpectreSBMDataModule`) | NOT EXPOSED — dataset identity baked into `_target_` class, not a string selector |
| `dataset.datadir` | `data/sbm/` | `cache_dir: null` (downloads to `~/.cache/tmgg/spectre/`) | DIFFERENT — different storage convention; no direct equivalent of `datadir` |
| `max_n_nodes` (derived) | derived from train+val max node count | `max_n_nodes: 20` (LIVE, hardcoded override) | **DIFFERENT** — upstream derives this dynamically from the dataset; we set it as a static integer in the extra_features block. The SPECTRE SBM fixture has nodes up to 187, so the correct value is 187, not 20. |
| `amsgrad` | `True` (hardcoded in `configure_optimizers`) | `amsgrad: true` (`discrete_sbm_official`) | MATCH |

---

## Entries we expose that have no upstream counterpart (EXPOSED)

| Our key | Value | Role |
|---|---|---|
| `scheduler_config` | `cosine_warmup`, warmup 2%, decay 80% | LR warmup+cosine decay; upstream has no scheduler |
| `eval_every_n_steps` | 1100 (LIVE) / 1000 (default) | Step-based generation cadence replacing `sample_every_val` |
| `spectral_k` | 4 | Eigenvector tracking for diagnostics |
| `gradient_clip_algorithm` | `norm` | Clip mode; upstream has no equivalent |
| `val_check_interval` (steps) | 1100 (LIVE) | Step-based validation; upstream uses `check_val_every_n_epochs` |
| `early_stopping` callback | patience=10, min_delta=1e-4, monitors `val/epoch_NLL` | Upstream has no early stopping |

---

## Notable mismatches

**`max_n_nodes = 20` vs. dataset maximum ≈ 187 (critical).** The
`ExtraFeatures` module normalises node counts by `max_n_nodes` and uses
it as the denominator when computing the spectral features' normalised
eigengap. The LIVE resolved config has `max_n_nodes: 20`, which is the
value from our synthetic `SyntheticCategoricalDataModule` fixture (20
nodes), not the SPECTRE SBM fixture (nodes in [44, 187]). This means
the normalised node count $n / n_\max$ will be > 1 for most SPECTRE SBM
graphs, producing out-of-range input features that upstream never sees.
This is likely a data-config composition bug: the `max_n_nodes` field
was not overridden when switching to `spectre_sbm`, so it inherited the
synthetic-data default. This is a strong candidate for causing
behavioural divergence.

**`gradient_clip_val = 1.0` vs. `null` (upstream disabled).** Upstream
passes `clip_grad=null` to the trainer, meaning no gradient clipping.
Our trainer default is `gradient_clip_val: 1.0`. During early training
with a fresh random init this clamps gradient norms that upstream
allows through unconstrained, which could slow convergence or alter the
loss trajectory in the first few thousand steps.

**`model.diffusion_steps` static default is 500, not 1000.** The
`discrete_sbm_official.yaml` file declares `timesteps: 500` but all
SBM runs have been launched with a CLI override to 1000. The static
default is therefore a latent correctness trap: any run that omits the
override silently uses T=500 instead of the upstream-mandated T=1000.

**LR scheduler present vs. absent.** Upstream uses a flat learning rate
throughout (`configure_optimizers` returns the bare AdamW, no
scheduler). Our config applies cosine warmup+decay (`scheduler_config:
cosine_warmup`), meaning the effective LR changes over training even
though the nominal `learning_rate: 2e-4` matches. The LR curve will
diverge significantly after the warmup phase ends.

**Seed divergence.** Upstream default is 0; our base is 42; the failing
run used 1. This is expected and intentional, but worth keeping in mind
when comparing logged loss curves step-for-step against upstream
reference runs.

**`train.num_workers`: 0 upstream vs. 4 ours.** This is a minor
infrastructure difference with no effect on numerics, but it means data
loading is asynchronous on our side, which can affect the exact ordering
of batches under some DataLoader shuffle implementations.
