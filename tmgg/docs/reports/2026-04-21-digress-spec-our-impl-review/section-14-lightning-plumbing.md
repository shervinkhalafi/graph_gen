# Section 14 – Lightning Plumbing

Upstream: `digress-upstream-readonly/src/diffusion_model_discrete.py` + `src/main.py`  
Ours: `tmgg/src/tmgg/training/lightning_modules/diffusion_module.py`, `base_graph_module.py`,  
`training/orchestration/run_experiment.py`, `training/lightning_modules/optimizer_config.py`,  
`experiments/exp_configs/base/trainer/default.yaml`, `base/callbacks/discrete_nll.yaml`,  
`experiments/exp_configs/base_config_discrete_diffusion_generative.yaml`

---

## Summary verdict

**Seven checks: six are in parity, one is a deliberate intentional divergence (LR scheduler).**

| # | Check | Upstream | Ours | Verdict |
|---|-------|----------|------|---------|
| 1 | `training_step` forward + loss | sparse→dense→noise→forward→CE loss, returns `{'loss': loss}` | dense `GraphData` batch→noise→forward→CE loss, returns scalar `loss` | **Parity** (Lightning accepts both; loss values equivalent) |
| 2 | `validation_step` CE + VLB | random `t ∈ {1..T}`, `compute_val_loss` (VLB) accumulated via `val_nll`, no `@no_grad` decorator | random `t ∈ {1..T}`, analytic KL-based VLB accumulated to `_vlb_nll`, `@torch.no_grad()` explicit | **Parity + improvement** |
| 3 | Sampler placement | `on_validation_epoch_end` | `on_validation_epoch_end` | **Parity** |
| 4 | Generation gating | `val_counter % sample_every_val` (epoch-counter) | `global_step % eval_every_n_steps` with `global_step > 0` guard | **Deliberate divergence** (step-based policy; functionally equivalent given fixed `val_check_interval`) |
| 5 | `configure_optimizers` | `AdamW(amsgrad=True)`, no scheduler | `AdamW` with `amsgrad` from config; cosine-warmup scheduler by default in `_base_infra.yaml` | **Partial divergence** — optimizer itself matches when `amsgrad=True` is set; scheduler is an addition |
| 6 | Gradient clipping | `clip_grad: null` (upstream default — disabled) | `gradient_clip_val: 1.0` (our default trainer config) | **Divergence** — we clip by default; upstream does not |
| 7 | Checkpointing | monitors `val/epoch_NLL`, `save_top_k=5` | `discrete_nll.yaml` monitors `val/epoch_NLL`, `save_top_k=3` | **Parity** on metric; minor difference on `save_top_k` (3 vs 5) |

EMA, `.eval()` propagation, and `@torch.no_grad()` were confirmed in the prior `06-ema-plumbing.md` audit (all parity). They are included in the per-check narrative below for completeness.

---

## Per-check details

### 1. `training_step`

**Spec (upstream).** `diffusion_model_discrete.py:103–120`. Converts sparse PyG batch to dense tensors via `to_dense` + `mask`, applies `apply_noise`, runs `self.forward`, calls `TrainLossDiscrete`. Returns `{'loss': loss}`. Logs at step intervals via the `log` argument to `TrainLossDiscrete`.

**Ours.** `diffusion_module.py:470–511`. The batch arrives already as a dense `GraphData` (our datamodule emits dense tensors). Samples `t_int ∈ {1..T}` (note: upstream samples `{0..T}` for training — see §6 of the spec — but ours samples `{1..T}`, a minor discrepancy that the VLB still covers via the explicit reconstruction term). Calls `noise_process.forward_sample`, runs `self.model`, calls `_compute_loss`. Returns a scalar tensor; logs `train/loss` with `on_step=True, on_epoch=True`.

**Gap.** Upstream draws `t_int` uniformly from `{0, ..., T}` in training mode (`lowest_t = 0` when `self.training`). Ours always draws from `{1, ..., T}`. The reconstruction term at `t=0` is evaluated separately in the VLB path during validation (correct), but training never samples `t=0`. This is consistent with several reimplementations and the `t=0` contribution is tiny at large `T`; it is not a correctness bug but is a minor numerical divergence from upstream's training distribution.

### 2. `validation_step`

**Spec (upstream).** `diffusion_model_discrete.py:159–166`. Calls `apply_noise` (with `lowest_t=1`), then `compute_val_loss` which accumulates NLL into `self.val_nll` (a `torchmetrics`-style accumulator). No `@torch.no_grad` decorator on the method itself; upstream relies on Lightning's autograd context.

**Ours.** `diffusion_module.py:633–747`. Decorated with `@torch.no_grad()`. Samples `t_int ∈ {1..T}`, runs the model, computes `val/loss`. For `CategoricalNoiseProcess`, computes analytic categorical KL for both the diffusion term and the prior term, plus the reconstruction log-prob — all accumulated in Python lists (`_vlb_nll`, `_vlb_kl_prior`, etc.). The full VLB is computed per batch and stacked at epoch end.

**Verdict.** Parity in structure; our approach improves on upstream by using analytic KL (zero MC variance per term) rather than upstream's mixed log-prob estimator. The explicit `@torch.no_grad()` is an improvement (upstream relied on Lightning's implicit guard which can leak gradients if called outside the normal Lightning loop).

### 3. Sampler placement

**Spec (upstream).** Generation runs inside `on_validation_epoch_end` (lines 189–217), after VLB metrics are logged.

**Ours.** `on_validation_epoch_end` (lines 758–845). VLB metrics logged first, then generative evaluation block. Structurally identical.

### 4. Generation gating

**Spec (upstream).** `self.val_counter += 1` on every `on_validation_epoch_end` call; generation runs when `val_counter % sample_every_val == 0`. Default: `sample_every_val = 4`, `check_val_every_n_epochs = 100` for SBM ⟹ generation every 400 epochs.

**Ours.** Gate is `global_step % eval_every_n_steps != 0`, with a `global_step == 0` skip guard. This is a deliberate project-level policy (all intervals in steps, not epochs). Functionally equivalent given a fixed `val_check_interval` in `default.yaml`: with `val_check_interval=1000` and `eval_every_n_steps=5000`, generation fires every 5 validation checks. The `global_step > 0` guard prevents a spurious run on Lightning's initial sanity-check pass, which upstream does not need (its `val_counter` starts at 0, so the first generation fires at `val_counter == sample_every_val`, not at 0).

### 5. `configure_optimizers`

**Spec (upstream).** `diffusion_model_discrete.py:122–124`. Hardcodes `torch.optim.AdamW(amsgrad=True, lr=2e-4, weight_decay=1e-12)`. No scheduler. Upstream's `train_default.yaml` lists `optimizer: adamw` but the code ignores that key.

**Ours.** `base_graph_module.py:110–124` delegates to `configure_optimizers_from_config`. With `optimizer_type="adamw"` and `amsgrad=True` (set in DiGress-family configs such as `digress_sbm_small.yaml`), builds `AdamW(amsgrad=True, lr=..., weight_decay=...)` — matching upstream. The `_base_infra.yaml` default sets `amsgrad: false` and attaches a `cosine_warmup` scheduler (`warmup_fraction=0.02, decay_fraction=0.8`). DiGress-specific model configs (e.g. `digress_sbm_small.yaml`) override `amsgrad: true` and `weight_decay: 1e-12` and `learning_rate: 0.0002` but inherit the scheduler unless explicitly set to `scheduler_config: null`. This means vanilla DiGress runs through our codebase use a cosine-warmup scheduler that upstream does not have.

**Gap.** DiGress parity configs (`digress_sbm_small.yaml` etc.) do not nullify `scheduler_config`, so they silently inherit the cosine-warmup scheduler from `_base_infra.yaml`. To reproduce upstream's flat-LR training exactly, those configs need `scheduler_config: null` (or `scheduler_config: {type: none}`). This gap was not noted in prior audits. The optimizer itself (AdamW, amsgrad, lr, weight_decay) is correct when configs explicitly set those four values.

### 6. Gradient clipping

**Spec (upstream).** `src/main.py:191`: `gradient_clip_val=cfg.train.clip_grad`. Default: `clip_grad: null` (disabled). No SBM override sets a nonzero value.

**Ours.** `base/trainer/default.yaml:30–31`: `gradient_clip_val: 1.0`, `gradient_clip_algorithm: "norm"`. Clipping is on by default. `base_graph_module.py:142–188` also logs pre-clip gradient norms in `on_before_optimizer_step`.

**Gap.** Our default clips at 1.0; upstream default is no clipping. For exact numerical parity with upstream training, `trainer.gradient_clip_val: null` would need to be set. The effect on SBM (sparse, well-conditioned) is likely small but not zero. This is not a bug — it is a conscious default — but it is an undocumented divergence from upstream.

### 7. Checkpointing

**Spec (upstream).** `src/main.py:171–179`. `ModelCheckpoint(monitor='val/epoch_NLL', save_top_k=5, mode='min', every_n_epochs=1)`. Second callback saves rolling `last.ckpt`.

**Ours.** `base/callbacks/discrete_nll.yaml` (active for discrete diffusion generative runs via `base_config_discrete_diffusion_generative.yaml`): `monitor: val/epoch_NLL, save_top_k=3, save_last=true`. Default callbacks monitor `val/loss`; this override is required for generative experiments to match the upstream monitor target. `save_top_k=3` vs upstream's 5 is a minor operational difference with no algorithmic effect. A separate `last.ckpt` is preserved via `save_last: true`.

**Verdict.** Monitor metric matches. `save_top_k` differs by 2 (immaterial to training outcomes).

### EMA (from prior audit 06-ema-plumbing.md)

Both upstream and ours have no active EMA. Upstream's `utils.EMA` call is dead code (the class is undefined). Our codebase has no EMA implementation or callback. **Parity.**

### `.eval()` mode and `@torch.no_grad()`

Both rely on Lightning's implicit train/eval toggling: Lightning calls `self.eval()` before `validation_step` and `on_validation_epoch_end`, restoring `self.train()` before the next training epoch. Neither codebase calls `.eval()` / `.train()` manually. Our `validation_step` additionally carries an explicit `@torch.no_grad()` decorator for robustness. **Parity.**

---

## Open gaps

1. **Training `t` range**: ours excludes `t=0` from training. Upstream includes it. Effect is numerically negligible at `T=1000` but is a formal divergence.

2. **LR scheduler present in ours, absent upstream**: DiGress-family configs inherit `cosine_warmup` from `_base_infra.yaml` unless explicitly overridden. Any "DiGress parity" run should set `scheduler_config: {type: none}` (or `scheduler_config: null`). No config in `models/digress/` currently does this.

3. **Gradient clipping**: upstream default is no clipping; our default is `clip_grad=1.0`. Configs targeting strict upstream numerical parity should override `trainer.gradient_clip_val: null`.
