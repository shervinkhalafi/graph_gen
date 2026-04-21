# Divergence triage — user instructions and agent responses

Per-divergence record of what the user asked for and what the investigation
found. 46 divergences total, enumerated exactly as in the 2026-04-21 review
summary. `†` = intentional tmgg divergence, `*` = appears accidental.

Format per item:

- **Divergence** — the technical description (verbatim from the summary).
- **Instruction** — what the user wants done, verbatim. "dito"/"see above"
  references are resolved inline.
- **Response** — investigation result, clean-fix sketch, or discussion point.

Legend in the summary column: `FIX`, `DISCUSS`, `DOC`, `NO-OP`.

---

# Decisions (2026-04-21)

User-approved decisions on Appendix B items. Execute in numeric order.

## Cleared clear-to-fix items (12 items, no decision needed)

Ship as single-section commits:

- #4 `from_pyg_batch` symmetry assertion
- #5 `num_nodes` from train+val only
- #8 `custom_vignac` default K=5 → recovers `p=4/5`
- #14 replace `denom.clamp(1e-6)` with positivity assert + upstream-style row-sum renorm (subsumes parts of #28)
- #15 runtime `Q_t_bar` row-sum assertion
- #16 sample `t=0` in training only (not validation)
- #25 one-line `masked_softmax` match: return `x` when all-masked
- #32 replace `log_pN` silent fallback with `raise`
- #35 parametrise `sbm_refinement_steps` on `GraphEvaluator`, default 100
- #39 `gradient_clip_val` default → `null`
- #40 early-stopping patience → 1000, document divergence, confirm `save_top_k=-1`
- #43 `timesteps=500` → `1000` in `discrete_sbm_official.yaml`

## Decision items

### D-1 — `remove_self_loops` placement (#2): **Option A**

Use `torch_geometric.utils.remove_self_loops` on the sparse `edge_index`
before `to_dense_adj`. Drop our post-densification `adj[:, diag, diag] = 0`.
Keeps the `E_class[:, diag, diag, :] = 0` from commit `82bcec26` as the CE
predicate invariant.

### D-2 — `collapse` flag on `.mask()` (#3): **MERGE**

Rename/merge `collapse_to_indices` into `GraphData.mask(collapse=False)`
kwarg. Callers can pass `collapse=True` when they want integer class
indices out. No behaviour change on the default path.

### D-3 — π estimator path (#13): **Option A — port sparse edge_counts, drop-in interface**

Port upstream's sparse `edge_counts` via a clean drop-in interface
that preserves our single-path dense architecture. Hot-loop training
remains `DataLoader → GraphData (dense) → model` untouched; π
estimation is preprocessing (runs once at datamodule setup), and
preprocessing is the right place to adopt upstream's sparse approach
for bit-identical statistical parity.

Sketch (see full detail in the re-ask section at the bottom):
- New helper `tmgg/src/tmgg/data/utils/edge_counts.py` (~25 LoC,
  direct port of upstream's routine).
- Data modules expose a `train_dataloader_raw_pyg()` accessor
  yielding PyG batches pre-densification.
- `NoiseProcess.initialize_from_data` switches to the new helper.
- Total diff: ~50 LoC additive, zero changes to the training hot
  loop.

### D-4 — `NoisedBatch` dataclass + `t_int` shape `(B, 1)` (#17, #18): **ACCEPT default**

Ship as recommended. Parity rationale:
- Upstream's `apply_noise` returns a dict keyed `{t_int, t, beta_t,
  alpha_s_bar, alpha_t_bar, X_t, E_t, y_t, node_mask}`. Every VLB
  computation downstream (`kl_prior`, `compute_Lt`, `reconstruction_logp`)
  reads from this dict by name.
- Our `NoisedBatch` dataclass exposes the same named scalars
  (`t_int, alpha_t_bar, alpha_s_bar, beta_t`) plus a typed
  `z_t: GraphData` bundle for the noised tensors. **Same information
  content**, different access syntax (`.z_t.E_class` vs `noisy_data['E_t']`).
- `t_int` shape `(B, 1)` matches upstream's shape exactly. When we
  `git grep` upstream for patterns like `t_int[:, 0]` or `t_int.squeeze(-1)`,
  our ports can preserve the same expressions without adapting.
- Parity check: every upstream VLB-path expression that reads a noise
  scalar has a direct one-to-one translation in our code. If the
  dataclass exposes the same fields with the same shapes, we can build
  a side-by-side correspondence table and verify no expression is
  silently different.

### D-5 — Unnormalised-posterior helper + zero-floor + symmetry toggle (#28, #29): **ACCEPT**

How this assures upstream parity:
- Upstream (`diffusion_utils.py:274-290`) computes
  `unnormalized = (z_t @ Qt^T) * (x_0 @ Qsb)`, then
  `prob = unnormalized / unnormalized.sum(dim=-1, keepdim=True)`, with
  the guard `denom[denom == 0] = 1e-5` applied to the row-sum before
  division.
- Our new helper `_sample_from_unnormalised_posterior(unnorm, *,
  zero_floor=1e-5)` computes `denom = unnorm.sum(dim=-1, keepdim=True);
  denom = denom.clamp(min=zero_floor); prob = unnorm / denom` — same
  arithmetic, same numerical output row-by-row.
- Current code clamps a *different* quantity (the contracted product,
  one step earlier in the chain). Replacing it with upstream's row-sum
  floor resolves both the structural and numerical divergence in one
  edit.
- Precondition `assert (unnorm >= 0).all()` makes the clamp meaningful
  (it can only fire on the degenerate "empty posterior" case, not on
  ordinary negative-rounding rows).
- `assert_symmetric_e: bool = True` toggle on the sampler reproduces
  upstream's per-reverse-step `assert (E_s == E_s.transpose(1,2)).all()`
  at `diffusion_model_discrete.py:649`. Upstream fires unconditionally;
  our toggle defaults to True for debug / test, can be disabled in
  production hot loops if the overhead matters.
- Expose `zero_floor` on the sampler config so strict-parity runs can
  set `zero_floor=1e-5` and hit upstream's exact value (also the
  default).

### D-6 — `use_marginalised_vlb_kl` toggle (#30): **ADD TOGGLE, default to upstream**

Default `False` = upstream's direct Bayes plug-in form (soft `x_0`
plugged straight into the posterior formula). Setting `True` selects
our marginalised form `Σ_c p(z_s|z_t, x_0=c) · p_θ(x_0=c|z_t)`.

Add a docstring/code comment at the selection site explaining:
- **Mathematical correctness**: both estimators are unbiased estimates
  of the same `KL(q(z_s|z_t,x_0) || p(z_s|z_t,θ))`; they coincide when
  `p_θ(x_0|z_t)` is one-hot (at convergence).
- **Why marginalised**: aligns the VLB KL with the actual sampling
  distribution the reverse chain draws from, reducing estimator
  variance during early training when `p_θ` is diffuse.
- **Why upstream plug-in**: simpler, the original DiGress formulation,
  what the paper reports.
- Default tracks upstream for strict parity; flip to ours for
  lower-variance VLB reporting.

### D-7 — `use_upstream_reconstruction` toggle (#31): **ADD TOGGLE, default to upstream**

Default `True` = upstream's `z_0` (via `Q_0`) + raw softmax scoring.
Setting `False` selects our `z_1` + marginalised posterior scoring.

Docstring:
- **Upstream form**: at `t=0`, `ᾱ_0 ≈ 1`, so `z_0 ≈ x_0` up to
  `Q_0`-sampling noise. Model is asked "given a near-clean graph,
  predict `x_0`"; score the softmax directly. This is the "first step
  of denoising" reconstruction likelihood.
- **Our form**: at `t=1`, sample `z_1` (slightly noisier than `z_0`),
  score via the reverse-chain's marginalised posterior. Aligns with
  the sampler's actual first reverse step rather than the theoretical
  denoiser endpoint.
- **Mathematical correctness**: both are valid estimators of the
  reconstruction term; they differ at O(1e-3) for T=1000 per the
  original docstring. Upstream is the published DiGress formulation.

### D-8 — `zero_output_diagonal` toggle on transformer (#33): **ADD TOGGLE, default to upstream**

Default `False` = upstream's behaviour (model output retains diagonal
logits, `.mask(node_mask)` only zeros padding). Setting `True` selects
our current `.mask_zero_diag()` (zeros both padding and diagonal).

Docstring:
- **Upstream**: allows the model to output arbitrary logits on the
  diagonal. These are dropped downstream by the target-side
  `encode_no_edge` zeroing + row predicate in CE, and by
  `triu+symmetrize` in the sampler. So diagonal model output is
  *information that's never used*, not harmful.
- **Ours (previous default)**: zeroed the diagonal of the output for
  "strictly cleaner predictions". Arguably cleaner but introduces a
  needless constraint on the network's latent representation (the
  diagonal channel is meaningful in attention-pooling; forcing zero
  is information loss).
- **Mathematical correctness**: both produce identical valid output
  after the downstream filters. Upstream's is the "honest" version.

### D-9 — Scheduler default (#38): options and a **likely default flip**

Three options on `_base_infra.yaml`'s scheduler block:

- **Option A**: remove scheduler config from `_base_infra.yaml`
  entirely; downstream configs that want one must add it explicitly.
  All existing configs that silently inherit would break.
- **Option B**: keep `cosine_warmup` as `_base_infra.yaml` default;
  specific upstream-match configs override to `{type: none}`.
- **Option C**: flip `_base_infra.yaml` default to
  `scheduler_config: {type: none}`; configs that want cosine_warmup
  override explicitly.

Upstream's default is no scheduler (flat LR). Under user directive
("change default probably if that is upstream default, not per config
override"), **Option C is the answer**: default matches upstream,
opt-in for scheduler.

Implementation: `_base_infra.yaml:scheduler_config.type: none` (or
delete the block so downstream uses a null-scheduler constructor);
add `scheduler_config: {type: cosine_warmup, warmup_fraction: 0.02,
decay_fraction: 0.8}` to whichever tmgg-specific experiment configs
want the scheduler. Likely candidates: `models/spectral/*.yaml` and
any eigenstructure-research config that relies on warmup.

### D-10 — Step-vs-epoch gate (#41): **KEEP STEP, no toggle, document**

No code change. Add:
- Docstring on the `on_validation_epoch_end` gate explaining that
  generation is gated by `global_step % eval_every_n_steps` rather
  than by epoch count.
- Conversion formula for comparing to upstream:
  ```
  upstream_sample_every_val (epochs) ↔
    ours eval_every_n_steps = upstream_value × len(train_dataloader)
  ```
- Flag for upstream-match runs: specify `eval_every_n_steps` as a
  product of `steps_per_epoch` and the upstream epoch count.

Step-based is the more general design (invariant under batch-size /
dataset-size changes); keeping it unflipped is the right call. Just
make sure users reproducing upstream can hit the same validation
cadence via the conversion.

### D-11 — `max_n_nodes` (#42): detailed explanation

**Current state.** `discrete_sbm_official.yaml` hardcodes
`max_n_nodes: 20` inside the `ExtraFeatures` block. The
`ExtraFeatures` constructor reads it at instantiation time and uses it
as the divisor for the normalised node-count feature and for padding
ceilings on eigenvalue/eigenvector tensors.

**Why this is bad.** The value is structurally *tied to the dataset*,
not to the model. The correct value is `dataset.max_n_nodes` (upstream
computes this dynamically at `abstract_dataset.py:95-100`). A user
running the same model config on SPECTRE (n up to 187) silently gets
`max_n_nodes=20`, producing:
- Node-count feature `n / 20 > 1` for ~95% of SPECTRE graphs
  (absurd).
- Eigenvector pad shape too small → runtime error or truncation.
- Cycle normalisation scaled wrong.

**Why Hydra interpolation alone isn't enough.** For `SpectreSBMDataModule`
the max node count is computed at `setup()` time (it loads the fixture
and counts). Hydra composition happens BEFORE setup. So
`max_n_nodes: ${data.num_nodes}` won't read the true value — `data`
hasn't run `setup()` yet.

**Three paths, each with its tradeoff.**

1. **Runtime injection** (recommended). In
   `DiffusionModule.on_fit_start`, read `self.trainer.datamodule.num_nodes`
   (populated by `setup()`) and write it into the `ExtraFeatures`
   instance:
   ```python
   def on_fit_start(self):
       actual_max = self.trainer.datamodule.num_nodes
       if self.model.extra_features.max_n_nodes != actual_max:
           self.log.warning(...)
           self.model.extra_features.max_n_nodes = actual_max
   ```
   Also add an assertion that the config's value is at least as large
   as the actual max (fail loud if the config is too small to pad to):
   ```python
   assert self.model.extra_features.max_n_nodes >= actual_max
   ```

2. **Config-time interpolation with a static upper bound exposed on
   the data config**. Require every data module to expose
   `num_nodes_max_static` as a top-level YAML field (a safe upper
   bound chosen at design time — e.g., `num_nodes_max_static: 200`
   for SPECTRE). Then the model config uses
   `max_n_nodes: ${data.num_nodes_max_static}`. Explicit, but requires
   every data config to set it.

3. **Both**. Interpolation in config (immediate correctness for
   config-driven Hydra composition) + runtime assertion (catches any
   case where config and actual disagree).

**Recommendation: path 3**, but only *option 2* portion initially if
we want to avoid touching `DiffusionModule.on_fit_start`. Single
commit change in SBM config:
```yaml
# discrete_sbm_official.yaml
model:
  model:
    extra_features:
      max_n_nodes: ${data.num_nodes_max_static}
```
Plus adding `num_nodes_max_static: 20` to synthetic-SBM data configs
and `num_nodes_max_static: 200` to SPECTRE data configs.

Runtime assertion (path 1) can be added later as a hardening step.

## Low-priority items — USER DECISIONS (2026-04-21)

### D-12 — Absorbing transition (#12): **PLAN + IMPLEMENT**

Build the correct absorbing-state transition from scratch (documenting
the upstream `u_e` vs `u_y` typo as a code comment, not copying it).
Target: `CategoricalNoiseProcess` gets an `absorbing` limit-distribution
variant alongside `uniform` and `empirical_marginal`. Include:
- `Q_t` construction: `(1 - β_t) · I + β_t · u ⊗ 1^T` where `u` is a
  one-hot indicator on the absorbing class.
- Unit test exercising forward kernel + posterior on a small graph.
- Docstring citing `noise_schedule.py:200-203` as the upstream bug
  and explaining why our version writes `u_y` separately.

### D-13 — `y_class` per-field CE wiring (#27): **WIRE UP**

Add `y_class` (and `y_feat` for symmetry) to `GRAPHDATA_LOSS_KIND`,
add `masked_y_ce(pred_y_logits, true_y, ...)` helper, register default
`lambda_y=0.0` in `lambda_per_field`, and extend the per-field loop to
handle the new field. Enables molecular datasets without changing SBM
behaviour (λ_y=0 → y-term contributes zero).

Note: `lambda_y` knob (originally #44 in the enumeration) is coupled
to this item — exposing the per-field helper without the config knob
would be incoherent. Treat #44 as subsumed by D-13.

### D-14 — `#28` wire-up clarification

User wrote "28: wire up" in the decision list. #28 was already
resolved under D-5 (unnormalised-posterior helper with zero_floor +
symmetry toggle). Interpreting "wire up" as **confirm D-5 implementation
is in scope for the current work batch**, not a separate decision.
If the user meant a different item (e.g., #44 which pairs with #27),
they should clarify — but #44 is now folded into D-13 above.

### D-15 — EMA callback + utility (#45): **PLAN + IMPLEMENT**

Add a minimal `ExponentialMovingAverage` utility (self-contained,
~30 LoC, either copied from `torch_ema` reference or written fresh)
at `tmgg/src/tmgg/training/ema.py`, and an `EMACallback` Lightning
callback at `tmgg/src/tmgg/training/callbacks/ema.py`.

Callback behaviour:
- `on_fit_start`: instantiate EMA wrapping `module.model.parameters()`.
- `on_train_batch_end`: `ema.update()`.
- `on_validation_start`: `ema.store()` + `ema.copy_to(model)` so
  validation/sampling use EMA weights.
- `on_validation_end`: `ema.restore()` so training resumes with live
  weights.

Config gate: `ema_decay: 0.0` disables the callback; non-zero enables
with that decay. Matches upstream's `cfg.train.ema_decay > 0` gate
pattern but with a real implementation.

Include a unit test verifying EMA parameters are updated and that
validation mode swaps in EMA weights then restores.

### D-16 — Upstream config-surface features (#46): **SPEC + PLAN + IMPLEMENT all four**

Four sub-items, each gets a spec-then-implement cycle.

**D-16a — `chain_saving_parameters`** (reverse-chain PMF snapshots for
visualisation):
- Spec: at validation generation, save the per-step PMF of a small
  number of chains to disk for later rendering (upstream generates
  trajectory plots from these). Parameters: `num_chains_to_save: int`
  (e.g., 3), `chain_save_path: str | None`, `snapshot_step_interval: int`
  (save every K reverse steps).
- Implementation site: `GraphEvaluator.evaluate` or a new
  `ChainSavingCallback` that hooks the sampler's reverse loop.

**D-16b — `final_model_samples_to_generate`** (post-training sample
dump):
- Spec: after `trainer.fit()` completes, if configured, run the
  sampler once with `N = final_model_samples_to_generate` samples
  and write to `results/final_samples.pt` + an evaluation summary.
- Implementation site: `DiffusionModule.on_fit_end` or a separate
  Lightning callback.

**D-16c — `evaluate_all_checkpoints`** (CLI subcommand iterating
saved checkpoints):
- Spec: `tmgg-discrete-gen evaluate_all_checkpoints --run_dir
  PATH` iterates all `*.ckpt` files in the checkpoint directory,
  loads each, runs the evaluator on `num_samples` generated graphs,
  and writes a CSV `{checkpoint: metrics...}` table.
- Implementation site: new CLI entry under
  `src/tmgg/cli/evaluate_all.py` (or extend the existing `tmgg-modal`
  CLI surface).

**D-16d — `sample_every_val`** parity:
- Already covered by D-10 (keep step-based gate, document conversion
  formula). No extra action under D-16.

Deliverables: four spec documents under
`docs/specs/2026-04-22-upstream-config-surface-{a,b,c}.md` before
implementing each, so the design can be reviewed incrementally.

---

## D-3 resolution rationale (user follow-up — RESOLVED as Option A)

**User's follow-up**: "why can we not port the sparse edge counts AND
keep it clean single path? there should be a way to do a drop-in
interface?"

**Answer: we can, and it's the right design.** My earlier framing
conflated two different senses of "single path":

1. **Hot-loop single path** (training forward pass): every batch flows
   `DataLoader → GraphData (dense) → noise_process.forward_sample →
   model → CE`. Immutable by D-3; unchanged regardless of which π
   estimator we pick.
2. **Preprocessing π estimation**: one-time computation at
   datamodule setup, used to fill the `π_E` / `π_X` stationary-PMF
   buffers on the `CategoricalNoiseProcess`. This runs once, not per
   batch.

π estimation is preprocessing, not hot-loop. Whether it reads from a
sparse `edge_index` or a dense `E_class` does not affect training
path single-ness. I was conservative for no good reason.

**Clean drop-in interface sketch** (preserves dense training, adopts
upstream sparse counting):

```python
# tmgg/src/tmgg/data/utils/edge_counts.py  (new)
def count_edge_classes_sparse(
    pyg_batch: torch_geometric.data.Batch,
    num_edge_classes: int,
) -> torch.Tensor:
    """Count one-hot edge-class occurrences across a PyG batch.

    Mirrors upstream DiGress's `AbstractDatasetInfos.edge_counts`
    (``digress-upstream-readonly/src/datasets/abstract_dataset.py:50-72``):
    iterate over edges in the sparse representation, one-hot the
    edge_attr, sum; add the implicit no-edge count from the total
    `N(N-1)` pair count minus actual edges.

    Returns a (K,) tensor of integer counts (un-normalised).
    """
    ...  # ~20 LoC port of upstream's routine

# tmgg/src/tmgg/diffusion/noise_process.py
def initialize_from_data(self, datamodule):
    """π_E / π_X estimator, preprocessing only."""
    total = torch.zeros(self.num_edge_classes)
    for pyg_batch in datamodule.train_dataloader_raw_pyg():  # ← new accessor
        total += count_edge_classes_sparse(pyg_batch, self.num_edge_classes)
    self._limit_e.copy_(total / total.sum())
    # similar for X
```

**Minimal surface change**:
- New file `data/utils/edge_counts.py`: ~25 LoC (upstream port).
- Data modules expose a `train_dataloader_raw_pyg()` method (or a
  `raw_pyg_iter` property) that yields PyG batches *before* the
  `GraphData` densification collator. Most data modules already hold
  the PyG loader internally; exposing it is additive, ~5 LoC per
  module.
- `NoiseProcess.initialize_from_data` switches to the new helper:
  ~10 LoC change.

Training hot loop: untouched.
Dense `GraphData` pipeline: untouched.
π values: bit-identical to upstream's.

**Single-path principle preserved**: the training flow is still
DataLoader → GraphData (dense) → model. The preprocessing pass is
just a separate one-time statistical query that reaches into the PyG
side, which is the canonical data representation anyway (GraphData
is a densified *view* of it).

**Resolution**: **Option A** — port the sparse `edge_counts`.
Upstream structural parity *and* clean architecture. Previous framing
was wrong; see the D-3 decision at the top of this document for the
final sketch and implementation scope.

**Upstream approach** (sparse edge_index pass via PyG DataLoader):
`abstract_dataset.py:50-72`. Iterates over the PyG train loader,
`F.one_hot(data.edge_attr)` on each batch, `sum(dim=0)` over edges,
computes the "no-edge" complement from `N_max*(N_max-1)` pair count,
normalises to a `(K,)` PMF.

**Our approach** (dense upper-triangle on already-densified
`GraphData.E_class`): iterate over training batches that are already
dense, extract `triu(E_class, 1)`, flatten, `sum(dim=0)`, normalise.

### Tradeoffs

| Dimension | Upstream (sparse `edge_index`) | Ours (dense upper-tri) |
|---|---|---|
| **Numerical result** | Identical `(K,)` PMF | Identical `(K,)` PMF |
| **Memory** | `O(|E|)` per batch — scales with edge count | `O(B × N² × K)` per batch — scales with max-N² |
| **Wall time** | Fast on sparse graphs; dominated by PyG DataLoader overhead | Fast on dense tensors; dominated by tensor ops |
| **Code organisation** | Requires a sparse iteration outside the normal densification path | Self-contained; operates on the same `GraphData` shape the rest of the pipeline consumes |
| **Cross-reference parity** | Direct line-by-line match with upstream; grep for `edge_counts` finds upstream's and our code identically | Diverges structurally from upstream; reviewer has to translate mentally |
| **Maintenance** | We'd need to keep a sparse path alive even though our training is dense | Single code path for all π computations |
| **Risk of drift** | Two independent paths (sparse for `edge_counts`, dense elsewhere) must agree; future refactor could diverge | One path; invariant by construction |
| **Debuggability** | Easier to diff against upstream line-by-line | Easier to inspect values in-situ with the rest of our debug tooling |
| **Porting cost** | ~20 LoC to add the sparse loop + retire the dense version | 0 LoC (already done) |

### Summary

**If priority is bit-for-bit structural parity with upstream for easier
auditing** → port upstream's sparse `edge_counts`.

**If priority is clean, single-path architecture** → keep dense upper-tri.
Numerical result is already identical; audits in §4 and the Opus
reviewer both verified this.

### Your call

A (port sparse) or B (keep dense)? I still lean B on engineering
grounds but the case for A is "reviewers shouldn't have to translate."


## §1 Problem representation

### 1. `X_class=None` synthesised as dX=2 vs upstream dX=1 — \*

- **Divergence**: We emit `[1-node_ind, node_ind]` (two channels) when
  `X_class is None`; upstream's structure-only path uses a trivial `dX=1`
  node type.
- **Instruction**: "should be okay, but need to verify no induced bugs."
- **Response**: Mechanism check. On SBM, the only consumer of `X_class` in
  the loss is `masked_node_ce` via the per-field loop. Padding rows in our
  synthesised target are `[0, 0]` post-commit `82bcec26`/`f6d99185`
  (`_read_field` multiplies by `node_ind`); valid rows are `[0, 1]`. The
  row predicate `(true != 0).any(-1)` keeps only valid rows → CE on a
  two-class target where every valid row is class 1. Upstream's `dX=1`
  trivially has no loss contribution. Mathematically: our `loss_X` is
  exactly `-log softmax(logits)[class 1]` summed over valid nodes; since
  `lambda_X=1` and this is a trivial classification the model converges
  to constant class-1 prediction and the loss floor is ~0. Contributes
  nothing to gradient on E. No induced bug. **NO-OP**.

### 2. `remove_self_loops` sparse vs post-densification `adj[diag]=0` — \*

- **Divergence**: Upstream removes self-loops in the sparse representation
  before `to_dense_adj`; ours densifies first, then zeros the diagonal.
- **Instruction**: "can we add remove_self_loops in the sparse section
  instead for no change, or add both if we require post densification
  because of entanglement? answer/discuss/get final decision before
  writing."
- **Response**: Checked `graph_types.py:488-504`. Our flow:
  ```
  adj = to_dense_adj(edge_index, batch_vec)     # dense
  adj[:, diag, diag] = 0.0                       # zero diag
  adj = (adj + adj.transpose(1, 2)).clamp(max=1) # symmetrise
  ```
  Upstream flow: `remove_self_loops(edge_index, edge_attr)` before
  `to_dense_adj`. **Both approaches produce the same dense tensor** when
  the input graph is simple and undirected — the diagonal in
  `to_dense_adj`'s output either comes from actual self-loop entries (which
  `remove_self_loops` strips) or from zero-initialised cells (which
  `to_dense_adj` leaves at zero). Our post-densification zero is
  defensively redundant.
  - **Option A (upstream-matching, small)**: import
    `torch_geometric.utils.remove_self_loops`, call it on `edge_index`
    before `to_dense_adj`. Drop our `adj[:, diag, diag] = 0`. Keeps the
    later `E_class[:, diag, diag, :] = 0` (commit `82bcec26`) as the CE
    predicate invariant.
  - **Option B (keep both)**: add upstream-style `remove_self_loops` AND
    retain our post-densification zero. Truly defensive, slight cost.
  - **Recommendation**: Option A. Upstream parity + one fewer
    explanation in the code. **DECISION NEEDED.**

### 3. `collapse=True` as separate `collapse_to_indices` method — \*

- **Divergence**: Upstream's `PlaceHolder.mask(collapse=True)` folds the
  class dimension back to integer indices; ours exposes that as a
  separate `collapse_to_indices` method.
- **Instruction**: "make a flag."
- **Response**: **FIX.** Rename `collapse_to_indices` → add `collapse`
  kwarg to `GraphData.mask()`. Behaviour: `mask(collapse=False)` (default)
  returns the current `GraphData` with same class dims;
  `mask(collapse=True)` also calls the index collapse. Keeps caller
  parity: anywhere upstream writes `data.mask(node_mask, collapse=True)`,
  ours becomes `data.mask().collapse_if(True)` or (cleaner)
  `data.mask(collapse=True)`. Straightforward mechanical change in
  `graph_types.py`.

### 4. No symmetry assertion inside `from_pyg_batch` — \*

- **Divergence**: `.mask()` asserts `E_class` symmetry, but
  `from_pyg_batch` doesn't — so the first symmetry check happens at
  loss-time.
- **Instruction**: "add assertions."
- **Response**: **FIX.** After the `adj = (adj + adj.transpose) / ... `
  step in `from_pyg_batch` (currently
  `graph_types.py:504`), add `assert torch.allclose(adj,
  adj.transpose(-2, -1))` before the stack. Fails loudly at data-
  construction time. Zero runtime cost at fp32.

### 5. `SpectreSBMDataModule.num_nodes` derived from all splits — \*

- **Divergence**: Upstream computes `max_n_nodes` from train+val only
  (`abstract_dataset.py:95-100`); ours uses all splits including test.
- **Instruction**: "match 1:1, leaks test info."
- **Response**: **FIX.** Narrow to train+val only in
  `SpectreSBMDataModule.setup()` / the derived-max computation. Correct
  evaluation hygiene. Trivial one-line change.

---

## §2 Discrete diffusion formalism

### 6. Architecture collapse (two-object → one-object) — †

- **Divergence**: We merge `PredefinedNoiseScheduleDiscrete` +
  `MarginalUniformTransition` into `CategoricalNoiseProcess +
  NoiseSchedule`.
- **Instruction**: "okay, if math and effective implementation behaviour
  truly identical, else flag and we discuss."
- **Response**: §5 review found no numerical divergence; `_mix_with_limit`
  is algebraically identical to `x @ Q̄_t` for one-hot `x` under marginal
  kernels. Confirmed MATCH at multiple audit passes. **NO-OP.**

### 7. Inline closed form `_mix_with_limit` — †

- **Divergence**: We skip materialising `Q̄_t` matrix; use
  `ᾱ x + (1-ᾱ) π` directly.
- **Instruction**: "if only API change and math + effective behaviour
  identical, fine, else flag and we discuss."
- **Response**: Math identity is exact: for one-hot x, `x @ Q̄_t =
  x @ (ᾱI + (1-ᾱ)·1 π^T) = ᾱ·x + (1-ᾱ)·π` (since `x @ 1 = 1` when x is
  a PMF). Effective behaviour identical, strictly cheaper in memory.
  **NO-OP.**

### 8. `custom_vignac` schedule: `p=4/5` vs `1-1/K` — \*

- **Divergence**: Upstream hardcodes `p=4/5` (K=5 molecular case); ours
  parameterises as `1-1/K`.
- **Instruction**: "if our parameterization recovers theirs exactly,
  keep, but set default to upstream."
- **Response**: For K=5, `1 - 1/5 = 0.8 = 4/5` — exact recovery. **FIX:**
  set the default value of K in `custom_vignac` to 5 so calling it
  without arguments gives `p=4/5` (matching upstream behaviour); keep
  the parameterisation so non-molecular callers can still pass other K.
  Trivial default change.

---

## §3 Noise schedule

### 9. Buffer registration order — \*

- **Divergence**: We buffer `alphas`, `alphas_bar`, `betas`; upstream
  only `betas`.
- **Instruction**: "ok, sounds correct."
- **Response**: Confirmed safer for `.to(device)` + `.half()` ops.
  **NO-OP.**

### 10. Bounds validation with informative errors — \*

- **Divergence**: We validate `t_int ∈ [0, T]`; upstream raises bare
  `IndexError`.
- **Instruction**: "ok."
- **Response**: **NO-OP** (kept as tmgg improvement).

### 11. `alpha_bar` 1e-5 floor on linear path — †

- **Divergence**: We clip `alpha_bar` at 1e-5 inside
  `_register_schedule_from_alpha_bar` (which is only invoked for
  `schedule_type="linear_ddpm"`). The `cosine_iddpm` path uses
  `_register_schedule_from_betas` instead and inherits upstream's
  `beta` clip at `0.9999`.
- **Instruction**: "what does upstream do here? are we deviating?"
- **Response**: Upstream has no `linear_ddpm` schedule at all — it
  implements `cosine` and `custom` only, both deriving `alpha_bar` from
  betas (docstring at `schedule.py:133-134` says so). So the 1e-5 floor
  exists in a code path upstream has no equivalent for; it's our
  extension for a schedule upstream doesn't implement. No deviation
  on shared paths: our `cosine_iddpm` path clips `betas` at 0.9999
  exactly like upstream (`schedule.py:151`, matching upstream
  `diffusion_utils.py:67`). `alpha_bar[T] ≈ 2.4e-10` matches upstream.
  **NO-OP**; could document in `schedule.py:146` comment that the 1e-5
  floor is only a linear-DDPM extension safeguard with no upstream
  counterpart (linear-DDPM itself is our extension).

---

## §4 Limit distribution

### 12. Absorbing variant absent — \*

- **Divergence**: Upstream has `AbsorbingStateTransition` (with a `u_e`
  vs `u_y` copy-paste bug); we don't implement it at all.
- **Instruction**: "implement the correct version, just for
  completeness, note fixed typo."
- **Response**: **FIX (small).** Add a `CategoricalNoiseProcess` variant
  (or an `absorbing` branch) that implements the correct transition —
  `Q_t = (1-β_t)·I + β_t · u ⊗ 1^T` where `u` is a one-hot on the
  absorbing class. Write a unit test that exercises the forward kernel.
  Add a doctring note citing the upstream typo
  (`noise_schedule.py:200,203` writes `u_e` twice where the second
  should have been `u_y`). Not needed for current experiments but worth
  having for future ablations / completeness. Low priority.

### 13. `π` estimation: upper-tri dense vs edge_index — †

- **Divergence**: We count over upper-triangular dense indices; upstream
  uses `edge_index` (both directions) + non-edge complement.
- **Instruction**: "can we use the upstream version without breakage?
  just for exact match?"
- **Response**: Discussion. Upstream's `edge_counts` in
  `abstract_dataset.py:50-72` does:
  ```python
  for data in loader:
      E = F.one_hot(data.edge_attr, num_classes=self.num_edge_classes).float()
      edge_types[1:] += E.sum(dim=0)
      # then adds the "no-edge" count for missing pairs
  ```
  Our version operates on dense `E_class` tensors. Both produce the same
  `(K,)` normalised PMF in the end (confirmed numerically in §4 review).
  Porting upstream's approach is feasible but requires either (a) keeping
  a sparse PyG iteration at `initialize_from_data` time, or (b) a
  separate dense↔sparse roundtrip. **DECISION NEEDED.** Cost is ~20 LoC
  in `noise_process.py` + a corresponding test. Benefit is strict
  structural parity; numerical result is unchanged. I lean toward
  **keeping ours** because it's cleaner given we already consume dense
  `GraphData` batches downstream — but if strict structural parity is the
  goal, port it.

---

## §5 Transition matrices

### 14. `denom.clamp(min=1e-6)` in posterior — \*

- **Divergence**: We clamp the posterior denominator at 1e-6 in
  `diffusion_sampling.py:180`; upstream does not.
- **Instruction**: "explain what happens on healthy and unhealthy
  posteriors, how do we differ in behaviour."
- **Response**:
  - **Healthy posterior** (denom > 1e-6 everywhere): identical output.
    Clamp is inert.
  - **Degenerate posterior** (denom ≈ 0 somewhere — e.g., the product
    `(z_t @ Qt^T) ⊙ (x̂_0 @ Qsb)` sums to zero at some position because
    `x̂_0` placed all its mass on classes that are incompatible with
    `z_t`'s current state):
    - Upstream: `prob = product / 0 = inf/nan`; downstream
      `multinomial` will error or sample pathologically.
    - Ours: `prob = product / 1e-6 ≈ 1e6 × product`, which renormalises
      later via multinomial's scale-invariance. Practically: we produce
      a uniform-ish distribution over whichever classes have any
      product mass, which then samples something reasonable.
  - **So the behaviour difference is**: upstream hard-fails on a
    degenerate prediction; ours soft-recovers. Our behaviour is more
    robust, theirs is more diagnostic. On a healthy trained model the
    divergence never fires.
  - CLAUDE.md prefers fail-loud. **FIX:** remove the clamp, add an
    `assert (denom > 0).all(), f"degenerate posterior at t=..."` just
    above the division. Trains the same, fails loudly in the failure
    regime.

### 15. Row-sum invariant: upstream asserts, we don't — \*

- **Divergence**: Upstream asserts `Q_t_bar` row-sums equal 1; we rely
  on algebra.
- **Instruction**: "add runtime assertion."
- **Response**: **FIX.** Add `assert torch.allclose(Q_t_bar.sum(dim=-1),
  torch.ones_like(...), atol=1e-5)` wherever we materialise or
  inline-compute posterior distributions. `_mix_with_limit` preserves
  sums algebraically, but a runtime assertion guards against future
  refactors. Zero perf cost on fp32 at normal batch sizes.

---

## §6 Forward noising

### 16. `t ∈ {1..T}` training range — \*

- **Divergence**: We hardcode `randint(1, T+1)`; upstream uses
  `lowest_t = 0 if training else 1`.
- **Instruction**: "fix, does the fix have any downstream implications?
  discuss before."
- **Response**: Implications of sampling `t=0` at training:
  - `alpha_bar[0] ≈ 1.0 - epsilon` (almost fully clean). Forward
    noising with t=0 gives `z_0 ≈ x_0` with tiny noise.
  - The model is trained on a near-identity reconstruction task for
    1/T of its gradient updates — useful for calibrating the denoiser
    at the endpoint of the reverse chain.
  - Upstream runs this by default; our exclusion is the 0.1% gap.
  - Downstream impact: `validation_step` (`diffusion_module.py:652`)
    uses `randint(1, T+1)` too — that's *correct* for validation (no
    identity task at validation time), so don't change there.
  - **Proposed fix**: change `training_step` at `diffusion_module.py`
    to use `randint(0, T+1)`. Leave `validation_step` alone. Document
    the asymmetry in a code comment.
  - **Subtle point**: at `t=0`, `forward_sample` returns `z_0 = x_0`
    (since ᾱ₀ = 1 exactly; actually we have `ᾱ[0] ≈ 0.9999` from the
    cosine clip). Edge case to test: the model should handle it.

### 17. `forward_sample` returns `GraphData`, not dict — †

- **Divergence**: We return a `GraphData` with just the noised tensors;
  upstream returns a dict with `t_int, t, beta_t, alpha_s_bar,
  alpha_t_bar, X_t, E_t, y_t, node_mask`.
- **Instruction**: "how can we make this cleaner? discuss."
- **Response**: Options:
  - **A. Return a richer `NoisedBatch` dataclass** with
    `z_t: GraphData` plus the schedule scalars. Type-safe, extensible.
    ~30 LoC.
  - **B. Keep `GraphData` return and pre-compute schedule scalars on
    demand** in the VLB path. Cheap but leaves the "silent coupling"
    — the VLB path has to know the schedule to re-derive.
  - **C. Match upstream dict exactly.** Pragmatic, but loses our
    typed-GraphData architecture at a single call site, weird.
  - **Recommendation**: **A**. Introduce
    ```python
    @dataclass
    class NoisedBatch:
        z_t: GraphData
        t_int: Tensor            # (B,) integer timesteps
        alpha_t_bar: Tensor      # (B,)
        alpha_s_bar: Tensor      # (B,) where s = t-1
        beta_t: Tensor           # (B,)
    ```
    Callers that only need `z_t` do `noised.z_t`; VLB callers use the
    scalars directly. Clean and explicit. **DISCUSS / DECISION.**

### 18. `t_int` shape `(B,)` vs `(B, 1)` — \*

- **Divergence**: Shape only.
- **Instruction**: "how do they differ, any bug possibilities?"
- **Response**: PyTorch broadcasting makes them numerically identical
  for every operation in the schedule/transition lookup path. Bug
  possibilities:
  - If code writes `t_int[:, 0]` assuming `(B, 1)`, our `(B,)` tensor
    would index on the batch dim → wrong result. Grep confirms no such
    usage in our codebase (the noise process handles both shapes via
    explicit reshape).
  - If code does `t_int + 1` expecting `(B, 1)` for broadcast against a
    `(B, 1, K)` tensor, our `(B,)` broadcasts against `(..., K)` ok.
  - **Verdict: NO-OP** but worth matching upstream's `(B, 1)` for
    consistency if we do the broader `forward_sample` refactor (#17).
    Ties cleanly into the `NoisedBatch` dataclass approach.

---

## §7 Extra features

### 19. `EigenvectorAugmentation` class — †

- **Divergence**: We have a class upstream doesn't; off by default.
- **Instruction**: "okay, any behaviour diff induced?"
- **Response**: Off by default means no behaviour diff when the default
  `extra_features_type='all'` is used. The failing run uses
  `extra_features_type='all'` (no `EigenvectorAugmentation`). No induced
  diff. **NO-OP**.

### 20. `DummyExtraFeatures + use_timestep` combination — †

- **Divergence**: We expose a toggle upstream doesn't.
- **Instruction**: "dito" (= "okay, any behaviour diff induced?").
- **Response**: Same reasoning. The failing run does not enable
  `DummyExtraFeatures`; upstream's extra_features pipeline runs end-to-end
  in ours too. **NO-OP**.

### 21. `cycles` mode dim-count cosmetic difference — \*

- **Divergence**: Tensor-shape cosmetics.
- **Instruction**: "explain, any behaviour diff?"
- **Response**: Both produce 4 cycle features (triangles, 4/5/6-cycles);
  shape metadata reported differently internally. Output tensor shape
  and values at the call site are identical. **NO-OP**.

### 22. Eigenvector masking expression — \*

- **Divergence**: `mask_2d(node_mask)` vs `mask.unsqueeze(2) *
  mask.unsqueeze(1)`.
- **Instruction**: "any behaviour diff?"
- **Response**: Produces the same boolean tensor. **NO-OP**.

---

## §8 Architecture

### 23. `use_gnn_q`, `use_spectral_q` flags — †

- **Divergence**: Optional GNN/spectral Q/K/V projections we expose;
  upstream doesn't have them.
- **Instruction**: "ok if no behaviour diff when off."
- **Response**: §8 review confirmed the linear path with both off is
  byte-for-byte equivalent to upstream. Failing run has both off.
  **NO-OP**.

### 24. `use_timestep=True` GraphTransformer flag — †

- **Divergence**: We can inject `t/T` into `y` inside the transformer
  wrapper when `use_timestep=True`. Upstream injects `t/T` as the last
  dimension of the `ExtraFeatures` output, not via a separate
  transformer-level flag.
- **Instruction**: "any behaviour diff/redundancy here? I thought we
  wanted noise process to own this via the feature? discuss."
- **Response**: Redundancy risk is real. Our `ExtraFeatures.__call__`
  already appends `t/T` to `y` at the feature-construction boundary
  (confirmed in §7 review). The separate `use_timestep=True` flag on the
  transformer wrapper would then append it a **second** time if both are
  on. Check: the discrete_sbm_official config has
  `use_timestep: true` — does this trigger the wrapper-level injection
  AND the extra_features injection?
  - Let me trace: `transformer_model.py:944` calls
    `self.extra_features(X, E, y, node_mask)` and that class handles
    timestep if configured. The `use_timestep` flag at the wrapper level
    *also* appends `t/T` to `y` before the transformer body runs.
    **Risk of double-injection.**
  - **DISCUSS / DECISION**: (a) decide which owner is canonical; (b) if
    extra_features owns it, make `use_timestep` flag on the wrapper a
    no-op (or remove it); (c) if the wrapper owns it, ExtraFeatures
    should not re-append. Priority: medium — this could mean our
    `y`-vector has an extra dim compared to upstream, which would
    trigger an input-dim mismatch at the first FiLM layer… unless the
    config's `input_dims.y` compensates. Worth a grep + careful read.

### 25. `masked_softmax` all-masked edge case — \*

- **Divergence**: Our `masked_softmax` returns zeros when *all* positions
  are masked; upstream returns a passthrough (the raw inputs).
- **Instruction**: "can we match upstream here safely? if not, how does
  this affect behaviour?"
- **Response**: The all-masked case in attention means: every row of the
  attention-score matrix has no valid key. Upstream's passthrough
  returns un-normalised raw scores (which are then dot-producted with
  values → arbitrary garbage that's downstream multiplied by 0 anyway
  because the value rows are also masked). Ours returns zeros, which
  gives the same final zero output via a different path.
  - Both produce zero downstream once the mask is re-applied on the
    output side.
  - **Safety**: matching upstream (returning passthrough) is safe —
    the downstream mask catches it.
  - **FIX-LITE:** change our `masked_softmax` to return the raw input
    tensor when the mask is all-False, wrapped in a `torch.where`. One
    line. Or keep our zero behaviour and add a comment that this is
    the intentional divergence. Either way, confirm it's behaviourally
    equivalent end-to-end by a unit test with an all-masked batch.

---

## §9 Training objective

### 26. `label_smoothing` kwarg exposed — †

- **Divergence**: We have a `label_smoothing: float = 0.0` knob upstream
  doesn't.
- **Instruction**: "ok if we match upstream now."
- **Response**: Default 0.0 preserves bit-for-bit parity. Pinned at
  `atol=1e-6` by `TestUpstreamParity`. **NO-OP**.

### 27. `y`-field CE absent from per-field loop — †

- **Divergence**: Our per-field loop in `_compute_loss` handles
  `X_class`, `E_class`, `X_feat`, `E_feat` but not `y_class`. Upstream
  has `loss_y` with `lambda_train[1]` weight.
- **Instruction**: "how would we need to wire? need to be able to match
  upstream."
- **Response**: For SBM the `y`-loss weight is 0 anyway, so exclusion is
  benign. For molecular data (future), upstream predicts molecular
  property y alongside X, E. **Wiring plan:**
  - Add `y_class` and/or `y_feat` to `GRAPHDATA_LOSS_KIND` dict.
  - Add `masked_y_ce(pred_y_logits, true_y, ...)` helper — simpler than
    node/edge CE because `y` is `(B, D_y)` without spatial/mask
    dimensions. Just a call through to `F.cross_entropy`.
  - Add `lambda_y` field to `lambda_per_field` defaulting to 0.
  - The existing per-field loop then handles it transparently.
  - ~30 LoC change, zero impact on SBM training. **FIX, low priority.**

---

## §10 Reverse sampling

### 28. Missing post-contraction renormalisation + zero-guard — \*

- **Divergence**: Upstream does `unnorm / sum` with `sum==0 → 1e-5`
  floor on the predicted posterior; we don't.
- **Instruction**: "important: fix, discuss how this can affect
  behaviour, sketch clean fix (we want to maintain our clean code and
  abstractions, so don't just hack this in)."
- **Response**:
  - **Current behaviour**: scale-invariant under `torch.multinomial` so
    the sampling distribution is unaffected for healthy posteriors. On
    a degenerate row (all-zero product), ours hard-errors inside
    `multinomial` because it requires nonnegative weights with nonzero
    sum; upstream's 1e-5 floor degrades gracefully to uniform-ish.
  - **Behaviour impact**: identical for a trained model; different only
    under pathological prediction. If the model predicts a posterior
    that's incompatible with `z_t` (e.g., due to numerical issues at
    extreme t values), ours crashes where upstream keeps sampling.
  - **Clean fix sketch** (new posterior helper in `diffusion_sampling.py`):
    ```python
    def _sample_from_unnormalised_posterior(
        unnorm: Tensor,                 # (bs, N, K) or (bs, N, N, K)
        *,
        zero_floor: float = 1e-5,
    ) -> Tensor:
        denom = unnorm.sum(dim=-1, keepdim=True)
        # Upstream parity: degenerate rows (sum ≈ 0) fall back to
        # uniform-ish via the floor. On healthy posteriors this is inert.
        denom = denom.clamp(min=zero_floor)
        prob = unnorm / denom
        # Optional: assert per-row sums are ~1 post-normalisation.
        flat_prob = prob.reshape(-1, prob.size(-1))
        return torch.multinomial(flat_prob, 1).reshape(prob.shape[:-1])
    ```
    Call from both `CategoricalNoiseProcess.posterior_sample_*` paths.
    Replaces the current ad-hoc `denom.clamp(min=1e-6)` with a single
    well-named helper that is the canonical upstream-parity sampling
    primitive. Add `assert (unnorm >= 0).all()` as a precondition.
  - Also consider exposing `zero_floor` on the `CategoricalSampler`
    config so strict-parity runs can force `zero_floor=1e-5` (upstream
    value).
  - **FIX.**

### 29. No runtime per-step symmetry assertion — \*

- **Divergence**: Upstream asserts E-symmetry after every reverse step
  in `sample_p_zs_given_zt`. We rely on structural guarantees from
  `triu + transpose`.
- **Instruction**: "see above" (= "important: fix, discuss behaviour,
  sketch clean fix with clean abstractions").
- **Response**: Behaviour impact: identical on correct code — the
  assertion is a defensive check that would fire only on a regression.
  **Clean fix sketch**:
  - Add an `assert_symmetric_e: bool = True` parameter to the Sampler /
    noise process sampling loop (default True for debug, can be disabled
    in production hot loops).
  - In the reverse step's post-sample code, wrap the symmetry check:
    ```python
    if self._assert_symmetric_e and z_t.E_class is not None:
        assert torch.allclose(
            z_t.E_class, z_t.E_class.transpose(-3, -2)
        ), f"E-symmetry broken at step t={t}"
    ```
  - Expose via the `CategoricalSampler` config. Zero perf cost in
    practice; catches sampling regressions immediately. **FIX.**

---

## §11 VLB

### 30. Marginalised `kl_diffusion` posterior — †

- **Divergence**: We use the marginalised form `Σ_c p(z_s|z_t, x_0=c) ·
  p_θ(x_0=c|z_t)`; upstream plugs the soft `x_0` into Bayes directly.
- **Instruction**: "upstream is closed form? does our api allow plugging
  in both?"
- **Response**: Upstream is closed-form in the sense that it plugs
  `pred_probs_X = F.softmax(pred.X)` directly into
  `posterior_distributions(X=pred_probs_X, E=..., ..., Qt=Qt, Qsb=Qsb,
  Qtb=Qtb)` which returns the posterior distribution parameters. No
  explicit marginalisation loop over classes.
  Our marginalised form sums `p(z_s | z_t, x_0=c)` over one-hot class
  indicators `c`, weighted by `p_θ(x_0=c|z_t)`. Algebraically this
  equals upstream's plug-in form when the soft `x_0` is treated as a
  mixture over one-hots. At convergence (one-hot `x_0`) they coincide
  exactly; during training they differ.
  - **API for both**: our `CategoricalNoiseProcess` already has
    `_posterior_probabilities` (plug-in / direct) and
    `_posterior_probabilities_marginalised`. So the infrastructure is
    there; the VLB path specifically picks the marginalised form at
    `diffusion_module.py:680-685`.
  - **Options**: (a) add a `use_marginalised_kl: bool = True`
    parameter to `DiffusionModule.__init__` (default keeps our current
    behaviour, flipping reproduces upstream). (b) Always use
    upstream's plug-in form. Decision depends on whether we think the
    marginalised form is a real improvement or just an accidental
    choice.
  - **DISCUSS / DECISION.** Either way, document the choice explicitly.

### 31. `reconstruction_logp` via `z_1`+marginalised vs `z_0`+softmax — †

- **Divergence**: We use `z_1` + marginalised posterior; upstream uses
  `z_0` + raw softmax scoring.
- **Instruction**: "explain behaviour diff, can we follow upstream with
  our impl cleanly?"
- **Response**:
  - **Behaviour diff**: at `t=0`, `ᾱ_0 ≈ 1-epsilon`; forward sampling
    gives `z_0 ≈ x_0` with tiny noise. Upstream then asks the model
    `what is x_0 given z_0?` and scores its softmax log-probability on
    the clean `x_0`. This is the reconstruction term of the ELBO.
  - At `t=1`, `ᾱ_1 ≈ 1-slightly-more-epsilon`; `z_1` is a tiny bit
    noisier. We score via `p(x_0 | z_1)` using the marginalised
    posterior rather than the softmax.
  - Numerical gap: bounded at ~1e-3 abs. for T=1000 (per docstring at
    `diffusion_module.py:600-607`).
  - **Clean port to upstream's approach**:
    ```python
    def _compute_reconstruction_upstream_style(self, batch):
        bs = int(batch.node_mask.shape[0])
        t_zero = torch.zeros(bs, device=device, dtype=torch.long)
        z_0 = self.noise_process.forward_sample(batch, t_zero)
        condition_zero = self.noise_process.process_state_condition_vector(t_zero)
        pred_zero = self.model(z_0, t=condition_zero)
        # score raw softmax against clean batch
        pred_probs = ...softmax...(pred_zero)
        return log_prob_of_clean_under_pred_probs(batch, pred_probs)
    ```
    Doable cleanly. **DISCUSS / DECISION**: do we want strict upstream
    parity for `val/epoch_NLL`, or keep our marginalised-via-sampler
    alignment? The current choice was documented as intentional at
    commit time; reverting is fine if parity wins.

### 32. `log_pN` silent zero fallback — \*

- **Divergence**: We silently default to zero when `_size_distribution
  is None`.
- **Instruction**: "remove silent fallback, wtf."
- **Response**: **FIX.** Replace at `diffusion_module.py:702-705`:
  ```python
  if self._size_distribution is None:
      raise RuntimeError(
          "DiffusionModule._compute_log_pN called before setup(): "
          "_size_distribution is None. The datamodule must populate "
          "a SizeDistribution in setup() — see SpectreSBMDataModule / "
          "SyntheticCategoricalDataModule."
      )
  ```
  One-line edit. Zero risk on normal training (path always populated);
  loud failure on misconfiguration. CLAUDE.md-compliant.

---

## §12 Masking / symmetry / diagonal

### 33. `mask_zero_diag()` inside `_GraphTransformer.forward` — †

- **Divergence**: We zero the diagonal of the model OUTPUT inside the
  transformer wrapper; upstream only applies `.mask(node_mask)` (padding
  only) post-model.
- **Instruction**: "inside forward we should not zero I think, because
  the latent graph is allowed whatever, unless the upstream zeros
  implicitly, can we make this a toggle to control easily while keeping
  things clean?"
- **Response**: The user's intuition is right. The model's OUTPUT at the
  diagonal is informationally useless but non-zero output doesn't hurt
  — downstream the CE row-predicate drops diagonal rows anyway (via
  `from_pyg_batch` zeroing the target diagonal), and the sampler's
  `triu+transpose` handles symmetry. So the wrapper-level
  `mask_zero_diag()` is defensive / cleanliness, not correctness.
  - Upstream does NOT zero the diagonal of predictions (grep confirms).
  - **Clean fix**: introduce `zero_output_diagonal: bool = True`
    parameter on `_GraphTransformer.__init__`. Default `True` preserves
    current behaviour; strict-parity runs set `False`.
    ```python
    if self.zero_output_diagonal and out.E_class is not None:
        out = out.mask_zero_diag()
    else:
        out = out.mask()  # padding only, matching upstream
    ```
    Add a comment explaining upstream parity.
  - **FIX**. User preference on default: keep `True` (tmgg default,
    cleaner) or flip to `False` (upstream match). **DECISION NEEDED.**

### 34. Explicit symmetry `AssertionError` in prior-draw — †

- **Divergence**: We assert symmetry immediately after drawing the prior
  noise; upstream asserts it per-step instead (post-sample).
- **Instruction**: "behaviour diff? else fine."
- **Response**: No behaviour diff — different assertion location, same
  structural invariant. Both fire only on regression. **NO-OP**.

---

## §13 Evaluation metrics

### 35. `compute_sbm_accuracy` default `refinement_steps=1000` — \*

- **Divergence**: Upstream's live caller passes `refinement_steps=100`;
  our function default is 1000 and we never pass an override.
- **Instruction**: "match upstream, document, parametrize."
- **Response**: **FIX.**
  - Add `sbm_refinement_steps: int = 100` parameter to
    `GraphEvaluator.__init__`.
  - Pass it through to `compute_sbm_accuracy(...,
    refinement_steps=self.sbm_refinement_steps)`.
  - Document in the class docstring: "Default 100 matches upstream
    DiGress's `SpectreSamplingMetrics.forward` live value at
    `src/analysis/spectre_utils.py:830`. Setting higher (e.g., the
    function's own default 1000) gives tighter block-model fits but is
    10× slower."
  - Small config change.

### 36. `ProcessPoolExecutor` vs `ThreadPoolExecutor` — †

- **Divergence**: Deliberate deviation for graph-tool thread safety.
- **Instruction**: "okay."
- **Response**: **NO-OP** (documented deliberate deviation).

### 37. Uniqueness via `is_isomorphic` (precise) — †

- **Divergence**: Stricter than upstream's default `precise=False`, but
  matches upstream's SBM-metrics path exactly.
- **Instruction**: "ok, but document divergence."
- **Response**: **DOC.** Add a comment in `GraphEvaluator.evaluate`
  or the uniqueness helper clarifying: "`precise=True` via
  `is_isomorphic` — stricter than upstream's standalone
  `eval_fraction_unique(precise=False)` default; matches upstream's
  `SpectreSamplingMetrics` / `eval_fraction_unique_non_isomorphic_valid`
  path that the paper actually uses."

---

## §14 Lightning plumbing

### 38. LR `cosine_warmup` scheduler inherited by default — †

- **Divergence**: `_base_infra.yaml` silently provides a scheduler;
  upstream has none.
- **Instruction**: "fix default."
- **Response**: **FIX.** In `models/digress/*.yaml` (or the relevant
  parent config), either:
  - (a) set `scheduler_config: {type: none}` explicitly for upstream-
    match configs, or
  - (b) remove scheduler from `_base_infra.yaml` defaults and require
    opt-in.
  - Option (a) is narrower; preserves the scheduler for other model
    configs that want it. Option (b) is more principled. **DECISION
    NEEDED** on scope. I lean toward (a) — the scheduler is useful for
    experiments, just not for strict parity.

### 39. `gradient_clip_val=1.0` default — †

- **Divergence**: Our trainer default; upstream `null`.
- **Instruction**: "fix default."
- **Response**: **FIX.** Change trainer YAML default to `null`. Runs
  that want clipping can opt in explicitly. Aligns with "document
  divergences, don't inherit silently."

### 40. Early-stopping callback — †

- **Divergence**: We have one on `val/epoch_NLL`; upstream has none.
- **Instruction**: "make very long patience, document in comment that
  this is divergence, make sure we keep all checkpoints."
- **Response**: **FIX.**
  - Bump `patience` in the early-stopping callback config to a large
    value (e.g., 1000 epochs) so it effectively never fires in a normal
    run, preserving tmgg's safety net without breaking long training
    comparability with upstream.
  - Add a YAML comment: "Early-stopping divergence vs upstream DiGress,
    which has no ES. Patience set very high to approximate upstream's
    'train to budget' behaviour while retaining a safety net."
  - Verify checkpointing config: `save_top_k` on the monitor checkpoint
    callback should be -1 (keep all) or a high value, NOT the default
    3. Grep the config to confirm.

### 41. `eval_every_n_steps` step-based vs `sample_every_val` epoch-based — †

- **Divergence**: We gate generation by global_step; upstream gates by
  val_counter.
- **Instruction**: "match behaviour, but I think ours is valid? discuss."
- **Response**: Both are valid. Key tradeoff:
  - **Epoch-based** (upstream): generation cadence tied to epoch
    count. Natural when epoch = fixed dataset pass.
  - **Step-based** (ours): cadence tied to gradient updates. Better
    when batch size or dataset size changes between runs (step is the
    invariant).
  - For fixed-size SPECTRE (128 train / batch 12 ≈ 11 batches/epoch),
    both are trivially convertible. For variable-size runs they
    diverge.
  - **Recommendation**: keep step-based as tmgg default (more general)
    but expose a `gate: "step" | "epoch"` parameter on the generation
    callback. Upstream-parity runs can select "epoch". Document in the
    callback docstring.
  - **FIX (minor) / DISCUSS.**

---

## §15 Hyperparameters

### 42. `max_n_nodes=20` baked into model config — \*

- **Divergence**: The discrete_sbm_official.yaml hardcodes
  `max_n_nodes=20` in the `ExtraFeatures` block. Correct for synthetic-
  n=20; breaks if swapped with SPECTRE (n up to 187).
- **Instruction**: "fix, should probably not be baked in, discuss clean
  fix."
- **Response**: **FIX (DISCUSS).** Options:
  - **A. Hydra interpolation**: change model config to
    `max_n_nodes: ${data.num_nodes}` with a Hydra interpolation pointing
    at the data config's node-count field. Automatically tracks which
    data the model is paired with.
  - **B. Runtime injection**: `DiffusionModule.on_fit_start` computes
    `max_n_nodes` from the actual datamodule and writes it into the
    `ExtraFeatures` instance. More dynamic but silent.
  - **C. Validation assertion**: keep the config value but add
    `assert max_n_nodes >= datamodule.num_nodes_max` at setup time,
    failing loudly if the pair is incompatible.
  - **Recommendation**: **A + C combined**. Interpolation gives the
    right value automatically; assertion catches the misconfigurations
    that slip through. Clean, explicit, Hydra-idiomatic. Applies to
    every `ExtraFeatures`-using model config.

### 43. `diffusion_steps=500` YAML default, 1000 via CLI — \*

- **Divergence**: Static YAML default disagrees with the CLI override
  used in runs.
- **Instruction**: "fix to match upstream."
- **Response**: **FIX.** Change `discrete_sbm_official.yaml`
  `timesteps: 500` → `timesteps: 1000` to match upstream's default and
  the actual run behaviour. Remove the CLI override pattern from the
  launch scripts. Single-line YAML change.

### 44. No `lambda_y` knob — †

- **Divergence**: We only have `lambda_E`.
- **Instruction**: "fix, make sure we can match upstream here."
- **Response**: **FIX.** Related to #27.
  - Add `lambda_y: float = 0.0` to the `lambda_per_field` config
    pattern.
  - Add to `DiffusionModule.__init__` signature.
  - When the `y_class` / `y_feat` per-field CE (from #27) lands, the
    `lambda_y` wiring plugs in automatically.
  - For SBM, `lambda_y=0` = current behaviour.

### 45. No EMA support — \*

- **Divergence**: We have no EMA at all; upstream has a dead-code EMA
  guard.
- **Instruction**: "fix, but sketch out how to do it cleanly."
- **Response**: **FIX (LOW PRIORITY).**
  - Add a standard `ExponentialMovingAverage` utility under
    `tmgg/src/tmgg/training/ema.py` (copy `torch_ema` semantics or
    write 30-line version).
  - Wire via a Lightning callback:
    ```python
    class EMACallback(Callback):
        def __init__(self, decay: float = 0.999):
            self.ema = None
        def on_fit_start(self, trainer, module):
            self.ema = ExponentialMovingAverage(
                module.model.parameters(), decay=decay
            )
        def on_train_batch_end(self, trainer, module, ...):
            self.ema.update(module.model.parameters())
        def on_validation_start(self, trainer, module):
            self.ema.store(module.model.parameters())
            self.ema.copy_to(module.model.parameters())
        def on_validation_end(self, trainer, module):
            self.ema.restore(module.model.parameters())
    ```
  - Config flag `ema_decay: 0.0` (default) → callback no-ops. Non-zero
    → callback active. Matches upstream's `cfg.train.ema_decay > 0`
    gate, but with a real implementation.
  - Not needed for the current parity fix since upstream's EMA is
    dead-code anyway. Ship when we want to benchmark EMA-trained
    samplers.

### 46. `chain_saving_parameters`, `final_model_samples_to_generate`, `evaluate_all_checkpoints`, `sample_every_val` missing — †

- **Divergence**: Reorganised functionality; config keys absent.
- **Instruction**: "how to expose functionality to be able to match
  upstream?"
- **Response**:
  - `chain_saving_parameters`: these control saving the reverse-chain
    at validation. Upstream writes per-step PMF snapshots to disk for
    later visualisation. Our `GraphEvaluator` doesn't do this — could
    add a `save_chain: bool` param + a `chain_sample_step_interval` to
    the evaluator. Low priority unless we want chain-visualisation
    output.
  - `final_model_samples_to_generate`: generate N samples after
    training for analysis. Currently we rely on explicit `modal run
    tmgg-discrete-gen` followup runs; could add an `on_fit_end`
    hook that generates + writes to `results/final_samples.pt`.
  - `evaluate_all_checkpoints`: iterate over saved checkpoints and
    re-evaluate. Not in our code. Could add a CLI subcommand.
  - `sample_every_val`: already covered by our `eval_every_n_steps`;
    see #41 for the step-vs-epoch choice.
  - **Priorities**: all LOW. None are needed for the core parity
    investigation. Tracking as backlog for full feature parity with
    upstream CLI. **DOC only for this session**.

---

## Summary: main takeaways

**Immediate FIX items (high-confidence, small)**:
- #4 `from_pyg_batch` symmetry assertion.
- #5 `num_nodes` from train+val only.
- #8 `custom_vignac` default K=5.
- #14 Replace `denom.clamp(1e-6)` with positivity assert.
- #15 Runtime `Q_t_bar` row-sum assertion.
- #16 Sample `t=0` in training only (not validation).
- #32 `log_pN` silent fallback → raise.
- #35 Parametrise `sbm_refinement_steps`, default 100.
- #38, #39 Fix scheduler and gradient-clip defaults.
- #40 Bump early-stopping patience + document.
- #43 Fix `timesteps=500` → `1000` in SBM config.

**FIX requiring design-discussion**:
- #2 `remove_self_loops` placement (Option A vs B).
- #3 `collapse` flag on `.mask()`.
- #13 `π` estimator path (dense vs edge_index).
- #17 `forward_sample` return type → `NoisedBatch` dataclass.
- #24 **Double-injection risk** for timestep (`use_timestep` flag vs
  `ExtraFeatures` path) — needs verification.
- #25 `masked_softmax` all-masked behaviour.
- #28 Clean unnormalised-posterior sampling helper + zero-floor config.
- #29 Symmetry-check callback on sampling.
- #30 Expose both posterior-KL forms via config.
- #31 Upstream-style reconstruction at `t=0`.
- #33 `zero_output_diagonal` toggle on the transformer wrapper.
- #41 Step-vs-epoch gate for generation.
- #42 `max_n_nodes` via Hydra interpolation + runtime assert.

**Low-priority / feature work**:
- #12 Absorbing variant with documented typo fix.
- #27 Wire `y_class` per-field CE.
- #44 `lambda_y` config knob.
- #45 EMA callback + utility.
- #46 Chain-saving / final-samples / eval-all-ckpts config.

**Flagged critical to verify before any fix ordering**:
- **#24**: is `use_timestep=True` causing a **double timestep injection**
  into `y`? If yes, this is a shape/semantic bug, possibly explaining
  the edge-collapse. If the config's `input_dims.y` is silently
  compensating for the extra dim, model weights are still being learned
  on a non-upstream `y` layout. Highest-priority investigation item on
  this list.

**NO-OP items** (keep tmgg behaviour, documented):
- #1, #6, #7, #9, #10, #11, #18, #19, #20, #21, #22, #23, #26, #34, #36,
  #37.

**Total**: 46 divergences triaged, 11 immediate fixes, 13 needing
design discussion, 5 low-priority features, 1 critical-to-verify, 16
no-ops.

---

# Appendix A — verification of open-question items

Sources read in verification: `tmgg/src/tmgg/models/digress/transformer_model.py:920-960`,
`tmgg/src/tmgg/models/digress/extra_features.py:180-215`,
`tmgg/src/tmgg/models/layers/masked_softmax.py`,
`tmgg/src/tmgg/data/datasets/graph_types.py:485-515`,
`tmgg/src/tmgg/diffusion/schedule.py:130-160`,
`tmgg/src/tmgg/diffusion/diffusion_sampling.py:170-185`,
plus the matching upstream files under `digress-upstream-readonly/src/`.

## #24 — Timestep double-injection: **NOT PRESENT**

Traced both flows end-to-end.

**Ours** (`_GraphTransformer.forward`, `transformer_model.py:943-950`):
```python
if self.extra_features is not None:
    extra_X, extra_E, extra_y = self.extra_features(X, E, y, node_mask)
    X = torch.cat([X, extra_X], dim=-1)
    E = torch.cat([E, extra_E], dim=-1)
    y = torch.cat([y, extra_y], dim=-1)
if self._use_timestep and t is not None:
    y = torch.cat([y, t.unsqueeze(-1)], dim=-1)
```
Our `ExtraFeatures.__call__` returns `extra_y = (n, y_cycles, n_components,
batched_eigenvalues)` — **no timestep inside**. The wrapper appends `t`
exactly once at the end.

**Upstream** (`diffusion_model_discrete.py:657-671`):
```python
def compute_extra_data(self, noisy_data):
    extra_features = self.extra_features(noisy_data)
    extra_molecular_features = self.domain_features(noisy_data)
    ...
    extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)
    t = noisy_data['t']
    extra_y = torch.cat((extra_y, t), dim=1)
    return utils.PlaceHolder(X=..., E=..., y=extra_y)
```
Upstream's `ExtraFeatures.__call__` returns `y = (n, y_cycles, n_components,
batched_eigenvalues)` — **also no timestep inside**. `compute_extra_data`
appends `t` exactly once at the end, identically.

**Verdict**: Both paths produce `y_final = original_y + extras + t` with
`t` appended last exactly once. **No double injection. #24 is benign**;
the `use_timestep=True` flag simply factored out what upstream does
implicitly inside `compute_extra_data`. Equivalent semantics, different
organisational choice. **NO-OP** (was flagged as "critical to verify" —
now cleared).

One minor observation worth noting: our extra_y layout is
`(n[1], y_cycles[4], n_components[1], batched_eigenvalues[K])` where K
depends on the `EigenFeatures` config. Upstream defaults to K=10;
our discrete_sbm_official uses K=5. That's a config difference (§15
hyperparameter table), not a double-injection risk.

## #2 — `remove_self_loops` placement: Option A is safe

Traced `graph_types.py:488-507`:
- `adj = to_dense_adj(edge_index, batch_vec)` — PyG densification.
- `adj[:, diag, diag] = 0.0` — zero self-loops (defensive, since
  `to_dense_adj` doesn't actively add self-loops but could pass them
  through from a sparse input that had them).

Upstream `utils.py:53-62` uses
`edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)` on
the sparse side before densification.

Both approaches are equivalent when self-loops are absent from the
source sparse graph. They diverge only if the upstream PyG `Batch`
actually contained self-loop entries — in which case upstream drops
them with their `edge_attr`, while ours drops them from `adj` after
densification (losing any class-specific self-loop attribute).

For tmgg's current use (structure-only SBM with no self-loops in data
by construction), both are equivalent. **DECISION**: Option A (sparse
`remove_self_loops`) is strictly safer for future datasets that might
carry self-loop edge attributes. Keep our post-densification zero as
well for defence in depth? — probably overkill but costs nothing.

## #25 — `masked_softmax` all-masked behaviour

Verified both files:
- **Ours** (`models/layers/masked_softmax.py:32-33`):
  `if mask.sum() == 0: return torch.zeros_like(x)`.
- **Upstream** (`models/layers.py:41-46`):
  `if mask.sum() == 0: return x`.

The all-masked branch can only fire for attention heads where every
key-position is masked — a degenerate configuration that does not
occur with node-mask-based masking on non-empty graphs. Downstream in
both codebases, the result is multiplied by a masked `V` whose rows
are zero at the same positions, so the final attention output is zero
either way. **Never fires in practice**; matching upstream (returning
`x`) is one line. **FIX-LITE** (one-line change), zero behavioural
impact.

## #28 — Zero-guard on posterior: healthy/degenerate behaviour confirmed

Verified `diffusion_sampling.py:178-182`:
```python
denom = M @ Qtb_M
denom = (denom * M_t).sum(dim=-1)
denom = denom.clamp(min=1e-6)
prob = product / denom.unsqueeze(-1)
```

Upstream `diffusion_utils.py:274-290` has no clamp; computes
`unnormalized_prob = ...` then later renormalises via `unnormalized /
unnormalized.sum(dim=-1, keepdim=True)` with the `sum==0 → 1e-5` floor
applied to that row-sum.

**Our divergence is structural, not just the value**: we clamp the
*dot-product denominator*; upstream clamps the *final row-sum*. Both
prevent NaN downstream, but the mathematics differs on degenerate
rows. For healthy posteriors (the only regime a trained model
produces), both equal. **FIX** (as sketched in the main table) replaces
our clamp with `assert (denom > 0).all()` + explicit row-sum
normalisation matching upstream's form. Cleaner, louder, and directly
porting upstream's shape.

## #30 — VLB posterior form: API supports both

Verified `DiffusionModule` already has both helpers:
- `_posterior_probabilities(z_t, batch, Qt, Qsb, Qtb)` — direct Bayes
  plug-in form (upstream-style).
- `_posterior_probabilities_marginalised(z_t, x0_param, ...)` —
  our marginalised form currently used in `kl_diffusion`.

Only the `validation_step` at `diffusion_module.py:680-685` has the
hardcoded selection (marginalised). **FIX**: add
`use_marginalised_vlb_kl: bool = True` to `DiffusionModule.__init__`,
select between the two helpers at that line. Default `True` preserves
tmgg behaviour; `False` selects upstream-style. One parameter, one
branch. **DECISION NEEDED** on default.

## #31 — Reconstruction `z_1`/marginalised vs `z_0`/softmax: clean port feasible

Current (`diffusion_module.py:596-631`) samples `z_1` and scores via
marginalised posterior. Upstream samples `z_0` and scores raw softmax.

The clean port sketched in the main table writes a separate method
`_compute_reconstruction_upstream_style`. Toggle via
`use_upstream_reconstruction: bool = False` on `DiffusionModule.__init__`.
Default `False` preserves current tmgg form (documented intentional
choice); `True` matches upstream for strict-parity runs. **FIX**.
**DECISION NEEDED** on default.

## #33 — `zero_output_diagonal` toggle: compatible with the API

Verified `_GraphTransformer.forward` path. The final masking step is
a `.mask_zero_diag()` on the predicted `GraphData`. Introducing a
`zero_output_diagonal: bool = True` parameter at `__init__` and
branching:
```python
if self.zero_output_diagonal:
    out = out.mask_zero_diag()
else:
    out = out.mask()
```
…is a clean 3-line change. Default `True` preserves current
behaviour (cleaner predictions); `False` matches upstream (predictions
retain diagonal logits). **FIX**. **DECISION NEEDED** on default.

## Other items

- **#3 (collapse flag)**: straightforward API change; no verification needed.
- **#4 (from_pyg_batch symmetry assertion)**: straightforward addition.
- **#5 (num_nodes train+val)**: one-line change in
  `SpectreSBMDataModule.setup()`. Verification trivial.
- **#13 (π estimator path)**: §4 review already verified both produce
  the same `(K,)` scalar. Porting upstream's approach is ~20 LoC;
  numerical result unchanged. Decision is stylistic.
- **#15 (row-sum assertion)**: add and done.
- **#16 (t=0 training)**: one-line change in `training_step`. Verified
  `validation_step` independently uses `randint(1, T+1)` so no
  cascade.
- **#17 (NoisedBatch dataclass)**: largest open design discussion. See
  main table.
- **#29 (per-step symmetry assertion)**: toggle on the sampler config.
  No verification needed beyond the main-table sketch.
- **#32 (log_pN silent fallback)**: verified one-line change at
  `diffusion_module.py:702-705`.
- **#35 (refinement_steps parametrise)**: verified `GraphEvaluator`
  class is the single site needing the change.
- **#38-#41 (scheduler, grad_clip, early_stopping, step-vs-epoch)**:
  config-level decisions only, no code verification needed.
- **#42 (max_n_nodes)**: Hydra interpolation is the recommended route
  (`max_n_nodes: ${data.num_nodes}`). Needs a small follow-up: add a
  runtime assertion `assert max_n_nodes >= datamodule.num_nodes_max`
  at setup time.
- **#43, #44, #45, #46**: config-level YAML edits + optional new
  callback/class.

---

# Appendix B — consolidated discussion + decision items

## Clear-to-fix (no decision needed, just do it)

These are mechanical ports or small changes with no behavioural
tradeoff. Sequence them in one commit per §.

1. #4 — add symmetry assertion in `from_pyg_batch`.
2. #5 — `num_nodes` from train+val only.
3. #8 — `custom_vignac` default K=5.
4. #14 — replace `denom.clamp(1e-6)` with positivity assert + match
   upstream row-sum normalisation (closes #28 structurally, subsuming
   the clamp-vs-floor question).
5. #15 — runtime `Q_t_bar` row-sum assertion.
6. #16 — sample `t=0` in training only (not validation). Verified
   no cascade.
7. #25 — one-line `masked_softmax` match: return `x` when all-masked
   (matches upstream; behaviourally equivalent).
8. #32 — replace `log_pN` silent fallback with raise.
9. #35 — parametrise `sbm_refinement_steps` on `GraphEvaluator`,
   default 100.
10. #39 — change trainer `gradient_clip_val` default from 1.0 to
    null.
11. #40 — bump early-stopping patience to 1000, add divergence comment,
    confirm `save_top_k=-1`.
12. #43 — fix `timesteps=500` → `1000` in `discrete_sbm_official.yaml`.

## Decision items (need user sign-off before fix)

Each has a recommended default that preserves tmgg's tested behaviour
while exposing strict-upstream-parity via a toggle. The question is
whether to flip defaults for any of them.

### D-1. `remove_self_loops` placement (#2)

- **Option A**: use upstream's sparse `remove_self_loops` before
  densification; drop our post-densification `adj[diag]=0`.
- **Option B**: keep both for defence in depth.
- **Recommendation**: A (smaller, cleaner).

### D-2. `collapse` flag on `.mask()` (#3)

Straightforward rename/merge. No default to decide; ship as additive.

### D-3. π estimation via upstream edge_index path (#13)

- **Option A**: keep our dense upper-triangle approach (cleaner given
  dense `GraphData`).
- **Option B**: port upstream's sparse `edge_counts`.
- **Recommendation**: A. Numerical identity already verified; B adds
  ~20 LoC for no result change.

### D-4. `forward_sample` return type (#17)

- Introduce a `NoisedBatch` dataclass bundling `z_t: GraphData` +
  `t_int, alpha_t_bar, alpha_s_bar, beta_t`. Single open question:
  naming + whether to move `t_int` shape from `(B,)` → `(B, 1)` to
  fully align upstream (resolves #18 jointly).
- **Recommendation**: do it; pick `(B, 1)` shape for cross-
  codebase grep-friendliness.

### D-5. Unnormalised-posterior sampling helper (#28, #29)

- Introduce `_sample_from_unnormalised_posterior(unnorm, *,
  zero_floor=1e-5)` with the upstream-parity row-sum normalisation
  + zero-floor. Adds `assert_symmetric_e: bool` toggle on the
  sampler with a default `True`.
- **Recommendation**: ship. Expose `zero_floor` on the sampler
  config for strict-parity knob.

### D-6. `use_marginalised_vlb_kl` toggle (#30)

- Add the toggle; two helpers already exist. Default `True`
  preserves current tmgg behaviour; `False` matches upstream.
- **Recommendation**: ship; default `True` (ours); flip via config
  for strict-parity runs.

### D-7. `use_upstream_reconstruction` toggle (#31)

- Same structure as D-6. Default `False` (current tmgg form); `True`
  for strict parity.
- **Recommendation**: ship; default `False`; document why tmgg's
  form is the current default.

### D-8. `zero_output_diagonal` toggle on `_GraphTransformer` (#33)

- Default `True` (clean tmgg predictions); `False` matches upstream.
- **Recommendation**: ship; default `True`.

### D-9. Scheduler default (#38)

- **Option A**: remove `cosine_warmup` from `_base_infra.yaml`
  defaults; require opt-in.
- **Option B**: override per-config in `models/digress/*.yaml` to
  `scheduler_config: {type: none}` for upstream-parity runs; keep
  base default.
- **Recommendation**: B (narrower blast radius).

### D-10. Step-vs-epoch generation gate (#41)

- Add `gate: "step" | "epoch"` parameter. Default `"step"` (current).
- **Recommendation**: ship; default `"step"`; upstream-parity runs
  set `"epoch"`.

### D-11. `max_n_nodes` interpolation (#42)

- Replace `max_n_nodes: 20` hardcode in `discrete_sbm_official.yaml`
  with Hydra interpolation: `max_n_nodes: ${data.num_nodes}` OR
  `${data.num_nodes_max}` (whichever is exposed). Add
  `assert max_n_nodes >= dataset.num_nodes_max` at setup.
- **Recommendation**: ship; one config change + one assertion.

## Low-priority / feature work (backlog)

- #12 absorbing transition (fix typo, implement for completeness).
- #27 wire `y_class` per-field CE.
- #44 `lambda_y` config knob.
- #45 EMA callback + utility.
- #46 expose missing upstream config functionality.

## No-op confirmations (keep tmgg behaviour, do not touch)

- #1, #6, #7, #9, #10, #11, #18 (subsumed by D-4),
  #19, #20, #21, #22, #23, #24 (**verified non-issue**),
  #26, #34, #36, #37.
