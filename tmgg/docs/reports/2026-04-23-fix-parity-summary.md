# Fix-parity wave — summary

**Branch**: `igork/fix-parity`
**Baseline commit**: `5a71c74f` (pre-parity main merge)
**Total commits**: 58

A two-stage effort on upstream-DiGress parity. The first stage (40 commits,
2026-04-21 through 2026-04-22) implemented the approved decisions from the
divergence triage — twelve cleared parity fixes, sixteen D-N decision
implementations, three D-16 specs with user-resolved open questions, and
the D-16a/b/c implementations. The second stage (18 commits, 2026-04-23)
closed every item the post-implementation review flagged as unhonored,
partial, or CLAUDE.md-violating — including two pyright-suppression
cleanups, six mechanical cleanups, three spec-alignment refactors, and
six test additions.

## What landed

### Stage 1 — original parity wave (commits `fc328528` through `e5288487`)

Twelve cleared parity fixes in `fc328528`–`edf3c19a`: diagonal symmetry
assert, sparse `remove_self_loops`, train+val node-count narrowing,
`custom_vignac` K=5 default, unnormalised-denom positivity guard,
row-stochastic kernel assertions, t=0 training sample, `masked_softmax`
passthrough, `_size_distribution` fail-loud, `sbm_refinement_steps=100`,
`gradient_clip_val=null`, early-stopping patience 1000 / `save_top_k=-1`,
`timesteps=1000` in the SBM official config.

Sixteen D-N implementations:

- **D-1**: sparse `remove_self_loops` pre-densification.
- **D-2**: merge `collapse_to_indices` into `GraphData.mask(collapse=...)`.
- **D-3**: port upstream sparse `edge_counts` for π estimation (new
  `data/utils/edge_counts.py` + `train_dataloader_raw_pyg` accessor).
- **D-4**: `NoisedBatch` dataclass bundling schedule scalars with
  `(B, 1)` shape.
- **D-5**: `_sample_from_unnormalised_posterior` helper +
  `assert_symmetric_e` toggle on `Sampler`.
- **D-6**: `use_marginalised_vlb_kl` config toggle, default upstream.
- **D-7**: `use_upstream_reconstruction` config toggle, default upstream.
- **D-8**: `zero_output_diagonal` toggle (later verified dead code and
  removed in stage 2).
- **D-9**: scheduler default flipped to `none`.
- **D-10**: step-based generation gate documentation.
- **D-11**: `max_n_nodes` via Hydra interpolation.
- **D-12**: absorbing categorical limit-distribution variant.
- **D-13**: `y_class` / `y_feat` per-field CE wiring with `lambda_y`.
- **D-15**: EMA utility + Lightning callback.
- **D-16a/b/c**: ChainRecorder, FinalSampleDumpCallback,
  `tmgg-discrete-eval-all` CLI, plus the three specs with user-resolved
  open questions.

The stage 1 work shipped with two pyright-suppression violations
(`EMACallback`, `FinalSampleDumpCallback`) that required follow-up
cleanup commits in the same stage.

### Stage 2 — cleanup wave (commits `fff0fe37` through `edc910e5`)

Eighteen commits closing every item the 5-batch + 2-synthesis review
flagged:

- **W1-1**: delete `final_model_samples_to_generate` legacy shim +
  migrate wave configs via a Hydra defaults block.
- **W1-2**: refactor `evaluate_checkpoint` to load the datamodule from
  checkpoint hparams (replaces the synthesis shortcut); wire
  `--reference_set` and `--use_ema` properly;
  `EMACallback.state_dict`/`load_state_dict`/`copy_shadow_into`.
- **W1-3**: `ChainRecorder.meta` required at construction.
- **W2-1**: delete `num_nodes=50` silent fallback + audit + fix 6 PyG
  data configs.
- **W2-2**: match upstream's `s_int=-1 → alpha_bar[T]` semantics; warn
  by default; `FAIL_STRICTLY=1` env var upgrades warning to
  `RuntimeError`.
- **W2-3**: regression test for the mask-aware posterior denom guard.
- **W2-4**: single parametrized test pinning eight parity-flip config
  defaults.
- **W2-5**: `ChainSavingCallback` with evaluator-threaded
  `ChainRecorder` (every Nth validation pass + at-fit-end).
- **W2-6**: Composite fan-out for `ChainRecorder` (dict-typed
  `chain_recorder` on `Sampler.sample`).
- **W2-5-followup** (`b2702d66`): update `fake_sample` stub for the
  `chain_recorder` kwarg.
- **W3-1**: delete dead `zero_output_diagonal` toggle; rewrite D-8
  disposition; delete triage-doc Appendix B.
- **W3-2**: defer `.item()` into the failure path in `from_pyg_batch`
  symmetry assert (CUDA-sync optimisation).
- **W3-3**: fix off-by-one citation in D-16a spec.
- **W3-4**: `evaluate_all_cli` autodetect `cuda` → `cuda:0`.
- **W3-5**: resolve wandb-name template test post-`edf3c19a` (case A:
  template selects `discrete_default` with T=500, not the flipped
  `discrete_sbm_official`; assertion stays).
- **W3-6**: D-12 absorbing-variant integration tests.
- **W4-1**: wire `_normalise_unnormalised_posterior` helper into
  `_posterior_probabilities_marginalised`; split the D-5 primitive
  into pure normalisation + thin sampling wrapper; move `zero_floor`
  from `Sampler` to `CategoricalNoiseProcess`.
- **W4-2**: K=5 default safety sweep — zero findings (all discrete
  configs use `cosine_iddpm`; `custom_vignac` default doesn't apply).

## Where to read further

- `docs/reports/2026-04-21-digress-upstream-spec.md` — upstream
  behaviour specification (15 sections).
- `docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md`
  — the 46-divergence triage with user-resolved D-1..D-16 decisions.
- `docs/specs/2026-04-22-upstream-config-surface-{a,b,c}.md` — the
  D-16 specs with user-resolved "Resolutions (2026-04-22)" sections.
- `docs/reports/2026-04-23-fix-parity-review/batches/batch-{A,B,C,D,E}-*.md`
  — five per-batch reviews covering every stage-1 commit.
- `docs/reports/2026-04-23-fix-parity-review/synthesis/synthesis-{1,2}.md`
  — two independent syntheses of the batch reviews.
- `/home/igork/.claude/plans/polished-munching-codd.md` — the stage 2
  cleanup plan this summary records the completion of.

## Behaviour changes for downstream W&B comparisons

The stage 1 and stage 2 commits change several user-facing numerical
quantities relative to pre-parity runs. Re-running an old config on
the new branch will produce different numbers; cross-branch
comparisons need to account for these shifts.

**`val/epoch_NLL` and `_vlb_reconstruction`** shift by approximately
1e-3 absolute per T=1000 step after D-7 flipped
`use_upstream_reconstruction` to `True`. Previously-recorded values
computed with the marginalised-posterior form cannot be directly
compared to post-flip values without this offset. Flip the toggle
back to `False` in the config if a direct comparison against an
older run is needed.

**`val/epoch_NLL` KL term** uses the direct Bayes plug-in form after
D-6 flipped `use_marginalised_vlb_kl` to `False`. At convergence
(one-hot `p_θ(x_0)`) the two forms coincide; during early training
they differ by a variance-reduction constant. Monitoring curves over
many epochs are unaffected at convergence.

**SBM edge-class schedule** picks up K=5 instead of K=2 in any
`custom_vignac` call site that omits `num_edge_classes`. The W4-2
audit confirmed no in-tree configs hit this path (all discrete
configs use `cosine_iddpm`), so this is documented for research users
who construct `NoiseSchedule` directly in notebooks or one-off
scripts.

**Training samples t=0** now, not just `t ∈ {1..T}` (parity #16).
Loss at t=0 is the reconstruction term; training loss shifts by a
small amount reflecting its inclusion.

**E_class diagonal** is now asserted symmetric at construction in
`from_pyg_batch` (parity #4). Any dataset producing asymmetric
inputs now raises `AssertionError` at construction rather than at
loss-time; upstream-compliant datasets are unaffected.

**Reference graphs for `tmgg-discrete-eval-all`** now come from the
datamodule's val or test split (per the `--reference_set` flag) after
W1-2. The pre-W1-2 synthesis-based path is gone. Comparing runs that
evaluated against synthesised graphs to post-W1-2 runs that evaluated
against the datamodule's test split is an apples-to-oranges
comparison and should not be performed.

**EMA weights at evaluation** are used when `--use_ema auto` (the
default) and the checkpoint carries an EMA shadow after W1-2. Pre-W1-2
eval always used live weights even when the checkpoint had an EMA
shadow; the published-quality model (EMA) is now evaluated by default
when it exists.

## Outstanding backlog

Empty. Every item the 2026-04-23 review flagged is closed by stage 2.
A future parity wave starting from this branch state would need a new
review to surface items.
