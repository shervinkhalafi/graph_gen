# Spec: chain-saving parameters (D-16a)

**Status:** Draft
**Date:** 2026-04-22
**Author:** igork
**Refs:**
- `docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md`
  (D-16a, parity item #46), lines 370-378.
- Upstream `digress-upstream-readonly/src/diffusion_model_discrete.py:491-595`
  (`sample_batch`'s `keep_chain` snapshotting block).
- Upstream `digress-upstream-readonly/configs/general/general_default.yaml:16-18`
  (`chains_to_save: 1`, `number_chain_steps: 50`).
- Upstream `digress-upstream-readonly/configs/experiment/sbm.yaml:11-12`
  (per-experiment override of the same fields).

**Normative language.** The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and
MAY follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Goal

Let a researcher record the per-step categorical PMF of a small number of
reverse-chain samples during validation generation, written to disk in a
deterministic schema that an offline rendering script can consume to produce
trajectory plots or animations. This unlocks visual debugging of stuck or
mode-collapsed generations and provides paper-grade trajectory figures
without re-running training.

## Upstream behaviour

`sample_batch` (`diffusion_model_discrete.py:491-595`) materialises two
buffers `chain_X` of shape `(number_chain_steps, keep_chain, n)` and
`chain_E` of shape `(number_chain_steps, keep_chain, n, n)` before entering
the reverse loop. Inside the loop (`:527-540`) it picks the write index as
`(s_int * number_chain_steps) // self.T`, which deterministically selects
`number_chain_steps` evenly spaced snapshots out of the full `T` reverse
steps. Only the first `keep_chain` graphs in the batch are tracked; the
written tensor is the *discrete* (collapsed) sampled state, not the soft
posterior parameters. After the loop, `chain_X` and `chain_E` are reversed
in time so frame `0` is the prior sample and frame `-1` is the final clean
sample, then the final frame is repeated ten times as visual padding
(`:560-562`). Upstream consumes the buffers immediately in
`visualization_tools.visualize_chain` (`:572-585`), writing rendered chains
to `chains/<run_name>/epoch<N>/chains/molecule_<i>/`. The configuration
surface lives at `general.chains_to_save` and `general.number_chain_steps`,
overridden per-experiment (SBM keeps `chains_to_save: 1`).

## Non-goals

This spec covers chain capture and on-disk serialisation only. It does not
ship a renderer, does not couple to molecule visualisation (DiGress's
`visualize_chain` assumes RDKit / matplotlib molecular plotting that is
domain-specific), and does not propose new metrics over chain trajectories.
It also does not change sampler semantics: the captured state is the same
discrete state the sampler already produces; the only addition is a side
output. EMA weight-swapping interaction is delegated to the existing
`EMACallback` (`src/tmgg/training/callbacks/ema.py`) and not redesigned
here.

## Design

### Surface (config + CLI)

Chain saving is configured as a nested block on the trainer-side config
that today drives `GraphEvaluator` instantiation. The block is opt-in; an
absent or `None` block disables capture entirely (no overhead in the
sampler hot loop).

```yaml
training:
  evaluator:
    chain_saving:
      num_chains_to_save: 3            # mirrors upstream `chains_to_save`
      snapshot_step_interval: 50        # mirrors upstream `number_chain_steps`
      chain_save_path: null             # null -> ${run_dir}/chains
      every_n_validations: 1            # 1 = every val pass; N = every Nth
      max_validations: null             # null = unbounded; cap on writes
```

`num_chains_to_save = 0` MUST disable capture even when the block is
present, mirroring the upstream gating idiom of "set the count to zero".
`snapshot_step_interval` is interpreted exactly as upstream's
`number_chain_steps`: the buffer holds that many evenly spaced frames from
the `T`-step reverse chain, *not* "every Kth step". The naming difference
(`interval` vs upstream's `chain_steps`) is deliberate — the implementation
computes the write index as `(s_int * snapshot_step_interval) // T` and
"snapshot count" reads cleaner than the upstream name when documented in
isolation. The spec records the rename for cross-reference.

`every_n_validations` and `max_validations` exist only on our side. DiGress
captures chains on every test invocation; our validation cadence is
step-based (`eval_every_n_steps`, see D-10) and a long run might trigger
hundreds of validation passes, so we want a coarse second-level throttle.

### Implementation site

We thread chain capture through `Sampler.sample`
(`src/tmgg/diffusion/sampler.py:268`) rather than introduce a separate
Lightning callback that re-runs sampling. Two reasons:

1. The sampler already owns the reverse loop and the per-step state we need
   to snapshot. A callback would have to either duplicate the loop or call
   `Sampler.sample` a second time with capture enabled; both are wasteful.
2. The existing `StepMetricCollector` hook on `Sampler.sample` (used by the
   per-step metric tap) demonstrates that the sampler is comfortable
   accepting an optional side-effect callable. A second optional argument
   for chain capture follows the same pattern.

Concretely, `Sampler.sample` gains a new optional parameter
`chain_recorder: ChainRecorder | None = None`. `ChainRecorder` is a small
class (in a new module `src/tmgg/diffusion/chain_recorder.py`) that owns
the per-snapshot buffers and exposes `record(t_int, s_int, z_t)` and
`finalise() -> ChainSnapshot`.
`DiffusionModule.on_validation_epoch_end` instantiates the recorder when
the config block is present and the throttle gates fire, passes it through
to `generate_graphs`, and on completion calls
`ChainSnapshot.save(path)`.

The alternative — a `ChainSavingCallback` that runs sampling itself — is
documented under Open questions for review.

### Data flow

At each reverse step inside `Sampler.sample`:

1. The sampler computes `posterior_param`, draws the next state `z_s`, and
   already calls `_record_step_metrics` when a collector is supplied.
2. New: if `chain_recorder is not None`, the sampler invokes
   `chain_recorder.record(t_int, s_int, z_s)` immediately after the next
   state is materialised. The recorder decides whether `(t_int, s_int)`
   maps to a snapshot slot; the sampler does not duplicate that logic.
3. After the loop, the sampler returns the trimmed graph list as today.
   `ChainSnapshot.save` is called by the caller (the Lightning module),
   not by the sampler — keeping I/O at the orchestration layer.

`ChainRecorder.record` slices the first `num_chains_to_save` graphs from
the batch (raising if the batch is smaller — an explicit user error,
captured in the validity checks below), pads the per-graph node count to
`max_n` using `node_mask` for the padded positions, and stores the soft
PMF (the categorical distribution before argmax collapse) for both
`E_class` and `X_class`. We choose to store the PMF rather than the
discrete state because it is strictly more informative — a renderer can
collapse the PMF cheaply, but the reverse direction loses information.
Upstream stored the discrete state because their renderer was molecule
visualisation that needed a categorical assignment; for graph-only chains
the PMF is the better artefact.

### Storage / artifact format

A single `.pt` file per write under `${chain_save_path}/`. Default path
when `chain_save_path` is null is `${run_dir}/chains/`, where `run_dir` is
the Lightning trainer's `default_root_dir`. The filename pattern is
`epoch=<E>-step=<S>-chains.pt` so the file sorts naturally and includes
both clocks.

Schema (single dict, written via `torch.save`):

| key             | dtype            | shape                                           | semantics |
|-----------------|------------------|-------------------------------------------------|-----------|
| `E_chain`       | `torch.float32`  | `(num_snapshots, num_chains, max_n, max_n, de)` | Soft per-edge PMF at each snapshot. |
| `X_chain`       | `torch.float32`  | `(num_snapshots, num_chains, max_n, dx)`        | Soft per-node PMF; `None`-shaped key when `X_class` is absent. |
| `node_mask`     | `torch.bool`     | `(num_chains, max_n)`                           | Per-graph node validity (constant across snapshots — node count is fixed at sampler entry). |
| `step_indices`  | `torch.long`     | `(num_snapshots,)`                              | The `t_int` value at which each snapshot was taken, in capture order (latest noise first; matches the reverse-loop traversal). |
| `meta`          | `dict[str, Any]` | -                                               | `{"global_step": int, "epoch": int, "T": int, "snapshot_step_interval": int, "noise_process": str, "ema_active": bool}`. |

The `meta` dict carries the information a renderer needs to label frames
and reproduce the run context. `ema_active` records whether EMA weights
were swapped in at capture time; downstream analyses that compare EMA vs
live trajectories will need this.

The schema is documented in the module docstring of
`src/tmgg/diffusion/chain_recorder.py` and exercised by a load-roundtrip
unit test under `tests/diffusion/test_chain_recorder.py`.

### Validity checks

`ChainRecorder.__init__` MUST raise `ValueError` when:

- `num_chains_to_save < 1`. (The "0 disables" path is handled at the
  Lightning module by not constructing a recorder; the recorder itself
  rejects zero to fail loud on misuse.)
- `snapshot_step_interval < 1` or `>= T`.

`ChainRecorder.record` MUST raise `ValueError` when the batch presented at
the first call has fewer graphs than `num_chains_to_save` — better to
fail at the first reverse step than to silently truncate.

`ChainSnapshot.save` MUST refuse to overwrite an existing file (raise
`FileExistsError`); the per-step naming pattern means collisions indicate
a real bug (two captures fired at the same global step).

## Open questions

1. Single-collector vs split-collector. The reverse loop already accepts a
   `StepMetricCollector`. Should `ChainRecorder` be folded into the same
   collector interface (one optional argument carries both metric and
   chain side-effects), or kept separate? Current proposal: keep separate
   — they have different lifetimes (the metric collector is per-step and
   stateless across calls; the chain recorder accumulates a buffer across
   the whole reverse chain and produces a finalised snapshot). Owner:
   igork.
2. Capture cadence. Default `every_n_validations: 1` writes a chain on
   every validation pass that fires generation, which on a long SBM run
   produces dozens of files. Is that acceptable, or should the default be
   `every_n_validations: null` with manual opt-in to per-pass capture and
   an end-of-training-only mode? Owner: igork.
3. Memory ceiling. For SPECTRE SBM (`n=187`, `T=1000`,
   `snapshot_step_interval=50`, `num_chains_to_save=3`) the soft-PMF
   buffer is `50 × 3 × 187 × 187 × 2 × 4 bytes ≈ 42 MB`, plus the X
   buffer when present. This is acceptable on disk but the in-memory
   accumulation runs on the sampling device. Should the recorder copy to
   CPU on each `record` call (slower, no GPU pressure) or accumulate on
   GPU and copy once at finalise (faster, ~84 MB GPU pressure)? Default
   proposal: accumulate on GPU; revisit if a hybrid run blows up the
   sampler.
4. Coupling to D-5's `assert_symmetric_e` toggle. The captured E PMF is
   the post-step soft posterior, which the sampler symmetrises before the
   next iteration. Should the recorder snapshot pre- or
   post-symmetrisation? Current proposal: post — it is the state actually
   used by the next reverse step and matches what the sampler returns at
   the end. Owner: TBD; depends on whether trajectory plots want to
   visualise the asymmetry at intermediate steps.
5. Compatibility with composite noise processes. When the noise process
   is a `CompositeNoiseProcess`, the snapshot per field needs a stable
   key naming. Current proposal: emit one `<field>_chain` key per
   declared field of the composite, e.g. `E_class_chain`, `E_feat_chain`.
   This generalises the schema above. Owner: TBD.

## Acceptance

The implementation is correct when the following hold:

A1. Enabling the `chain_saving` block on a SPECTRE SBM run produces, on
the second validation pass after the gate fires, a file
`${run_dir}/chains/epoch=<E>-step=<S>-chains.pt` whose schema matches the
table above and which `torch.load` reads back to tensors of the declared
shapes.

A2. The reverse-chain output graphs (the return value of
`Sampler.sample`) are bit-identical with and without a chain recorder
attached, given the same RNG seed. Capture must be a pure side effect.

A3. The recorder raises `ValueError` when constructed with
`num_chains_to_save = 0`, `snapshot_step_interval = 0`, or
`snapshot_step_interval >= T`, and raises `ValueError` at first
`record(...)` call when the input batch is smaller than
`num_chains_to_save`.

A4. A roundtrip unit test under `tests/diffusion/test_chain_recorder.py`
constructs a fake reverse trajectory, drives the recorder through it,
saves the snapshot, reloads it, and asserts all keys and shapes round-trip
exactly.

A5. The captured snapshot's `step_indices` tensor matches upstream's
write-index formula `(s_int * snapshot_step_interval) // T` evaluated for
each `s_int` in `reversed(range(0, T))`, deduplicated to the unique
snapshot frames.


## Resolutions (2026-04-22)

User responses to the open questions, applied as the implementation
contract:

- **Q1 (recorder vs metric collector)**: keep `ChainRecorder` as a
  separate optional argument on the sampler. Different concerns —
  metrics aggregate scalars per step, the recorder snapshots full
  per-position PMFs.

- **Q2 (cadence)**: configurable. Default `chain_save_every_n_val: 20`
  (every 20th validation pass) plus `chain_save_at_fit_end: true`
  unconditionally. Both knobs ship on the evaluator config block.

- **Q3 (memory placement)**: accumulate on GPU during the reverse
  loop. If a future large dataset OOMs, fall back to per-step CPU
  copy as a follow-up; not needed for SPECTRE n=187 at the upper
  bound (≈21 MB per pass).

- **Q4 (pre- vs post-symmetrisation snapshot)**: post-symmetrisation,
  matching upstream (`triu + transpose` is applied before the
  snapshot is taken). The user-facing artifact is what the sampler
  actually emitted at each step, not the model's pre-symmetrised
  output.

- **Q5 (CompositeNoiseProcess key naming)**: prefix snapshot keys
  with the sub-process name (e.g. `categorical/E_class`,
  `gaussian/E_feat`); reconverge at render time by merging the
  per-sub-process tensors back into a single GraphData per step.
  No deep gotcha — the disjoint-fields invariant on
  `CompositeNoiseProcess` guarantees each sub-process owns a unique
  subset of GraphData fields, so the merge is just
  `merged.replace(**{f: tensor for sub in subs for f, tensor in sub.items()})`.
  Edge case: if sub-processes use independent schedules (rare; the
  canonical case shares one timestep), the prefix-keyed snapshots
  stay separate timelines rather than merging — same `t` ⇒ same
  frame, different `t` ⇒ different timelines.
