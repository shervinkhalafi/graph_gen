# Spec: final-model sample dump (D-16b)

**Status:** Draft
**Date:** 2026-04-22
**Author:** igork
**Refs:**
- `docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md`
  (D-16b, parity item #46), lines 380-386.
- Upstream `digress-upstream-readonly/src/diffusion_model_discrete.py:229-300`
  (`test_step` and the post-test sample dump loop).
- Upstream `digress-upstream-readonly/configs/general/general_default.yaml:20-22`
  (`final_model_samples_to_generate: 10000`,
  `final_model_samples_to_save: 30`,
  `final_model_chains_to_save: 20`).
- Upstream `digress-upstream-readonly/configs/experiment/sbm.yaml:13-15`
  (per-experiment override; SBM caps at 40).
- `src/tmgg/training/callbacks/ema.py` (the existing callback pattern we
  mirror).

**Normative language.** The key words MUST, MUST NOT, SHOULD, SHOULD NOT,
and MAY follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Goal

Provide a deterministic post-training generation step that draws a
configurable number of samples from the trained model, saves them to a
canonical run-dir path, and runs the standard `GraphEvaluator` pass over
the saved batch. This produces a reproducible artefact a researcher can
re-evaluate offline, separates the published-quality sample set from the
training-time validation cadence, and matches upstream's
`final_model_samples_to_generate` knob without forcing per-validation
duplication.

## Upstream behaviour

`test_step` and `on_test_epoch_end` in
`diffusion_model_discrete.py:229-300` run after `trainer.fit()` and
implement the dump loop. Three knobs gate it
(`general_default.yaml:20-22`):

- `final_model_samples_to_generate` — total number of generated graphs.
- `final_model_samples_to_save` — count written to a textual dump file.
- `final_model_chains_to_save` — count of trajectories visualised
  (relevant only when chain saving is on; see D-16a).

The loop calls `sample_batch` repeatedly with `bs = 2 *
cfg.train.batch_size` until the requested count is exhausted, accumulating
into a Python list (`:262-276`). It writes a textual dump at
`generated_samples<i>.txt` (`:278-298`) where `i` is the next free
suffix, then runs `self.sampling_metrics(samples, ..., test=True)` to
recompute MMD-style metrics on the dumped set. The textual format is
specific to molecules (`atoms`, `bond_list`); the code path is otherwise
domain-neutral.

Our codebase has no equivalent: validation generation uses
`eval_every_n_steps` cadence (D-10), and `on_test_epoch_end` is
explicitly TODO at `diffusion_module.py:885-886`.

## Non-goals

This spec does not define the textual dump format from upstream
(`generated_samples<i>.txt` was molecule-specific). The graph-level
artefact is a `torch.save`d list of `GraphData`. It does not redesign the
test-step path; the dump callback runs at `on_fit_end` and is independent
of `trainer.test(...)` entirely. It does not introduce a new sampler or a
new evaluator — it composes the existing ones. It does not handle
distributed sampling beyond what `Sampler.sample` already supports.

## Design

### Surface (config + CLI)

Add a top-level training config block:

```yaml
training:
  final_sample_dump:
    num_samples: 10000        # 0 disables; matches upstream default
    save_path: null           # null -> ${run_dir}/final_samples.pt
    sample_batch_size: null   # null -> 2 * train.batch_size (upstream rule)
    evaluate: true            # run GraphEvaluator over the dumped set
    evaluation_reference: val # "val" | "test" | "both"
```

`num_samples = 0` disables the entire pass — no callback registration, no
generation, no I/O. The block being absent from the config has the same
effect (the runner does not register the callback).

`sample_batch_size = null` mirrors upstream's `2 * train.batch_size` rule
exactly. We expose it as an explicit knob because the right value depends
on inference-time GPU memory, which can differ from training memory
(activations are smaller during generation but the batch shape is the
same). A user MAY pin it to a fixed integer.

`evaluate: true` runs `GraphEvaluator.evaluate` over the dumped samples
and logs the result dict to W&B under the `final/` namespace
(`final/degree_mmd`, `final/sbm_accuracy`, …). When `false`, the dump
file is still written; this is useful when the researcher plans to
evaluate offline against a held-out reference.

`evaluation_reference: both` runs the evaluator twice and emits both
prefixes (`final/val/<metric>` and `final/test/<metric>`).

### Implementation site

We introduce a Lightning callback at
`src/tmgg/training/callbacks/final_sample_dump.py` rather than adding the
logic to `DiffusionModule.on_fit_end`. The reasons mirror the `EMACallback`
rationale (`src/tmgg/training/callbacks/ema.py`):

1. The Lightning module is already large (~1000 LoC) and its
   responsibility is the training mechanics. Post-training I/O and
   summary metrics belong at the orchestration layer.
2. A callback can be unit-tested against a stub `LightningModule` without
   pulling the full `DiffusionModule` into the test fixture.
3. The user can compose the dump with EMA by callback ordering: EMA
   swaps weights at `on_validation_start` / `on_validation_end`; the
   dump callback fires at `on_fit_end`, which the EMA callback does not
   currently hook. We extend `EMACallback` (or add a sibling helper) so
   `on_fit_end` swaps EMA weights into the live model for the duration
   of the dump call. See open question 3 below.

The runner registers the callback when `training.final_sample_dump.num_samples > 0`.

`on_fit_end(trainer, pl_module)` is the only hook the callback overrides.
It pulls the sampler, noise process, evaluator, and datamodule reference
graphs from `pl_module` (using `getattr(..., None)` and raising
informatively if any required attribute is absent — the
`_require_backbone_parameters` pattern from `EMACallback` generalises
here), drives the generation loop, writes the artefact, and emits
metrics.

The alternative — `DiffusionModule.on_fit_end` — is rejected because
introducing the dump logic on the module forces every other Lightning
module that wants the same artefact (e.g. a future `DenoisingModule`
running discrete diffusion) to duplicate it, and because moving I/O onto
the module mixes concerns the EMA refactor deliberately split.

### Data flow

`on_fit_end` runs in this order:

1. Resolve config: `num_samples`, `sample_batch_size = 2 *
   trainer.datamodule.batch_size` when null, `save_path =
   trainer.default_root_dir / "final_samples.pt"` when null.
2. Pull collaborators: `sampler = pl_module.sampler`,
   `noise_process = pl_module.noise_process`,
   `evaluator = pl_module.evaluator`. Raise `RuntimeError` if any is
   `None` — this callback has no meaningful behaviour without all three.
3. Pull node-count distribution. Re-use the same path
   `DiffusionModule.generate_graphs` uses today (the datamodule's
   `node_count_distribution` or equivalent). The callback MUST sample
   per-graph `n` in the same way validation generation does, so the
   final dump distribution matches the training-time evaluation.
4. Generation loop: while `remaining > 0`, call `sampler.sample(model,
   noise_process, num_graphs=min(remaining, sample_batch_size),
   num_nodes=<sampled or fixed>, device=pl_module.device)`. Append to a
   list. Decrement `remaining`. Log `final/sample_progress` at each
   iteration so a long dump shows progress in W&B.
5. Persist: `torch.save({"graphs": samples, "meta": {...}}, save_path)`
   where `meta` records `{"num_samples": int, "epoch": int,
   "global_step": int, "noise_process": str, "ema_active": bool,
   "wall_seconds": float, "git_sha": str | None}`. The wrapper dict
   makes future schema additions backward-compatible without breaking
   readers; readers MUST handle the case where `meta` keys are missing
   (Postel's law applies at the artefact boundary).
6. If `evaluate: true`: pull `refs =
   trainer.datamodule.get_reference_graphs(stage,
   evaluator.eval_num_samples)` for each requested stage, call
   `evaluator.evaluate(refs=refs, generated=samples)`, log every entry
   of `results.to_dict()` under `final/<stage>/<metric>`. Fail-loud on
   `None` returns (the evaluator returns `None` only when reference
   counts are too low; that is a configuration error post-training).

### Storage / artifact format

Single file. Default location is
`${trainer.default_root_dir}/final_samples.pt`. The wrapper schema:

| key      | type                  | semantics                                                                     |
|----------|-----------------------|-------------------------------------------------------------------------------|
| `graphs` | `list[GraphData]`     | One entry per generated graph, post-`Sampler.sample` trim (per-graph node count). |
| `meta`   | `dict[str, Any]`      | Provenance dict (see step 5 above).                                           |

A `list[GraphData]` is chosen over a stacked dense tensor because graphs
have heterogeneous node counts (the sampler trims each entry to its real
`n`). A stacked tensor would force re-padding to `max_n`, wasting memory
on a long-tail size distribution. The codebase has no existing precedent
for `torch.save` of `GraphData` (a search of `src/tmgg/` returns no hits;
see commit message for confirmation), so this spec defines the pattern;
future readers MUST go through a small loader helper
`tmgg.evaluation.final_samples.load(path)` that validates the dict
schema. The loader lives in the same module as the callback's writer.

### Validity checks

The callback MUST raise `ValueError` at `__init__` when:

- `num_samples < 0`.
- `sample_batch_size` is set and `<= 0`.
- `evaluation_reference` is not in `{"val", "test", "both"}`.

The callback MUST raise `RuntimeError` at `on_fit_end` when:

- Any of `sampler`, `noise_process`, `evaluator` is `None` on the
  module. (Don't silently skip — the user opted in via config.)
- `save_path` already exists. The callback writes once per fit; an
  overwrite indicates either a bug or a re-fit collision the user should
  notice.

## Open questions

1. EMA weight-swap at `on_fit_end`. `EMACallback` today swaps at
   validation only. If both callbacks are active, do we want the final
   dump to use EMA weights (the published-quality model) or live
   weights? Default proposal: EMA, matching DiGress and the standard
   convention. Implementation requires either extending `EMACallback`
   with `on_fit_end` swap hooks, or having `FinalSampleDumpCallback`
   detect a sibling `EMACallback` on the trainer and call its `store` /
   `copy_to` / `restore` directly. The first option is cleaner; record
   as the recommendation. Owner: igork.
2. Reference-set choice. `evaluation_reference: val` is the safest
   default — the test set should remain held out for the final paper
   run. Should we default to `null` (skip evaluation) and force the user
   to opt in? Owner: igork.
3. Modal sandbox persistence. On Modal, `trainer.default_root_dir`
   resolves inside the container; the artefact must land on a Modal
   volume to survive container termination. Verify the launch wrapper
   already maps the run dir onto a persistent volume; if not, the
   callback needs an explicit `save_path` resolution that points to a
   known-mounted path. Owner: igork (ops).
4. Sample-batch-size default. Upstream's `2 * train.batch_size` rule was
   tuned for their architectures. For our larger models on smaller GPUs
   it may OOM. Should the default be `min(2 * train.batch_size, 64)`?
   Owner: TBD; collect data once a real run executes.
5. Distributed runs. The current sampler runs single-device. If we ever
   shard validation across GPUs, the callback would need to gather
   samples from all ranks before writing. Out of scope for v1; flag
   here to revisit.

## Acceptance

A1. Setting `training.final_sample_dump.num_samples: 40` on the SPECTRE
SBM config and running `trainer.fit()` to completion produces the file
`${run_dir}/final_samples.pt` whose `graphs` list contains exactly 40
`GraphData` instances and whose `meta` dict has the keys listed in step 5
above.

A2. With `evaluate: true` and `evaluation_reference: val`, the W&B run
shows `final/val/degree_mmd`, `final/val/clustering_mmd`,
`final/val/spectral_mmd`, `final/val/sbm_accuracy` (and any other key
returned by `EvaluationResults.to_dict`) at one logged entry each, all
emitted at the global step at which fit ended.

A3. Setting `num_samples: 0` (or omitting the block) registers no
callback, leaves no artefact, and adds no W&B entries under `final/`.

A4. The reader helper `tmgg.evaluation.final_samples.load(path)` round-
trips the saved dict, returning the `graphs` list and the `meta` dict
unchanged for a freshly written artefact.

A5. The callback raises `RuntimeError` at `on_fit_end` when the
trained module has no sampler attached (e.g. the user enabled the
callback against a `DenoisingModule` that omits it), with a message
naming the missing collaborator.


## Resolutions (2026-04-22)

User responses to the open questions, applied as the implementation
contract:

- **Q6 (reference-set provenance)**: both. Validation passes during
  training continue to evaluate against the val set as today; the
  end-of-fit dump uses the **test** set for the published-quality
  numbers. The callback exposes both reference handles via the
  datamodule and selects test at `on_fit_end`.

- **Q7 (EMA semantics at on_fit_end)**: swap to EMA weights for the
  dump if and only if an `EMACallback` (D-15) is registered on the
  trainer. The callback inspects `trainer.callbacks` for a registered
  `EMACallback` instance, calls its `store + copy_to` shim before
  sampling, and `restore` after. If EMA is not enabled, sample with
  live weights.

- **Q8 (Modal volume persistence)**: write to the existing
  `tmgg-outputs` Modal volume (mounted at `/data/outputs` per
  `src/tmgg/modal/_lib/volumes.py`). Default path resolution:
  `${OUTPUTS_MOUNT}/final_samples/${run_name}.pt` when the runtime is
  Modal, `${run_dir}/final_samples.pt` locally. The path-selection
  helper inspects an env signal that already distinguishes the two
  contexts (or falls back to a CLI/config flag — confirm the canonical
  one at implementation time).

- **Q9 (DDP / parallel sampling)**: sequential v1 confirmed.
  Distributed-rank batching is a follow-up; for v1 the dump runs on
  rank 0 only and other ranks no-op so DDP doesn't double-sample.
  Document the rank gate in the callback.
