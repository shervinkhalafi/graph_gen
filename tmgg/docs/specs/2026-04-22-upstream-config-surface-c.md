# Spec: evaluate-all-checkpoints CLI (D-16c)

**Status:** Draft
**Date:** 2026-04-22
**Author:** igork
**Refs:**
- `docs/reports/2026-04-21-digress-spec-our-impl-review/divergence-triage.md`
  (D-16c, parity item #46), lines 388-396.
- Parity commit `2e58eb1f` ("fix(config): early-stopping patience 1000,
  save_top_k -1") — the change that motivates this CLI by retaining
  every training checkpoint.
- Upstream `digress-upstream-readonly/configs/general/general_default.yaml:27`
  (`evaluate_all_checkpoints: False`).
- `src/tmgg/experiments/discrete_diffusion_generative/evaluate_cli.py`
  (the existing single-checkpoint evaluator we generalise).
- `src/tmgg/evaluation/graph_evaluator.py:93-140` (the
  `EvaluationResults` schema we serialise as CSV).

**Normative language.** The key words MUST, MUST NOT, SHOULD, SHOULD NOT,
and MAY follow [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Goal

Provide a one-shot CLI that walks every Lightning checkpoint in a run
directory, loads each, generates a fixed number of samples, runs
`GraphEvaluator`, and writes a CSV of metrics keyed by checkpoint. This
unlocks training-curve plots for generative metrics (MMD, SBM accuracy)
that the training loop does not log per checkpoint, and lets a researcher
post-hoc isolate the best checkpoint by a generative criterion rather
than the validation NLL the trainer monitors.

## Upstream behaviour

DiGress exposes `general.evaluate_all_checkpoints: False`
(`general_default.yaml:27`). When enabled, the upstream `main.py` walks
the checkpoint directory after fit completes and runs `trainer.test`
against each, accumulating metrics into the W&B run (the loop itself is
in `main.py`, not the diffusion module). Our setup never had the
equivalent; the parity commit `2e58eb1f` flipped `save_top_k = -1` so
every checkpoint is now retained, but we do not yet exploit them.

This CLI is the offline, decoupled version of the same workflow. It does
not piggy-back on `trainer.test`; it loads each checkpoint into a fresh
`DiffusionModule`, drives the sampler directly, and writes a CSV. The
researcher can then load the CSV in any analysis tool to plot any column
against `step`.

## Non-goals

This spec does not extend `tmgg-discrete-eval` (the single-checkpoint
evaluator at
`src/tmgg/experiments/discrete_diffusion_generative/evaluate_cli.py`)
in-place — the new CLI lives alongside it and reuses its
`evaluate_checkpoint` core. It does not introduce a parallel sampler, a
new evaluator, or a new metric. It does not handle distributed sampling.
It does not orchestrate Modal launches for the evaluation pass; the
researcher runs the CLI on a host with the saved checkpoints accessible.

## Design

### Surface (config + CLI)

A new entry point `tmgg-discrete-eval-all` wired in `pyproject.toml`'s
`[project.scripts]` block. Naming mirrors the existing
`tmgg-discrete-eval` (single-checkpoint) and stays inside the
`tmgg-discrete-*` namespace so shell completion groups them.

```
tmgg-discrete-eval-all \
    --run-dir <path>                 \
    --num-samples <int>              \
    [--num-nodes <int>]              \
    [--checkpoint-glob "*.ckpt"]     \
    [--output-csv <path>]            \
    [--dataset {sbm,erdos_renyi,...}] \
    [--kernel {gaussian,gaussian_tv}] \
    [--sigma <float>]                \
    [--device <str>]                 \
    [--seed <int>]                   \
    [--skip-existing]                \
    [--limit <int>]                  \
    [--sort-by {step,name,mtime}]
```

Required: `--run-dir`, `--num-samples`. The remainder match the
single-checkpoint CLI argument names exactly so a user can reuse muscle
memory; only `--checkpoint-glob`, `--output-csv`, `--skip-existing`,
`--limit`, and `--sort-by` are new.

Default `--output-csv` is
`<run-dir>/all_checkpoints_eval.csv`. Default `--checkpoint-glob` is
`*.ckpt`. Default `--sort-by` is `step` (parsed from the canonical
checkpoint filename pattern). Default `--device` is `cpu`, matching
`tmgg-discrete-eval`.

`--skip-existing` reads the CSV (when it exists), collects the
`checkpoint_path` column, and skips any checkpoint already present. This
makes the CLI restart-safe — interrupted runs resume without re-doing
finished work.

`--limit N` evaluates only the first `N` checkpoints in sort order.
Useful for quick sanity passes ("does the CLI work end-to-end before I
spend an hour on 100 checkpoints?") without a separate dry-run flag.

### Implementation site

A new module `src/tmgg/experiments/discrete_diffusion_generative/evaluate_all_cli.py`
sits next to the existing `evaluate_cli.py`. Its `main(argv)` function is
registered as `tmgg-discrete-eval-all` in `pyproject.toml`'s
`[project.scripts]` block. The module imports
`evaluate_checkpoint` from the sibling `evaluate_cli.py` and calls it
once per checkpoint; no re-implementation of the sampling or evaluation
logic.

The argparse parser is kept narrow — `argparse`, not `click` or `typer`,
matches the existing CLI surface (`evaluate_cli.py` uses `argparse`).
Switching frameworks is out of scope.

The new module also exposes a programmatic entry
`evaluate_all_checkpoints(run_dir, num_samples, ...) -> pandas.DataFrame`
so test fixtures and notebooks can drive the same workflow without
shelling out. The CLI `main` is a thin wrapper over this function.

### Data flow

1. **Discover checkpoints.** `glob(run_dir / checkpoint_glob)`. Filter
   to files. Parse each filename for the step counter using a regex
   `r"step=(?P<step>\d+)"`; checkpoints that do not match the pattern
   are kept but assigned `step = -1` and a warning is logged.
2. **Sort.** `step` (ascending), `name` (alphabetical), or `mtime`
   (file modification time ascending). The default `step` matches what
   a training-curve plot expects.
3. **Filter.** Apply `--limit`. Apply `--skip-existing` by reading the
   output CSV's `checkpoint_path` column when the file exists; if it
   does not exist, the filter is a no-op.
4. **Per-checkpoint loop.** For each remaining checkpoint:
   - Call `evaluate_checkpoint(checkpoint_path, dataset_type=...,
     num_samples=..., num_nodes=..., mmd_kernel=..., mmd_sigma=...,
     device=..., seed=...)`. The function already handles the model
     reconstruction dance (`_load_diffusion_module`).
   - Time the call (`time.perf_counter`); record `eval_seconds`.
   - Append a row to an in-memory list of dicts.
   - After each checkpoint, flush the accumulated rows to the CSV
     (overwriting the file, header included). This is the simplest
     restart-safe approach: a crash mid-run leaves the CSV consistent
     with whatever finished, and `--skip-existing` picks up where it
     left off. Per-row append (without re-writing the header) is harder
     to make robust and not worth the complexity for run sizes
     < 1000 checkpoints.
5. **Return / print.** The function returns the DataFrame. The CLI
   `main` prints `Wrote N rows to <output_csv>` and exits `0`.

The loop is sequential. Parallelism is rejected for v1 (see open
question 3).

### Storage / artifact format

CSV with header. One row per checkpoint. Columns, in order:

| column                  | dtype   | source / semantics                                                                 |
|-------------------------|---------|------------------------------------------------------------------------------------|
| `checkpoint_path`       | str     | Absolute path to the `.ckpt` file.                                                 |
| `step`                  | int     | Parsed from filename; `-1` when unparseable.                                       |
| `epoch`                 | int     | Lightning's `current_epoch` from the checkpoint hparams when available; `-1` else. |
| `num_samples_generated` | int     | The actual count returned by the sampler (defensive — should equal `--num-samples`). |
| `eval_seconds`          | float   | Wall-clock time for the per-checkpoint evaluation.                                 |
| `degree_mmd`            | float   | `EvaluationResults.degree_mmd`.                                                    |
| `clustering_mmd`        | float   | `EvaluationResults.clustering_mmd`.                                                |
| `spectral_mmd`          | float   | `EvaluationResults.spectral_mmd`.                                                  |
| `orbit_mmd`             | float   | `EvaluationResults.orbit_mmd`. Empty cell when `None` (orca unavailable).          |
| `sbm_accuracy`          | float   | `EvaluationResults.sbm_accuracy`. Empty cell when `None` (graph-tool unavailable). |
| `planarity_accuracy`    | float   | `EvaluationResults.planarity_accuracy`. Empty cell when skipped.                   |
| `uniqueness`            | float   | `EvaluationResults.uniqueness`. Empty cell when skipped.                           |
| `novelty`               | float   | `EvaluationResults.novelty`. Empty cell when no train graphs supplied.             |

The CSV header is fixed: every row has every column, even when an
underlying metric is `None` (encoded as the empty string). A
schema-stable CSV is easier to load in pandas/polars than a sparse one.
The column set is a strict superset of `EvaluationResults.to_dict()`
keys; if the evaluator ever grows new keys, this CLI MUST be updated in
the same commit.

The CSV is written at `<run-dir>/all_checkpoints_eval.csv` by default.
The CLI does not touch any other file under `run-dir`.

### Validity checks

The CLI MUST raise (via `argparse.ArgumentTypeError` at parse time, or
`FileNotFoundError` / `ValueError` at run time):

- `--run-dir` does not exist or is not a directory.
- `--num-samples < 1`.
- `--checkpoint-glob` matches zero files in `run-dir`.
- `--limit` is set and `< 1`.
- `--device` cannot be passed to `torch.device()` (delegate to torch's
  exception).

When a single checkpoint fails to load (e.g. a corrupt file, a pre-
refactor `.ckpt` that no longer matches the current `DiffusionModule`),
the CLI MUST log a warning naming the file and the exception, write a
row with `step` parsed but every metric column empty, and continue. This
is the one place we deviate from CLAUDE.md's "fail loud" stance: the
whole point of the CLI is to walk a directory of checkpoints, and a
single bad file should not abort 99 good ones. The empty-metrics row in
the CSV is the visible failure signal.

## Open questions

1. Reference-set construction. `evaluate_checkpoint` constructs
   reference graphs via `generate_reference_graphs(dataset_type=...,
   num_samples=..., num_nodes=..., seed=...)` rather than reusing the
   datamodule the checkpoint was trained against. That is fine when the
   user knows the dataset, but it means the CLI will silently evaluate
   against the wrong reference if the user mistypes `--dataset`. Should
   the CLI inspect the checkpoint's hparams to infer the dataset and
   warn on mismatch with `--dataset`? Owner: igork.
2. EMA weights. If the checkpoint stores both live and EMA shadow
   weights, which does the CLI evaluate? The current
   `_instantiate_checkpoint_model` rebuilds from `model_config` and
   `Lightning.load_from_checkpoint` repopulates from `state_dict` —
   this loads live weights. To evaluate the EMA shadow, the CLI would
   need to detect an `ema_state` key in the checkpoint and copy it into
   the model. Default proposal: evaluate live weights for v1 (matches
   `tmgg-discrete-eval`'s current behaviour); add a `--use-ema` flag
   in a follow-up once the EMACallback writes its shadow into the
   checkpoint. Owner: TBD.
3. Parallelism. Sequential evaluation of 100 checkpoints at 1000
   samples per checkpoint takes hours. A naïve `multiprocessing.Pool`
   would let one wall-clock pass cover many checkpoints, but
   `compute_sbm_accuracy` already pins graph-tool's worker count to 1
   for thread-safety; doubling the parallelism layer is fragile. v1
   stays sequential. Owner: out of scope for D-16c; revisit only when
   wall time becomes a real constraint.
4. Device handling. The single-checkpoint CLI defaults `--device cpu`.
   For 100 checkpoints, GPU evaluation would help, but loading and
   evicting models repeatedly on a GPU has overhead. Should the CLI
   keep the default at `cpu` and document `--device cuda` for users
   with the headroom, or detect availability and pick automatically?
   Default proposal: stay explicit (`cpu`) for parity with
   `evaluate_cli.py`. Owner: igork.
5. CSV column drift. The CSV column order is fixed in this spec. If a
   future commit adds a new key to `EvaluationResults.to_dict()`, do
   downstream readers need to handle the new column at the end, or
   should we sort columns lexicographically to make append-stability
   easier? Default proposal: keep the order from this spec; document
   that new columns append at the end. Owner: igork.

## Acceptance

A1. Pointing `tmgg-discrete-eval-all --run-dir <path-to-real-run>
--num-samples 100` at a SBM run with N retained checkpoints produces a
CSV at `<run-dir>/all_checkpoints_eval.csv` with exactly N rows, each
row's `checkpoint_path` resolving to a real file, and each row's
`degree_mmd`, `clustering_mmd`, `spectral_mmd` populated.

A2. Re-running with `--skip-existing` after the first invocation
produces no additional work and exits `0` without modifying the CSV.

A3. `--limit 3` evaluates exactly the first three checkpoints in the
selected sort order and writes a 3-row CSV.

A4. A corrupted checkpoint inserted into the directory results in a
warning log line and an empty-metrics row, while neighbouring valid
checkpoints continue to evaluate cleanly.

A5. The programmatic `evaluate_all_checkpoints(...)` function returns a
`pandas.DataFrame` with the same column set as the CSV header, and the
CLI's CSV is `df.to_csv(path, index=False)`.

A6. A follow-up analysis script (out of scope to implement here, but
the CSV format permits it) loads the CSV, plots `degree_mmd` against
`step`, and shows a smooth descent for a healthy SBM run.


## Resolutions (2026-04-22)

User responses to the open questions, applied as the implementation
contract:

- **Q10 (default device)**: autodetect with GPU preference. Probe
  `torch.cuda.is_available()` and select `cuda:0` when present;
  otherwise `cpu`. Expose a `--device {auto,cpu,cuda}` override on the
  CLI; default `auto`.

- **Q11 (CSV column-drift policy)**: easy answer — union of all keys
  per row, write `NaN` for missing columns. Pandas does this natively
  when re-running and concatenating: load existing CSV (if present),
  build a new row from the current `EvaluationResults` keys, append,
  write. Future metric additions surface as new columns with `NaN` in
  rows from earlier runs; nothing breaks. Schema versioning is a
  later concern if at all.

The two cross-spec answers from the D-16b resolution apply here too:

- **Q6 (reference-set provenance, shared with D-16b)**: the eval-all
  CLI's `--reference-set {val,test}` flag defaults to `val` (matches
  training-time evaluation) and the user opts into `test` for
  published-quality numbers, mirroring the D-16b dump's split.

- **Q7 (EMA semantics, shared with D-16b)**: when a checkpoint
  carries an EMA shadow (Lightning saves it as part of the callback
  state), the CLI loads the EMA weights into the model before
  sampling. Add a `--use-ema {auto,true,false}` flag with `auto`
  default that uses EMA whenever the checkpoint contains it.
