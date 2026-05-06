- Unless explicitly specified by the user or confirmed with a question, we never silence pyright errors or warnings, and we never do "graceful fallback", we always _address_ the error/warning, and we always fail loudly and informatively with an exception

## Testing

Run the test suite with:

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```
- the semantic groupings are modeltype,dataset,asymetric, noiselevels,input embeddings, ablations across the digress components + the hyperparmater settings. the only averaging we allow is across seeds,across hyperparmaters we pick the best (but also report the distribution) and we group by these high level research question comparisons

## Debugging Modal runs

When a run on Modal (`tmgg-spectral`) crashes, follow `docs/debugging-modal.md`.
Key point: `modal app logs` streams live only — there is no historical
log retrieval from the CLI. Reproduce under a live stream (`DETACH=0` on
the launch wrapper, tee'd to a file) or attach `modal app logs tmgg-spectral`
in parallel immediately after launch.

## Run log

Track every launched Modal run via two pieces:

- `runlog.md` (repo root) is the index — quick-status table, panel
  sections that link out, cross-cutting findings, open questions,
  backfill checklist.
- `run_details/<launch-date-iso>/<config>_<run_id>_details.md` is the
  per-run detail file — identity, timeline, fetched data, structured
  diagnostics block (MMDs, loss, gradient health, throughput),
  visualisations, notes.

On launch, create the detail file and add a one-line link in
`runlog.md`. On state change, update the detail file (status,
ended_at, notes) and refresh the runlog row. On data fetch, edit the
detail file's `Fetched` block. On scoring, fill `Diagnostics` and link
visualisations under `Visuals`. The runlog index is the single source
of truth for "what's running, what finished, what's been pulled,
what's been measured" — check it before relaunching the same config.

### Cross-panel measurement summary

Second file to keep current alongside `runlog.md` when checking
results: [`docs/eval/2026-05-06-ablations_measurment.md`](docs/eval/2026-05-06-ablations_measurment.md).
It is the consolidated cross-panel view — per-step MMD CSV, raw
last-cycle and ratio tables, anchor conversions to absolute MMD²
via our measured train↔test baselines, and the best-per-metric / gap
table against DiGress paper and HiGen anchors. When new eval cycles
land or runs finish/crash, refresh both files together: `runlog.md`
captures *operational* state, this file captures *measurement*
state. Update the snapshot timestamp at the top when you do.

### Paper-bound supplementary bundles

Self-contained bundles intended for inclusion in a paper as figure
source / supplementary tables live under `paper-artifacts/<topic>/`
and have stable schemas. Each bundle contains: a long-format data
CSV, raw history parquets per run, configs (pre-fix + post-fix),
baseline + anchor context markdown, snapshots of relevant analysis
docs, and a `HOW-TO-UPDATE.md` per folder. The first example is
`paper-artifacts/repro-ablations/`. Use this when the bundle's
purpose is "drop into the paper / share externally with stable
schemas". Ad-hoc analysis reports continue to live under
`wandb_export/<analysis-slug>/`.

## W&B Data Exports

Exported W&B data lives in `wandb_export/`. Check existing exports before re-exporting to avoid redundant API calls.

**Current exports (as of 2026-02-03):**

| File | Project | Runs | Notes |
|------|---------|------|-------|
| `graph_denoise_team_spectral_denoising_runs.parquet` | spectral_denoising | 2345 | Main experiments including DiGress variants |
| `graph_denoise_team_tmgg-None_runs.parquet` | tmgg-None | 12 | Recent DiGress experiments |
| `graph_denoise_team_tmgg-stage2_validation_runs.parquet` | tmgg-stage2_validation | ~215 | Stage 2 validation |
| `graph_denoise_team_00_initial_experiment_widening_runs.parquet` | 00_initial_experiment_widening | ~269 | Early experiments |

**Export tools:** `wandb-tools/export_runs.py`, `wandb-tools/aggregate_runs.py`, `wandb-tools/analyze_runs.py`

## TODOs

- [ ] Make `wandb-tools/export_runs.py` auto-deduplicate and cache exports: check existing parquet files, compare timestamps/run counts, only fetch new runs since last export (incremental sync)
