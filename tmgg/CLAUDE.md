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
