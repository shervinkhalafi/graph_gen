# scripts

The refresh pipeline. See `../HOW-TO-UPDATE.md` for the operational
protocol.

## `refresh.py`

Self-contained `uv run` script with PEP-723 inline deps.

```bash
cd paper-artifacts/repro-ablations
uv run scripts/refresh.py [--date YYYY-MM-DD] [--summary "<one-line>"] [--skip-media]
```

Reads `../data/runs_index.source.yaml`, queries wandb for each run,
and writes:

- `../data/runs_index.csv`
- `../data/all_metrics_long.csv`
- `../data/per_run_history/<run_id>.parquet`
- `../media/per_run/<run_id>/*.png`
- `../snapshots/{runlog,ablations-measurement}-<date>.md`
- `../CHANGELOG.md` (append)

Pure-function helpers (`classify_metric_namespace`, `build_run_slug`)
are unit-tested in `tests/paper_artifacts/test_refresh_helpers.py`.
