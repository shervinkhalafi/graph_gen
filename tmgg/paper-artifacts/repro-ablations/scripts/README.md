# scripts

The refresh pipeline. See `../HOW-TO-UPDATE.md` for the operational
protocol.

## `refresh.py`

Self-contained `uv run` script with PEP-723 inline deps. Hits the
W&B API; needs `WANDB_API_KEY` for a `<TEAM-ENTITY>` member
(via `doppler run` if your `~/.netrc` is for a different account).

```bash
cd paper-artifacts/repro-ablations
doppler run -- uv run scripts/refresh.py [--date YYYY-MM-DD] [--summary "<one-line>"] [--skip-media]
```

Reads `../data/runs_index.source.yaml`, queries wandb for each run,
and writes:

- `../data/runs_index.csv`
- `../data/all_metrics_long.parquet`
- `../data/per_run_history/<run_id>.parquet`
- `../media/per_run/<run_id>/*.png`
- `../snapshots/{runlog,ablations-measurement}-<date>.md`
- `../CHANGELOG.md` (append)

Pure-function helpers (`classify_metric_namespace`, `build_run_slug`)
are unit-tested in `tests/paper_artifacts/test_refresh_helpers.py`.

## `quickstatus.py`

Read-only companion to `refresh.py`. No W&B calls — pulls from the
already-refreshed `data/runs_index.csv` + `data/per_run_history/`
parquets and emits markdown rows that match the runlog's
quick-status table schema. Intended to be copy-pasted into
`runlog.md` after a refresh.

```bash
cd paper-artifacts/repro-ablations
uv run scripts/quickstatus.py [--postfix] [--dataset {sbm,enzymes}]
```

Flags filter the rows; default emits all 28 runs in the index.
Stability mirrors the runlog heuristic (grad_norm < 5.0 → ✓,
otherwise ⚠/✗); `runs_index.source.yaml` health overrides
(`invalidated_mask_bug`, `blew_up`) take precedence.
