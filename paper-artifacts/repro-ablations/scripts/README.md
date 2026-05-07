# scripts

The refresh pipeline. See `../HOW-TO-UPDATE.md` for the operational
protocol.

## `refresh.py`

Self-contained `uv run` script with PEP-723 inline deps. Hits the
W&B API; needs `WANDB_API_KEY` set in the environment for a member
of the entity that owns the per-variant projects.

```bash
cd paper-artifacts/repro-ablations
export WANDB_API_KEY="..."
uv run scripts/refresh.py [--date YYYY-MM-DD] [--summary "<one-line>"] [--skip-media]
```

Reads `../data/runs_index.source.yaml`, queries wandb for each run,
and writes:

- `../data/runs_index.csv`
- `../data/all_metrics_long.parquet`
- `../data/per_run_history/<run_id>.parquet`
- `../media/per_run/<run_id>/*.png`
- `../snapshots/{runlog,ablations-measurement}-<date>.md`
- `../CHANGELOG.md` (append)


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
