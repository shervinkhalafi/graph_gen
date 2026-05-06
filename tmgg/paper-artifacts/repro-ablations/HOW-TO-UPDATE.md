# How to update this bundle

The bundle is "live": as new runs launch, old runs finish, and eval
cycles accumulate, the data needs to be refreshed. This document is
the operational protocol.

## When to refresh

- A new run launches under one of the panel variants.
- A previously-running run finishes, crashes, or is killed.
- A new eval cycle lands (changes the per-step trajectories).
- Before any paper revision.

## Prereqs

- `wandb` authenticated for the `graph_denoise_team` entity.
- `uv` available on PATH.
- Repo at the commit you want the snapshots to reflect.

## Steps

### 1. Update `data/runs_index.source.yaml`

If a *new run* launched, append a stub entry:

```yaml
- run_id: <8-char-id>
  dataset: <sbm | enzymes>
  variant: <vignac | vignac_spectral | pearl | pearl_spectral | pearl_gnnconv_norm | pearl_gnnconv_raw>
  postfix: <true | false>
  config_name: <hydra experiment name>
  parent_run_id: <id-of-parent-or-null>
  continuation_type: <original | auto_reassign_force_fresh | auto_reassign_resume | manual_resume_from_checkpoint | post_fix_replacement>
  health: <stable | elevated_grad | blew_up | invalidated_mask_bug>
  notes: "<one-line context>"
```

If an *existing run's status* changed (e.g. running → crashed),
update its `health` and `notes`. The `final_state` and `final_step`
fields are wandb-derived and refresh automatically.

If a *run got reassigned by Modal*, add the auto-reassign child as a
new entry pointing to the parent.

### 2. Run the refresh script

```bash
cd paper-artifacts/repro-ablations
uv run scripts/refresh.py --summary "post-eval-cycle-N refresh"
```

Optional flags:
- `--date YYYY-MM-DD` — override the snapshot date suffix (default: today UTC).
- `--skip-media` — skip wandb-image downloads (faster for dry-runs).

### 3. Verify

```bash
# Index row count matches the YAML
yaml_count=$(uv run --with pyyaml python -c "import yaml,pathlib;print(len(yaml.safe_load(pathlib.Path('data/runs_index.source.yaml').read_text())['runs']))")
csv_count=$(($(wc -l < data/runs_index.csv) - 1))
echo "yaml=$yaml_count csv=$csv_count"

# Long CSV has rows for each run
uv run --with pandas python -c "
import pandas as pd
idx = pd.read_csv('data/runs_index.csv')
long = pd.read_csv('data/all_metrics_long.csv')
missing = set(idx.run_slug) - set(long.run_slug)
print('runs without metrics:', missing or 'none')
"

# Each run has at least one image (where wandb logged any)
ls media/per_run/ | wc -l
```

### 4. Inspect the diff and commit

```bash
git status paper-artifacts/repro-ablations/
git add -p paper-artifacts/repro-ablations/
git commit -m "data(paper-artifacts): refresh repro-ablations YYYY-MM-DD"
```

## What gets snapshotted vs mutated

| path | behaviour |
|------|-----------|
| `data/runs_index.csv` | mutated in place |
| `data/all_metrics_long.csv` | mutated in place |
| `data/per_run_history/*.parquet` | mutated in place (full overwrite) |
| `media/per_run/<run_id>/*.png` | mutated in place |
| `snapshots/*.md` | accumulate (existing same-day snapshots not overwritten — script warns) |
| `CHANGELOG.md` | append-only |
| `data/runs_index.source.yaml` | hand-edited |
| `configs/`, `context/` | hand-edited (rare) |

## Adding a brand-new dataset or variant

1. Add the new hydra config under `configs/{pre-fix,post-fix}/` (copy
   from `src/tmgg/experiments/exp_configs/experiment/`).
2. Extend the `variant` enum in
   `paper-artifacts/repro-ablations/scripts/refresh.py` (look for the
   `_wandb_project_for` helper — variant doesn't appear there, but
   the `runs_index.source.yaml` validator may need it).
3. Update `data/DATA-DICTIONARY.md` with the new enum value.
4. Add the new run to `runs_index.source.yaml`.
5. Run the refresh script.

## Adding a new dataset's MMD baseline

1. Compute the baseline JSON via `tmgg-mmd-baselines` Modal entry
   point.
2. Drop the JSON into `context/mmd_baselines/`.
3. Update `context/BASELINES-CONTEXT.md` with the new dataset's row.
4. Update `context/ANCHORS.md` with the per-dataset anchors.

## Troubleshooting

- **wandb 429 / rate-limit.** The script does sequential per-run
  queries; if you hit limits, sleep a few minutes and re-run. Re-runs
  are idempotent.
- **"could not fetch <key> for <run_id>"** — the run hasn't logged
  that media key yet (e.g. eval hasn't happened). Skip; re-run after
  the next eval cycle.
- **Snapshot already exists for today.** The script does not overwrite;
  delete the existing snapshot file by hand if you really want a
  same-day refresh, or use `--date` to specify a different suffix.
- **CSV row count mismatch.** Some runs may have failed to fetch (see
  stderr). Investigate the per-run error and re-run.
