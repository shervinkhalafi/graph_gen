# How to update this bundle

The bundle is "live": as new runs launch, old runs finish, and eval
cycles accumulate, the data needs to be refreshed. This document is
the operational protocol.

The bundle and the repo-root [`runlog.md`](../../runlog.md) form a
matched pair: the bundle is the canonical *data* (per-run parquets,
runs_index.csv, dated snapshots), `runlog.md` is the human-readable
*index* with cross-cutting prose. The refresh script also snapshots
`runlog.md` (and `docs/eval/<date>-ablations_measurment.md`) into
`snapshots/`, but the *live* `runlog.md` is hand-edited. Workflow:
refresh the bundle first (this script) → then update the live
`runlog.md` quick-status table from the freshly-pulled parquets →
commit both together.

## When to refresh

- A new run launches under one of the panel variants.
- A previously-running run finishes, crashes, or is killed.
- A new eval cycle lands (changes the per-step trajectories).
- Before any paper revision.

## Prereqs

- `wandb` authenticated for the `graph_denoise_team` entity. The
  on-host machine's `~/.netrc` may hold a different (non-team) wandb
  token; in that case use `doppler run` to inject the team-member key
  (see "Running with doppler" below).
- `uv` available on PATH.
- `doppler` CLI available and configured for this project (only if
  using the doppler-managed key).
- Repo at the commit you want the snapshots to reflect.

### Running with doppler

If your `~/.netrc` wandb token isn't for a `graph_denoise_team`
member, use the team-member key from doppler:

```bash
cd paper-artifacts/repro-ablations
doppler run -- uv run scripts/refresh.py --summary "..."
```

`doppler run` injects `WANDB_API_KEY` (and any other doppler-managed
secrets) into the environment before invoking the script; the wandb
SDK reads `WANDB_API_KEY` and overrides `~/.netrc`. The doppler
project + config used for this repo is whatever `doppler configure`
prints (typically `shervin-graph` / `prd`).

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
# preferred — works regardless of which wandb account the host's ~/.netrc holds
doppler run -- uv run scripts/refresh.py --summary "post-eval-cycle-N refresh"

# alternative — only if your ~/.netrc is already for a graph_denoise_team member
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

# Long parquet has rows for each run
uv run --with pandas --with pyarrow python -c "
import pandas as pd
idx = pd.read_csv('data/runs_index.csv')
long = pd.read_parquet('data/all_metrics_long.parquet')
missing = set(idx.run_slug) - set(long.run_slug)
print('runs without metrics:', missing or 'none')
"

# Each run has at least one image (where wandb logged any)
ls media/per_run/ | wc -l
```

### 4. Regenerate the runlog quick-status table rows (optional)

Once `runs_index.csv` and the per-run parquets are fresh, regenerate
the markdown rows for the runlog's quick-status table from a single
command, then copy-paste the changed rows into `runlog.md`:

```bash
cd paper-artifacts/repro-ablations
uv run scripts/quickstatus.py --postfix          # active panel only
uv run scripts/quickstatus.py --postfix --dataset enzymes
uv run scripts/quickstatus.py                    # all 28 runs
```

The script reads `data/runs_index.csv` + `data/per_run_history/<run_id>.parquet`
(no W&B calls, fast) and emits markdown matching the runlog's schema:

```
| Config | Run ID | Launched (UTC) | Status | Step (cycles) | degree MMD² | grad_norm | Stable? | Detail |
```

Numbers: latest non-NaN gen-val degree MMD² (annotated with the eval
step if not at the latest training step), latest grad_norm_total,
eval-cycle count from distinct degree-MMD steps. Stability mirrors the
runlog's heuristic (`grad_norm < 5.0` → ✓, otherwise ⚠/✗); health
overrides from `runs_index.source.yaml` (`invalidated_mask_bug`,
`blew_up`) take precedence.

After pasting rows into `runlog.md`, also bump the snapshot timestamp
on the "Quick status table" header and add a session note in
"Cross-cutting findings" if anything material moved.

### 5. Inspect the diff and commit

```bash
git status paper-artifacts/repro-ablations/ runlog.md
git add -p paper-artifacts/repro-ablations/ runlog.md
git commit -m "data(paper-artifacts): refresh repro-ablations YYYY-MM-DD"
```

Bundle and runlog should be committed together when the runlog rows
were sourced from this refresh; commit the bundle alone if you only
ran refresh.py and didn't touch the runlog.

## What gets snapshotted vs mutated

| path | behaviour |
|------|-----------|
| `data/runs_index.csv` | mutated in place |
| `data/all_metrics_long.parquet` | mutated in place; **gitignored, not committed** (large derived view of the per-run parquets) |
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
