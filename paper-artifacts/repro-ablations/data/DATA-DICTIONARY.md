# Data dictionary

Schema reference for `runs_index.csv` and the per-run parquets.

## `runs_index.csv`

One row per run; eight rows total (one per Table 2 cell). Primary key:
`run_id` (also unique on `run_slug`).

| column | type | nullable | description |
|--------|------|:--------:|-------------|
| `run_slug` | str | no | `<config_name>_<run_id>`. Stable. |
| `run_id` | str | no | wandb 8-char ID. Immutable. |
| `display_name` | str | no | wandb display name. Mutable; do not use for joins. |
| `config_name` | str | no | paper-aligned config name (e.g. `digress_pearl_gcat_sbm`). Matches `configs/<config_name>.yaml`. |
| `wandb_project` | str | no | wandb project string. |
| `wandb_url` | str | no | full wandb URL. Entity is anonymised to `<TEAM-ENTITY>`. |
| `dataset` | enum | no | `sbm` \| `enzymes`. |
| `variant` | enum | no | `vignac` (= DiGress baseline), `pearl`, `pearl_spectral`, `pearl_gnnconv_norm` (= GCAT). |
| `launched_at_utc` | iso8601 | no | from wandb `createdAt`. |
| `ended_at_utc` | iso8601 | yes | null while running. |
| `final_state` | enum | no | `running`, `finished`, `crashed`, `failed`, `killed`, `cancelled`, `oom`, `preempted`. |
| `final_step` | int | no | last `trainer/global_step` (or `_step` fallback). |
| `health` | enum | no | `stable`, `elevated_grad`, `blew_up`. |
| `notes` | str | yes | one-line free text. |

### Health labels

| value | meaning |
|-------|---------|
| `stable` | grad_norm < 5, lr in expected range, no numerical issues at point of observation. |
| `elevated_grad` | grad_norm 1–5, training still converging but with margin to investigate. |
| `blew_up` | grad_norm ≫ 100 or lr → ∞. |

## Per-run parquets

`per_run_history/<run_id>.parquet` is the wandb history exported
verbatim — wide format, one row per `_step`, scalar-only. Schema is
the union of every metric key the run logged.

Useful columns:

| key prefix | what's in it |
|------------|--------------|
| `gen-val/*_mmd` | squared MMD² values for degree, clustering, orbit, spectral distributions. V-statistic biased estimator. |
| `train/loss_*` | per-step / per-epoch training loss. |
| `val/epoch_NLL` | validation NLL each cycle. |
| `diagnostics-train/opt-health/*` | gradient norms, effective LR, per-layer cosine + SNR. |
| `impl-perf/train/step_time_s` | per-step wall-clock. |
| `trainer/global_step` | training-loop step (preferred x-axis for plotting). |

### Units caveat — MMD values are squared

`gen-val/*_mmd` are **squared MMD² values** (V-statistic biased
estimator). No square root before comparing to DiGress paper Table 1
ratios or HiGen Table 1 raw values. See
`../context/mmd-units-and-protocol.md`.
