# DiGress Panel Modal Logging Persistence Audit

Date: 2026-04-15
Context: Go/no-go check before the first `run-digress-arch-panel-modal.zsh` launch. Each panel run spawns a detached Modal container that outlives the local shell; any channel that writes only to ephemeral container storage is lost the moment the function returns (Modal destroys the filesystem on exit, and `modal app logs` streams live only — there is no historical CLI retrieval).

## 1. Volume mount layout

`src/tmgg/modal/_lib/volumes.py` declares two persistent volumes attached to every training function:

| Mount path | Volume name | Created lazily? |
|---|---|---|
| `/data/datasets` | `tmgg-datasets` | yes (`create_if_missing=True`) |
| `/data/outputs`  | `tmgg-outputs`  | yes |

Both are Modal `Volume` objects (not `NetworkFileSystem`), attached via `volumes=get_volume_mounts()` on `modal_run_cli`, `modal_run_cli_fast`, `modal_run_cli_debug`, and on the evaluation / checkpoint-listing functions. Modal `Volume` objects persist indefinitely — there is no TTL or retention policy set in code; the data stays until the volume is manually deleted via the Modal CLI / dashboard.

## 2. `${paths.output_dir}` resolution on Modal

`src/tmgg/experiments/exp_configs/_base_infra.yaml` (line 74) sets `paths.output_dir: ${hydra:runtime.output_dir}`. That interpolation is only meaningful inside a `@hydra.main` decorated entry point.

On Modal, `tmgg-modal run` composes the config via `resolve_config()` in `src/tmgg/modal/_lib/config_resolution.py` (lines 153–162), which overrides the interpolation before serializing the YAML:

```python
experiment_name = cfg.get("experiment_name", "tmgg_training")
output_dir = f"{OUTPUTS_MOUNT}/{experiment_name}/{run_id}"
cfg.paths.output_dir = output_dir
cfg.paths.results_dir = f"{output_dir}/results"
```

Since `OUTPUTS_MOUNT = "/data/outputs"`, every Modal run gets `paths.output_dir = /data/outputs/discrete_diffusion/<run_id>` — inside the persistent `tmgg-outputs` volume. Anything the Lightning loggers, callbacks, or app code writes under `${paths.output_dir}` survives the container exit.

## 3. CSV logger persistence

`src/tmgg/experiments/exp_configs/base/logger/discrete_wandb.yaml` configures `csv.save_dir = ${paths.output_dir}/csv`. Given the resolution above, this resolves to `/data/outputs/discrete_diffusion/<run_id>/csv/<experiment_name>/metrics.csv` inside the `tmgg-outputs` volume. Persists across container exit, retrievable via `modal volume get tmgg-outputs ...` after the run completes.

## 4. Text logger (stdout/stderr)

`src/tmgg/modal/_functions.py` (`_run_cli_impl`, lines ~341–365) spawns the experiment CLI via `subprocess.Popen` with `stdout=PIPE, stderr=STDOUT`, then iterates the pipe with `print(line, end="")` and an `output_lines` in-memory list. The accumulated `combined_output` is **only** used for W&B URL extraction and for the error-tail in the final exception / confirmation record.

There is no `tee`, no `logging.FileHandler`, no `sys.stdout` redirect to a file on `/data/outputs`. The per-line `print()` surfaces the subprocess text to Modal's log stream, which `modal app logs tmgg-spectral` can tail live — but Modal does not persist history retrievable after the container scales down.

The `append_confirmation()` sink at `/data/outputs/confirmation.jsonl` captures only lifecycle events (started, completed, failed, with the final 500–1200 chars of output as `error`), not the full stdout stream.

Net effect: **stdout is ephemeral** under the current layout. For a detached panel run, the only way to keep stdout is to either (a) attach `modal app logs tmgg-spectral 2>&1 | tee local.log` in parallel to the launch, or (b) patch `_run_cli_impl` to tee into `${paths.output_dir}/stdout.log`.

## 5. Validation figures

`src/tmgg/training/lightning_modules/diffusion_module.py` lines 804–814 routes visualizations through `tmgg.training.logging.log_figures`. `log_figure()` (lines 190–212) branches on logger type:

- `WandbLogger` → `logger.experiment.log({tag: wandb.Image(figure)})` — persists on wandb.ai servers.
- `CSVLogger` → saves PNG to `Path(logger.log_dir) / "figures" / <tag>_<global_step>.png`, i.e. under `${paths.output_dir}/csv/...`, which is on the `tmgg-outputs` volume.

Both paths persist. wandb retrieval via the project page; local copies via `modal volume get`.

## 6. Checkpoints

The DiGress-panel base config (`base_config_discrete_diffusion_generative.yaml`) pulls `base/callbacks: discrete_nll` via `override /base/callbacks: discrete_nll`. That callback (`src/tmgg/experiments/exp_configs/base/callbacks/discrete_nll.yaml`) declares a Lightning `ModelCheckpoint` with `save_top_k: 3, save_last: true`, monitor `val/epoch_NLL`, filename template `model-step={step:06d}-val_nll={val/epoch_NLL:.4f}`.

The trainer config `base/trainer/default.yaml` pins `default_root_dir: ${paths.output_dir}` and `enable_checkpointing: true`, and `src/tmgg/training/orchestration/run_experiment.py` constructs `ModelCheckpoint(dirpath=Path(config.paths.output_dir) / "checkpoints", ...)`. The panel launcher does **not** override any of these. Checkpoints land at `/data/outputs/discrete_diffusion/<run_id>/checkpoints/*.ckpt` on the `tmgg-outputs` volume.

## Persistence summary table

| Channel | Where it lands on Modal | Persists across container exit? | Retrievable after run completes? | Knob to fix (if needed) |
|---|---|---|---|---|
| wandb metrics | wandb.ai project `digress-arch-panel` | yes (wandb servers) | yes, via wandb UI / API | — |
| csv logger | `/data/outputs/discrete_diffusion/<run_id>/csv/...` on `tmgg-outputs` volume | yes | yes, via `modal volume get tmgg-outputs` | — |
| stdout text | Modal live log stream only (print→PIPE, no disk tee) | **no** | **no** (Modal CLI streams live, no history) | Attach `modal app logs tmgg-spectral | tee` in parallel to launch, OR patch `_run_cli_impl` to tee into `${paths.output_dir}/stdout.log` |
| validation figures | wandb (`wandb.Image`) AND `/data/outputs/.../csv/<experiment>/figures/*.png` | yes (both sinks) | yes (wandb + volume) | — |
| checkpoints | `/data/outputs/discrete_diffusion/<run_id>/checkpoints/*.ckpt` on `tmgg-outputs` volume | yes (`ModelCheckpoint` from `discrete_nll` callback; `enable_checkpointing: true` from trainer default, not overridden) | yes, via `modal volume get` | — |

## Verdict

READY TO LAUNCH.

Every channel except raw stdout persists to `tmgg-outputs` or to wandb. All loss curves, metrics, generated-graph visualizations, and checkpoints needed to compare the five architectures survive container exit and are retrievable without SSH into Modal infrastructure. Stdout is ephemeral, but the panel launcher is detached and the training subprocess surfaces all failures through the confirmation log's `error` tail plus the last-N-lines of `combined_output` captured at exit time — enough to diagnose crashes without a full disk tee.

Operator note: if a panel run crashes, the only way to retrieve full stdout is `modal app logs tmgg-spectral | tee local.log` attached in parallel (immediately after the launch, since Modal does not retain history). Attach it defensively on the first panel run to validate nothing surprising shows up in the console stream; subsequent panels can rely on wandb + csv + confirmation log alone.
