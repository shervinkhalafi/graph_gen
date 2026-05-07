# Pickup Doc — Profiling task (2026-05-01)

How to deploy, run, interpret, and iterate on the train+eval profile
pair built in commit `926e78ec`. Written so a fresh session can pick
up without re-discovering the wiring.

## Why this exists

Round 5 of the smallest-config sweep landed Greedy
(n_layers=4, dx=16, dim_ffX=16, dim_ffy=32) as winner. Greedy is only
**1.21× faster wall-clock** than Shervin's base despite being **18×
smaller in transformer params** — wall-clock is bottlenecked
elsewhere. The user explicitly cares about wall-clock, not compute. The
profile task exists to identify where the wall-clock minutes actually
go so round 6 attacks the right knob.

## What was built

Three files, all under commit `926e78ec`:

| Path | Role |
|------|------|
| `src/tmgg/profile/runners.py` | `run_train_profile()` + `run_eval_profile()` — Hydra-aware, profiler-wrapped entrypoints. |
| `src/tmgg/modal/_profile_functions.py` | Modal app `tmgg-profile`; two `@app.function`s on `GPU_CONFIGS["fast"]` (A100). |
| `scripts/profiling/launch_profile.py` | Parallel `.spawn()` launcher; appends audit rows to `rounds.jsonl`. |

Outputs land on the `tmgg-outputs` Modal volume at
`/outputs/profiles/<run_tag>/{train,eval}/{trace.json, summary.txt}`.

## Standard procedure

```bash
# 1. Deploy the Modal app (one-time; redeploy after editing code).
doppler run -- uv run modal deploy -m tmgg.modal._profile_functions

# 2. Spawn both profiles in parallel.
doppler run -- uv run python -m scripts.profiling.launch_profile

# 3. Wait — both functions are short (train ~3-5min, eval ~10-15min).
#    Watch progress on https://modal.com/apps or via `modal app logs tmgg-profile`.

# 4. Pull artefacts back. <run_tag> is printed by step 2 (default = greedy_<UTC>).
modal volume get tmgg-outputs profiles/<run_tag>/ ./local_profiles

# 5. Read the wall-time tables.
cat local_profiles/<run_tag>/train/summary.txt   # train inner loop hotspots
cat local_profiles/<run_tag>/eval/summary.txt    # eval cycle hotspots

# 6. Open trace.json in chrome://tracing/ or perfetto.dev for flame-graph view.
```

## Reading the summary tables

Each `summary.txt` is a `key_averages().table()` printout sorted by
CUDA time. Top three rows usually account for >70% of wall time. Look
for:

- **Train profile dominators** likely to find: `aten::matmul`,
  `aten::scaled_dot_product_attention`, edge-feature update ops,
  `optimizer.step`. If `dataloader::*` or `cudaStreamSynchronize` is
  high, the bottleneck is data feeding, not compute.
- **Eval profile dominators** likely to find: the diffusion sample
  loop's repeated `forward` calls (T=500 timesteps × 32 graphs =
  16,000 forwards), MMD computation (orca subprocess shows as a CPU
  bubble), graph rendering for visualizations.

Anything that surprises is interesting. The "obvious" answer is "edge
attention dominates wall-clock" — confirm or refute from the table.

## What to do with the findings

Map findings → round-6 plan:

| Hotspot | Round-6 cut |
|---------|-------------|
| Edge attention / per-edge ops | `de=8→4`, `dim_ffE=8→4` (edge-side narrowing) |
| Eval forward-pass loop | Reduce diffusion T (default 500 → 250 or 100); requires `noise_process` config edit |
| MMD / orca subprocess | Skip orbit_mmd on every eval (compute every Nth instead); orca is slow |
| DataLoader stalls | Bump `num_workers`, `persistent_workers=True`, `pin_memory=True` in datamodule |
| `cudaStreamSynchronize` bubbles | Enable `torch.compile()` on the model — currently NOT used (verified; nothing in `tmgg.training` calls `torch.compile`) |
| Optimizer step | Bigger batch size (under-utilization); halves optimizer steps for same data |

## Iterating — non-default configurations

```bash
# Profile a different config (round-6 candidate, e.g. de=4, dim_ffE=4):
doppler run -- uv run python -m scripts.profiling.launch_profile \
  --run-tag greedy_edge_cut \
  --overrides "models/discrete@model=discrete_sbm_official,+data=spectre_sbm,trainer.precision=bf16-mixed,model.model.n_layers=4,model.model.hidden_dims.dx=16,model.model.hidden_dims.de=4,model.model.hidden_dims.dy=8,model.model.hidden_dims.n_head=8,model.model.hidden_dims.dim_ffX=16,model.model.hidden_dims.dim_ffE=4,model.model.hidden_dims.dim_ffy=32,+model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures,+model.model.extra_features.extra_features_type=all,+model.model.extra_features.max_n_nodes=200"

# Profile only training (skip eval, faster turnaround):
doppler run -- uv run python -m scripts.profiling.launch_profile --no-eval --run-tag train_only

# Profile only eval against a different checkpoint:
doppler run -- uv run python -m scripts.profiling.launch_profile --no-train \
  --checkpoint-path /outputs/<run_id>/checkpoints/last.ckpt \
  --run-tag eval_specific_ckpt

# Smaller eval profile (cheaper but noisier):
doppler run -- uv run python -m scripts.profiling.launch_profile --num-eval-samples 8

# Longer train profile to capture slower-developing regimes:
doppler run -- uv run python -m scripts.profiling.launch_profile \
  --num-train-steps 500 --warmup-steps 50 --active-steps 100
```

## Default-checkpoint caveat

`scripts/profiling/launch_profile.py:DEFAULT_GREEDY_CKPT` points at:
```
/outputs/discrete_diffusion_DiffusionModule_dSpectreSBMDataModule_lr2e-4_wd1e-12_L4_s0_fresh_20260501T111030/checkpoints/last.ckpt
```

That's the round-5 Greedy pod. If the file doesn't exist on the
volume (e.g. the `last.ckpt` was overwritten or the run dir got cleaned
up), the eval profile will hard-error. Verify with:

```bash
modal volume ls tmgg-outputs discrete_diffusion_DiffusionModule_dSpectreSBMDataModule_lr2e-4_wd1e-12_L4_s0_fresh_20260501T111030/checkpoints/
```

If absent, list available recent ckpts and pass `--checkpoint-path`
explicitly.

## Audit trail

Every spawn appends a row to
`docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl` with
`kind=profile_train_launched` or `kind=profile_eval_launched`. Useful
for grouping profile runs with sweep rounds in retrospect.

## Known issues / gotchas

1. **Modal SDK type stubs are missing.** `_profile_functions.py`
   uses `modal.App`, `modal.Secret`, etc. — basedpyright flags these
   as "not a known attribute" but they work at runtime. Same situation
   as `_eval_all_functions.py`. Don't be surprised by the noise.
2. **`get_volume_mounts()` returns `dict[str, Any]`.** Modal's stricter
   `volumes=` typing wants `dict[str | PurePosixPath, Volume |
   CloudBucketMount]`. Existing pattern is to ignore the impedance;
   widening the helper's return type would cascade through every
   `_functions.py`. Out of scope for the profiling task.
3. **Profile output directory must live under `/outputs/`** so it
   persists on the Modal volume after the function returns. The
   default routing handles this; only matters if you bypass the
   launcher.
4. **Lightning's `PyTorchProfiler` writes its own filename.** Default
   is `train.pt.trace.json` (not `trace.json`). Both are present in
   the output directory. The launcher's `summary.txt` extraction
   handles the naming.

## What's NOT done (potential follow-ups)

- **No host-CPU profile mode.** Both runners assume `device="cuda"`.
  If you need a quick local sanity check without Modal, you'd need to
  flip the device flag and remove the GPU dependency.
- **No `with_stack=True` (Python frame attribution).** The current
  profile sees framework op names but not the user-code frame. If a
  hotspot's function is ambiguous (e.g. multiple call sites of
  `aten::matmul`), re-run with `with_stack=True` set in the runner.
  Costs trace size and slowdown.
- **No FlopCounter integration.** torch.profiler emits times, not
  FLOPs. If you want per-op FLOP counts for compute vs wall-clock
  comparison, add `flops` extra to the profiler config.
- **No automated trace-summary diff** between two profile runs (e.g.
  Greedy vs Shervin base). Manual `diff` of the two `summary.txt`
  files works for now.

## Sweep state at handoff

- Round 5 closed; Greedy resolved as winner (commit `38025505`).
- No active Modal training pods (verified: `watch_runs` reports 0
  running launches).
- Future-work menu in `docs/experiments/sweep/smallest-config-2026-04-29/progress.md`
  → "Future-work suggestions (post round 5)" section.
- Round 6 NOT yet planned. The profile findings should inform which
  item from the future-work menu to attack first.
