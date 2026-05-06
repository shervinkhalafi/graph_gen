# `discrete_sbm_pearl_gnnconv_raw_repro` / `g1g6xpx1` ⚠ blew up — killed

**Launched:** 2026-05-05 18:56 UTC
**Ended:** 2026-05-06 09:36 UTC (killed — runtime 14.67h, step 271049)
**Status:** failed (container stopped via `modal container stop` 2026-05-06 ≈09:35 UTC; final W&B state `failed`)

## Identity

| field | value |
|-------|-------|
| config | `discrete_sbm_pearl_gnnconv_raw_repro` |
| wandb_project | `discrete-sbm-pearl-gnnconv-raw-repro` |
| run_id | `g1g6xpx1` |
| volume_path | `tmgg-outputs:/data/outputs/discrete_sbm_pearl_gnnconv_raw_repro/g1g6xpx1/` |
| gpu_tier | `fast` |
| W&B URL | <https://wandb.ai/graph_denoise_team/discrete-sbm-pearl-gnnconv-raw-repro/runs/g1g6xpx1> |

## Fetched

- **status:** no — live W&B summary pulled 2026-05-06 07:25 UTC for the diagnostics below.
- **local_path:** —

## Diagnostics

> Live W&B summary, runtime 44533s.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² | **0.4329** | far worse than panel norm (~0.184) |
| clustering MMD² | 0.1400 | |
| orbit MMD² | 0.1419 | |
| spectral MMD² | 0.2698 | |
| MMD ratios | _pending — irrelevant for divergent runs_ | |
| train_loss_epoch | 1.072 | |
| val_NLL | 6577.79 | second-worst on panel after `qao36vwu` |
| mean_step_kl | 0.00892 | |
| grad_norm_total | **2.16e+16** | ⚠ |
| effective_lr | **2.48e+9** | ⚠ optimizer state corrupted |
| epoch | 22811 | |
| global_step | 228119 | |
| step_time_s | 0.208 | |

**Health:** ✗ blew up — same divergence signature as the first run `qao36vwu` of this variant.

## Anchor comparison

> Computed against the cached train↔test baseline at `data/eval/mmd_baselines/spectre_sbm.json` (see [`docs/eval/mmd-units-and-protocol.md`](../../docs/eval/mmd-units-and-protocol.md) for unit semantics). All MMD values are V-statistic squared MMD (GraphRNN/GRAN convention).

- `r_run` = MMD²(gen-val, this run) / MMD²(train, test).
- `r_higen` = HiGen Table 1's reported MMD² for DiGress on this dataset, divided by our baseline. Expresses HiGen's reproduction in our pipeline's ratio units.
- `r_paper` = DiGress paper Table 1 ratio, verbatim. Verbatim from arXiv:2209.14734v3 Table 1 SBM row.

| metric | run mmd² | baseline mmd² | r_run | r_higen | r_paper |
|--------|---------:|--------------:|------:|--------:|--------:|
| degree     | 0.4329 | 3.4125e-04 | 1268.56 | 3.81 | 1.60 |
| clustering | 0.1400 | 3.3118e-02 | 4.23 | 1.50 | 1.50 |
| orbit      | 0.1419 | 3.0991e-02 | 4.58 | 1.40 | 1.70 |
| spectral   | 0.2698 | 2.8182e-03 | 95.74 | n/a | n/a |

**Caveat:** run-side MMD is gen↔val; baseline is train↔test. [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md) Task 3 §2 flags this. Effect likely O(1) given i.i.d. splits, but a train↔val baseline would tighten the comparison.

## Visuals

- none — would not be diagnostic on a divergent run.

## Notes

Second attempt at the `_raw_` SBM variant. First attempt `qao36vwu` (2026-05-04 22:10 → 2026-05-05 18:51 UTC) also blew up; this relaunch reproduces the same failure mode and on a worse trajectory (degree MMD 0.43 vs 0.30 on the first run).

**Killed via `modal container stop` 2026-05-06 09:36 UTC** to free Modal capacity. Modal then reassigned the underlying function call's input to a new container, spawning a chain of further reassigns: `uuifd9v3` (failed in 4 min) → `bepjqwqz` (failed in 3 min) → `g6y8ubfg` (crashed in 7 min). Eventually the call was cancelled at the function-call level (via the Modal web UI) and the chain terminated. Together with the first run `qao36vwu`, this is the second confirmed numerical instability of the `BareGraphConv` Q/K/V variant on raw (un-normalised) adjacency. The variant is unviable as currently configured.

**Together with the enzymes-side `dt0ux9zh`** — which also shows huge `grad_norm_total` (7600) and out-of-band `effective_lr` (1.10e-3) on the same `_raw_` config — the variant is unviable as configured. The `_norm_` sister variant (normalised adjacency) trains cleanly on both datasets; the issue is specifically the un-normalised Q/K/V projection.

**Action:** kill, fix (normalisation + gradient clipping + lower lr), or document-as-known-broken. See `runlog.md` "Backfill checklist" item 8.
