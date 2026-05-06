# Run Log

Single source of truth for launched Modal training runs. Per-run detail
sits in `run_details/<launch-date>/<config>_<run_id>_details.md`. This
file is the index plus cross-cutting analysis.

## How to use

1. **On launch:** create `run_details/<YYYY-MM-DD>/<config>_<run_id>_details.md` (date is the launch date, UTC), fill the identity / timeline / fetched / diagnostics blocks per the format below, and add a row to the relevant panel section here in `runlog.md` linking the file.
2. **On state change:** edit the per-run detail file (status, ended_at, notes); update the corresponding row in this file's *quick status table* if the state flips status / step / health.
3. **On data fetch:** edit the detail file's `Fetched` block (status / local_path).
4. **On scoring:** fill the `Diagnostics` block in the detail file with structured numbers + interpretation, plus link any visualisations under `Visuals`.

The runlog is the index — figure files, summary parquets, and detail
markdown each get linked from a per-run row, never embedded.

## Field reference (per-run detail file)

| Field | Meaning |
|-------|---------|
| `config` | Hydra `+experiment=` value, e.g. `discrete_sbm_pearl_spectral_repro`. |
| `wandb_project` | Project string from the YAML (`wandb_project:`). One project per variant by current convention. |
| `run_id` | Modal/W&B run ID. The trainer writes to `/data/outputs/{experiment_name}/{run_id}/` on the `tmgg-outputs` volume — that path is the canonical artefact location. |
| `volume_path` | `tmgg-outputs:/data/outputs/{experiment_name}/{run_id}/`. Always derivable from `config` + `run_id`; record explicitly so the entry stays self-contained. |
| `launched_at` | UTC timestamp at the moment `tmgg-modal run` was issued. Format: `YYYY-MM-DD HH:MM UTC`. |
| `ended_at` | UTC timestamp when the run finished, crashed, or was killed. Empty while still running. |
| `status` | `running` / `finished` / `crashed` / `failed` / `oom` / `preempted` / `killed`. (W&B `crashed` typically = container died mid-step; `failed` = nonzero exit from training process.) |
| `gpu_tier` | `fast` / `cheap` / explicit GPU type. Default `fast` (A100) per `scripts/run-digress-repro-modal.zsh`. |
| `fetched` | `no` / `partial` / `yes`. `partial` = parquet/summary only, no checkpoints. |
| `local_path` | Host-side directory holding fetched artefacts. Empty if `fetched=no`. |
| `Diagnostics` | Structured sub-block — see "Diagnostics block format" below. |
| `Visuals` | List of plot/sample paths plus a one-line interpretation each. |
| `Notes` | Crash hypothesis, mid-run config tweak, lineage links, anything load-bearing for future-you. |

### Diagnostics block format

The `Diagnostics` block in each detail file is a table, not prose. Always
note the snapshot source (which `summary.json` / parquet / W&B query the
numbers came from) and the snapshot step/runtime so future readers can
tell stale data from fresh data.

```
> Snapshot from <source path or W&B query>, runtime <s>.

| metric | value | comment |
|--------|------:|---------|
| degree MMD² (gen-val) | … |
| clustering MMD² | … |
| orbit MMD² | … |
| spectral MMD² | … |
| MMD ratios | … or "pending — needs data/eval/mmd_baselines/<dataset>.json" |
| train_loss_epoch | … |
| val_NLL | … |
| mean_step_kl | … |
| grad_norm_total | … |
| effective_lr | … |
| epoch | … |
| global_step | … |
| step_time_s | … |

**Health:** ✓ stable / ⚠ flag / ✗ blew up — one-line justification.
```

**Field source map** (W&B summary key → diagnostics field):

| Diagnostics field | W&B summary key |
|-------------------|-----------------|
| MMDs              | `gen-val/{degree,clustering,orbit,spectral}_mmd` |
| Loss              | `train/loss_epoch`, `val/epoch_NLL`, `gen-val/mean_step_kl` |
| Gradient health   | `diagnostics-train/opt-health/grad_norm_total`, `.../effective_lr`, `.../grad_cosine/<layer>`, `.../grad_snr/<layer>` |
| Step counts       | `epoch`, `trainer/global_step` |
| Throughput        | `impl-perf/train/step_time_s` |

**Caveat on units:** `gen-val/*_mmd` values are **squared MMD** values
(V-statistic biased estimator, per `mmd_metrics.py:compute_mmd`),
*not* square-root MMD distances. The "MMD" in our W&B keys, parquet
columns, and field names follows the GraphRNN/GRAN/DiGress/HiGen
convention of dropping the squared label, but every number is MMD².
**Don't sqrt** when comparing to HiGen Table 1 (also MMD²). **Don't
double-square** when comparing to DiGress Table 1 (ratios of MMD²).
Full detail and rationale: `docs/eval/mmd-units-and-protocol.md`.

## Volume / fetch conventions

- Modal output volume: `tmgg-outputs`, mounted at `/data/outputs` inside containers (see `src/tmgg/modal/_lib/volumes.py`).
- Per-run layout: `/data/outputs/{experiment_name}/{run_id}/` containing `checkpoints/`, `eval_manifest.jsonl`, generated samples, etc.
- Pull a run locally with `modal volume get tmgg-outputs /data/outputs/{experiment_name}/{run_id} ./data/runs/{experiment_name}/{run_id}`.
- W&B exports of *summary* data live in `wandb_export/`; checkpoints stay on the Modal volume unless explicitly pulled.

### Visualisation conventions

Plots / generated-sample renders / loss curves should live next to the
data they came from, never as one-off files in the repo root:

- **Per-run figures from local checkpoint pulls:** `data/runs/<exp>/<run_id>/figures/<name>.png`. Mirror the volume layout so the figure sits beside the checkpoint and `eval_manifest.jsonl` it summarises.
- **Per-panel / per-report figures (cross-run comparisons):** `wandb_export/<analysis-slug>/figures/<name>.png`, with a sibling `report.typ` or `report.md` that interprets them. Example precedent: `wandb_export/sbm-repro-report-2026-05-05/figures/`.
- **Whatever you produce, link it from the detail file's `Visuals` line and add a one-sentence interpretation.** The detail file is the index — the figure file alone does not document its own meaning.

Generated-sample image grids are bulky; check whether they belong in
git or whether `data/` (gitignored) is the right home. By default they
go to `data/runs/<exp>/<run>/samples/`; copy into `wandb_export/` only
when included in a published report.

## Backfill query

The W&B inventory below was pulled 2026-05-06 07:12–07:25 UTC via
`/tmp/query_repro_runs_json.py` and `/tmp/query_specific_runs.py`
against the 11 per-variant projects under `graph_denoise_team`. Re-run
those scripts (or their source-controlled successor — see backfill
checklist) to refresh.

---

## SBM repro panel (DiGress on Spectre SBM)

Six variants from `scripts/run-digress-repro-modal.zsh` keys
`sbm`, `sbm-vignac-spectral`, `sbm-pearl`, `sbm-pearl-spectral`,
`sbm-pearl-gnnconv-norm`, `sbm-pearl-gnnconv-raw`. Axis A = extra
features (Vignac all / R-PEARL); axis B = Q/K/V projection (Linear /
SpectralProjectionLayer / BareGraphConv-norm / BareGraphConv-raw).

Cross-cutting note: every "headline" SBM run that crashed at
runtime ≈ 86200–86260s (≈ 24h) hit step 370k–515k before death. That
runtime is suspiciously close to the Modal 24h container timeout —
these look like timeouts rather than mid-training crashes.

### `discrete_sbm_vignac_repro`

- [`12s2b4a7`](run_details/2026-05-04/discrete_sbm_vignac_repro_12s2b4a7_details.md) — launched 2026-05-04 16:23 UTC, crashed (≈24h timeout), ✓ stable. Anchor for the panel.

### `discrete_sbm_vignac_repro_exact` (paper-anchor parity)

- [`lptjvfbe`](run_details/2026-05-06/discrete_sbm_vignac_repro_exact_lptjvfbe_details.md) — launched 2026-05-06 09:39 UTC, **cancelled 2026-05-06 at step ~15k**. Killed because `_GraphTransformer.forward` calls `.mask_zero_diag()` on the post-`mlp_in_E` hidden state (zeroes hidden E diagonal) where upstream DiGress calls `.mask(node_mask)` (padding-only, diagonal preserved until output mask) — outputs not byte-equivalent to upstream even with identical weights (state-dict diff `X=0.0087, E=0.0010` → 0 under monkey-patch). Re-launch after `transformer_model.py:834` fix. See [`docs/eval/2026-05-06-mmd-ratio-analysis.md`](docs/eval/2026-05-06-mmd-ratio-analysis.md) "GDPO reference" for prior context.
- [`2026-05-06-sbm-vignac-1`](run_details/2026-05-06/discrete_sbm_vignac_repro_exact_2026-05-06-sbm-vignac-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC, running with `force_fresh=false` (preempt-resume via wandb-id sidecar). `fc-01KQYRNJ9S0STFDZBEPFBY5WZQ`.

### `discrete_sbm_vignac_spectral_repro`

- [`e5pd9drt`](run_details/2026-05-05/discrete_sbm_vignac_spectral_repro_e5pd9drt_details.md) — launched 2026-05-05 21:22 UTC, **killed 2026-05-06 ~13:30 UTC** (mask-bug invalidation). No post-fix successor in the 2026-05-06 relaunch batch (architectural variant deferred pending budget).

### `discrete_sbm_pearl_repro`

- [`s07qwx3b`](run_details/2026-05-04/discrete_sbm_pearl_repro_s07qwx3b_details.md) — launched 2026-05-04 21:38 UTC, crashed (≈24h timeout), ✓ stable. Cleanest improvement signal: orbit MMD ~0.10 vs vignac ~0.14.
- [`2026-05-06-sbm-pearl-1`](run_details/2026-05-06/discrete_sbm_pearl_repro_exact_2026-05-06-sbm-pearl-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_sbm_pearl_repro_exact`, running, preempt-resume enabled. `fc-01KQYRNSEK23MRX1ATF9T3BMYZ`.

### `discrete_sbm_pearl_spectral_repro`

- [`jbraoj7o`](run_details/2026-05-04/discrete_sbm_pearl_spectral_repro_jbraoj7o_details.md) — launched 2026-05-04 21:52 UTC, crashed (≈24h timeout), ✓ stable. Within noise of plain pearl.
- [`2026-05-06-sbm-pearl-spectral-1`](run_details/2026-05-06/discrete_sbm_pearl_spectral_repro_exact_2026-05-06-sbm-pearl-spectral-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_sbm_pearl_spectral_repro_exact`, running, preempt-resume enabled. `fc-01KQYRNZRZ0MD33VYHM6TDCB7T`.

### `discrete_sbm_pearl_gnnconv_norm_repro`

- [`rarihsee`](run_details/2026-05-04/discrete_sbm_pearl_gnnconv_norm_repro_rarihsee_details.md) — launched 2026-05-04 22:09 UTC, crashed (≈24h timeout), ⚠ elevated grad_norm (1.49 vs ~0.08 elsewhere) but converged.
- [`2026-05-06-sbm-pearl-gnnconv-1`](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_norm_repro_exact_2026-05-06-sbm-pearl-gnnconv-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_sbm_pearl_gnnconv_norm_repro_exact`, running, preempt-resume enabled. `fc-01KQYRP5Z0SB93YEFW4Y6JSMZE`.

### `discrete_sbm_pearl_gnnconv_raw_repro`

Variant **terminally killed 2026-05-06 09:35–09:59 UTC** after numerical instability across two real runs and three short auto-reassigns. Function call cancelled via Modal web UI to break the reassign chain.

- [`qao36vwu`](run_details/2026-05-04/discrete_sbm_pearl_gnnconv_raw_repro_qao36vwu_details.md) — launched 2026-05-04 22:10 UTC, crashed at 21h, ✗ blew up (grad_norm 2.4e18, lr Infinity).
- [`g1g6xpx1`](run_details/2026-05-05/discrete_sbm_pearl_gnnconv_raw_repro_g1g6xpx1_details.md) — launched 2026-05-05 18:56 UTC, ran 14.7h until killed 2026-05-06 09:36 UTC. Same blow-up signature.
- [`uuifd9v3`](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_raw_repro_uuifd9v3_details.md), [`bepjqwqz`](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_raw_repro_bepjqwqz_details.md), [`g6y8ubfg`](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_raw_repro_g6y8ubfg_details.md) — Modal auto-reassign chain after the kill, each <10 min. Chain ended via call cancellation 2026-05-06 09:59 UTC.

---

## Enzymes repro panel (DiGress on PyG ENZYMES)

Five variants — same axes as SBM but no Vignac+spectral combination yet
(no `discrete_enzymes_vignac_spectral_repro`). All five launched within
a 5-minute window on 2026-05-05 (≈ 15:37–15:42 UTC).

### `discrete_enzymes_vignac_repro`

- [`l1nk0622`](run_details/2026-05-05/discrete_enzymes_vignac_repro_l1nk0622_details.md) — launched 2026-05-05 15:42 UTC, **finished at 525k step (550k target)**, ✓ stable. Degree MMD² ≈ 0.187 vs HiGen-reported DiGress 0.004 → ~10⁴× anchor gap. Mask-bug invalidated; not a usable reference.
- [`2026-05-06-enzymes-vignac-1`](run_details/2026-05-06/discrete_enzymes_vignac_repro_exact_2026-05-06-enzymes-vignac-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_enzymes_vignac_repro_exact`, running, preempt-resume enabled. `fc-01KQYRPC8G81QSZ9QQ8E05RCPC`.

### `discrete_enzymes_pearl_repro`

- [`ge461v1o`](run_details/2026-05-05/discrete_enzymes_pearl_repro_ge461v1o_details.md) — launched 2026-05-05 15:42 UTC, ran 18.3h to step 398k. Killed as collateral of the `_gnnconv_raw` cancellation 2026-05-06 ≈09:35 UTC; W&B state crashed once heartbeat timed out.
- [`bhqss75w`](run_details/2026-05-06/discrete_enzymes_pearl_repro_bhqss75w_details.md) — Modal auto-reassign 2026-05-06 10:01 UTC. `force_fresh=True` (launcher default) discarded `ge461v1o`'s checkpoint and started from step 0; user killed after 8 min.
- [`vejeny0f`](run_details/2026-05-06/discrete_enzymes_pearl_repro_vejeny0f_details.md) — manual relaunch with `force_fresh=false +run_id=…fresh_20260505T154023` resumed from `ge461v1o`'s `last.ckpt` at step ~395k. **Killed 2026-05-06 ~13:30 UTC** as part of mask-bug cleanup.
- [`2026-05-06-enzymes-pearl-1`](run_details/2026-05-06/discrete_enzymes_pearl_repro_exact_2026-05-06-enzymes-pearl-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_enzymes_pearl_repro_exact`, running, preempt-resume enabled. `fc-01KQYRPK5RP70YGEZ795ASDM7K`.

### `discrete_enzymes_pearl_spectral_repro`

- [`4n28svrj`](run_details/2026-05-05/discrete_enzymes_pearl_spectral_repro_4n28svrj_details.md) — launched 2026-05-05 15:37 UTC, **killed 2026-05-06 ~13:30 UTC** (mask-bug invalidation). Best orbit MMD on (buggy) enzymes panel: 0.097.
- [`2026-05-06-enzymes-pearl-spectral-1`](run_details/2026-05-06/discrete_enzymes_pearl_spectral_repro_exact_2026-05-06-enzymes-pearl-spectral-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_enzymes_pearl_spectral_repro_exact`, running, preempt-resume enabled. `fc-01KQYRPSDQAY1P5BAW1Y1ABYFD`.

### `discrete_enzymes_pearl_gnnconv_norm_repro`

Three attempts pre-fix; the first two failed with healthy gradients at point of failure → cause is infra, not numerical.

- [`txfr1vms`](run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_txfr1vms_details.md) — launched 2026-05-05 15:42 UTC, failed at 4.4h. Diagnostics not yet pulled.
- [`ly0d6lyi`](run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_ly0d6lyi_details.md) — launched 2026-05-05 20:06 UTC, failed at 11h, ✓ stable at point of failure.
- [`zyawhwrx`](run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_zyawhwrx_details.md) — launched 2026-05-06 07:07 UTC, **killed 2026-05-06 ~13:30 UTC** (mask-bug invalidation; only 75k cycle, never matured).
- [`2026-05-06-enzymes-pearl-gnnconv-1`](run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_exact_2026-05-06-enzymes-pearl-gnnconv-1_details.md) — **post-fix re-launch** 2026-05-06 13:46 UTC under `discrete_enzymes_pearl_gnnconv_norm_repro_exact`, running, preempt-resume enabled. `fc-01KQYRPZKRW6PJHP1QDMKKVDAY`.

### `discrete_enzymes_pearl_gnnconv_raw_repro`

Variant **terminally killed 2026-05-06 09:35–09:44 UTC** after one numerically-divergent run and one short auto-reassign. Function call cancelled cleanly via Modal web UI; chain ended with no further reassigns.

- [`dt0ux9zh`](run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_raw_repro_dt0ux9zh_details.md) — launched 2026-05-05 15:42 UTC, ran 17.9h until killed 2026-05-06 09:36 UTC. ✗ blew up (grad_norm 7600, lr 1.1e-3, orbit MMD 0.54).
- [`b7lqqac8`](run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_raw_repro_b7lqqac8_details.md) — Modal auto-reassign 2026-05-06 09:38 UTC, failed in 6 min (chain end).

---

## Quick status table

Snapshot 2026-05-06 20:13 UTC. `degree_mmd` is the headline metric
(raw MMD², gen-val); see per-run detail files for full MMD tuples plus
loss / gradient / throughput. **Stable** flags whether gradient norm
and effective_lr are in the expected range (healthy ≈ grad_norm < 5,
lr ~1e-7..1e-6).

| Config | Run ID | Launched (UTC) | Status | Step | degree MMD² | grad_norm | Stable? | Detail |
|--------|--------|----------------|:------:|-----:|------------:|----------:|:-------:|--------|
| `discrete_sbm_vignac_repro`               | `12s2b4a7` | 2026-05-04 16:23 | crashed (≈24h) | 430k | 0.185 | 0.071 | ✓ | [link](run_details/2026-05-04/discrete_sbm_vignac_repro_12s2b4a7_details.md) |
| **`discrete_sbm_vignac_repro_exact`**     | `lptjvfbe` | 2026-05-06 09:39 | cancelled (mask bug) | 15k  | _no eval logged_ | 0.036 | n/a — invalidated | [link](run_details/2026-05-06/discrete_sbm_vignac_repro_exact_lptjvfbe_details.md) |
| `discrete_sbm_vignac_spectral_repro`      | `e5pd9drt` | 2026-05-05 21:22 | killed (mask bug) | 91k  | 0.225 (step 75k) | 0.107 | n/a — invalidated | [link](run_details/2026-05-05/discrete_sbm_vignac_spectral_repro_e5pd9drt_details.md) |
| `discrete_sbm_pearl_repro`                | `s07qwx3b` | 2026-05-04 21:38 | crashed (≈24h) | 514k | 0.186 | 0.089 | ✓ | [link](run_details/2026-05-04/discrete_sbm_pearl_repro_s07qwx3b_details.md) |
| `discrete_sbm_pearl_spectral_repro`       | `jbraoj7o` | 2026-05-04 21:52 | crashed (≈24h) | 403k | 0.187 | 0.080 | ✓ | [link](run_details/2026-05-04/discrete_sbm_pearl_spectral_repro_jbraoj7o_details.md) |
| `discrete_sbm_pearl_gnnconv_norm_repro`   | `rarihsee` | 2026-05-04 22:09 | crashed (≈24h) | 443k | 0.184 | 1.49  | ⚠ (high norm) | [link](run_details/2026-05-04/discrete_sbm_pearl_gnnconv_norm_repro_rarihsee_details.md) |
| `discrete_sbm_pearl_gnnconv_raw_repro` (1) | `qao36vwu` | 2026-05-04 22:10 | crashed (~21h) | 376k | 0.299 | 2.4e18 | ✗ blew up | [link](run_details/2026-05-04/discrete_sbm_pearl_gnnconv_raw_repro_qao36vwu_details.md) |
| `discrete_sbm_pearl_gnnconv_raw_repro` (2) | `g1g6xpx1` | 2026-05-05 18:56 | failed (killed) | 271k | 0.433 | 2.2e16 | ✗ blew up | [link](run_details/2026-05-05/discrete_sbm_pearl_gnnconv_raw_repro_g1g6xpx1_details.md) |
| `discrete_sbm_pearl_gnnconv_raw_repro` (3, reassign) | `uuifd9v3` | 2026-05-06 09:40 | failed | — | — | — | ✗ chain | [link](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_raw_repro_uuifd9v3_details.md) |
| `discrete_sbm_pearl_gnnconv_raw_repro` (4, reassign) | `bepjqwqz` | 2026-05-06 09:46 | failed | 339 | — | — | ✗ chain | [link](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_raw_repro_bepjqwqz_details.md) |
| `discrete_sbm_pearl_gnnconv_raw_repro` (5, reassign) | `g6y8ubfg` | 2026-05-06 09:52 | crashed | 1.2k | — | — | ✗ chain end | [link](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_raw_repro_g6y8ubfg_details.md) |
| **`discrete_enzymes_vignac_repro`**       | `l1nk0622` | 2026-05-05 15:42 | finished (mask-bug invalidated) | 550k | 0.190 | 0.212 | n/a — invalidated | [link](run_details/2026-05-05/discrete_enzymes_vignac_repro_l1nk0622_details.md) |
| `discrete_enzymes_pearl_repro` (orig)     | `ge461v1o` | 2026-05-05 15:42 | crashed (collateral kill) | 398k | 0.186 | 0.189 | n/a — invalidated | [link](run_details/2026-05-05/discrete_enzymes_pearl_repro_ge461v1o_details.md) |
| `discrete_enzymes_pearl_repro` (force-fresh restart) | `bhqss75w` | 2026-05-06 10:01 | failed (killed; force_fresh wrong) | 3.6k | — | — | _killed by user_ | [link](run_details/2026-05-06/discrete_enzymes_pearl_repro_bhqss75w_details.md) |
| `discrete_enzymes_pearl_repro` (resumed)  | `vejeny0f` | 2026-05-06 10:17 | killed (mask bug) | 430k (resumed) | _no eval logged_ | 0.302 | n/a — invalidated | [link](run_details/2026-05-06/discrete_enzymes_pearl_repro_vejeny0f_details.md) |
| `discrete_enzymes_pearl_spectral_repro`   | `4n28svrj` | 2026-05-05 15:37 | killed (mask bug) | 462k | 0.188 | 0.150 | n/a — invalidated | [link](run_details/2026-05-05/discrete_enzymes_pearl_spectral_repro_4n28svrj_details.md) |
| `discrete_enzymes_pearl_gnnconv_norm_repro` (1) | `txfr1vms` | 2026-05-05 15:42 | failed | 103k | _not pulled_ | _—_ | _infra_ | [link](run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_txfr1vms_details.md) |
| `discrete_enzymes_pearl_gnnconv_norm_repro` (2) | `ly0d6lyi` | 2026-05-05 20:06 | failed | 251k | 0.184 | 0.294 | ✓ at fail (infra cause) | [link](run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_norm_repro_ly0d6lyi_details.md) |
| `discrete_enzymes_pearl_gnnconv_norm_repro` (3) | `zyawhwrx` | 2026-05-06 07:07 | killed (mask bug) | 96k  | 0.181 (step 75k) | 0.472 | n/a — invalidated | [link](run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_zyawhwrx_details.md) |
| `discrete_enzymes_pearl_gnnconv_raw_repro` | `dt0ux9zh` | 2026-05-05 15:42 | failed (killed) | 439k | 0.211 | 7600 | ✗ blew up | [link](run_details/2026-05-05/discrete_enzymes_pearl_gnnconv_raw_repro_dt0ux9zh_details.md) |
| `discrete_enzymes_pearl_gnnconv_raw_repro` (reassign) | `b7lqqac8` | 2026-05-06 09:38 | failed | 1.8k | — | — | ✗ chain end | [link](run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_raw_repro_b7lqqac8_details.md) |
| **`discrete_sbm_vignac_repro_exact`** (post-fix) | `2026-05-06-sbm-vignac-1` (`cgfv3f85`) | 2026-05-06 13:46 | running | 41.8k (1 cycle) | 0.187 | 0.132 | ✓ | [link](run_details/2026-05-06/discrete_sbm_vignac_repro_exact_2026-05-06-sbm-vignac-1_details.md) |
| **`discrete_sbm_pearl_repro_exact`** (post-fix) | `2026-05-06-sbm-pearl-1` (`k4iiw5sg`) | 2026-05-06 13:46 | running | 44.0k (1 cycle) | 0.181 | 0.200 | ✓ | [link](run_details/2026-05-06/discrete_sbm_pearl_repro_exact_2026-05-06-sbm-pearl-1_details.md) |
| **`discrete_sbm_pearl_spectral_repro_exact`** (post-fix) | `2026-05-06-sbm-pearl-spectral-1` (`qukgm6zu`) | 2026-05-06 13:46 | running | 41.2k (1 cycle) | 0.210 | 0.138 | ✓ | [link](run_details/2026-05-06/discrete_sbm_pearl_spectral_repro_exact_2026-05-06-sbm-pearl-spectral-1_details.md) |
| **`discrete_sbm_pearl_gnnconv_norm_repro_exact`** (post-fix) | `2026-05-06-sbm-pearl-gnnconv-1` (`5qchu8c4`) | 2026-05-06 13:46 | running | 44.0k (1 cycle) | 0.188 | 0.312 | ✓ | [link](run_details/2026-05-06/discrete_sbm_pearl_gnnconv_norm_repro_exact_2026-05-06-sbm-pearl-gnnconv-1_details.md) |
| **`discrete_enzymes_vignac_repro_exact`** (post-fix) | `2026-05-06-enzymes-vignac-1` (`8nhefhnl`) | 2026-05-06 13:46 | running | 178.7k (2 cycles) | 0.180 | 0.165 | ✓ | [link](run_details/2026-05-06/discrete_enzymes_vignac_repro_exact_2026-05-06-enzymes-vignac-1_details.md) |
| **`discrete_enzymes_pearl_repro_exact`** (post-fix) | `2026-05-06-enzymes-pearl-1` (`7yi627fv`) | 2026-05-06 13:46 | running | 134.0k (1 cycle) | 0.183 | 0.217 | ✓ | [link](run_details/2026-05-06/discrete_enzymes_pearl_repro_exact_2026-05-06-enzymes-pearl-1_details.md) |
| **`discrete_enzymes_pearl_spectral_repro_exact`** (post-fix) | `2026-05-06-enzymes-pearl-spectral-1` (`ths6e1da`) | 2026-05-06 13:46 | running | 113.6k (1 cycle) | 0.188 | 0.345 | ✓ | [link](run_details/2026-05-06/discrete_enzymes_pearl_spectral_repro_exact_2026-05-06-enzymes-pearl-spectral-1_details.md) |
| **`discrete_enzymes_pearl_gnnconv_norm_repro_exact`** (post-fix) | `2026-05-06-enzymes-pearl-gnnconv-1` (`xsmz6yql`) | 2026-05-06 13:46 | running | 159.2k (2 cycles) | 0.218 | 0.141 | ✓ | [link](run_details/2026-05-06/discrete_enzymes_pearl_gnnconv_norm_repro_exact_2026-05-06-enzymes-pearl-gnnconv-1_details.md) |

## Cross-cutting findings

> **Session note 2026-05-06: cancellation, reassign, and resume.** During the cleanup of the divergent `_gnnconv_raw` chains, three operationally important things happened that future-us should remember.
>
> 1. **`modal container stop` reassigns inputs, doesn't cancel them.** Stopping `g1g6xpx1`'s container spawned a chain `uuifd9v3 → bepjqwqz → g6y8ubfg`; only cancelling at the *function-call* level (Modal web UI → app → function → call → cancel) breaks the chain. The enzymes side stopped after one reassign (`b7lqqac8`) because the call was cancelled cleanly via the UI; the SBM side took three reassigns before the call was cancelled.
> 2. **Container stops can collateral-damage healthy runs that share a start-time bucket.** `ge461v1o`'s container was at the same 17:40 CEST timestamp as the divergent `dt0ux9zh`'s container; the user's stop signal hit both. Modal then auto-reassigned `ge461v1o`'s input to a fresh container — but with the project's default `force_fresh=True`, that reassign discarded the run's 18.3h of training and started from step 0 (run `bhqss75w`).
> 3. **Resume from checkpoint via `force_fresh=false` + explicit `+run_id`.** To rescue `ge461v1o`'s training we relaunched with `force_fresh=false +run_id=<original-fresh-suffix-name>`, which routed Lightning's `_find_last_checkpoint` to the existing `last.ckpt` and resumed at step ~395k (run `vejeny0f`). The mechanism is documented in `src/tmgg/training/orchestration/run_experiment.py:310,462–472`. The `+` prefix is required because `run_id` is not in the base config.
>
> **Session note 2026-05-06 ~20:13 UTC: post-fix 6h25m snapshot.** All
> 8 post-fix runs are healthy (no crashes, no preempts). Each has logged
> 1 eval cycle (SBM, trainer step ~22k) or 2 cycles (`8nhefhnl`,
> `xsmz6yql` on enzymes, trainer step 75k + 150k). Degree / clustering /
> orbit / spectral MMDs all sit in the same basin as the
> mask-bug-invalidated runs at comparable steps — fix did not move
> headline metrics. Implication: the diagonal-mask divergence was a
> correctness defect (state-dict non-equivalence to upstream), not the
> source of the published-anchor gap. Anchor-gap analysis stands.
> Best post-fix metrics so far: SBM orbit 0.0909 (`cgfv3f85`), ENZYMES
> clustering 0.0967 (`7yi627fv`). See [`docs/eval/2026-05-06-ablations_measurment.md`](docs/eval/2026-05-06-ablations_measurment.md) post-fix tables for full numbers.
>
> **Ratio comparison vs published anchors** — see [`docs/eval/2026-05-06-mmd-ratio-analysis.md`](docs/eval/2026-05-06-mmd-ratio-analysis.md). Headlines:
> - SBM clustering converges towards DiGress paper r=1.5 (HiGen reproduces this exactly; our healthy panel sits at 4.0); orbit similar, ~2× from paper.
> - SBM degree blows up to r≈540 vs paper r=1.6 — real undertraining (our run terminated at 78% of intended steps), the smallest-baseline / most-amplified metric.
> - ENZYMES clustering at r≈10 vs HiGen-implied 7.95 (1.3× off, plausibly converging). Degree and orbit hundreds-of-times worse than HiGen's reproduction. DiGress paper has no ENZYMES anchor.
> - The `_raw_` variant is unviable across both datasets — degree ratios 877–1269 on SBM, orbit 3107 on ENZYMES.
> - Caveat: our `vignac_repro` is **not byte-equivalent** to upstream DiGress (`dim_ffy=2048` vs 256, `amsgrad=true`, code is a tmgg port). See the analysis file's "Reproduction caveats" section.

1. **`gnnconv_raw_repro` is numerically unstable across datasets.** Both SBM runs (`qao36vwu`, `g1g6xpx1`) and the enzymes run (`dt0ux9zh`) show grad_norm in the 1e3–1e18 range and effective_lr at 1e-3 or higher (healthy is ~3e-7). The `_norm_` sister variant uses normalised adjacency and stays stable, so the issue is specifically the un-normalised Q/K/V projection. Either fix the normalisation, add gradient clipping, or drop the variant.
2. **`gnnconv_norm` SBM run has elevated grad_norm (1.49) but trains stably.** Worth checking the per-layer `grad_cosine` / `grad_snr` rows in the W&B history before declaring this variant fully clean.
3. **`vignac_spectral_repro` is ~3× slower per step than the Linear-Q/K/V variants** (0.56s vs 0.18s). Step counts at the same wall-clock differ by that factor; cross-variant comparison must control for step count, not wall-clock.
4. **PEARL benefits orbit MMD on SBM but not enzymes.** SBM: vignac=0.142, pearl=0.095 (improvement). Enzymes: vignac=0.119, pearl=0.198 (regression). Spectral attention recovers the orbit gain on enzymes (0.097). Hypothesis: PEARL's orbit signal is dataset-specific.

## Open questions

1. **24h crash signature.** All five top-of-panel SBM runs and the first `gnnconv-raw` run terminated within minutes of the 24h mark at high step counts. Modal default function timeout, deliberate max-runtime, or something else? Confirm against the `tmgg-spectral` Modal app definition before relaunching with longer wall-clock.
2. **`enzymes-pearl-gnnconv-norm` failures.** Two consecutive `failed` states (not `crashed`) at 4h and 11h respectively, with healthy gradients at the point of failure. Cause is infra (preempt? OOM? Modal volume hiccup?). Pull `modal app logs tmgg-spectral` for `ly0d6lyi` and `txfr1vms` before relying on `zyawhwrx`.
3. **Volume contents per run.** None of the entries have verified `volume_path` contents — the path is *expected* per the `_lib/config_resolution.py` convention but not yet inspected. Spot-check one with `modal volume ls tmgg-outputs /data/outputs/<exp>/<run_id>/`.
4. **MMD anchor gap on enzymes.** Our raw degree MMD ≈ 0.187 across all healthy enzyme variants vs HiGen's reported DiGress 0.004 → MMD² gap of ~10⁴×. Either we are massively undertrained or the kernel/sigma differs from HiGen's. PICKUP doc Task 2/3 will resolve.

## Backfill checklist (next session)

1. ~~Run IDs and exact launch UTCs~~ — done 2026-05-06 07:12 UTC.
2. ~~Verify `discrete_sbm_vignac_spectral_repro` actually launched~~ — done; run `e5pd9drt`.
3. ~~MMDs and basic ML diagnostics for all panel runs~~ — done 2026-05-06 07:25 UTC for SBM-from-export and live-from-W&B for the rest. Exception: `txfr1vms` diagnostics not pulled (only metadata).
4. Spot-check `volume_path` for at least one run per dataset to confirm `/data/outputs/{experiment_name}/{run_id}/` layout matches expectation.
5. Once `data/eval/mmd_baselines/{spectre_sbm,pyg_enzymes}.json` exist (PICKUP doc Task 1), compute ratios per run and link the analysis file in each detail file's `Diagnostics`.
6. Pull the still-running runs' final summaries once they finish; update `Fetched`, `local_path`, and re-snapshot `Diagnostics` in their detail files.
7. **Pull `modal app logs tmgg-spectral`** for the failed runs (`ly0d6lyi`, `txfr1vms`) — diagnostics show no numerical issue, so cause is infra. Determine before further relaunches.
8. **Decide on `gnnconv_raw_repro`** — kill, fix (normalisation + grad clipping + lower lr), or document-as-known-broken. Both SBM runs and the enzymes run blew up; variant is not currently usable.
9. Commit `/tmp/query_repro_runs_json.py` and `/tmp/query_specific_runs.py` somewhere durable (e.g. `wandb-tools/runlog_inventory.py`) so future inventory refreshes don't depend on /tmp scripts.
10. Pull or render visualisation artefacts (loss curves, gradient histograms, generated-sample grids) into `wandb_export/<analysis>/figures/` or `data/runs/<exp>/<run>/figures/` per the conventions above, and link them under each detail file's `Visuals` line. Currently only the SBM panel has any figures (`wandb_export/sbm-repro-report-2026-05-05/figures/`).
11. Pull diagnostics for `txfr1vms` (currently metadata-only) to complete the enzymes-norm trio.
12. Add a step-equal cross-variant comparison section (or per-dataset detail aggregator) — see chat history 2026-05-06; SBM has step-equal tables in `wandb_export/sbm-repro-report-2026-05-05/report.typ`, but they are incomplete (missing `vignac_spectral`) and there is no enzyme equivalent.
