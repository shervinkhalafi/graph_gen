# Smallest-config search — round-by-round progress log

Append-only. One section per round per dataset. Round-0 is the
diagnostic pre-rank baseline derived from the v1 long-run; it has
no Modal launches and no rounds.jsonl entries.

See `docs/superpowers/specs/2026-04-29-smallest-config-search-design.md` §4.4
for the round-section scaffold.

## Round 0 (no launches) — diagnostic pre-rank for SPECTRE SBM

### Source
v1 long-run at
`2026-04-28-working-runs-analysis/modal_volume/discrete_sbm_vignac_repro/discrete_sbm_vignac_repro_DiffusionModule_lr1e-3_wd1e-4_s666/csv/discrete_sbm_vignac_repro/version_1/metrics.csv`
(231k+ steps, single seed=666, full DiGress config).

### What the diagnostics pre-rank says
v1 predates the new `diagnostics-train/opt-health/*` namespace, so we
substitute the legacy `train/weight_norm/tf_layers_*` trajectories as
a coarse proxy. Layers whose weight norm grows slowest (or saturates
earliest) are the candidates for the first axis cut.

The eight transformer blocks all start tightly clustered around
weight-norm 52.4–54.5 at step 5500, but spread to 54.6–61.8 by
step 231k. The terminal-norm and growth-ratio distributions tell
a consistent story: `tf_layers_7` ends with the lowest terminal
norm (54.58) and the lowest early-to-late ratio (1.0485), and
`tf_layers_1` is the second-lowest on both axes (55.93 terminal,
1.056 ratio) — these two blocks accumulated the least extra
capacity through training. By contrast `tf_layers_0` is the
highest-norm block (61.81, ratio 1.135), consistent with the
first layer carrying a disproportionate share of the input
feature transformation, and the middle stack (`tf_layers_2..6`)
falls in a 57–60 / ratio 1.08–1.15 band that grew steadily.
The MLP output heads (`mlp_out_X`, `mlp_out_y`) are exactly
flat (ratio 1.0), and `mlp_in_E` barely moves (ratio 1.013) —
output capacity is weakly axis-dependent, so cutting hidden
width before depth would risk hitting a real bottleneck first.

### Pre-rank order (cheapest cut first)
1. **`n_layers`** — `tf_layers_7` is barely growing (5% over
   226k steps) and `tf_layers_1` is close behind; an 8-block
   stack with one near-pass-through layer is over-provisioned.
   Going from 8 → 6 should be the lowest-risk cut.
2. **`dim_ffy`** — the `y` (global-feature) MLPs (`mlp_in_y`,
   `mlp_out_y`) show modest or zero growth, and the global
   stream is a known auxiliary in DiGress; trimming `dim_ffy`
   before `dx` likely hits an axis with the most slack after
   depth.
3. **`dx`** — `tf_layers_0` and `mlp_in_X` both grew strongly
   (ratios 1.135 and 1.295), so the X stream is doing real
   work; cut it after the obviously over-provisioned axes.
4. **`T`** (timesteps) — orthogonal to the architecture
   diagnostics and most expensive to validate (changes the
   schedule, not just block size); leave for last.

### Round-1 plan derived from pre-rank
- For SPECTRE SBM: anchor at full config + first axis cut of
  `n_layers` from `8` to `6`. 2 runs total (anchor + cut).
- For PyG ENZYMES: anchor run only — no v1 long-run exists, so the
  anchor run is also the path-D fallback that pins ENZYMES's
  `spectral_mmd` threshold.

### S* used for round 1
`s_star.yaml` reports `s_star_operational: 100000`. The
`gen-val/*` sample counts are insufficient (3 finite points
each, need ≥ 8) and the NLL fit is degenerate, so the
structural-quality S* is unreached. Round 1 operates off the
heuristic 100k-step cap; the operational S* will be replaced
with a real gen-val/* fit once a v1-equivalent run with
`eval_every_n_steps <= 10000` lands.

### Eval cadence for round 1 — cosine/U-bowl

Per spec §11.1, generation evaluations follow the cosine/U-bowl
schedule from `scripts/sweep/eval_schedule.py`. SBM defaults:
`rho_max = 1/4000`, `rho_min = 1/20000`, `s_p = 35000`,
`s_k = 70000`. With `total_steps = 100000` and `n_evals = 24`,
the integer schedule (computed by
`uv run python -m scripts.sweep.eval_schedule --dataset spectre_sbm
--total-steps 100000 --n-evals 24`) is:

```
[2590, 5237, 8011, 11004, 14369, 18403, 23855, 33237, 44142,
 50291, 54605, 58115, 61192, 64014, 66689, 69289, 71873, 74499,
 77229, 80150, 83392, 87196, 92125, 100000]
```

The first half is dense in `[0, ~14k]` (warmup), middle is sparse
around `[33k, 50k]` (chance plateau), late stretch is dense in
`[60k, 80k]` (expected knee at `s_k = 70k`), plus the iterate-beyond
entry at 100000. The ENZYMES round-1 anchor run uses the same
parameters (no v1-derived ENZYMES priors yet to motivate different
defaults).

The training-side consumer that fires `val_check` at exactly these
steps is a deferred patch (spec §10 Phase 0.4). Until that lands,
the wrappers' default `val_check_interval = 10000` continues to
fire — round 1's actual eval cadence is therefore uniform every
10000 steps, NOT the cosine bowl. The schedule list is committed
ahead of the consumer so round-1 vibes-augmented analysis can
already reason about the intended placement, and so the consumer
patch (when it lands) can be slotted in without changing the round
artefacts.

### In-flight watch entries for round 1
(populated during the 15-min watch loop while runs train; see
`watches.jsonl` for the structured log)

### Synthesis (every round, cheap)
This is round 0; the synthesis is the pre-rank above. Round 1 will
decide whether the pre-rank survives contact with new opt-health
diagnostics that weren't available in v1.

## Round 1 — anchor + first axis cut (n_layers 8 → 6)

### What round 0 told us
The v1 long-run's transformer-block weight-norm trajectories ranked
`n_layers` as the cheapest axis to cut: `tf_layers_7` grew only 5%
over 226k steps and `tf_layers_1` was second-lowest, both consistent
with an over-provisioned 8-block stack. `dim_ffy` and `dx` showed
real growth, so depth before width is the right starting move. No
new opt-health namespace was available in v1 to cross-check, so
round 1 also serves as the first opportunity to see
`diagnostics-train/opt-health/*` on the anchor config.

### What I'm trying next and why
Three launches: the SBM anchor at full DiGress config (path-D
in-house baseline against HiGen-derived raw-MMD anchors), the SBM
cut at `n_layers=6` (the round-0 pre-rank's first move), and the
ENZYMES anchor at full config (also pins ENZYMES's `spectral_mmd`
threshold since neither DiGress nor HiGen reports it). Step cap
100k from `s_star_operational`; eval cadence is the cosine/U-bowl
16-eval schedule per dataset.

### Configs to launch
| dataset | model_arch | axis_changed | axis_value | seed | step_cap | run_uid |
|---|---|---|---|---|---|---|
| spectre_sbm | digress_official | anchor | full | 0 | 100000 | smallest-cfg/spectre_sbm/r1/anchor/5b20d928 |
| spectre_sbm | digress_official | n_layers | 6 | 0 | 100000 | smallest-cfg/spectre_sbm/r1/n_layers/358d818c |
| pyg_enzymes | digress_official | anchor | full | 0 | 100000 | smallest-cfg/pyg_enzymes/r1/anchor/5b20d928 |

### If I'm wrong, the cheapest disconfirming config is:
`n_layers=7` — interpolates between `n_layers=8` (anchor) and
`n_layers=6` (round-1 cut). If `n_layers=6` fails to cross threshold
but `n_layers=7` passes, the pre-rank is partially right (last block
is removable, but the second-to-last isn't). If both `n_layers=7`
and `n_layers=6` fail, the depth axis isn't actually slack and we
pivot to `dim_ffy` per the round-0 pre-rank order.

### Wrapper-misconfig kill + relaunch (operational note)
Original SBM launches went to W&B project `discrete-diffusion`
because `run-upstream-digress-sbm-modal.zsh:20` defaulted
`WANDB_PROJECT=discrete-diffusion`. Both SBM pods cancelled via
`scripts.sweep.kill_call` within minutes of spawn (no training had
started). Wrapper default patched to `tmgg-smallest-config-sweep`;
`launch_round.py` now asserts the rendered wrapper command's
`wandb_project=<value>` matches `CANONICAL_WANDB_PROJECT` before
appending a launched row (regression tests pin both failure modes).
`find_pending_launches` in `fetch_outcomes.py` updated to pair by
`(run_uid, latest-outcome-ts)` so the relaunch isn't shadowed by
the cancel-outcome that shares its `run_uid`. Cancel-outcomes
written to rounds.jsonl with `failure_kind=cancelled_wrapper_misconfig`;
relaunches landed in the canonical project and verified by the new
guard. See `skill-feedback.md` round-1 entry for the full forensics.

Active call IDs (the relaunched pair + the original ENZYMES anchor):
- SBM anchor: `fc-01KQEB0K2RZ8HY01BKSNMW6T4E`
- SBM n_layers=6: `fc-01KQEB11GXA2D7AAY2MS4NRY8Y`
- ENZYMES anchor: `fc-01KQEAMD3RW1XN40QZ4Y8YA75Q`

### In-flight watch entries for round 1

**Watch wakeup #1 (T+~15min after launch).** Two of three pods
crashed within minutes; SBM anchor still running.

- **SBM n_layers=6** (`fc-01KQEB11GXA2D7AAY2MS4NRY8Y`): KILLED.
  Trainer subprocess exited 1 during checkpoint restoration. The
  state_dict of a previously-cached n_layers=8 model was loaded into
  the new n_layers=6 model, which lacks `tf_layers.7.norm_y2.{weight,bias}`.
  Root cause: the wrapper's run_id formula
  (`discrete_diffusion_DiffusionModule_lr2e-4_wd1e-12_s0`) does not
  include `n_layers`, so this run shared a Modal output dir with a
  prior n_layers=8 SBM run. The wandb run
  https://wandb.ai/graph_denoise_team/tmgg-smallest-config-sweep/runs/m1uwn3es
  exists but only logged init metadata.
  Fix path: add `+force_fresh=true` to the round.yaml overrides for
  the n_layers cut, or include `n_layers` in the wrapper run_id formula.
  Marked `failure_kind=checkpoint_state_dict_mismatch`.
- **ENZYMES anchor** (`fc-01KQEAMD3RW1XN40QZ4Y8YA75Q`): KILLED.
  Hydra preflight crashed: `GraphDataModule.__init__() got an
  unexpected keyword argument 'num_nodes'`. Root cause:
  `base_config_discrete_diffusion_generative.yaml` defines
  `data.num_nodes=20` as a top-level field for the synthetic
  `SyntheticCategoricalDataModule` default; `+data=pyg_enzymes` merges
  (does not replace) the data dict, so the leftover `num_nodes=20` is
  passed to `GraphDataModule.__init__` which strict-rejects unknown
  kwargs. SBM works because `SpectreSBMDataModule` has a `**_metadata`
  catch-all that swallows the leftover. Fix path: add `~data.num_nodes`
  to the ENZYMES wrapper, or change `pyg_enzymes.yaml` to nullify the
  legacy field. Marked `failure_kind=config_schema_mismatch`.
- **SBM anchor** (`fc-01KQEB0K2RZ8HY01BKSNMW6T4E`): KEEP. Modal call
  still running. W&B project `tmgg-smallest-config-sweep` does not yet
  exist (the project is created when the first run logs its first
  metric); no gen-val data available for the watcher flowchart yet.
  Recheck at wakeup #2.

**Watch wakeup #2 (T+~30min after launch).** Modal call still running.
W&B-side flowchart could not run: this host's `~/.netrc` authenticates
to wandb as `igkgh-personal`, but the sweep writes under the
`graph_denoise_team` entity (Modal pods authenticate via Doppler
secrets, a different identity). The project lookup for
`tmgg-smallest-config-sweep` returns `Could not find project` —
which is auth, not absence (a direct fetch of the n_layers=6 wandb
run id `m1uwn3es` also fails the same way, and listing all projects
under `graph_denoise_team` returns 0 even though several known
projects exist). Per `CLAUDE.md` rule "re-authentication issue → stop
and tell the user, don't look for alternatives", halting the
autonomous watch loop until wandb auth is sorted.

While the loop is halted, the SBM anchor continues training on Modal
regardless — its terminal outcome can be fetched once auth lands.
Side fix shipped this wakeup: `watch_runs.find_running_launches` is
now a re-export of `fetch_outcomes.find_pending_launches` so the
watcher and fetcher cannot diverge again on pairing semantics
(regression test in `test_watch_runs.py`). Without this fix the
watcher would have ignored the relaunch-after-kill case and reported
0 running launches the first time it ran post-fix.

**Auth resolution (between wakeup #2 and #3).** `WANDB_API_KEY` lives
in the project's Doppler `prd` config — the same source the Modal
wrappers' `mise run modal-deploy` step injects into trainer pods.
Invoking the watcher under `doppler run -- uv run python -m
scripts.sweep.watch_runs` authenticates as `igorkraw` (default entity
under doppler) and resolves the `graph_denoise_team/tmgg-smallest-
config-sweep` lookup. Local watcher invocations from this point onward
use the doppler prefix.

**Bugfix relaunch (between wakeup #2 and #3).**
`round-1-relaunch-after-bugfixes.yaml` re-launched the two failed
configs after the corresponding code fixes landed:
- `GraphDataModule.__init__` now accepts (and ignores) `num_nodes`
  and `num_graphs` as explicit ignored params (matches
  `SpectreSBMDataModule`'s already-existing `**_metadata` absorber);
  the legacy-kwarg rejection contract for `noise_levels` /
  `noise_type` is preserved by using explicit ignored params instead
  of a generic `**_metadata` swallow. ENZYMES anchor relaunch:
  `fc-01KQEGC2BFGPW6KFYAZTS6C95R`.
- SBM `n_layers=6` relaunch uses `force_fresh=true` (plain, not
  `+force_fresh` — Hydra rejects the append form because the trainer
  config already declares the key) so `generate_run_id()` appends
  `_fresh_<UTC>` to the run_id and dodges the Modal output-dir
  collision. Relaunch: `fc-01KQEGBNS2GN2MYTG1JVVV8194`, run_id
  `discrete_diffusion_DiffusionModule_lr2e-4_wd1e-12_s0_fresh_20260430T061359`.

**Watch wakeup #3 (T+~3h after original launch).** Three pods now
visible to the watcher.

- **SBM anchor** (`5b20d928`, fc-...VV8194): step 2154, single gen-val
  cycle landed (around step 1798); next eval at step 3903 per the
  cosine schedule. Snapshot sha `3b007ad6` unchanged from wakeup #2's
  observation since no new gen-val cycle. Mostly-passing metrics
  early: `modularity_q=0.42` (≥ 0.3 ✓), `spectral_mmd=0.005`
  (≤ 0.01 ✓), `degree_mmd=0.0016` (≤ 0.0013×1.5 ✓),
  `clustering_mmd=0.067` (≤ 0.0498×1.5 ✓), `orbit_mmd=0.078`
  (✗ early), `sbm_accuracy=0.344` (✗ early). Recommendation: keep.
- **SBM n_layers=6 relaunch** (`52104237`, fc-...8194): step 393,
  trainer up and stepping. force_fresh suffix confirmed in the run_id.
  No gen-val yet (first at step 3903). Recommendation: keep.
- **ENZYMES anchor relaunch** (`5b20d928`, fc-...95R): step 233.
  GraphDataModule fix held — Hydra preflight passed, training started.
  No gen-val yet. Recommendation: keep.

**Watch wakeup #4 (T+~3.5h after original launch).** All three pods
have gen-val data now. Step pacing: SBM anchor advanced 2154 → 4678
in 15 min ≈ 168 steps/min, projecting ~70h to reach 100k — well over
the original ~13h estimate. Worth flagging: A100 + bf16-mixed at
batch_size=12 may be slower than projected, or async-eval workers are
serializing more than expected. Cost estimate revision pending.

- **SBM anchor** (`5b20d928`, step 4678): `modularity_q=0.412` (≥ 0.3 ✓),
  `spectral_mmd=0.0065` (≤ 0.01 ✓), `clustering_mmd=0.067`
  (≤ 0.0747 ✓). Failing early: `degree_mmd=0.0067` (> 0.00195 ceil;
  got worse from 0.0016 at wakeup #3 — normal early variance),
  `orbit_mmd=0.110`, `sbm_accuracy=0.44`. Recommendation: keep.
- **SBM n_layers=6 relaunch** (`52104237`, step 3462): first gen-val
  landed. `modularity_q=0.406`, `spectral_mmd=0.0073`,
  `clustering_mmd=0.073` — within tolerances; `degree_mmd=0.017`,
  `orbit_mmd=0.133`, `sbm_accuracy=0.406`. Metrics close to the
  anchor at the same step regime — too early to prefer either.
  Recommendation: keep.
- **ENZYMES anchor** (`5b20d928`, step 2436): first gen-val landed.
  HiGen anchors are `degree=0.004`, `clustering=0.083`, `orbit=0.002`.
  Observed: `degree=0.0076` (✗ at 1.5× ceil 0.006), `clustering=0.057`
  (✓ pass), `orbit=0.0091` (✗ at 1.5× ceil 0.003); `spectral_mmd=0.033`
  (this run pins the path-D in-house threshold for ENZYMES).
  `modularity_q=0.37`, `planarity_accuracy=0.375`. Recommendation: keep.

**Bookkeeping.** 11 abandoned round-0 smoke launches (no associated
W&B runs, never paired with outcomes since round 0) cleared at the
same time with two `failure_kind=abandoned_smoke` outcome rows (one
per smoke run_uid). `find_pending_launches` shrank from 14 to 1 (the
SBM anchor). Without this cleanup, every future watch wakeup would
have re-traversed the abandoned set and crashed `watch_runs.py` on
the first project lookup.

### Synthesis (every round, cheap)
Round 1's effective evidence is just the SBM anchor — both axis cuts
failed for infrastructure reasons, not scientific ones. The n_layers=6
cut crashed in checkpoint restoration (wrapper run_id reuses a Modal
output dir across architectures), and the ENZYMES anchor crashed in
Hydra preflight (base config's synthetic-default data block leaks
`num_nodes` into `GraphDataModule` because that data class lacks the
`**_metadata` swallow that `SpectreSBMDataModule` has). Neither
failure tells us anything about the scientific hypotheses (does
n_layers=6 cross threshold? Is ENZYMES's spectral_mmd around HiGen's
level?). Recommended round-2 plan: ship two wrapper fixes (run-id
includes n_layers OR force_fresh on cuts; `~data.num_nodes` on ENZYMES
wrapper), then relaunch (a) SBM n_layers=6 with `force_fresh`,
(b) ENZYMES anchor. The SBM anchor result from round 1 stays as the
in-house baseline regardless. The wrapper-misconfig (W&B project)
incident from earlier is unrelated to either round-1 crash and is
already locked out by the canonical-project guard.
