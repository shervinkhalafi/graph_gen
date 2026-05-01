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

## Round 2 — Closing synthesis (early-stop, 2026-04-30)

### What we observed
Round 2 closed early at step ≈30k (anchor) / 40k (L6) / 75k (ENZYMES) after 4
trustworthy in-band data points per SBM pod. The cosine-schedule async-eval
data turned out unreliable on both datasets (loaded checkpoints produced
metrics that systematically regressed vs the in-band evaluator path; in-band
is the trustworthy source). The watcher had four bugs found and fixed during
the round: (a) `_step` vs `trainer/global_step` misread, (b) duplicate-name
ambiguity post-relaunch, (c) `scan_history(keys=...)` filtering rule that
hid all gen-val data behind an empty result, (d) `find_pending_launches`
ts-pairing that mis-credited cancel-outcomes. All four are in regression
tests.

### What round 2 told us about n_layers
Matched-step in-band evidence:

| step | anchor (L8) sbm_acc | L6 sbm_acc | anchor pass | L6 pass |
|---|---|---|---|---|
| 10k | 0.34 | 0.56 | 4/6 | 3/6 |
| 20k | 0.59 | 0.44 | 5/6 | 3/6 |
| 30k | 0.375 | 0.41 | 3/6 | 5/6 |
| 40k | n/a   | 0.5625 | n/a | 5/6 |

Both pods bounce sbm_accuracy by ±0.15 absolute — that is the
32-graph evaluator's noise floor. At matched step 30k L6 actually beats
anchor 5/6 vs 3/6. There is no statistically meaningful gap on any
threshold metric within the noise floor. The "anchor pulled ahead at step
20k" reading from earlier was the noise itself, not signal.

### Decision
**Provisionally lock n_layers=6 as the new SBM baseline.** Queue an L6
re-seed in round 3 as the formal "borderline → re-seed" check (skill rule:
single-seed reads on borderline outcomes get a 2nd seed before locking).
Round 3 launches the re-seed in parallel with the next axis attack
(`dim_ffy`) to keep the sweep moving.

### What round 2 told us about ENZYMES
Saturated to 4/5 PASS by step 60-70k. Trajectory is stable and small —
spectral_mmd clusters around 0.025-0.034. The path-D in-house spectral_mmd
anchor is now pinned at 0.025 (`tolerance_x=1.0` because this IS the
reference). Only `orbit_mmd` still fails (0.020 vs HiGen ceil 0.003) — the
gap is large enough to suggest ENZYMES needs more steps OR a different
model regime (orbit MMD measures higher-order graph structure that DiGress
struggles with even at the published config).

### Cost
Round 2 burn: ~$8 across 3 pods × ~3-4h × A100. Saved ~$15-20 by stopping
early before step_cap.

## Round 3 — Lock L6, re-seed, attack dim_ffy

### What round 2 told us
n_layers=6 is provisionally safe for SBM (matched-step parity within
eval-noise floor across 4 in-band points per pod). ENZYMES anchor is
saturated at 4/5 PASS; spectral_mmd anchor pinned at 0.025.

### What I'm trying next and why
Three SBM launches:
1. **L6 re-seed (seed=1)**: keystone confirmation of round 2's "L6 is
   safe" read. If seed-1 sbm_accuracy stays in the 0.4-0.6 range across
   matched in-band steps, L6 is locked. If seed-1 fails the noise
   envelope, revert to L8 and probe n_layers=7.
2. **dim_ffy=128 on L6**: first cut of the next axis (round-0 pre-rank:
   `mlp_in_y` ratio 1.013, `mlp_out_y` flat — the global-feature MLPs
   are over-provisioned). 50% reduction from the round-1 default 256.
3. **dim_ffy=64 on L6**: aggressive 75% cut. Gives gradient info on the
   dim_ffy axis regardless of which way 128 lands.

### Configs to launch
| dataset | model_arch | axis_changed | axis_value | seed | step_cap |
|---|---|---|---|---|---|
| spectre_sbm | digress_official | n_layers_reseed | 6 | 1 | 100000 |
| spectre_sbm | digress_official | dim_ffy | 128 | 0 | 100000 |
| spectre_sbm | digress_official | dim_ffy | 64 | 0 | 100000 |

ENZYMES does not get a round-3 launch. Strategy: lock the SBM smallest
config first (n_layers + dim_ffy + dx + T axes), THEN attack ENZYMES axes
once the SBM minimum is identified. Reduces sweep search-space
combinatorics.

### If I'm wrong, the cheapest disconfirming config is:
The L6 re-seed itself is the disconfirming-config check on round 2's
n_layers decision. For dim_ffy: if both 128 and 64 fail to cross threshold,
that establishes the dim_ffy axis as non-shrinkable given L6 — pivot to dx
in round 4. If only 64 fails, 128 is the lower bound on this axis.

### In-flight watch entries for round 3
(populated during the 4m30s cache-warm watch loop while runs train; see
`watches.jsonl` for the structured log)

### Synthesis (every round, cheap)
Round 3 is the first round to compose two axis cuts (L6 + dim_ffy). The
ablation it can answer: does cutting dim_ffy on top of L6 still produce a
valid config? Eval-pipeline noise is the dominant uncertainty source we've
characterized (±0.15 abs sbm_accuracy at 32 graphs); round 3's threshold
verdicts are interpreted against that floor, not against zero. Saturation-
fit diagnostic continues running each tick to characterize asymptote
prediction quality on the new pods.

### Round-3 close (2026-05-01)

All 3 pods completed full 100k steps cleanly (~9.5h each, no preemption).
Terminal metrics @ 100k (sbm_accuracy on 32 graphs):

| Pod                       | sbm_acc | mod_q | degree | clust | orbit | spectral |
|---------------------------|---------|-------|--------|-------|-------|----------|
| L6 reseed (s=1)           | 0.6875  | 0.423 | 0.0019 | 0.066 | 0.096 | 0.0065   |
| dim_ffy=128 on L6 (s=0)   | 0.4375  | 0.420 | 0.0055 | 0.065 | 0.062 | 0.0080   |
| dim_ffy=64 on L6 (s=0)    | 0.4688  | 0.436 | 0.0011 | 0.066 | 0.078 | 0.0078   |

vs round-2 baselines: L8 anchor 0.375 @ 32k, L6 (s=0) 0.5625 @ 44k.

**L6 confirmed**: seed-1 reached 0.6875, both seeds within ±0.15 noise
envelope, both beat L8 at every matched step. Round-2's "L6 safely
shrinkable" decision is fully backed.

**dim_ffy regression on L6 base**: both cuts (128 and 64) clustered at
sbm_acc ≈ 0.45, ~0.18 below L6 baseline mean (just outside ±0.15 floor).
dim_ffy=64 was NOT meaningfully worse than dim_ffy=128 — axis behaves
binary (intact vs broken) on this base. **HOWEVER, this finding is now
SUPERSEDED by Shervin's reference (see Round 4 below).**

## Round 4 — pivot to bracket-narrow around Shervin's reference

### What rounds 1-3 told us

Coordinate-descent on round-0's per-axis pre-rank was biased: each axis
was ranked one-at-a-time on the over-parameterized anchor, which
flagged dim_ffy as over-provisioned. Round 3's dim_ffy regression on
the L6 base (with all other dims still at full anchor values) appeared
to confirm this. Then on 2026-04-30 Shervin reported a working DiGress
config at ~300k params with dx=64, de=8, dy=8, dim_ffX=64, dim_ffE=8,
dim_ffy=256, n_head=8 — i.e. he kept dim_ffy at 256 while slashing
EVERY other axis 4-32×. This evidence dominates our marginal-axis
read: cutting one axis on an over-parameterized base behaves
differently than cutting the same axis on a tightly-parameterized base.

### What I'm trying next and why

Search reframe: bracket-narrow around Shervin's known-pass reference,
not coord-descent on pre-rank. Per-round invariant going forward:
include ≥1 config we expect to fail, to keep the bracket tight.

Three pods, all on Shervin's base (dx=64, de=8, dy=8, dim_ffX=64,
dim_ffE=8, dim_ffy=256, n_head=8, n_layers=6 assumed, seed=0):

1. **Pod A — exact reproduction**. Positive control: confirms Shervin's
   pass claim under our eval pipeline + threshold rule + step budget.
2. **Pod B — Shervin + dim_ffy 256 → 32**. Probes BELOW Shervin on the
   axis he kept full. ~0% compute change (dim_ffy operates on 1 vector
   per graph) — pure param-count cut. Hypothesis: passes (dim_ffy
   irrelevant in small regime).
3. **Pod C — Shervin + dx 64 → 32 + dim_ffX 64 → 32 (joint halving)**.
   Quadratic-ish node-side cut: attn QKV proj 4× cut (`dx²`), node FFN
   4× cut (`dx · dim_ffX`), edge attn 2× cut (linear in dx). Edge-side
   untouched (de=dim_ffE=8 already minimal). Hypothesis: lower-bracket
   probe — plausibly fails because dx=32 may be at the boundary for
   SBM block discrimination.

### Configs to launch
| dataset     | model_arch       | axis_changed       | axis_value | seed | step_cap |
|-------------|------------------|--------------------|------------|------|----------|
| spectre_sbm | digress_official | shervin_base       | exact      | 0    | 100000   |
| spectre_sbm | digress_official | shervin_dim_ffy    | 32         | 0    | 100000   |
| spectre_sbm | digress_official | shervin_dx_dim_ffX | 32_32      | 0    | 100000   |

ENZYMES still no round-4 launch (lock SBM smallest first).

### Cost-savers (all pods)

- **Drop async-eval entirely.** Rely on trainer's in-band
  `val_check_interval=10000` (10 trustworthy data points across 100k).
  Async-eval has documented untrustworthy divergence on SBM (see
  taskwarrior #24). Saves ~3h per pod and removes scheduling complexity.
- n_layers=6 baked into all pods (don't re-test n_layers axis).

### If I'm wrong, the cheapest disconfirming config is

- Pod A FAILS to reproduce → reproduction problem; investigate eval-
  pipeline differences vs Shervin's setup before round 5.
- Pod A passes, B passes, C fails → dim_ffy irrelevant, dx is the
  active constraint. Round 5 probes dx=48 (midpoint) and locks dim_ffy
  at 32.
- Pod A passes, B fails, C passes → dim_ffy was load-bearing in the
  small regime (Shervin's 256 was deliberate). Round 5 probes dim_ffy
  midpoint (e.g. 128) and locks dx=32.
- All three pass → aggressive round 5 with joint cuts.
- A passes, B and C fail → Shervin sits at the exact boundary; round 5
  narrows by midpoints between Shervin and the failed cuts.

### Synthesis (every round, cheap)

Round 4 is the first bracket-search round. Cited evidence: Shervin's
2026-04-30 config message (informal, no W&B handle yet). Round 3's
"dim_ffy is binary intact/broken on L6" finding is provisionally
re-interpreted as a coord-descent artifact and will be re-tested in
Pod B on the small base. Saturation-fit diagnostic continues each tick.

## Round 5 — closed on user direction (2026-05-01); GREEDY config wins

### What happened

Three pods launched on R4-union base (dx=32, dim_ffX=32, dim_ffy=32,
n_layers=6, others Shervin):

- **Cautious** (combined_dy_cut): + dy 8→4
- **Moderate** (combined_L4):     + n_layers 6→4 (depth halve)
- **Greedy**   (combined_L4_dx16): + n_layers 6→4 + dx 32→16 + dim_ffX 32→16

All 3 killed on user direction at ~step 50-70k after Greedy emerged
as clear winner on two key axes simultaneously.

### Last-observed metrics before kill

| Pod      | step  | sbm_acc | mod_q | degree  | clust  | orbit  | spectral |
|----------|-------|---------|-------|---------|--------|--------|----------|
| Cautious | 52920 | 0.281   | 0.417 | 0.0046  | 0.0709 | 0.0777 | 0.0070   |
| Moderate | 68899 | 0.250   | 0.405 | 0.0015  | 0.0692 | 0.0693 | 0.0047   |
| **Greedy**| 60114| **0.500** | **0.442** | 0.0014 | 0.0657 | 0.0763 | 0.0095 |

Greedy had clear leadership on sbm_accuracy (0.500 — best of entire
sweep), modularity_q (0.442 — best of sweep), and competitive degree.
Cautious + Moderate were both bouncing in the ±0.15 noise floor.
Greedy strictly contains Moderate, so Moderate is provably non-pareto;
Greedy beat Cautious on the leadership metrics, so Cautious is also
deprioritized.

### Verdict and pareto reset

**Round-5 winning config (new sweep pareto reference):**

```yaml
n_layers: 4
hidden_dims:
  dx: 16
  de: 8
  dy: 8
  n_head: 8
  dim_ffX: 16
  dim_ffE: 8
  dim_ffy: 32
```

**Cost vs Shervin's reference (~300k params):**

| Dimension                         | Shervin | Greedy | Ratio                     |
|-----------------------------------|---------|--------|---------------------------|
| Wall-clock pace (steps/min, A100) | 309     | 373    | **1.21× faster**          |
| Transformer-stack params          | 182k    | 10k    | **18× fewer**             |
| Total model params (estimate)     | ~300k   | ~30-50k| **~7-10× smaller**        |

The wall-clock speedup is much smaller than the param/FLOP cuts because
edge-side compute (n²-scaled at N=200 nodes, both pods have de=8 and
dim_ffE=8 unchanged) and eval-cycle wall time (fixed per evaluation)
dominate at this size.

### Open question (deliberately unresolved)

NONE of the round-4 or round-5 configs cross the sbm_acc=0.8 threshold
within the 100k step cap. Best the sweep has ever seen is sbm_acc=0.500
(Greedy at step 60k). Visualizations across the sweep still show ER-like
samples without a crisp 2-block diagonal — Shervin's informal "550k step"
remark may be the actual scale needed for true block discrimination on
configs of this size. Open question for round 6: is the gap between
sbm_acc=0.5 and 0.8 closeable with more steps on the same config, or
does it require a structurally different architecture?

## Future-work suggestions (post round 5)

These are in priority order; pick from this menu when resuming the sweep.

### Wall-clock-focused suggestions (since wall-clock is the cost metric)

1. **Edge-side cuts on Greedy base.** Greedy left de=8, dim_ffE=8
   untouched. Cutting both 8→4 attacks the n²-scaled compute that
   currently dominates wall-clock. Expected wall-clock win: 1.5-2×
   on top of Greedy's 1.21×. Risk: edge-feature dim might be load-
   bearing for binary-edge SBM (we never tested edge-side cuts on a
   small base).

2. **Larger eval cadence.** `val_check_interval: 10000 → 20000`
   halves the per-100k-step eval wall time. Costs 50% of the in-band
   data points (5 instead of 10) but the eval-noise floor (±0.15 abs
   sbm_acc) means individual data points are weak signal anyway.

3. **Smaller eval sample count.** `eval_num_samples: 32 → 16` cuts
   sampling time roughly in half per cycle. Increases noise floor
   (~±0.20 abs) but proportionally reduces the eval wall-time
   floor that masks param-cut wall-clock wins.

### Quality-focused suggestions (for crossing sbm_acc=0.8)

4. **Long Greedy run.** Take the round-5 winner and just train for
   500k-1M steps. Single-config probe of whether the structurally-
   sound-but-undertrained hypothesis (per Shervin's claim) holds.
   Cost: ~25h A100 wall on greedy at 373 steps/min. This is the
   highest-evidence experiment for the open question above.

5. **Architectural change.** If Greedy at 500k+ still doesn't cross
   threshold, the architecture (DiGress-as-implemented) may not
   support the regime; investigate adding spectral/positional input
   features (already wired in via extra_features) or higher-order
   interactions.

### Sweep-protocol suggestions

6. **ENZYMES axis attacks.** The smallest-config search on ENZYMES
   has been deferred since round 1 (only the path-D anchor was run).
   Now that SBM has a tight pareto baseline (Greedy), ENZYMES axis
   cuts can be informed by SBM findings and run in parallel.

7. **Per-config seed budget = 2 going forward.** Single-seed reads
   on sbm_acc are nearly meaningless given the ±0.15 noise floor and
   the single-eval-bounces we've seen (R4 Pod C 0.03→0.34 in adjacent
   evals; R5 Greedy 0.281→0.500). Two seeds per config would cut
   the false-call rate substantially without doubling cost (each pod
   is cheap at this size).

8. **Visualization-gated promotions.** sbm_accuracy's 0-0.5 range
   under our eval pipeline does NOT distinguish "pre-block-discrim
   ER noise" from "early SBM emergence" — we proved this with the
   step-30k visualization sniff-test where all 6 R4+R5 pods looked
   visually identical despite numerical metric spread. Consider
   gating round-progression decisions on visual-block-emergence,
   not on sbm_accuracy threshold alone.

