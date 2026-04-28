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

### Synthesis (every round, cheap)
This is round 0; the synthesis is the pre-rank above. Round 1 will
decide whether the pre-rank survives contact with new opt-health
diagnostics that weren't available in v1.
