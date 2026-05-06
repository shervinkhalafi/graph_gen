# Pickup Doc — MMD ratio reporting (2026-05-06)

Carries forward from the 2026-05-05 conversation in which we wired up
on-disk train↔test MMD baselines (`compute_mmd_baselines.py` +
`tmgg.evaluation.mmd_baselines`). Tomorrow's job: **launch container
runs to populate the cache for every dataset, then use the cached
baselines to compute DiGress-style ratios from existing W&B runs**.
Plus a **hyperparameter audit**: confirm the kernel / sigma / batch
configurations DiGress and HiGen actually used so our cached baselines
are numerically comparable to their reported numbers.

Repo state at handoff: branch `main`, HEAD `e157de0f`, both new files
added but **not yet committed**. The five SBM repro runs from
2026-05-04 and the five enzyme repro runs launched 2026-05-05 are still
running (or just-finished) on Modal under `tmgg-spectral`.

## Why this exists

DiGress (Vignac et al. 2023) Appendix F.1 reports MMD as a ratio
`r = MMD²(generated, test) / MMD²(training, test)` rather than raw
MMD². The denominator is dataset-determined — once per dataset, then
reused across every training run. Our pipeline emits raw MMD² (the
`gen-val/{degree,clustering,orbit,spectral}_mmd` keys); without the
ratio, we cannot directly compare to DiGress Table 1 (1.6 / 1.5 / 1.7
/ 74% on SBM) or to HiGen Table 1's enzyme row (degree 0.004,
clustering 0.083, orbit 0.002 — HiGen is the *only* published source
for DiGress on ENZYMES because the DiGress paper did not evaluate that
dataset).

Yesterday's session established:

1. The DiGress paper does **not** publish hyperparameters for ENZYMES
   (no config in `cvignac/DiGress`, no config in `Karami-m/HiGen_main`
   either; HiGen's only DiGress reference is a code attribution for
   `KNodeCycles`). HiGen's reported DiGress numbers come from a custom
   run Mahdi never released.
2. Our pipeline uses raw MMD² with `gaussian_tv` kernel, defaults
   matching upstream DiGress per
   `digress-upstream-readonly/src/analysis/spectre_utils.py` —
   confirmed line-by-line for SBM. Worth re-confirming for ENZYMES /
   planar tomorrow (see "Audit task" below).
3. The five enzyme repro runs and four healthy SBM runs are
   meaningfully far from anchor numbers (degree MMD ~0.18 vs HiGen
   0.004 on enzymes; ~140× gap on SBM degree). Either training is far
   from converged or there is a units mismatch we have not yet caught.
   The ratio computation will resolve which.

## What was built (2026-05-05)

Two files, **uncommitted**:

| Path | Role |
|------|------|
| `src/tmgg/evaluation/mmd_baselines.py` | Schema (`MMDBaseline`, `MMDBaselineParams`), atomic writer (`save_baseline`), validating loader (`load_baseline`), path helper (`baseline_path`), ratio helper (`compute_ratios`). Re-exported from `tmgg.evaluation`. |
| `scripts/compute_mmd_baselines.py` | uv-script CLI. Instantiates each datamodule, runs `setup()`, materialises NetworkX graphs, computes the four MMD² baselines (degree, clustering, spectral via `compute_mmd_metrics`; orbit via `compute_orbit_mmd`), writes one JSON per dataset to `data/eval/mmd_baselines/{dataset}.json`. |

Cache schema (one file per dataset):

```json
{
  "dataset": "spectre_sbm",
  "n_train": 128, "n_test": 40,
  "params": {"kernel": "gaussian_tv", "degree_sigma": 1.0,
             "clustering_sigma": 0.1, "spectral_sigma": 1.0,
             "orbit_sigma": 30.0, "clustering_bins": 100,
             "spectral_bins": 200},
  "fingerprint": "<sha256[:16]>",
  "mmd_squared": {"degree_mmd": ..., "clustering_mmd": ...,
                  "spectral_mmd": ..., "orbit_mmd": ...},
  "mmd":         {"degree_mmd": ..., ...},
  "computed_at": "...", "git_sha": "...", "extra": {...}
}
```

The fingerprint hashes all `params` fields, so changing any sigma /
bin / kernel produces a new fingerprint and `load_baseline` rejects
files where the stored fingerprint disagrees with the params block.
Both squared and unsquared MMD are stored; DiGress's ratio uses
`mmd_squared`, HiGen's table reports `mmd`.

Currently registered datasets in
`scripts/compute_mmd_baselines.py::DATASET_BUILDERS`:

- `spectre_sbm` — `SpectreSBMDataModule()` (128/32/40 split, Martinkus
  fixture)
- `spectre_planar` — `SpectrePlanarDataModule()` (Martinkus planar
  fixture)
- `pyg_enzymes` — `GraphDataModule(graph_type="enzymes",
  graph_config={...}, train_ratio=0.7, val_ratio=0.1, seed=42)` per
  `configs/data/pyg_enzymes.yaml`

## Tomorrow's tasks

### Task 1 — Launch containers to compute all baselines

**Goal:** populate `data/eval/mmd_baselines/{dataset}.json` for every
dataset we care about. Local execution works (each baseline takes a
few minutes), but the user explicitly wants containers — both for
parallelism across datasets and for ORCA binary availability (orca is
not in the host PATH; `tmgg-spectral` Modal image bundles it via
`src/tmgg/modal/_lib/image.py`).

**Suggested approach:** add a Modal app `tmgg-mmd-baselines`
modelled on `src/tmgg/modal/_profile_functions.py`. One
`@app.function` per dataset, all on cheap CPU-only configs (no GPU
needed; this is statistics + ORCA, both CPU). Output to the
`tmgg-outputs` Modal volume at
`/outputs/mmd_baselines/{dataset}.json`, then pull back via
`modal volume get`.

Sketch:

```python
# src/tmgg/modal/_mmd_baseline_functions.py
app = modal.App("tmgg-mmd-baselines", include_source=False)

@app.function(
    name="compute_baseline",
    image=experiment_image,  # bundles ORCA
    cpu=4.0,
    memory=8192,
    timeout=1800,
    volumes=get_volume_mounts(),
)
def compute_baseline(dataset: str, params_dict: dict) -> dict:
    from tmgg.evaluation.mmd_baselines import (
        MMDBaselineParams, save_baseline,
    )
    from scripts.compute_mmd_baselines import compute_baseline as compute
    from pathlib import Path

    params = MMDBaselineParams(**params_dict)
    baseline = compute(dataset, params)
    save_baseline(baseline, root=Path("/outputs/mmd_baselines"))
    return baseline.to_dict()
```

Launcher: spawn one function per dataset with `.spawn()`, await all,
pull results with `modal volume get tmgg-outputs /mmd_baselines ./data/eval/`.

**Datasets to cover (priority order):**

1. `spectre_sbm` — anchored by both DiGress paper (ratios) and HiGen
   raw MMD. Highest value.
2. `pyg_enzymes` — anchored only by HiGen; computing the baseline
   tells us whether our 0.18 degree MMD on the live runs is "training
   not converged" or "bandwidth mismatch".
3. `spectre_planar` — third pinned dataset in `anchors.yaml`.
4. Optional follow-ups if the molecular runs become relevant:
   `qm9`, `moses`, `guacamol`, `comm20` (none currently registered;
   add to `DATASET_BUILDERS` per their respective Hydra configs in
   `src/tmgg/experiments/exp_configs/data/`).

**Cost estimate:** four CPU containers at ~5 min each, parallel —
under $1 of Modal CPU time. Local execution would also work in ~20
minutes total but loses the ORCA binary already-built advantage and
keeps host CPU free for other work.

### Task 2 — Use baselines to ratio-score existing runs

Once `data/eval/mmd_baselines/{dataset}.json` exists for the three
datasets, the SBM repro report at
`wandb_export/sbm-repro-report-2026-05-05/report.typ` should grow a
new section comparing:

| variant | degree raw MMD² | degree ratio (vs train baseline) | DiGress paper ratio |
|---------|----------------:|--------------------------------:|--------------------:|
| vignac  | 0.0335 (=0.183²) | r = 0.0335 / baseline_sbm_degree² | 1.6 |
| ...     |                  |                                  |     |

Concretely, in `scripts/sbm_repro_report.py`:

1. Load `tmgg.evaluation.load_baseline("spectre_sbm")`.
2. For each variant's latest summary, square the W&B-logged
   `gen-val/*_mmd` values and call
   `tmgg.evaluation.compute_ratios(squared, baseline)`.
   - Caveat: our `compute_mmd` returns the V-statistic *already
     squared*. Re-confirm by reading `mmd_metrics.py` lines 446–460.
     If `gen-val/*_mmd` is also already squared, no squaring needed.
     This is the single most important thing to verify before
     publishing ratios — getting it wrong gives sqrt-scale errors.
3. Add a "DiGress-comparable ratio" table to the Typst report.

Repeat for the enzyme repro runs (no Typst report yet; could write a
small `scripts/enzymes_repro_report.py` if useful, or just emit a
markdown summary).

**Important sanity check:** if the ratios for the working runs land in
the [1, 5] range, we are likely calibrated. If they are 100×+, the
units are wrong somewhere — either `gen-val/*_mmd` is `sqrt(MMD²)`
instead of `MMD²`, or the kernel bandwidth differs from what DiGress
used at evaluation time, or our train/test split differs from what
HiGen used (HiGen reports DiGress numbers from a custom run, so this
is plausible).

### Task 3 — Hyperparameter audit (DiGress + HiGen)

The user explicitly asked to **double-check the hyperparameters used
by DiGress and HiGen for computing MMDs**. Yesterday's quick scan
confirmed the SBM path matches upstream; this task does the audit
properly across all four metrics and both papers.

**Sources to compare:**

| Source | Path | What to read |
|--------|------|--------------|
| Our pipeline | `src/tmgg/evaluation/mmd_metrics.py`, `graph_evaluator.py:compute_orbit_mmd` | `gaussian_tv`, `degree_sigma=1.0`, `clustering_sigma=0.1`, `spectral_sigma=1.0`, `orbit_sigma=30.0`, `clustering_bins=100`, `spectral_bins=200`, V-statistic (biased) |
| DiGress upstream | `digress-upstream-readonly/src/analysis/spectre_utils.py` lines 36–490 (degree/clustering/spectral/orbit `_stats` functions) | Confirmed lines 72, 270, 335, 490: all use `gaussian_tv` when `compute_emd=False`. Clustering uses `sigma=1.0/10=0.1`. Orbit uses `sigma=30.0, is_hist=False`. **Note:** `orbit_stats_all` line 488 uses `kernel=gaussian` (not `gaussian_tv`) when `compute_emd=True`. Verify which path the DiGress paper Table actually used. |
| DiGress paper | `arXiv:2209.14734v3`, Appendix F.1 | Verbatim quote: "We do not report raw numbers but ratios computed as follows: r = MMD(generated, test)² / MMD(training, test)²". Squared. Confirms ratio convention. **Open question:** does the paper specify which kernel? Last session did not check; pull the appendix tomorrow. |
| HiGen repo | `github.com/Karami-m/HiGen_main` | Reads from GraphRNN/GRAN MMD metrics (per the README: "structure-based metrics from GRAN ... are employed"). The repo has no DiGress runner — Mahdi's DiGress baseline was a separate run. We do NOT have his exact eval config. |
| HiGen paper | `arXiv:2305.19337` | Reports raw MMD on enzymes (degree 0.004, clustering 0.083, orbit 0.002 for "DiGress" row). **Verify in tomorrow's audit:** does HiGen specify the kernel/sigmas, or just say "GRAN protocol"? GRAN's metrics ship in the GraphRNN repo at `github.com/JiaxuanYou/graph-generation`. |
| GRAN reference impl | Cloned somewhere? Or fetch fresh from `lrjconan/GRAN` per HiGen README | Compare GRAN's `evaluation/eval_helper.py` against ours. This is the ground truth for the "GraphRNN/GRAN convention" both DiGress and HiGen claim to follow. |

**Specific things to check:**

1. **Batch size at evaluation.** Our pipeline computes MMD over
   `eval_num_samples=40` generated graphs (set in
   `configs/models/discrete/discrete_sbm_official.yaml:66`) vs all
   train graphs. Does DiGress use the full test set against
   `samples_to_generate` generated graphs? Upstream's
   `general/general_default.yaml` sets `final_model_samples_to_generate`
   per dataset; for SBM `experiment/sbm.yaml:11` shows
   `samples_to_generate: 40` — matches us at validation, but final
   evaluation uses 40 too (line 14). ENZYMES does not have an
   upstream config so we cannot confirm there directly.

2. **Test set vs validation set as reference.** Upstream
   `spectre_utils.py` is called from `diffusion_model_discrete.py`
   with the **test** set as `graph_ref_list`. Our `gen-val/*_mmd`
   keys evaluate against the **val** set. This is a definitional
   difference: `gen-val/degree_mmd` is `MMD²(gen, val)` not
   `MMD²(gen, test)`. The baseline we are computing is
   `MMD²(train, test)` — so the ratio mixes val and test denominators.
   Two fixes:
   - Compute baselines for both `MMD²(train, val)` and
     `MMD²(train, test)` and pick the right one per W&B key.
   - Or audit when in training the `gen-val/*_mmd` keys are emitted
     and confirm the reference set; check
     `src/tmgg/training/orchestration/run_experiment.py` and
     callbacks. **High priority: do this before reporting any
     ratios.**

3. **Spectral histogram range.** Our `compute_spectral_histogram`
   uses `(-1e-5, 2.0)` with 200 bins (line 195 in
   `mmd_metrics.py`). DiGress upstream `spectre_utils.py:spectral_stats`
   (around line 234) — read the bin range there. If different, our
   spectral MMD numbers are calibrated differently.

4. **Degree histogram support.** Our `compute_degree_histogram` uses
   `np.arange(0, max_d + 2)` (data-dependent). DiGress upstream
   `degree_stats`: read lines 36–73. If they use a fixed support
   while we use data-dependent, the kernel TV distance differs.

Output of the audit: a short comparison table in `docs/` (e.g.
`docs/eval/2026-05-06-mmd-protocol-audit.md`) listing **for each of
the four metrics** the kernel, sigma, bin/range config, sample size,
reference set (train/val/test), and statistic (biased V vs unbiased
U) used by (a) us, (b) DiGress upstream, (c) HiGen repo / paper.
Flag every divergence.

## Pointers and pre-existing context

Already-validated reference material the next agent should read
before touching anything:

| Path | Why |
|------|-----|
| `docs/experiments/sweep/smallest-config-2026-04-29/methodology.md` §2.3 | Establishes the unit-mismatch reasoning between DiGress's MMD-ratio reporting and our raw MMD pipeline. The `anchors.yaml` decision to use HiGen's raw numbers as the SBM anchor is recorded there. |
| `docs/experiments/sweep/smallest-config-2026-04-29/anchors.yaml` | Per-metric anchor targets and source citations. The full quote from DiGress Appendix F.1 lives at the top of the `spectre_sbm` block. |
| `src/tmgg/evaluation/mmd_metrics.py` lines 393–460 | `compute_mmd` docstring explicitly notes "biased V-statistic ... matches upstream DiGress". This is load-bearing for ratio comparability. |
| `src/tmgg/evaluation/graph_evaluator.py` lines 173–213 | `compute_orbit_mmd` defaults: `kernel="gaussian_tv"`, `sigma=30.0`. ORCA binary required (Modal image has it; host does not). |
| `digress-upstream-readonly/src/analysis/spectre_utils.py` | Authoritative DiGress eval source. Search for `compute_mmd(` — every call site reveals kernel + sigma + is_hist for each metric. |
| `digress-upstream-readonly/configs/experiment/sbm.yaml` | DiGress upstream SBM hyperparameters: 50000 epochs, batch 12, 40 samples to generate at val, 1000 diffusion steps. |
| `wandb_export/sbm-repro-report-2026-05-05/report.typ` | Today's frozen panel of the five SBM repro runs. The "Like-to-like comparison at common eval steps" tables list the raw MMD values that need ratioing. |
| `src/tmgg/modal/_profile_functions.py` | Template for tomorrow's `tmgg-mmd-baselines` Modal app. Copy-paste-friendly. |
| `scripts/profiling/launch_profile.py` | Template for the parallel-spawn launcher. |
| `src/tmgg/modal/_lib/image.py` | Confirms ORCA binary is in the Modal image (search `_ORCA_SOURCE_RELATIVE_DIR`). |

## Standard procedure (tomorrow)

```bash
# 0. Sanity check the new code (already imports clean):
uv run python -c "from tmgg.evaluation import load_baseline, compute_ratios; print('OK')"

# 1. Local dry-run on one dataset to confirm the script works end-to-end
#    BEFORE wasting Modal time. spectre_sbm fixture is the smallest:
uv run scripts/compute_mmd_baselines.py --dataset spectre_sbm 2>&1 | tee /tmp/mmd-sbm.log
ls data/eval/mmd_baselines/spectre_sbm.json
cat data/eval/mmd_baselines/spectre_sbm.json | jq '.mmd_squared'

# 2. If (1) succeeds, build the Modal app sketched above and deploy:
doppler run -- uv run modal deploy -m tmgg.modal._mmd_baseline_functions

# 3. Spawn one function per dataset in parallel (write a launcher under
#    scripts/baselines/ similar to scripts/profiling/launch_profile.py).
doppler run -- uv run python -m scripts.baselines.launch_baselines

# 4. After the spawns finish, pull results back:
modal volume get tmgg-outputs /mmd_baselines ./data/eval/

# 5. Verify all expected files landed and fingerprints agree:
ls data/eval/mmd_baselines/
uv run python -c "
from tmgg.evaluation import load_baseline
for ds in ['spectre_sbm', 'spectre_planar', 'pyg_enzymes']:
    b = load_baseline(ds)
    print(f'{ds}: train={b.n_train} test={b.n_test} fp={b.params.fingerprint()}')
    print(f'  mmd²: {b.mmd_squared}')
"

# 6. Update scripts/sbm_repro_report.py to emit ratio columns.
#    Re-run it, regenerate the report.

# 7. Run the hyperparameter audit (Task 3) — produces
#    docs/eval/2026-05-06-mmd-protocol-audit.md.

# 8. If any divergence found in audit, decide: do we re-emit baselines
#    under the corrected kernel/sigma, or accept the gap and document it?
```

## Risks and pitfalls

- **Squared vs unsquared confusion.** Our `compute_mmd` returns the V-
  statistic *already squared*, but the variable name `mmd` reads as
  "MMD" (i.e. unsquared). Same ambiguity in W&B keys (`gen-val/*_mmd`).
  The cache file stores both forms with explicit names (`mmd_squared`,
  `mmd`) to defuse this. **Before writing any ratios, verify which one
  the W&B value actually is.** Read `compute_mmd_metrics` and trace
  what gets logged in the training callback.
- **Train↔test vs train↔val asymmetry.** As above (Task 3 item 2).
  Picking the wrong reference set silently shifts ratios.
- **HiGen's DiGress numbers are non-reproducible from public artefacts.**
  Even with the audit done, our ratios will only be comparable to
  *DiGress's own* paper (SBM) and conditionally to HiGen's reproduction
  (ENZYMES) under the assumption that HiGen used GRAN-default kernels.
  If our audit reveals our defaults already match GRAN, that
  assumption is well-grounded. If not, ENZYMES anchor uncertainty
  remains.
- **Modal CPU containers may not match local CPU containers
  numerically.** ORCA is deterministic, but parallel histogram
  computation in `compute_graph_statistics` is not (ThreadPoolExecutor
  ordering). Pass `--max-workers 1` to the script in the container to
  rule this out as a confounder when comparing across runs.
- **Don't commit `data/eval/mmd_baselines/*.json` to the repo.** The
  global `.gitignore` already excludes `/data/`, so they stay local.
  If we want them shareable, put them in `wandb_export/` instead and
  document. Per the user's repo conventions (CLAUDE.md), specs/plans
  from superpowers stay out of committed history.

## Open questions for the user

1. Should the baselines live in `data/eval/mmd_baselines/` (gitignored,
   regenerable) or `wandb_export/mmd_baselines/` (shared with team via
   sync)? Default in current code is the former; flip via `--out`.
2. Add `comm20`, `qm9`, `moses`, `guacamol` to the dataset registry
   tomorrow, or only when those repros become active?
3. After the audit lands, if we discover a kernel-bandwidth mismatch,
   do we (a) re-emit `gen-val/*_mmd` from the W&B run history under
   corrected bandwidths (re-running `evaluate_cli.py` against saved
   checkpoints), (b) accept the gap as documented uncertainty, or
   (c) treat the audit as advisory and only fix forward for new runs?

## Closing the loop

When the ratio columns land in the SBM repro report and the enzyme
repro report (or summary), the original question — "are these runs
actually as bad as the raw MMDs suggest, or are we comparing apples
to oranges?" — gets a quantitative answer. The current best guess
from yesterday is "training is far from converged" but the ratio
gives us an independent measurement under the paper's reported units.
