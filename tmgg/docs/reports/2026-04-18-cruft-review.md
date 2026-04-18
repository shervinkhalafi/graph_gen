# 2026-04-18 — Cruft Review

Critical pass over the tmgg codebase for duplicate code, LLM-isms, redundant
files, dead configs, and stale docs. Scope is `/tmgg/` only; sibling
directories (`baseline_codes/`, `merge-main/`, `denoising/`, etc.) are out of
scope. Code quality and architecture are excluded except where they produce
code volume.

Findings are ranked by cost/benefit. "Safe to delete" means the file has zero
inbound references from Python, launchers, or other configs and is either
superseded or ephemeral. "Consolidate" means the content is load-bearing but
duplicated.

## Verdict at a glance

The repo is in decent shape but carries measurable cruft in four categories:

1. **Copy-paste launchers and debug wrappers** at the repo root — worst concentration of duplicate shell code (≈4 DEBUG wrappers of 15 lines each, differing by one Hydra override).
2. **Archival grid-search configs** — `configs/` root tree carries ~1,300 files in stage-variant directories that no launcher or Python code references.
3. **Dated reports / plans never moved out of root** — five `.md` files at the tmgg root predate the `docs/reports/` convention and should migrate or be deleted.
4. **Scaffolded docstring / comment bloat** — repeated "mirrors upstream DiGress" justifications (15+ occurrences, 6+ files), one 39-line module docstring with an ASCII table that should be a `NamedTuple`, and a 72-line class docstring.

One orphaned experiment (`modified_attention`) has live model code but no
launcher, no base-config references, and no config consumers. Either finish
wiring or delete.

No duplicate data-generation code, no duplicate MMD implementations, no dead
model families. Core src/ is clean.

## 1. Code duplication (src/tmgg)

### 1.1 `_ListDataset` — inlined copy

`src/tmgg/data/data_modules/spectre_sbm.py:204-214` reimplements the exact
`_ListDataset` wrapper defined in
`src/tmgg/data/data_modules/multigraph_data_module.py:67-77`. The spectre file
even carries a comment saying "Re-use the list-wrapper from
multigraph_data_module rather than import the private symbol" — then
reimplements it instead of importing. This is the single clearest duplicate in
the Python code.

- **Fix**: promote `_ListDataset` to a shared module (e.g.
  `tmgg/data/data_modules/_list_dataset.py`) or drop the underscore prefix and
  import. 12 LOC recovered, single maintenance point.

### 1.2 Datamodule dataloader scaffolding

All six datamodule classes repeat the `train_dataloader` / `val_dataloader` /
`test_dataloader` trio. The boilerplate is already abstracted via
`_make_dataloader`; what remains is Lightning's protocol demand, not cruft.
**No action.**

### 1.3 Denoising runner stubs

Five runner modules under `src/tmgg/experiments/*_denoising/runner.py` are
each ≤35 lines and delegate to `run_experiment(cfg)` with a different Hydra
config. This is thin-entrypoint boilerplate, not duplication — the configs
they bind are where the actual variance lives. **No action.**

### 1.4 No SBM / MMD / model duplication

- `generate_sbm_adjacency` and `generate_sbm_batch` exist once in
  `src/tmgg/data/datasets/sbm.py`.
- MMD implementation lives once in `evaluation/mmd_metrics.py` (524 lines);
  `graph_evaluator.py` is an aggregator, not a second implementation.
- Model families (spectral, GNN, attention, transformer, hybrid) are
  well-separated; each variant is actively used by a corresponding experiment.

## 2. Dead / orphaned code and configs

### 2.1 `modified_attention` experiment is orphaned

| Path | Status |
|------|--------|
| `src/tmgg/experiments/exp_configs/base_config_mod_attention.yaml` | zero references |
| `src/tmgg/experiments/exp_configs/models/modified_attention/mod_attention.yaml` | zero references |
| `src/tmgg/models/mod_attention.py` (380 LOC) | model code exists |

No launcher (`run-*.zsh`) mentions `mod_attention` or `modified_attention`.
Grep found zero `.zsh` hits.

- **Fix**: either wire into a launcher / experiment panel, or delete the three
  files above. Keep one, delete the other two only after confirming no
  downstream dependency.

### 2.2 `configs/` root tree carries ~1,300 archival files

`configs/` at the repo root is a 3,126-file machine-generated grid-search
archive split across 10 stage directories. Only `stage2c/`,
`stage3_pyg_dist/`, and `stage3_pyg_dist_relaunch/` are referenced from
`wandb-tools/` post-hoc analysis scripts. The rest (`stage1`, `stage1b`,
`stage1c`, `stage1d_asymmetric`, `stage1f_digress_spectral`, `stage2`,
`stage2b`) are unreferenced from code, launchers, and tests.

- **Fix**: move `stage1*` + `stage2b/` to `configs/_archive/` (or delete
  outright if results are persisted in W&B). ~1,328 files recovered.

## 3. Root-level cruft

### 3.1 DEBUG launcher clones (highest-density cruft)

Four files at the repo root:

```
DEBUG-run-upstream-digress-sbm-modal-no-viz.zsh         (370 bytes)
DEBUG-run-upstream-digress-sbm-modal-skip-orbit.zsh     (327 bytes)
DEBUG-run-upstream-digress-sbm-modal-skip-sbm.zsh       (337 bytes)
DEBUG-run-upstream-digress-sbm-modal-skip-sbm-orbit.zsh (390 bytes)
```

Verified by diff: each is a 15-line wrapper that differs from its siblings by
exactly one Hydra override and one `print` comment. Pure copy-paste.

- **Fix**: collapse into one `DEBUG-run-upstream-digress-sbm-modal.zsh` that
  takes a `--skip` flag (`no-viz`, `orbit`, `sbm`, `sbm+orbit`). 3 files
  recovered.

### 3.2 Launcher scripts with untracked siblings

- `run-upstream-digress-sbm-modal.zsh` (tracked, 93 lines) is the canonical
  launcher.
- `run-upstream-digress-sbm-modal-a100.zsh` (**untracked**, 24 lines) is an
  instance-type variant.
- `run-denoising-sbm-panel-a10g.zsh` (**untracked**, 109 lines) is a different
  panel.

Untracked launchers are either in-progress drafts or drift. Either commit or
fold the instance-type delta into a flag on the parent script.

### 3.3 Dated reports and session logs at root

| File | Age | Referenced | Fate |
|---|---|---|---|
| `2026-02-18-audit-2.md` | 2 months | no | move to `docs/reports/` or delete |
| `2026-02-18-digress-report.md` | 2 months | no | move to `docs/reports/` or delete |
| `distributional_audit_log.md` | 2 months | no | move to `docs/reports/` or delete |
| `section4_experiments.md` | 3 months | no | delete — paper-section notes |
| `section4_implementation_plan.md` | 3 months | no | delete — paper-section notes |
| `tea_debug.log` | 2 days, empty | no | delete and gitignore |
| `.coverage` (86 KB, tracked) | 2 months | no | delete, add to gitignore |

### 3.4 Ambiguous utilities

- `calculate_equal_params.py` (4 months old) — SBM parameter calculator, no
  current callers. Either fold into `scripts/` with a README note or delete.
- `integration_test.py` (5 weeks old) — standalone test; does not follow
  `tests/` layout. Relocate or delete.
- `sync_wb` — referenced from `.modalignore`; **keep**.

### 3.5 Lock file confusion

Three package-lock mechanisms coexist:

- `uv.lock` (Mar 16) — current, `uv`-managed.
- `requirements.lock` (5 weeks old) — stale.
- `requirements-dev.lock` (5 weeks old) — stale.

`uv.lock` is the source of truth given the CLAUDE.md guidance ("Use uv for
dependencies"). Delete the two `.lock` files or document their purpose.

## 4. Docs cruft (docs/)

### 4.1 Unstaged accumulation

The working tree has 9 untracked report / plan files:

```
docs/how-to-run-experiments.md
docs/reports/2026-03-06-post-class-cleanup/ (6 files)
docs/reports/2026-03-12-tmgg-review/ (6 files)
docs/reports/2026-03-16-redundancy-audit-a3.md
docs/reports/2026-04-05-datamodule-noise-interface-review.md
docs/reports/2026-04-13-datamodule-noise-interface-execution-plan.md
docs/reports/2026-04-15-upstream-digress-parity-audit.md
docs/superpowers/
```

Untracked documentation is cruft by default: unreachable, unindexed. Decide
per-item: commit the load-bearing ones (the April parity audit and noise
interface review are high-value), delete the rest.

### 4.2 Superseded / mergeable reports

- `docs/reports/2026-03-16-redundancy-audit-a3.md` (53 lines) — narrow,
  superseded by later audits. **Delete.**
- `docs/reports/2026-04-15-debugging-log.md` (490 lines, tracked) overlaps with
  `docs/reports/2026-04-15-bug-modal-sigabrt.md` (243 lines). Merge the
  transcript into the bug report under "Investigation notes" and drop the
  session log.

### 4.3 Plans directory — ~10 executed plans

All plans dated before 2026-04-15 correspond to completed work visible in git
log (eliminate-config-pipeline, simplify-modal-execution, fix-layer-violations,
training-loop-unification, models-cleanup, datamodule-noise-sampler waves
1-5, hydra-data-interpolation-refactor, etc.). Active plan set is limited to
the April 15 unified-graph-features spec + plan.

- **Fix**: move executed plans to `docs/plans/archive/`. Keeps the top-level
  `docs/plans/` scoped to what's actually in flight.

### 4.4 Top-level docs — one real overlap

`get-started.md` and the untracked `how-to-run-experiments.md` both cover CLI
basics. `experiments.md` also overlaps. Intent isn't clear from filenames
alone.

- **Fix**: define a doctype split (e.g. `get-started.md` = 5-minute quickstart,
  `how-to-run-experiments.md` = full CLI reference, `experiments.md` = the
  experiment-system design) and prune overlap. Or delete `get-started.md` and
  let the README handle onboarding.

### 4.5 `docs/superpowers/` — ambiguous

Untracked, contains 5 MD files of design notes. No link in from any other doc.
Either commit + integrate into `docs/` navigation or delete.

### 4.6 Other doc dirs

- `docs/bugs/stage1_missing_runs.md` — from 2026-01-05, resolved same day.
  Delete or archive.
- `docs/eigenstructure_study/` — valid research artifact, keep.
- `docs/reference/` — well-organized, no overlap, keep.

## 5. LLM-isms in Python source

Not pervasive but concentrated in a few files. The strongest signals:

### 5.1 "Upstream DiGress" re-justification (15+ occurrences)

Same reason explained three to four times per module. Worst offenders:

- `src/tmgg/diffusion/noise_process.py` — "Mirrors upstream DiGress" appears
  at lines 1066, 1128, 1164 (three adjacent docstrings, each restating the
  same parity rationale).
- `src/tmgg/evaluation/mmd_metrics.py` — upstream MMD justification repeated
  at 351, 405, 424.
- `src/tmgg/data/datasets/spectre_sbm.py` — module docstring (line 3),
  split docstring (line 28), and constants comment (line 64) all point back
  to upstream.

- **Fix**: state the parity principle once, per module if needed, and drop
  per-function reiterations. Rough estimate: 40-60 lines recoverable.

### 5.2 Docstring bloat

- `src/tmgg/data/datasets/spectre_sbm.py` lines 1-39 — 39-line module
  docstring containing a 13-line ASCII table describing the on-disk tuple
  structure plus a formal References section citing SPECTRE and DiGress
  papers. The ASCII table should be a `NamedTuple` or a comment next to the
  loader; the References section is over-formal for an internal module.
- `src/tmgg/models/spectral_denoisers/base_spectral.py` lines 1-72 — 72-line
  class docstring with exhaustive enumeration of optional PEARL knobs.
  Parameters belong in the parameter signature, not the narrative.

- **Fix**: convert ASCII tables to typed structures, drop internal-module
  References sections, trim parameter enumerations. ~80 lines recoverable
  across the two files.

### 5.3 Module-opener boilerplate ("This module provides/computes/handles…")

9 modules open with this formula: `mmd_metrics.py`, `graph_evaluator.py`,
`pyg_datasets.py`, `embedding_study/__init__.py`, and six others. It's not
harmful but it's a template smell — first sentence of a module docstring
should say the distinctive thing, not repeat the filename.

### 5.4 `from __future__ import annotations` — 75/158 files

Inconsistent coverage (≈47%) on a Python 3.12 project where it's optional.
Not cruft, but a signal of file-by-file generation without a unified pass.
Pick a policy (all or none) and enforce.

### 5.5 Lower-severity signals

- **Em-dashes**: 104 across 59 files. Moderate, not pathological. No action.
- **Explanatory comments** (`# This loops…`): scattered, not systemic. No
  action.
- **`# Rationale:` / `# Note:` / `# Why:` markers**: 25 occurrences across 20
  files, not the worst signal. No action.

## 6. What is clean

Recording this so the review is balanced and so future reviewers know what
was audited without changes needed:

- **Data generation** (`data/datasets/sbm.py`, `graph_generation.py`) — one
  canonical SBM generator, no duplicates.
- **Evaluation** (`evaluation/mmd_metrics.py`, `graph_evaluator.py`) — single
  MMD source, clean aggregation layer.
- **Model architectures** (`models/`) — every variant has a corresponding
  active experiment; no dead families.
- **LightningModules** — `digress_checkpoint_compat.py` looks heavy (494
  lines) but is specialized for upstream checkpoint format translation, not
  cruft.
- **Reference docs** (`docs/reference/`) — organized by subsystem, no overlap.
- **Experiments directory** — thin runner stubs are boilerplate, not
  duplication; configs carry the variance.

## 7. Recommended cleanup order

Ordered by blast-radius-over-benefit (smallest, highest-value first):

1. **Delete root-level cruft**: `tea_debug.log`, `.coverage`, `section4_*.md`,
   legacy `requirements*.lock`. Add to `.gitignore` where relevant. (10
   min, zero risk.)
2. **Collapse DEBUG launcher quartet** into one parametrized wrapper. (15
   min, zero risk.)
3. **Migrate dated root reports** to `docs/reports/`. (15 min, zero risk.)
4. **Stage the untracked docs**: commit the 2026-04-* reports + plans; delete
   the 2026-03-16 redundancy audit; decide `docs/superpowers/` intent. (30
   min.)
5. **Consolidate `_ListDataset`** into a shared module. (10 min, trivial
   diff.)
6. **Decide `modified_attention` fate**: wire or delete (config + model). (20
   min once decision is made.)
7. **Archive `configs/stage1*/` and `stage2b/`** into `configs/_archive/`.
   (10 min, ~1,300 files moved.)
8. **Archive executed plans** under `docs/plans/archive/`. (15 min.)
9. **Trim top LLM-ism offenders**: dedupe "upstream DiGress" justifications in
   `noise_process.py` / `mmd_metrics.py`, collapse the `spectre_sbm.py`
   ASCII table, shorten `base_spectral.py` class docstring. (45 min.)
10. **Resolve doc overlaps**: define scope for `get-started.md` vs
    `how-to-run-experiments.md` vs `experiments.md`. (30 min, needs a
    judgment call.)

Total estimated recovery: ~1,400 files (mostly archival configs), ~150-200
lines of code, a handful of decisions about ambiguous experiments.
