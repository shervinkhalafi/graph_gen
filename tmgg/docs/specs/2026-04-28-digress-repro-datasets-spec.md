# DiGress repro datasets: Planar (SPECTRE) + QM9 + MOSES + GuacaMol

**Date:** 2026-04-28
**Owner:** igork
**Branch target:** `igork/cheap-progress-telemetry` (or a fresh branch
on top, depending on how this lands)
**Vendored upstream reference:**
`/home/igork/900_personal/900.000_research/900.003_shervin_graphgen/digress-upstream-readonly/`
(read-only; we mirror its preprocessing rules in our own code rather
than importing from it)

## Context

We have an SBM repro path
(`tmgg.data.data_modules.spectre_sbm.SpectreSBMDataModule`,
`exp_configs/experiment/discrete_sbm_vignac_repro.yaml`) that
faithfully reproduces DiGress's SPECTRE-SBM experiment and lets us
study training dynamics (see
`docs/reports/2026-04-28-hp-tuning-and-knee-identification/README.md`
and the recently-landed telemetry in branch
`igork/cheap-progress-telemetry`).

To run the full DiGress repro panel — Tables 1-3 (graph-only) and
Tables 4-6 (molecular) of Vignac et al., ICLR 2023 — we need four
new datasets wired through our existing categorical-diffusion
pipeline:

| Dataset | DiGress paper section | Source | Eval metrics |
|---|---|---|---|
| Planar (SPECTRE) | §4.1, Table 1 | `planar_64_200.pt` | degree/clustering/spectral/orbit MMD, planarity_accuracy, uniqueness |
| QM9 (no H) | §4.2, Table 4 | DiGress's QM9 CSV | validity, uniqueness, novelty |
| MOSES | §4.3, Table 5 | molecularsets/moses | validity, uniqueness, novelty, FCD, SNN, IntDiv, Filters, ScaffoldSplit |
| GuacaMol | §4.4, Table 6 | BenevolentAI/guacamol | validity, uniqueness, novelty, KL on physchem properties, FCD on ChEMBL |

The intended outcome is that the existing
`DiffusionModule` + `CategoricalNoiseProcess` pipeline trains all
four datasets unmodified, with a per-dataset experiment YAML
selecting the dataset and (where applicable) the model dimensions
that match each table's published architecture.

## Goal

Produce a **single PR** that adds:

1. The Planar SPECTRE datamodule, mirroring `spectre_sbm.py`.
2. A compositional molecular pipeline (vocabulary → SMILES codec →
   per-dataset Dataset → DataModule → per-metric classes →
   evaluator) that supports QM9, MOSES, and GuacaMol with all of
   each paper's reported metrics.
3. Four new experiment YAMLs slotting into `tmgg-discrete-gen`.
4. One parameterised launcher script
   (`scripts/run-digress-repro-modal.zsh <dataset>`) plus a thin
   backwards-compatible wrapper for the existing SBM script.
5. Modal image bump: `rdkit`, `fcd_torch`, `moses`, `guacamol`
   added as **required** deps in `pyproject.toml`; ChemNet weights
   warmed at deploy time.

Implementation strategy per the user directive: **burn through the
complete implementation end-to-end before writing any tests.** Tests
land in the same PR but as the final phase; the implementation must
be self-consistent (mypy/pyright clean, importable, runnable on a
toy fixture) before the test phase begins.

## Non-goals

- Reproducing DiGress's *training* numbers exactly (their seed/init
  isn't byte-equivalent across PyTorch versions). We target the
  metric *shapes* and within-±5% parity on QM9 (smallest, fastest
  to converge); MOSES + GuacaMol numerical parity is deferred to
  real sweeps post-merge.
- Implementing other DiGress evaluation knobs (conditional
  generation, property prediction). This spec is the
  unconditional-generation panel only.
- Vendoring the upstream code into our import graph. We use it as
  a read-only reference for preprocessing rules and metric formulas;
  every line of new code is ours.
- Optional-extras dependency story. The user chose required deps;
  RDKit + FCD/MOSES/GuacaMol packages are in the base install.

## Architecture: compositional class layout

Slots into existing tmgg hierarchies:
`BaseGraphDataModule(pl.LightningDataModule, abc.ABC)`,
`Dataset[T]` per-dataset subclasses, `GraphEvaluator`-shaped class
with `.evaluate(refs, generated) → flat dataclass.to_dict()`.

```
tmgg.data.datasets.molecular/  (NEW)
  vocabulary.py
    AtomBondVocabulary               # frozen, hashable; classmethod presets
                                     #   .qm9() / .moses() / .guacamol()
  codec.py
    SMILESCodec                      # parameterised by AtomBondVocabulary;
                                     # only place rdkit is imported.
                                     # encode(smi) → GraphData | None
                                     # decode(GraphData) → str | None
                                     # encode_dataset(iter) → iter[GraphData]
  dataset.py
    MolecularGraphDataset            # ABC: Dataset[GraphData].
                                     # Caches preprocessed shards under a
                                     # codec-hashed dir; concrete subclasses
                                     # override download_smiles_split()
                                     # and make_codec().
  qm9.py
    QM9Dataset(MolecularGraphDataset)
  moses.py
    MOSESDataset(MolecularGraphDataset)
  guacamol.py
    GuacaMolDataset(MolecularGraphDataset)

tmgg.data.data_modules.molecular/  (NEW)
  base.py
    MolecularDataModule              # composable base — composes a
                                     # MolecularGraphDataset subclass with
                                     # the existing categorical collator.
                                     # Per-dataset subclasses below pin the
                                     # dataset_cls + dataset-specific defaults
                                     # (max_n_nodes, vocab preset hint, etc.).
  qm9.py
    QM9DataModule(MolecularDataModule)
  moses.py
    MOSESDataModule(MolecularDataModule)
  guacamol.py
    GuacaMolDataModule(MolecularDataModule)

tmgg.data.data_modules.spectre_planar.py (NEW; mirrors spectre_sbm.py)
    SpectrePlanarDataset(Dataset[Data])
    SpectrePlanarDataModule(BaseGraphDataModule)

tmgg.data.datasets.spectre_planar.py (NEW; mirrors datasets/spectre_sbm.py)
    SPECTRE_PLANAR_URL = "https://raw.githubusercontent.com/.../planar_64_200.pt"
    download_spectre_planar_fixture(...)
    load_spectre_planar_fixture(...)
    split_spectre_planar(...)        # mirrors upstream's seed-0 randperm rule

tmgg.evaluation.molecular/  (NEW)
  metric.py
    MolecularMetric                  # ABC, .compute(generated, reference) → float | dict
  rdkit_metrics.py
    ValidityMetric, UniquenessMetric, NoveltyMetric
  moses_metrics.py
    FCDMetric, SNNMetric, IntDivMetric, FiltersMetric, ScaffoldSplitMetric
  guacamol_metrics.py
    KLDivPropertyMetric, FCDChEMBLMetric
  evaluator.py
    MolecularEvaluationResults       # flat dataclass mirroring EvaluationResults
    MolecularEvaluator               # composes [MolecularMetric] + a SMILESCodec.
                                     #   .evaluate(refs: list[GraphData],
                                     #             generated: list[GraphData])
                                     #     → MolecularEvaluationResults
                                     # classmethod presets:
                                     #   .for_qm9() / .for_moses() / .for_guacamol()
```

### Why these boundaries

- **AtomBondVocabulary** is frozen + hashable so it can be a cache
  key for codec output. Per-dataset preset classmethods make
  construction declarative; mutation is impossible.
- **SMILESCodec** confines all RDKit imports to one module. The
  codec hash (SHA-256 of `repr(vocab) + repr(kwargs)`) is the cache
  invalidation key for preprocessed shards.
- **MolecularGraphDataset** is the first place `GraphData` objects
  exist; it caches them on disk, so the per-step cost of training is
  zero RDKit calls.
- **MolecularDataModule** is the composable base; per-dataset thin
  subclasses (`QM9DataModule`, `MOSESDataModule`, `GuacaMolDataModule`)
  pin the `dataset_cls` and dataset-specific defaults (max_n_nodes,
  vocab preset). Mirrors the SPECTRE side's pattern
  (`SpectreSBMDataModule` / `SpectrePlanarDataModule`) so the
  per-dataset surface stays uniform across graph + molecular work.
- **MolecularMetric** is the unit of eval composition. Heavy
  stateful pieces (ChemNet embedder, fingerprint cache) live inside
  the metric class for clear lifetime — they live as long as the
  evaluator does, then go.
- **MolecularEvaluator** mirrors `GraphEvaluator`'s surface so
  `DiffusionModule.on_validation_epoch_end` accepts either via duck
  typing on `.evaluate(refs, generated)` returning a flat
  dataclass with `.to_dict()`.

### How `DiffusionModule` accommodates two evaluator types

`DiffusionModule.evaluator: GraphEvaluator | MolecularEvaluator | None`.
The `.evaluate(refs, generated) → results` contract is the same.
Inside `on_validation_epoch_end`, the existing loop:

```python
for key, value in results.to_dict().items():
    if value is not None:
        self.log(f"gen-val/{key}", value, on_epoch=True)
```

works unchanged. No changes to `DiffusionModule` itself except the
type union.

## File layout (NEW only)

```
src/tmgg/data/
  datasets/
    spectre_planar.py                   # ~120 lines, mirrors spectre_sbm.py
    molecular/
      __init__.py
      vocabulary.py                     # ~150 lines, 3 presets
      codec.py                          # ~250 lines, the only rdkit user
      dataset.py                        # ~200 lines, ABC + cache logic
      qm9.py                            # ~80 lines
      moses.py                          # ~100 lines
      guacamol.py                       # ~100 lines
  data_modules/
    spectre_planar.py                   # ~150 lines, mirrors spectre_sbm.py
    molecular/
      __init__.py
      base.py                           # ~120 lines, MolecularDataModule
      qm9.py                            # ~30 lines, QM9DataModule
      moses.py                          # ~30 lines, MOSESDataModule
      guacamol.py                       # ~30 lines, GuacaMolDataModule

src/tmgg/evaluation/
  molecular/
    __init__.py
    metric.py                           # ~50 lines, ABC
    rdkit_metrics.py                    # ~150 lines, 3 metrics
    moses_metrics.py                    # ~400 lines, 5 metrics + ChemNet wrapper
    guacamol_metrics.py                 # ~250 lines, 2 metrics
    evaluator.py                        # ~200 lines, composer + presets

src/tmgg/experiments/exp_configs/
  data/
    spectre_planar.yaml                 # mirrors spectre_sbm.yaml
    qm9_digress.yaml
    moses_digress.yaml
    guacamol_digress.yaml
  experiment/
    discrete_planar_digress_repro.yaml
    discrete_qm9_digress_repro.yaml
    discrete_moses_digress_repro.yaml
    discrete_guacamol_digress_repro.yaml

scripts/
  run-digress-repro-modal.zsh           # parameterised: $1=dataset; uses model.
                                        # eval_every_n_steps + max_steps from env

# Existing wrapper kept for backward compat:
run-discrete-sbm-vignac-repro-modal-a100.zsh  # → calls
                                              # scripts/run-digress-repro-modal.zsh sbm

tests/
  data/
    test_atom_bond_vocabulary.py
    test_smiles_codec.py
    test_molecular_dataset.py
    test_molecular_datamodule.py
    test_spectre_planar_datamodule.py
  evaluation/
    test_molecular_metrics.py
    test_molecular_evaluator.py
  training/
    test_diffusion_module_molecular.py  # slow-marked
  modal/
    test_molecular_image.py             # slow-marked + modal-marked
```

## Dependencies (added to `pyproject.toml`)

```toml
[project]
dependencies = [
    # ...existing...
    "rdkit>=2024.3",        # SMILES parsing, validity check, canonical form
    "fcd_torch>=1.0.7",     # ChemNet embedder for FCD
    "moses>=0.4",           # MOSES Filters + IntDiv + SNN + ScaffoldSplit
    "guacamol>=0.5",        # GuacaMol Distribution-Learning benchmarks + ChEMBL ref
]
```

Cold install: ~500 MB additional disk, ~3-5 min on a clean venv.
ChemNet weights (~50 MB) downloaded once on first FCD call to
`~/.cache/tmgg/molecular/chemnet/`.

Modal image: the existing `experiment_image` definition in
`src/tmgg/modal/_functions.py` already does
`uv pip install --system -e /app/tmgg`, so the new deps land
automatically. Add a one-shot deploy-time hook
`_warm_molecular_caches_in_image` that pulls ChemNet weights so
training containers don't pay the download cost on the first
validation cycle.

## Data flow + caching

```
prepare_data():    download → ~/.cache/tmgg/<dataset>/raw/<files>
setup():           SMILES → SMILESCodec → ~/.cache/tmgg/<dataset>/preprocessed/<codec_hash>/<split>.pt
__getitem__():     read shard → GraphData (no rdkit calls)
DataLoader:        existing categorical collator (unchanged)
```

**Cache invalidation by hash.** The shard directory is keyed by
SHA-256 of `repr(vocab) + repr(codec_kwargs)`. Any change to
vocabulary or codec parameters re-preprocesses from scratch
automatically. The codec's `__repr__` is stable across runs; we
write a unit test that asserts that.

**Shard format.** `torch.save([GraphData, ...])` per shard.
`torch.load(..., weights_only=False)` is required to restore
`GraphData` (our own dataclass, not a `state_dict`). Per the
project security memory the rule "never `weights_only=False` on
third-party files" applies — these are first-party files we wrote
ourselves; the codec's docstring + this spec document the
exception.

**Shard sharding.** MOSES (~2M mols, ~3 GB) and GuacaMol
(~1.6M mols, ~7 GB) are split into ≤ 50k-mol shards so a single
shard fits in <1 GB RAM and supports concurrent dataloader workers.
QM9 (~134k mols, ~60 MB) gets one shard per split.

**Modal volume strategy.** Reuse the existing `tmgg-datasets`
volume (mounted at `/data/datasets` per
`tmgg.modal._lib.volumes.DATASETS_MOUNT`). Preprocessed shards
land at `/data/datasets/<dataset>/preprocessed/<codec_hash>/<split>.pt`
on Modal; raw downloads land at `/data/datasets/<dataset>/raw/`.
The local cache mirror at `~/.cache/tmgg/<dataset>/` is used only
for host-side dev runs. No new volume needed; no changes to
`tmgg.modal._lib.volumes` required.

## Error handling

- **SMILES parse failure** in `SMILESCodec.encode`: return `None`,
  drop the molecule, increment a counter logged at preprocessing end.
- **Valence violation** in `SMILESCodec.decode` (used by
  `ValidityMetric`): return `None`, count as invalid.
- **Atom count > `max_atoms`**: drop with counter; per-dataset
  `max_atoms` is a class attribute on the dataset subclass.
- **Network failure on download**: 3× retry with exponential
  backoff, then raise `RuntimeError` naming the URL. Modal users
  hit the volume cache after the first run.
- **ChemNet weights missing**: downloaded on first
  `FCDMetric.compute()` call; cached under
  `~/.cache/tmgg/molecular/chemnet/`. Same retry policy.
- **rdkit / moses / guacamol import failure** at module import:
  raises an `ImportError` with the actionable message
  "molecular deps missing — `uv pip install -e .` to install."
  We chose required-deps so this only triggers if someone broke
  their install.

## Phasing — implement first, test last

**Phase 0 — Modal image + deps (½ day).**

Add the four new packages to `pyproject.toml`. Verify the existing
`uv pip install -e .` runs cleanly. Add the
`_warm_molecular_caches_in_image` deploy hook. Sanity-check the
Modal image build produces a working RDKit + ChemNet on the
container (`python -c "from rdkit import Chem; ..."`).

**Phase 1 — SPECTRE-Planar datamodule (½ day).**

`tmgg.data.datasets.spectre_planar` (download/load/split helpers
mirroring `spectre_sbm`) and
`tmgg.data.data_modules.spectre_planar` (the DataModule class).
The fixture `planar_64_200.pt` is already at
`~/.cache/tmgg/spectre/planar_64_200.pt` (sha256
`063dc3e675a5c63144e56aa974ca961abfcba02914368192a09b21a364df38fc`,
6.6 MiB, downloaded 2026-04-28). Add the experiment YAML
`discrete_planar_digress_repro.yaml`. The existing `GraphEvaluator`
already has `planarity_accuracy` + the MMDs; no eval changes
needed. Smoke-runnable on Modal at end of phase, but no test
files written yet.

**Phase 2 — Molecular vocabulary + codec (1 day).**

`AtomBondVocabulary` with three preset classmethods (constants
mirrored from upstream's `qm9_dataset.py:atom_decoder`,
`moses_dataset.py:atom_decoder`, `guacamol_dataset.py:atom_decoder`).
`SMILESCodec` with the encode/decode round-trip. RDKit-only; no
caching, no datasets yet. Round-trip a 100-mol QM9 sample manually
to confirm canonical-SMILES match rate before moving on.

**Phase 3 — Molecular Dataset + DataModule (1 day).**

`MolecularGraphDataset` ABC with on-disk shard cache.
`QM9Dataset`, `MOSESDataset`, `GuacaMolDataset` subclasses.
`MolecularDataModule` generic class. Three data YAMLs.
`prepare_data` + `setup` work end-to-end on QM9 (smallest);
preprocessing wall-clock and dropped-mol counts captured.

**Phase 4 — Molecular metrics (2 days).**

`MolecularMetric` ABC. RDKit metrics
(`Validity`/`Uniqueness`/`Novelty`). MOSES metrics
(`FCD`/`SNN`/`IntDiv`/`Filters`/`ScaffoldSplit`) — uses `fcd_torch`
+ `moses` package. GuacaMol metrics (`KLDivProperty`,
`FCDChEMBL`) — uses `guacamol` package. Each metric is a class
with a `.compute(generated, reference)` method.

**Phase 5 — MolecularEvaluator + experiment YAMLs (1 day).**

Compose metrics into `MolecularEvaluator`. Three classmethod
presets (`for_qm9`, `for_moses`, `for_guacamol`). Three experiment
YAMLs (`discrete_qm9_digress_repro`, `discrete_moses_digress_repro`,
`discrete_guacamol_digress_repro`) wiring the matching
DataModule + Evaluator.

**Phase 6 — Launcher (½ day).**

`scripts/run-digress-repro-modal.zsh <dataset>` parameterised
wrapper. Existing
`run-discrete-sbm-vignac-repro-modal-a100.zsh` becomes a thin
backwards-compatible wrapper that calls
`./scripts/run-digress-repro-modal.zsh sbm "$@"`. Four new
matching wrappers for symmetry.

**Phase 7 — Testing (1-2 days).**

Now write all tests under `tests/` per the testing-strategy table
above. `pytest -m "not slow"` is the fast lane; FCD + Modal
preflight tests carry `@pytest.mark.slow` and `@pytest.mark.modal`.

**Phase 8 — Validation runs (1-2 GPU-days).**

Per the validation criteria:

1. Train-set parity check on each molecular dataset (encode/decode
   round-trip ≥ 99% canonical-SMILES match on 1k random training
   mols). Pure CPU; runs in CI as a slow-marked test.
2. End-to-end smoke on Modal for each of the four new datasets:
   1k training steps + one validation cycle. No NaN/Inf in any
   `gen-val/*` metric. Logs of all four reach W&B.
3. QM9 short-budget parity: train QM9 for 50k steps; assert
   `gen-val/validity` lands within ±5% of DiGress Table 4's
   reported number. (±5% rather than ±2% to cover unavoidable
   PyTorch-version / CUDA-determinism drift.) MOSES + GuacaMol
   numerical parity is deferred — they're days-long runs that
   belong to real sweeps post-merge.

**Total estimate:** ~8-10 engineering days end-to-end before
testing + ~1-2 days of testing + ~1-2 GPU-days for validation runs.

## Validation criteria summary

The PR is mergeable when:

- All four new experiment YAMLs train end-to-end on Modal without
  modifying `DiffusionModule` or `CategoricalNoiseProcess`.
- Each evaluator emits the metric set DiGress reports for its
  dataset; key set is checked against an expected set in a unit
  test (`test_molecular_evaluator::test_for_qm9_metric_keys`,
  etc.).
- Train-set encode/decode round-trip ≥ 99% canonical-SMILES match
  on each molecular dataset.
- QM9 final-step `gen-val/validity` within ±5% of DiGress Table 4
  on the published config and a 50k-step budget.
- `uv run pytest tests/ -x -m "not slow"` passes (≤ 30s).
- `uv run pytest tests/ -m "slow"` passes (≤ 5 min, includes FCD
  ChemNet download).
- Documented namespace mapping in the spec — every metric's W&B
  key is named explicitly so downstream dashboard work has a
  contract.

## Launcher script template

```zsh
# scripts/run-digress-repro-modal.zsh
#!/usr/bin/env zsh
set -euo pipefail

: "${USE_DOPPLER:=1}"
: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=fast}"
: "${PRECISION:=bf16-mixed}"
: "${MODAL_DEBUG:=0}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/tmgg-mpl-cache}"
mkdir -p "${MPLCONFIGDIR}"; export MPLCONFIGDIR

DATASET="${1:?usage: $0 <sbm|planar|qm9|moses|guacamol> [hydra-overrides...]}"
shift

case "$DATASET" in
  sbm)        EXP="discrete_sbm_vignac_repro" ;;
  planar)     EXP="discrete_planar_digress_repro" ;;
  qm9)        EXP="discrete_qm9_digress_repro" ;;
  moses)      EXP="discrete_moses_digress_repro" ;;
  guacamol)   EXP="discrete_guacamol_digress_repro" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

run_prefixed() {
  if [[ "${USE_DOPPLER}" == "1" ]]; then doppler run -- "$@"; else "$@"; fi
}

[[ "${DEPLOY_FIRST}" == "1" ]] && run_prefixed mise run modal-deploy

if [[ "${MODAL_DEBUG}" == "1" ]]; then md=true; else md=false; fi

typeset -a cmd
cmd=(
  uv run tmgg-modal run tmgg-discrete-gen
  +experiment="${EXP}"
  trainer.precision="${PRECISION}"
  modal_debug="${md}"
  --gpu "${GPU_TIER}"
)
[[ "${DETACH}" == "1" ]]   && cmd+=(--detach)
[[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
(( $# > 0 ))               && cmd+=("$@")

print -r -- "Launching DiGress repro: dataset=${DATASET} (${GPU_TIER}, ${PRECISION})"
printf ' %q' "${cmd[@]}"; print
run_prefixed "${cmd[@]}"
```

## Open questions to resolve during implementation

- **Atom vocabulary for "no-H" QM9.** Upstream uses `(C, N, O, F)`;
  charges are folded into atom classes if present. Confirm against
  `digress-upstream-readonly/src/datasets/qm9_dataset.py:atom_decoder`
  in Phase 2 and pin the constants.
- **MOSES split convention.** DiGress paper uses MOSES's published
  train/test/scaffold split. We mirror the file structure from the
  `moses` package; if its layout has changed since DiGress's paper,
  document the divergence in the data YAML.
- **GuacaMol KL property bins.** The Distribution-Learning benchmark
  uses bin counts pinned in the `guacamol` package. We rely on the
  package's defaults; if a future package version changes them,
  the spec's validation criterion ("matches DiGress Table 6 within
  ±5%") may shift accordingly.

## References

- Vignac et al., *DiGress: Discrete Denoising diffusion for graph
  generation*, ICLR 2023. (Tables 1-6 are the parity targets.)
- Polykovskiy et al., *Molecular Sets (MOSES): A benchmarking
  platform for molecular generation models*, 2018.
- Brown et al., *GuacaMol: Benchmarking Models for de novo
  molecular design*, 2019.
- Martinkus et al., *SPECTRE: Spectral Conditioning Helps to
  Overcome the Expressivity Limits of One-shot Graph Generators*,
  ICML 2022. (Source of the planar fixture.)
- `digress-upstream-readonly/src/datasets/{qm9,moses,guacamol,
  spectre}_dataset.py` — read-only reference for preprocessing.
- `digress-upstream-readonly/src/analysis/{rdkit_functions,
  spectre_utils}.py` — read-only reference for metric formulas.
- `tmgg.data.data_modules.spectre_sbm` — pattern Planar mirrors.
- `tmgg.evaluation.graph_evaluator` — pattern
  `MolecularEvaluator` parallels.
- `docs/reports/2026-04-28-hp-tuning-and-knee-identification/README.md`
  — downstream consumer of this work; reads the new `gen-val/*`
  keys to anchor the HP sweep.
- Branch `igork/cheap-progress-telemetry` — final telemetry
  namespace this spec inherits (`train/`, `val/`,
  `diagnostics-train/{opt-health,progress}/`,
  `diagnostics-val/progress/`, `gen-val/`, `impl-perf/train/`).
