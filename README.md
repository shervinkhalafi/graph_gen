# Supplementary code — *Graph Convolutional Attention: A Spectral Perspective on Graph Denoising and Diffusion*

This bundle is the supplementary code for an anonymous NeurIPS 2026 submission.
It accompanies the paper PDF and contains the training framework, the
reproduction launcher for the Table 2 ablation panel, the per-run
training history of the runs reported in the paper, and the analysis
docs used to build the tables.

## Scope of this bundle

What you can reproduce from here:

- **Table 2** — the DiGress ablation panel: 4 architecture variants ×
  2 datasets (SPECTRE-SBM and PyG ENZYMES), single seed (666),
  matching the paper's training recipe (5.5 × 10⁵ steps, batch 12,
  AdamW + AMSGrad, lr 2 × 10⁻⁴, eval every 75 k steps).

What is **out of scope** for this bundle:

- **Table 1** (GT vs. GCAT generative sweep, 9 datasets) — different
  architecture series, separate codebase axis. Not part of the
  ablation panel and not driven by the launchers below.
- **Appendix B.3** spectral-diversity FVE study — CPU-only analysis,
  not a generative result. Not included.
- **Note (see paper §297–298):** the spectral-attention SBM accuracy
  column for non-baseline variants is flagged as provisional in the
  paper text. The same caveat applies here — the numbers in Table 2
  stand, but the SBM accuracy interpretation for the spectral-attention
  variant should be treated with the paper's stated uncertainty.

## Layout

```
.
├── src/tmgg/                         # Training framework (PyTorch Lightning + Hydra)
│   ├── data/                         # Datasets (SPECTRE SBM, PyG ENZYMES, …)
│   ├── models/                       # Architectures, including the four Table 2 variants
│   ├── training/                     # Lightning modules, callbacks, optimiser-health diagnostics
│   ├── evaluation/                   # MMD metrics (degree, clustering, orbit, spectral)
│   └── modal/                        # Modal cloud-runner adapters
├── scripts/
│   ├── run-digress-repro-modal.zsh   # Panel launcher — one-cell-per-invocation
│   └── compute_mmd_baselines.py      # Train↔test MMD baselines for ratio anchoring
├── paper-artifacts/repro-ablations/  # Self-contained data bundle for the panel
│   ├── README.md                     # What this bundle contains
│   ├── data/runs_index.csv           # 28 runs (8 paper Table 2 + 20 pre-fix lineage)
│   ├── data/per_run_history/*.parquet # Full W&B history per run, MMD evals, gradients
│   ├── data/DATA-DICTIONARY.md       # Schema for runs_index + per-run parquets
│   ├── configs/{pre-fix,post-fix}/   # Hydra YAMLs at submission and at the data freeze
│   ├── context/                      # MMD-units protocol + paper anchors + train↔test baselines
│   └── media/per_run/<run_id>/       # Sample-grid PNGs from W&B
├── pyproject.toml + uv.lock          # uv-managed dependencies
└── mise.toml                         # Optional Python + uv pinning (`mise run setup`, `mise run repro`)
```

## Setup

We use [`uv`](https://github.com/astral-sh/uv) (Astral) for dependency
management. Optionally [`mise`](https://mise.jdx.dev/) for tool
pinning (Python version + uv version).

```bash
# Option A: with mise (pins Python + uv)
mise install        # installs Python 3.12 + uv per .mise.toml
mise run setup      # uv sync

# Option B: without mise (assumes Python 3.12 + uv on PATH)
uv sync
```

Compiled-from-source dependencies:

- **ORCA** (orbit-counting binary; needed for orbit MMD). Compile in
  place once after `uv sync`:

  ```bash
  cd src/tmgg/evaluation/orca && g++ -O2 -std=c++11 -o orca orca.cpp && cd -
  ```

- **graph-tool** (only needed for SBM-accuracy evaluation on
  SPECTRE-SBM). Install via conda-forge:

  ```bash
  conda install -c conda-forge graph-tool
  ```

## Reproduction recipe

Training for the panel is dispatched to [Modal](https://modal.com)
(A100-40GB by default). Reviewers without Modal access can read the
already-trained run histories directly from
`paper-artifacts/repro-ablations/data/per_run_history/*.parquet` (see
"Inspecting the trained runs" below).

To launch a Table 2 cell from scratch:

```bash
# Prereq: env vars set
export WANDB_API_KEY="..."         # for run logging
export TMGG_TIGRIS_BUCKET="..."    # output volume; see "Modal credentials" in src/tmgg/modal/

# Deploy the Modal apps once
uv run modal deploy -m tmgg.modal._functions
uv run modal deploy -m tmgg.modal._eval_all_functions

# Or set DEPLOY_FIRST=1 to roll deploy + launch into one call:
DEPLOY_FIRST=1 ./scripts/run-digress-repro-modal.zsh sbm-pearl-gnnconv-norm
```

The 8 Table 2 cells map to these `<dataset-key>`s (run one at a time;
each takes ≈18 hours of A100):

| Variant | SBM key | ENZYMES key |
|---|---|---|
| DiGress (baseline)               | `sbm`                       | `enzymes`                       |
| + R-PEARL                        | `sbm-pearl`                 | `enzymes-pearl`                 |
| + R-PEARL + spectral attention   | `sbm-pearl-spectral`        | `enzymes-pearl-spectral`        |
| + R-PEARL + GCAT (D⁻¹ᐟ²AD⁻¹ᐟ²)   | `sbm-pearl-gnnconv-norm`    | `enzymes-pearl-gnnconv-norm`    |

The launcher uses `bf16-mixed` precision on A100 by default. Other
precisions/GPU tiers via `PRECISION=…` and `GPU_TIER=…`. See
`./scripts/run-digress-repro-modal.zsh --help`-style header in the
script for the full list.

## Inspecting the trained runs (no compute required)

`paper-artifacts/repro-ablations/` is a self-contained bundle of the
runs we report against. No training needed to read it.

```python
import pandas as pd

idx = pd.read_csv("paper-artifacts/repro-ablations/data/runs_index.csv")
print(idx[["run_id", "config_name", "final_state", "final_step"]])

# Per-run W&B history (training metrics, MMD evals, gradient health):
hist = pd.read_parquet("paper-artifacts/repro-ablations/data/per_run_history/cgfv3f85.parquet")
mmd_cols = [c for c in hist.columns if c.startswith("gen-val/") and c.endswith("_mmd")]
print(hist[mmd_cols].dropna(how="all").tail())
```

## What's where in the paper-artifacts bundle

`paper-artifacts/repro-ablations/context/` holds the analysis primitives:

- `ANCHORS.md` — paper-anchor reference numbers (DiGress, HiGen) for ratio comparison.
- `BASELINES-CONTEXT.md` — train↔test MMD² baselines and how they were measured.
- `mmd-units-and-protocol.md` — protocol detail (V-statistic biased estimator, σ pinning, sample count).
- `mmd_baselines/{spectre_sbm.json,pyg_enzymes.json}` — the per-dataset baseline JSONs.

`paper-artifacts/repro-ablations/configs/` holds the Hydra YAMLs in
two folders: `pre-fix/` (configs at the time the pre-fix runs were
trained) and `post-fix/` (configs paired with the eight Table 2
runs). The diffs between the two folders document the diagonal-mask
correctness fix.

## Caveats and known issues

- **Single seed (666).** Per the paper's protocol; we don't carry a
  seed-distribution claim in Table 2.
- **Modal-bound training.** The runners are Modal adapters; the
  per-run history parquets in `paper-artifacts/` let reviewers
  inspect outcomes without re-running. Rewriting for a local
  single-GPU loop is a manual step on top of the Lightning module
  (`src/tmgg/training/lightning_modules/`).
- **`<TEAM-ENTITY>` placeholder.** Reviewers who choose to re-run
  must set their own W&B entity via the Hydra `wandb_entity` override
  (or by editing `src/tmgg/experiments/exp_configs/_base_infra.yaml`).
- **Spectral-attention SBM accuracy is provisional** (see paper
  §297–298). The numbers in Table 2 stand; the interpretation note
  in the paper text applies here as well.
