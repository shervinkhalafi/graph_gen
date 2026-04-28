# Validate TMGG sampling with GDPO's DiGress-SBM pretrained weights

## What this does

Loads GDPO's public DiGress SBM checkpoint (`.local-storage/digress-checkpoints/gdpo_sbm/gdpo_sbm.ckpt`, architecture bit-identical to Vignac's `configs/experiment/sbm.yaml`) into our `GraphTransformer`, samples 40 graphs via our `Sampler` + `CategoricalNoiseProcess` + cosine schedule at T=1000, and computes MMD / SBM-accuracy / uniqueness / planarity metrics against the SPECTRE SBM test set via our `GraphEvaluator`. Results go to `outputs/`.

The goal is **parity validation of our sampler / noise process / extras pipeline**, not reproduction of Vignac's paper numbers. GDPO's SBM weights were bitwise-confirmed to be an independent training run from Vignac's (see `.local-storage/digress-checkpoints/README.md`), so run-to-run variance dominates — we're checking that our pipeline works with any valid DiGress checkpoint, using GDPO's as a known-good example.

## How to run

### Local

```bash
# Full run (40 samples × 1000 diffusion steps; ~45 min on a laptop RTX 2000 Ada, several hours on CPU)
uv run python3 analysis/digress-loss-check/validate-gdpo-sbm/validate.py --device cuda --batch-size 4

# Smoke test (4 samples, quick sanity check, ~5 min CPU)
uv run python3 analysis/digress-loss-check/validate-gdpo-sbm/validate.py --num-samples 4
```

On first run the script auto-downloads the SPECTRE SBM fixture (~20 MB) to `~/.cache/tmgg/spectre/sbm_200.pt` via `SpectreSBMDataModule`.

### Modal (recommended for the full 40-sample run)

```bash
# A10G (24 GB VRAM; fits the full batch in one pass) — should finish in 2-5 min
./analysis/digress-loss-check/validate-gdpo-sbm/run-modal.zsh

# Alternative tiers
./analysis/digress-loss-check/validate-gdpo-sbm/run-modal.zsh --gpu a100 --num-samples 200
./analysis/digress-loss-check/validate-gdpo-sbm/run-modal.zsh --gpu t4   # cheapest, slower
```

The Modal app (`modal_app.py`) bakes the 29 MB checkpoint into the image via `add_local_file` so no manual volume upload is needed. Outputs are written to the container's `/data/outputs/validate-gdpo-sbm/<timestamp>/` on the shared `tmgg-outputs` volume, and the local entrypoint also copies them back to `outputs/modal-<timestamp>/` on your machine.

To re-download the checkpoint copies from the volume later:

```bash
modal volume get tmgg-outputs /data/outputs/validate-gdpo-sbm/
```

## What outputs look like

- `outputs/metrics.json` — raw metric values plus `state_dict_load.missing`/`unexpected` key lists, reference values, seed, wall-clock.
- `outputs/metrics_vs_reference.png` — side-by-side bar plot of our MMDs/accuracies next to Vignac's pinned values.
- `outputs/report.md` — markdown table with per-metric delta and a pass/fail heuristic (|Δ/ref| < 0.5 for MMDs, |Δ| < 0.2 for sbm_accuracy), plus a node-count-distribution sanity check.

## Reference numbers (Vignac, `cvignac/DiGress` README)

| metric | value |
|---|---|
| Test NLL | 4757.903 |
| `spectre` (= degree MMD) | 0.0060 |
| `clustering` | 0.0502 |
| `orbit` | 0.0462 |
| `sbm_acc` | 0.675 |
| `sampling/frac_unique` | 1.0 |
| `sampling/frac_unique_non_iso` | 1.0 |
| `sampling/frac_unic_non_iso_valid` | 0.625 |

Vignac's SBM checkpoint switch.ch URL is dead; these are the numbers he pinned for the checkpoint he lost. GDPO's independent retrain using Vignac's exact configs will produce different weights and therefore somewhat different (but still SBM-like) samples.

## Pass/fail heuristic

The script flags a `REVIEW` row in the report if:

- MMD metric: `|Δ / Vignac_value|` exceeds 50%
- SBM accuracy: `|Δ|` exceeds 0.2
- Other scalars: `|Δ|` exceeds 0.1

All-PASS is the expected happy path. A single `REVIEW` is probably noise. Multiple `REVIEW`s together — especially degree + clustering + SBM-accuracy — point at a real bug in our pipeline.

**The primary parity signal is `state_dict_load.missing`/`unexpected` counts.** Both must be 0. A non-zero count means our `GraphTransformer`'s layer layout has diverged from upstream DiGress, and all downstream metrics are unreliable.

## Known caveats

1. **Independent training run.** See above. Single-digit-percent metric deltas are expected; large deltas (>50%) are the signal.
2. **No NLL reported.** The DiGress test-NLL formula lives inside our Lightning `DiffusionModule.test_step`; wiring it up here without also wiring up the full Lightning wrapper adds ~100 lines and doesn't move the validation needle much, so the script omits it. If you want NLL comparison, load the ckpt into a `DiffusionModule` via the existing `evaluate_cli.py` route after this script confirms the sampler path is healthy.
3. **Orbit MMD requires the ORCA binary.** The `GraphEvaluator` silently returns `None` for `orbit_mmd` when `orca` is missing from `src/tmgg/evaluation/orca/`; the report will show a blank row. Build it per the main repo README if you want the metric.
4. **SBM accuracy requires `graph-tool`.** Same story — `None` without `graph-tool`.
5. **Node-count distribution.** Generated graphs use node counts sampled from the *train* split's empirical `SizeDistribution` (matches upstream DiGress). The report prints a sanity summary; ranges should overlap heavily with the SPECTRE test set (roughly 44–187 nodes).

## Why this exists

We need to know that our sampler + noise process are wired correctly before trusting any of our own training runs' metrics. This script takes that question out of "does our full stack work end-to-end" and reduces it to "does our inference path match upstream DiGress's behaviour when given upstream-shaped weights." A clean all-PASS here rules out sampler bugs as the cause of any future training-parity discrepancies.
