# MMD ratio analysis — SBM and ENZYMES repro panels

Snapshot: **2026-05-06**.

This document compares our SBM and ENZYMES repro panel runs against
two anchors:

- **DiGress paper** (Vignac et al., ICLR 2023, `arXiv:2209.14734v3`)
  Table 1, which reports MMD ratios `r = MMD²(gen, test) / MMD²(train, test)`
  for SBM. Paper does not evaluate ENZYMES.
- **HiGen paper** (Karami, `arXiv:2305.19337`) Table 1, which reports
  raw MMD² values for "DiGress" rows on SBM and ENZYMES (HiGen
  reproduced DiGress; the paper does not).

For unit semantics — what "MMD" means in each citation, what our
pipeline emits, and how to convert — see
[`docs/eval/mmd-units-and-protocol.md`](mmd-units-and-protocol.md).
The cached train↔test baselines used here live at
`data/eval/mmd_baselines/{spectre_sbm,pyg_enzymes}.json` and were
produced by `tmgg-mmd-baselines` on Modal on 2026-05-06.

## Why ratios matter, briefly

DiGress paper reports ratios because raw MMD² is highly
dataset-dependent — `MMD²(train, test)` for SBM is ~1e-4 on degree
but ~3e-2 on clustering. A model that under-fits clustering (raw MMD²
0.1) would look "200× off" in absolute terms but "3× off" in ratio
units. The ratio normalises away that intrinsic dataset variance.

Anchoring to ratios also makes our results comparable to
hyper-parameter studies whose authors don't publish raw values. HiGen
publishes raw values; DiGress publishes ratios. Computing both columns
side by side here lets us anchor to whichever source applies.

## SBM repro panel

Baseline (our pipeline): `degree=3.41e-04`, `clustering=3.31e-02`,
`orbit=3.10e-02`, `spectral=2.82e-03` (V-statistic MMD², over the
full 128-train / 40-test SPECTRE-SBM splits).

| run | run_id | health | r_run degree | r_run cluster | r_run orbit | r_run spectral |
|-----|--------|:------:|-------------:|--------------:|------------:|---------------:|
| `vignac_repro`               | `12s2b4a7` | ✓ | 541.5 | 4.00 | 4.59 | 75.4 |
| `pearl_repro`                | `s07qwx3b` | ✓ | 543.9 | 4.04 | **3.05** | 75.1 |
| `pearl_spectral_repro`       | `jbraoj7o` | ✓ | 547.1 | 4.05 | 3.19 | 73.5 |
| `pearl_gnnconv_norm_repro`   | `rarihsee` | ⚠ | 538.6 | **3.98** | 3.83 | 77.0 |
| `pearl_gnnconv_raw_repro` (1)| `qao36vwu` | ✗ | 876.5 | 4.09 | 4.60 | 80.4 |
| `pearl_gnnconv_raw_repro` (2)| `g1g6xpx1` | ✗ | 1268.6 | 4.23 | 4.58 | 95.7 |
| **HiGen-implied** (their Table 1 / our baseline) | — | — | **3.81** | **1.50** | **1.40** | n/a |
| **DiGress paper** Table 1 | — | — | **1.6** | **1.5** | **1.7** | n/a |

Health flags: ✓ stable, ⚠ elevated grad_norm but converged, ✗ blew up
(`grad_norm` 1e3–1e18, optimizer state corrupted).

### SBM reads

1. **Clustering is the cleanest anchor in the panel.** The DiGress
   paper ratio is 1.5; HiGen's reproduction lands at 1.50 (bit-identical);
   our healthy variants sit at 3.98–4.05, just 2.7× the paper. With more
   training the gap is plausibly closeable. **Crucially, HiGen ≡ paper
   here** — strong evidence that the clustering MMD protocol is
   bandwidth-stable across independent implementations and that our
   clustering pipeline is calibrated correctly.
2. **Orbit is also close to converged.** Paper r=1.7, HiGen 1.40, our
   best run (`pearl`) 3.05. About 2× from paper. PEARL's orbit
   improvement on SBM is real and visible at this baseline.
3. **Degree is dramatically off.** Healthy panel sits at r≈540 vs
   paper r=1.6 (340× gap). Even *HiGen*'s degree ratio (3.81) is 2.4×
   above DiGress paper — suggesting HiGen never matched DiGress on
   degree either. Our 540× sits well past HiGen's. The diagnosis:
   the model is not learning the SBM block-degree distribution at the
   eval points we logged. Since it's the smallest baseline value
   (3.41e-04), small absolute generation errors blow up huge ratios.
4. **Spectral has no published anchor.** Our 73–77 ratio is a number
   to characterise our protocol, not an anchor comparison. Worth
   pinning a tmgg-internal anchor when a converged SBM run lands.
5. **The `_raw_` variant is unviable across two runs**, with degree
   ratios 877 and 1269 (5–10× the panel norm). Numerical instability
   confirmed by the gradient state — see the per-run detail files.

## ENZYMES repro panel

Baseline (our pipeline): `degree=3.00e-04`, `clustering=1.04e-02`,
`orbit=1.73e-04`, `spectral=2.85e-03` (V-statistic MMD², over
420-train / 120-test PyG-ENZYMES splits at 70/10/20 ratio).

| run | run_id | health | r_run degree | r_run cluster | r_run orbit | r_run spectral |
|-----|--------|:------:|-------------:|--------------:|------------:|---------------:|
| `vignac_repro`             | `l1nk0622` | ✓ | 623.2  | 10.61 | 686.0  | 71.9 |
| `pearl_repro`              | `ge461v1o` | ✓ | 619.5  | 10.38 | 1143.3 | 70.4 |
| `pearl_spectral_repro`     | `4n28svrj` | ✓ | 625.2  | **10.36** | **559.0**  | 69.0 |
| `pearl_gnnconv_norm_repro` | `ly0d6lyi` | ✓¹ | **613.5** | 10.56 | 852.9  | 71.3 |
| `pearl_gnnconv_raw_repro`  | `dt0ux9zh` | ✗ | 704.9  | **9.02** | 3106.6 | 68.6 |
| **HiGen-implied** (their Table 1 / our baseline) | — | — | **13.34** | **7.95** | **11.55** | n/a |
| **DiGress paper** Table 1 | — | — | n/a | n/a | n/a | n/a |

¹ Run terminated (failed) but gradients were healthy at point of
failure — infra cause, not numerical.

### ENZYMES reads

1. **Clustering is once again the cleanest anchor**, and the only
   metric where we can plausibly converge to the published number.
   HiGen-implied ratio is 7.95; our healthy panel sits at 10.36–10.61
   (1.3× off). With more training, this anchor is reachable.
2. **Degree at ~620× is ~50× off HiGen's 13.34 ratio.** Same diagnosis
   as SBM — the model is not learning the degree distribution, and
   the small baseline magnitude amplifies the visible ratio.
3. **Orbit shows enormous variance across variants:** `pearl_spectral`
   559× (best), `vignac` 686×, `gnnconv_norm` 853×, `pearl` 1143×,
   `gnnconv_raw` 3107× (divergent). The `pearl_spectral` win on orbit
   is consistent with the cross-variant finding from `runlog.md`:
   spectral attention recovers orbit on enzymes that plain PEARL
   regresses. Still, all variants are well past HiGen's 11.55.
4. **Spectral has no published anchor for ENZYMES** in either DiGress
   paper or HiGen paper.
5. **The `_raw_` variant is again unviable** — orbit at 3107×, same
   pattern as SBM.

## Reproduction caveats

These ratios should be read against the audit performed on
2026-05-06 of our `vignac_repro` configs against upstream DiGress.
The configs are **not byte-equivalent reproductions**:

| Item | Upstream Vignac | Our `discrete_sbm_vignac_repro` | Effect on ratios |
|------|------|------|------|
| `dim_ffy` | 256 | **2048** (matches GDPO checkpoint, not Vignac config) | More capacity than DiGress paper; could either help or hurt depending on optimization landscape. Material. |
| seed | 0 | 666 (matches GDPO) | O(σ) noise, not load-bearing. |
| `amsgrad` | (PyTorch default = false) | true | Material; changes optimiser convergence. |
| Eval cadence | every 1100 / 4400 steps | every 5000 / 75000 steps | Logging only; doesn't affect training. |
| Code path | `cvignac/DiGress/src/...` | `tmgg.models.digress.*` (port) | Semantically aligned but not byte-equivalent. Micro-bugs are possible. |
| Run completion | 50000 epochs (≈550k steps) | **430k steps reached** before 24h Modal timeout | 78% of intended training. Real undertraining. |

For ENZYMES, **DiGress paper does not configure or evaluate ENZYMES at
all**. Our `discrete_enzymes_vignac_repro` is the SBM recipe
transplanted onto ENZYMES — a heuristic baseline, not a paper
reproduction. HiGen's reported ENZYMES numbers come from a custom
DiGress run by HiGen's authors that has never been released.

## Open questions

1. **Train↔val baseline** — our run-side MMDs are `gen↔val` but the
   cached baselines are `train↔test`. PICKUP doc Task 3 §2 flags this.
   Effect should be O(1) given i.i.d. splits, but worth recomputing
   `MMD²(train, val)` baselines and re-running the ratio table to
   tighten the comparison.
2. **Degree ratio's 340× SBM gap** — is this resolvable with more
   training, or are we hitting a limit? Recommendation: run one
   `vignac_repro` to full 550k steps (need to lift the 24h Modal
   timeout) and re-snapshot ratios. If degree comes down, undertraining.
   If not, look at the `count_node_classes_sparse` / `K=1` change for
   regressions.
3. **Spectral protocol audit** (PICKUP doc Task 3 §3) — our spectral
   r_run sits at 70–80 across all variants and datasets, suggestively
   uniform. May indicate the spectral histogram bin count / range
   differs systematically from upstream's. Check
   `mmd_metrics.compute_spectral_histogram` against
   `digress-upstream-readonly/src/analysis/spectre_utils.py:spectral_stats`.
4. **`_raw_` numerical instability** — separate engineering question;
   see `runlog.md` open questions §2 and §3.

## See also

- [`runlog.md`](../../runlog.md) — index of running and finished runs.
- Per-run detail files in `run_details/<launch-date>/` carry their own
  Anchor comparison block.
- [`docs/eval/mmd-units-and-protocol.md`](mmd-units-and-protocol.md)
  — what MMD vs MMD² means in our pipeline and across the literature.
- [`docs/experiments/sweep/smallest-config-2026-04-29/anchors.yaml`](../experiments/sweep/smallest-config-2026-04-29/anchors.yaml)
  — pinned target values, with HiGen vs DiGress provenance per metric.
- [`PICKUP-MMD-RATIOS-2026-05-06.md`](../../PICKUP-MMD-RATIOS-2026-05-06.md)
  — open work on baseline computation and the kernel/sigma audit
  that motivated this whole effort.
