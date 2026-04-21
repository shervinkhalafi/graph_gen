# Potential bug: DiGress-on-SBM generation metrics frozen despite training-loss progress

Status: suspected, not yet reproduced locally. Based on Modal-volume run artifacts pulled 2026-04-21.

## Summary

On the upstream-matching DiGress SBM run on Modal (`tmgg-outputs` volume), training loss
drops substantially (min `train/loss_step` ≈ 0.24, `val/loss` 1.54 → 1.19 across ~10k
steps) while **every graph-quality metric is effectively frozen across 10 validation
checkpoints** spanning training steps 6599 → 16499. Generation appears mode-collapsed:
`val/gen/uniqueness` is 3-9% (i.e. ~90% of the 128 generated samples per validation pass
are duplicates), `val/gen/sbm_accuracy` is 0 everywhere, and `val/gen/spectral_mmd` has
standard deviation ≈ 7e-5 over the 10 checkpoints.

Short SBM DiGress runs on the SPECTRE fixture (2026-04-16, lr=1e-3, 2000 steps) do not
show this; their generation metrics drift over training. So the fault is specific to the
upstream-matching regime (synthetic p_intra=1.0 / p_inter=0.0 on 200 graphs, lr=2e-4,
wd=1e-12, long training with resumes).

## Evidence

Data files (relative to `src/tmgg/` working dir):

- `analysis/digress-loss-check/upstream_s1_lr2e-4/metrics_v7.csv` — 1450 rows, steps 5510-16499
- `analysis/digress-loss-check/upstream_s1_lr2e-4/versions/metrics_v5.csv` — 175 rows, steps 5510-6841
- `analysis/digress-loss-check/upstream_s1_lr2e-4/config.yaml` — resolved Hydra config
- `analysis/digress-loss-check/loss_vs_graphquality.png` — side-by-side loss + generation plots
- `analysis/digress-loss-check/comparison.png` — full per-run grid

Contrast artifacts:

- `analysis/digress-loss-check/waveA_s201/metrics.csv` and `panel_s201/metrics.csv` —
  2000-step DiGress on SPECTRE, show normal generation-metric drift.

### Constancy check (10 val checkpoints of v7)

```
val/gen/spectral_mmd     distinct=3/10  range=[0.726113, 0.726337]  std=6.6e-5
val/gen/degree_mmd       distinct=4/10  range=[0.767095, 0.786939]  std=5.8e-3
val/gen/clustering_mmd   distinct=4/10  range=[0.252690, 0.280734]  std=8.2e-3
val/gen/orbit_mmd        distinct=4/10  range=[0.000361, 0.000368]  std=2.0e-6
val/gen/sbm_accuracy     distinct=1/10  range=[0.000000, 0.000000]  std=0
val/gen/uniqueness       distinct=3/10  range=[0.031250, 0.093750]  std=2.0e-2
```

Training-signal trajectory over the same span: `train/loss_step` drops from ~2.5 to a
minimum of 0.24 at step 11299; `val/loss` trends 1.54 → 1.19. The model is demonstrably
still updating; the sampler/evaluator output is not.

## Candidate causes (to be falsified or confirmed by local investigation)

Ranked by my prior on likelihood:

1. **`CategoricalSampler` collapse under degenerate training data.** With
   `p_intra=1.0, p_inter=0.0` every training graph is edge-isomorphic (two disjoint
   10-cliques). Combined with `limit_distribution=empirical_marginal` as the reverse-chain
   prior, the sampler may always recover the same attractor from different initializations.
   Check `src/tmgg/diffusion/sampler.py` for temperature / argmax paths during reverse
   denoising, and confirm the initial state is actually randomized per sample.
2. **Evaluator caching.** `GraphEvaluator` (`src/tmgg/evaluation/graph_evaluator.py`)
   might memoize a sample batch or reuse a fixed reference graph set. With `eval_num_samples=128`
   and observed uniqueness ~0.06 across many checkpoints, the ratio is suspiciously
   stable.
3. **Reference-vs-samples confusion.** The MMD numerators could be computed against the
   training set with the samples cached, making the metric insensitive to training.
4. **Extra-features mismatch with fixture.** The upstream config has
   `extra_features.max_n_nodes=20`, correct for the synthetic p=1/0 data but not for
   SPECTRE graphs (n up to 187). In this specific run the data module is
   `SyntheticCategoricalDataModule` (n=20), so this should be consistent. Low priority.

## Incidental finding: silent config absorption

`SpectreSBMDataModule.__init__` swallows `graph_type`, `num_nodes`, `num_graphs`,
`graph_config`, `train_ratio`, `val_ratio` via `**_metadata`
(`src/tmgg/data/data_modules/spectre_sbm.py:76-83`). In the short-run configs those keys
are present but dead — the module always loads the 200-graph SPECTRE fixture. This is
defensible for hydra-composition convenience but would bite anyone trying to reshape the
training distribution via `data.graph_config.*=...` overrides: changes get silently
ignored. Worth raising a loud `TypeError` or `UserWarning` on unused kwargs.

## Reference: launch scripts that produced these runs

- `tmgg/run-upstream-digress-sbm-modal.zsh` — upstream-matching DiGress entry
  (lr=2e-4, wd=1e-12, `+data=spectre_sbm`). Note: the actual 2026-04-15 run used
  `SyntheticCategoricalDataModule` (p=1/0), so the script's SPECTRE intent was not in
  effect for the artifacts analysed here. Post-parity-audit wiring
  (`docs/reports/2026-04-15-upstream-digress-parity-audit.md`) has not yet produced a
  long run on the volume.
- `tmgg/run-digress-arch-panel-modal.zsh` — architecture panel (2000 steps), produced the
  wave A and wave B DiGress comparison runs.

---

# Debugging log

## 2026-04-21 — Parity audit against upstream DiGress (cvignac/DiGress @ 780242b)

Six parallel code-parity audits were run against the upstream reference at
`../../../digress-upstream-readonly/`. Each audit was independently verified by
a second Opus reviewer agent. All audit and review files live in
`parity-audit/`.

### Layers audited (verdict / review verdict)

| Layer | Audit | Reviewer | Outcome |
|---|---|---|---|
| Noise process + schedule + transition matrices (`Qt`, `Qt_bar`, `empirical_marginal`, posterior) | `02-noise-process.md` | `review-02-noise-process.md` | MATCH × 7, all confirmed |
| Sampler + reverse chain (per-sample `z_T`, no argmax, no module-scope RNG) | `01-sampler.md` | `review-01-sampler.md` | MATCH × 6, all confirmed |
| Training loss + DiffusionModule (`lambda_E`, reduction, masking, symmetry, class indexing) | `03-loss.md` | `review-03-loss.md` | MATCH × 8 + 1 benign divergence, all confirmed |
| Evaluator + MMD (caching, references, kernel, SBM Wald test, sample→graph decoding) | `05-evaluator.md` | `review-05-evaluator.md` | MATCH × 7, all confirmed |
| EMA + Lightning plumbing (no EMA anywhere, no stale weights, `@torch.no_grad` parity) | `06-ema-plumbing.md` | `review-06-ema-plumbing.md` | MATCH × 7, all confirmed |
| `ExtraFeatures` (clean-vs-noisy leak, cycles, eigenfeatures, y-layout, timestep injection) | `04-extra-features.md` | (not reviewed — audit re-launched after session interruption) | MATCH × 6 |

### What this rules out

- **Sampler mode-collapse from code mechanics.** Initial `z_T` is drawn per-sample per-position via `.multinomial(1)`; no broadcast bug, no argmax step. The benign missing post-contraction renormalisation is scale-invariant under `multinomial`.
- **Wrong prior.** `empirical_marginal` aggregates to a single `(K,)` scalar on both sides; `pi_E ≈ [0.526, 0.474]` for `p_intra=1.0, p_inter=0.0` on n=20 two-block SBM — a genuinely high-entropy prior, not a near-delta.
- **Stale weights at sampling.** No EMA anywhere in our code. `Sampler` receives a live reference to `self.model`; no deep-copy, no cached checkpoint.
- **Evaluator caching.** `GraphEvaluator` has zero sample/stat/MMD caching across `evaluate()` calls. References are drawn from the val dataloader, not synthesised from `p_intra`/`p_inter`.
- **Class-index inversion.** Channel 0 = no-edge, channel 1 = edge is consistent in prediction, target, sampler output, and evaluator decoder on both sides.
- **Training-loss mis-weighting.** `lambda_E=5.0` is applied once to `E_class`, matching upstream's `lambda_train[0]=5`. Masking of diagonal and padding positions is correct.
- **`val/loss` confounded with sampler.** `val/loss` is the training CE evaluated at a random `t` on the val set; the sampler runs only in `on_validation_epoch_end` under a separate metric namespace.
- **Clean-graph leak into `ExtraFeatures`.** The call site in `transformer_model.py:944` passes `data.X_class` / `data.E_class` — the noisy `z_t` fields, not `batch.X` / `batch.E`. All three call sites (training, validation, sampler) feed noisy tensors.
- **Missing timestep injection.** `t/T` is appended as the last dimension of `y` on both sides, producing 11 y-dimensions after `ExtraFeatures(extra_features_type='all')`.

### Incidental findings worth noting (not causes of the collapse)

- `noise_process.py:159-173` (`_read_categorical_x`) synthesises `[0, 1]` one-hot from `node_mask` when `X_class is None`. For structure-only SBM this yields a deterministic node channel — fine for the pipeline, but flagged if the data schema ever changes.
- `PEARLEmbedding.forward` at `models/layers/pearl_embedding.py:137` sets `torch.Generator(...)manual_seed(42)` per call in eval mode. It is **not** on the SBM DiGress path (config targets `GraphTransformer` + `ExtraFeatures`, not `spectral_denoisers.*`), so it cannot affect the frozen run. Would be a latent bug for spectral-denoiser configs.
- Our `_compute_loss` softmaxes then log-soft-CEs, upstream log-softmaxes via `F.cross_entropy`. Gradient-equivalent, numerically slightly less stable; unrelated to the symptom.
- Upstream file `src/metrics/train_metrics.py` (not `train_loss.py`) — cosmetic correction to `03-loss.md`.

### What remains unaudited

The parity audit covered all the layers that consume model predictions, compute loss, generate samples, or score them. It did **not** cover:

1. **Data pipeline.** Specifically whether `SyntheticCategoricalDataModule(p_intra=1.0, p_inter=0.0)` produces 200 *distinct* graphs (up to edge realisation, since with p=1/0 the Bernoulli generator is deterministic given the partition) or 200 *identical* graphs. If the training set is effectively a single graph repeated 200 times, the problem is degenerate — the model can fit the target distribution perfectly by memorising a single edge pattern indexed by node position, and the observed "mode collapse" is not a bug but expected behaviour on a pathological dataset. Given `p_intra=1.0, p_inter=0.0` and fixed `num_blocks=2, num_nodes=20, train_ratio=0.64`, every realisation of the SBM is the *exact same adjacency* up to the (deterministic) block partition — this strongly suggests the latter.
2. **Size distribution used at sampling.** `SyntheticCategoricalDataModule.get_size_distribution()` returns what? If it's a point mass at n=20 the sampler always generates fixed-n graphs, which is correct for this config. A mismatch (e.g., sampling n∈[44,187] on a model trained at n=20) would produce junk, but would not freeze MMDs.
3. **Interaction between noisy-graph eigenfeatures and mode-collapse.** With `max_n_nodes=20` and k=2 eigenvectors of the Laplacian, the model has exactly one rank-breaking signal per node. At `z_T` (≈ Bernoulli(0.47)), these eigenvectors are nearly random; the model cannot pick a stable community assignment from noise alone. If training has collapsed the denoiser to predict a *fixed* community assignment regardless of input (position-indexed), the sampler reliably reproduces one graph, matching the observed `uniqueness 3-9%`. This is a hypothesis about model *behaviour*, not a parity bug.

### Revised hypothesis

The symptom signature — `train/loss_step → 0.24`, `val/loss` drops monotonically, `val/gen/*` metrics frozen — is consistent with **a fully-correct implementation trained on a degenerate dataset**. With `p_intra=1.0, p_inter=0.0` and a fixed partition, the training distribution is effectively a point mass on one labelled graph. The DiGress transformer fits that point mass well (low train loss), but at sampling time it must reproduce that graph *without access to the community labels that indexed it during training*. The learned denoiser collapses onto whatever fixed partition the spectral extra-features happened to prefer during training, producing near-identical samples regardless of `z_T`.

This hypothesis is not yet confirmed. It would be confirmed by:
- inspecting `SyntheticCategoricalDataModule._generate()` output for fixed `num_blocks=2, p=1/0, seed=1` and checking whether the 200 graphs are edge-identical,
- reading the size-distribution wiring to rule out inference-time size mismatch,
- running a cheap local probe: load a trained checkpoint from `discrete_diffusion_DiffusionModule_lr2e-4_wd1e-12_s1`, sample 32 graphs, hash them — if they collide to <5 distinct classes, the collapse is real; if not, the CSV metrics are stale.

The short-run SBM DiGress on the SPECTRE fixture (not p=1/0 synthetic) did **not** exhibit this collapse — spectral_mmd drifted, uniqueness stayed at 1.0 — which is consistent with the dataset-pathology hypothesis: once the training distribution has genuine diversity, the sampler produces diverse outputs.

### Recommendation

Do not launch another long upstream-matching run with `p_intra=1.0, p_inter=0.0` until the data pipeline is audited. If the synthetic generator produces 200 identical graphs, either (a) raise the inter-block probability to add variability (e.g. `p_inter=0.01`, closer to SPECTRE's 0.005), (b) use the SPECTRE fixture end-to-end as the upstream paper does, or (c) accept that this particular configuration is a degenerate corner and move on.

### Would dropout fix this?

Short answer: **no, not as a root-cause fix.** It would mask the symptom, not
cure it.

Reasoning:

- The diagnosed root cause is *dataset* degeneracy — 200 edge-identical graphs
  → point-mass target distribution → model fits perfectly but has no
  non-memorisation signal to learn from. Dropout is a *model*-side
  regulariser; it cannot introduce diversity that is not in the training
  data.
- Dropout during training would perturb intermediate activations, forcing the
  transformer to not rely on any single attention channel. But the *targets*
  are still deterministic given node indices (intra-block = 1, inter-block = 0),
  so the model still converges to the same node-index-indexed lookup, just
  more slowly.
- Dropout at sampling time (keeping the model in `train()` mode during the
  reverse chain) would inject stochasticity into each reverse step, producing
  visibly different samples. But the diversity would be noise around the same
  collapsed mode, not coverage of different valid SBM partitions. MMD metrics
  would stop being frozen to 7 decimal places — they'd wiggle at the 1–2 %
  level — while remaining high, which is arguably worse: it would disguise
  the collapse.
- Upstream DiGress does not use dropout in the SBM config; the original
  implementation achieves non-collapsed generation by training on the SPECTRE
  fixture (200 genuinely distinct graphs with variable n, 2–5 communities),
  not by adding stochastic regularisation.

A model-side change that *would* plausibly help, if one insists on keeping
the p=1/0 synthetic data, is **random node relabelling per epoch** — either
by permuting the dataloader output (shuffle the adjacency rows/columns
jointly with node-feature rows) or by adding random sign-flips to the
Laplacian eigenvectors inside `ExtraFeatures` (upstream's convention for
handling the eigenvector sign ambiguity — worth a separate spot-check of our
`extra_features.py` since the audit did not verify sign-invariance
handling). Both attack the "model memorised node-index-to-block mapping"
failure mode directly.

But the cheaper and more principled fix is to stop training on a
point-mass distribution. Either `p_inter > 0` (say 0.005 to match SPECTRE)
or switching to the actual SPECTRE fixture removes the degeneracy at the
source. The short 2026-04-16 DiGress runs on SPECTRE already demonstrate
that the implementation generates diverse graphs (uniqueness = 1.0) when
given a non-degenerate training set.

### Audit gap worth a follow-up

`ExtraFeatures` audit verified that the Laplacian eigenvector features are
computed correctly, but did **not** verify whether random sign-flipping of
eigenvectors is applied at training time (upstream handles the ±v sign
ambiguity explicitly). If our implementation uses a deterministic sign
convention (e.g., first non-zero component positive), the model gets a
stable positional encoding that is still vulnerable to the same
memorisation failure mode. This is worth checking before dismissing the
parity audit as complete.

## 2026-04-21 (continued) — Visual audit reveals the hypothesis was wrong

Downloaded the 8 adjacency-sample PNGs emitted by the evaluator at validation
time (steps 6600, 11000, 16500 for the frozen upstream-match run; steps 500,
1000, 1500, 2000 for the panel/SPECTRE run) and cross-checked them with two
independent Haiku visual-audit agents (reports in
`parity-audit/visual-upstream-s1.md`, `parity-audit/visual-panel-s201.md`).

### What the pictures actually show

**Upstream-match run (frozen metrics).** References are two disjoint K_10
cliques (e=90 out of max 190, density 0.47). Generated graphs are
**near-complete K_20 graphs** with e=165 edges (density ~0.87). Four samples
within a single checkpoint look essentially the same — dense interconnected
webs with no community structure — and the pattern persists with negligible
change across steps 6600 → 16500. **This is edge-collapse, not
memorisation-collapse.**

**Panel run on SPECTRE.** References are variable-size SBM graphs (n ∈
[91, 169], 2-5 communities, density ~0.05). Generated graphs at step 2000 do
show block-like cluster structure and varying node counts (n ∈ [53, 142]) —
partial recovery, with spectral_mmd drifting 0.013 → 0.007 over training.
This run is not broken in the same way; it is under-trained on a small
128-graph dataset.

### Why this matters

The previous hypothesis in this report was "model memorises the training
distribution → outputs the same K_10+K_10 graph every time → frozen
metrics". **That is wrong.** Memorisation would produce e≈90 graphs matching
the reference. Outputting the prior would also produce e≈90 graphs
(empirical edge marginal ≈ 0.47 → 90 edges on n=20). Instead the model
collapses to an **edge-majority-everywhere** prediction regardless of input:
P(edge) ≈ 0.87 at most positions, ≈ 0 at a small deterministic set of
"hole" positions, giving e=165 consistently.

### What could cause a P(edge) ≈ 0.87 collapse

None of these are confirmed. They are the candidates remaining after the
parity audit cleared the sampler, noise process, loss, evaluator, EMA, and
extra-features layers.

1. **Eigenvector sign-flip augmentation missing (UNAUDITED gap).** Upstream
   DiGress randomly flips the sign of each Laplacian eigenvector at training
   time to make the model invariant to the ±v ambiguity. Our
   `ExtraFeatures` audit verified the eigenvalue computation but did not
   check sign-flip augmentation. If we use a deterministic sign convention
   (e.g., first non-zero component positive), the model at training sees
   one fixed sign; at inference the noisy-graph eigenvectors could come out
   with any sign, and the model gets "confusing" positional encodings it
   never saw. This does not obviously predict edge-majority collapse, but
   it is a real parity gap.
2. **Class-imbalance amplification via `lambda_E = 5.0`.** On p_intra=1.0,
   p_inter=0.0 data, 90/190 = 47% of valid edge positions are class 1
   (edge). With 5x weight on the edge CE, the gradient signal pushes the
   model to minimise edge-class errors even at the cost of no-edge
   accuracy. Combined with an `_epsilon_renormalise` that smooths targets
   away from hard zero (1e-7 per class), the model could have converged
   to a high-P(edge) solution that minimises weighted CE but is
   distributionally wrong.
3. **Time-embedding mis-injection at inference.** The loss audit noted that
   our training uses `t ∈ [1, T]` while upstream uses `t ∈ [0, T]`. At
   sampling both run `t` from T down to 1. Unlikely to cause an 87% edge
   bias on its own.
4. **Silent class-index swap in the training-set encoder.** Loss-audit
   check #8 confirmed channel 0 = no-edge, channel 1 = edge throughout,
   but it did not verify that the **data loader** for
   `SyntheticCategoricalDataModule` emits the same convention as the
   `SpectreSBMDataModule`. If the synthetic path encodes edges on channel
   0 and no-edges on channel 1 (inverse of the SPECTRE path), the model
   would train on the complement and generate ≈ 190-90 = 100 edges. That
   does not match the observed 165 either, so probably not this.
5. **Interaction between `empirical_marginal` prior and the reverse
   chain.** The noise audit confirmed `pi_E ≈ [0.526, 0.474]`. At `z_T` the
   initial distribution is high-entropy, so the chain starts sensibly. But
   the posterior mixing formula `(z_t @ Qt.T) ⊙ (x̂_0 @ Qt_bar_{t-1})`
   combines current state with x̂_0 prediction. If x̂_0 is biased toward
   edges (reason 2 above), every reverse step amplifies that bias
   geometrically.

### Falsification steps that do NOT require a new training run

1. Load the last checkpoint of `discrete_diffusion_DiffusionModule_lr2e-4_wd1e-12_s1`,
   run the model in eval mode on a single noisy batch `z_t` at t=T/2, and
   inspect the raw logits of `pred.E_class`. If `softmax(logits)[..., 1]`
   is close to 0.87 at most positions, the model IS predicting edge-everywhere
   — confirming reason 2 or something similar. If it's ~0.47, the bug is in
   the sampler's posterior mixing, not the predictions.
2. `rg "sign_flip|random_sign|sign\(eigvec" tmgg/src/tmgg/models/digress/`
   vs the same in upstream. A direct check on the audit-gap (reason 1).
3. Compare `SyntheticCategoricalDataModule` edge-encoding convention against
   `SpectreSBMDataModule` by printing one batch's `E_class` tensor for both
   and verifying channel 0 represents the same semantic (no-edge) in both
   paths.

### Panel run (working-er) observation

On SPECTRE, generation at step 2000 shows visibly emerging community
structure — not yet close to references but clearly not edge-collapsed.
This proves the TMGG pipeline can produce block-structured graphs given
a non-degenerate training set. The bug manifesting in the upstream-match
run is specific to either the synthetic p=1/0 dataset encoding, the
`lambda_E`-amplified class imbalance on that dataset, or both.

### Revised recommendation

Before any new training run, spend ~30 minutes on falsification step 1
(inspect raw logits from the last saved checkpoint). That single probe will
localise the fault to either the predictor (reason 2 or the sign-flip gap)
or the posterior-mixing / decoding path (reasons 3-5). Without it, any
"fix" is guesswork against a revised but still unconfirmed hypothesis.

## 2026-04-21 (continued) — Candidate-cause audits

Five Haiku code-parity audits ran in parallel, one per candidate cause listed
above. Reports at `parity-audit/candidate-{1..5}-*.md`.

| # | Candidate | Verdict |
|---|---|---|
| 1 | Missing eigenvector sign-flip augmentation | **NO GAP** — neither upstream nor ours randomises eigenvector signs; both use `torch.linalg.eigh` output as-is. My earlier claim that upstream has it was wrong. |
| 2 | `lambda_E = 5.0` × `_epsilon_renormalise(1e-7)` asymmetric gradient attenuation | **CONFIRMED PROBABLE** — real parity divergence. Ours smooths both targets and predictions by 1e-7 then renormalises; upstream uses hard integer targets through `F.cross_entropy`. With 5× weighting on the edge CE, softened targets attenuate the gradient penalty on over-predicting edges. Plausible mechanism for P(edge)≈0.87 collapse on a training set where the edge marginal is 0.47. |
| 3 | Synthetic vs SPECTRE data-module class-index inversion | **CONSISTENT** — both paths converge on `GraphData.from_pyg_batch` → `E_class = stack([1-adj, adj], dim=-1)`. Channel 0 = no-edge on both. |
| 4 | Time-embedding mis-injection | **CONSISTENT** — training and sampling both normalise `t_int / T` to `[1/T, 1]` before injection into `y`. Matches upstream. |
| 5 | Posterior-mixing Q-index off-by-one / transpose bug | **MATCH** — `Qt`, `Qsb`, `Qtb`, transposes, and division all indexed correctly in both codebases. |

### Net result

Of the five candidates, only **#2 survives**. The only confirmed
parity divergence across the entire audit — sampler, noise process,
training loss, evaluator, EMA, extra-features, sign-flip, class-index,
time-embedding, posterior mixing — is the label-smoothing/
epsilon-renormalisation asymmetry in our training loss.

### Why #2 alone might not explain the full picture

- `lambda_E = 5.0` and epsilon = 1e-7 are both small perturbations; the
  gradient attenuation they jointly induce is also small. Whether it is
  large enough to drive convergence from P(edge) = 0.47 (data marginal)
  to P(edge) = 0.87 (observed) is quantitative and cannot be resolved by
  code reading alone.
- On the panel/SPECTRE run, the **same** loss code runs with the **same**
  `lambda_E = 5.0` and `_epsilon_renormalise(1e-7)`. That run does NOT
  edge-collapse — it produces visibly block-structured graphs by step
  2000 (see `visual-panel-s201.md`). If #2 were the sole cause, both
  runs should collapse the same way. The data-distribution difference
  (SPECTRE density ~0.05 vs synthetic density ~0.47) clearly matters.

So the most defensible reading is: **#2 is a genuine but perhaps not
sufficient cause**. The p=1/0 synthetic dataset's high edge marginal
interacts with #2 to push the model past some threshold where
edge-over-prediction becomes stable. On the low-density SPECTRE dataset
(where only 5% of positions are edges), the same asymmetry is present
but not catastrophic.

### What this does not settle

The available evidence is all code-based. The quantitative question — does
the #2 asymmetry plausibly account for a 0.47→0.87 drift, or is there
another latent bug? — cannot be answered without either (a) a controlled
ablation (train with `_epsilon_renormalise` disabled) or (b) loading the
saved checkpoint and inspecting the raw `pred.E_class` logits. Both were
ruled out by the user for this investigation.

The conservative conclusion for a writeup or issue report is: "One
parity divergence found: label-smoothing applied to both targets and
predictions with epsilon=1e-7. Combined with `lambda_E=5` and a
high-edge-marginal training distribution, this is a plausible mechanism
for the observed edge-collapse, but has not been empirically validated."

## 2026-04-21 (continued) — Full-pipeline deep audit

Six additional Haiku read-only audits ran in parallel covering every
remaining conceptual layer of our implementation. Reports at
`parity-audit/deep-{noise-schedule,architecture,extra-features,synthetic-data,vlb-kl,collation-graphdata}.md`.

### Layer-by-layer results

| Layer | File(s) audited | Verdict |
|---|---|---|
| Noise schedule | `diffusion/schedule.py`, `diffusion/diffusion_math.py` | MATCH — one benign divergence: we clamp `alpha_bar` at 1e-5, upstream lets it go to ~2.4e-10 at t=T. Makes our `z_T` prior slightly less close to the stationary limit. |
| GraphTransformer architecture | `models/digress/transformer_model.py` + layer modules | MATCH — when `use_gnn_q`/`use_spectral_q` flags are off (the default, and what the failing config uses), the neural net is byte-for-byte equivalent to upstream. Two non-destructive feature additions (GNN/spectral projection layers, optional timestep appending) are off on the failing run. |
| Extra features edge cases | `models/digress/extra_features.py` | MATCH across 8 checks — eigenvalue/eigenvector k-handling when n<k, `max_n_nodes` usage, cycle normalisation, Laplacian definition, connected-component threshold, top-k selection order, y-concatenation order, timestep placement across train/val/sample all match. |
| VLB / KL path | `training/lightning_modules/diffusion_module.py` (VLB methods only), `diffusion/noise_process.py` helpers | MATCH across 7 checks — decomposition, `kl_prior`, `compute_Lt`, `reconstruction_logp`, `log_pN`, masking (our path masks internally in `_categorical_kl_per_graph`, so the dead `mask_distributions` helper is safe to delete), `T` multiplier all correct. |
| Collation & `GraphData` | `data/datasets/graph_types.py`, `data/data_modules/multigraph_data_module.py` | MATCH across 8 checks — E encoding (channel 0 = no-edge), diagonal convention (both codebases produce all-zero diagonals filtered by `(true != 0).any(-1)`), node_mask, padding, y construction, symmetry enforcement all match. |
| **Synthetic data module** | `data/data_modules/synthetic_categorical.py`, `experiment_utils/data/sbm.py` | **STRUCTURAL FINDING** — see below. |

### Structural finding: the training distribution is a point mass

The `SyntheticCategoricalDataModule` with `p_intra=1.0, p_inter=0.0,
num_blocks=2, num_nodes=20, num_graphs=200, seed=1` produces **200
bit-identical graphs** with a **fixed canonical partition**: nodes 0-9
always in block 0, nodes 10-19 always in block 1, edge between i and j
iff they are in the same block. Zero structural diversity. No
per-graph partition resampling. No node relabeling. The 128/32/40
train/val/test split matches upstream SPECTRE counts, and every graph
in every split is the same graph.

This is a data-side pathology, not a parity bug against upstream (which
uses the SPECTRE fixture with genuine variability). But it has strong
implications for interpreting the frozen-metrics run:

- The model's training target is literally one graph. Best-possible
  x̂_0 prediction, if the model can identify the partition, is a
  deterministic point mass.
- A permutation-equivariant transformer has no intrinsic way to tell
  which nodes are in which block. The only positional signal available
  is `ExtraFeatures(noisy_z_t)` — Laplacian eigenvectors of the current
  noisy graph. At high t those eigenvectors are nearly random; at low
  t they carry partition information.
- At high t (most of training), the theoretical-best x̂_0 prediction
  is the data marginal, `P(edge) ≈ 0.47` at every position. CE loss
  floor at that regime is ≈ 0.69 nats per edge position.
- We observe `train/loss_step` bottoming at **0.24**, well below the
  0.69 marginal-prediction floor. This means the model *is* finding
  some form of positional signal during training — most likely through
  the fixed-canonical-partition leaking into attention via node-index
  order, or through how the `y` graph-level features interact with
  edge predictions. But this learned shortcut does not transfer to
  sampling (where node order during generation is still fixed but the
  model's internal representation apparently doesn't stabilise on the
  training partition), yielding the edge-collapsed output (e≈165).

### Consolidated root-cause status

Across the full audit — 6 parity audits + 5 sanity reviews + 5
candidate-cause audits + 6 deep behavioural audits, on every
conceptual layer of the pipeline — the **only confirmed code-level
parity divergence** is `_epsilon_renormalise(1e-7)` in the training CE
(applied to both predictions and targets). Option B (see
`/home/igork/.claude/plans/dapper-napping-lark.md`) is being
implemented by an Opus subagent to remove it.

The symptom (edge-collapse to e≈165) has two plausible co-causes:

1. **The parity bug (being fixed).** Epsilon smoothing on targets
   combined with `lambda_E = 5.0` attenuates the gradient on "correctly
   predict no-edge" relative to upstream's hard CE, biasing the model
   toward high-P(edge) solutions.
2. **The data pathology (unfixed).** The training set has zero
   structural diversity, so any positional shortcut the model finds at
   training time is specific to that fixed canonical partition. The
   sampler at inference does not reproduce the shortcut's conditions
   reliably, yielding collapse.

Even if #1 is fully resolved, #2 is likely to produce pathological
behaviour on any future run with this specific data config. The
conservative next step after Option B lands is to re-run with one of:

- the SPECTRE fixture end-to-end (what upstream does; proven on our
  side by the short 2026-04-16 panel run which does not edge-collapse)
- `p_inter = 0.005` or similar on synthetic data (breaks the point-mass)
- random node-relabeling per graph in the synthetic generator (breaks
  the canonical-partition shortcut)

### Audit-gap: none remaining

No unaudited conceptual surface of the pipeline remains. If Option B
does not resolve the frozen metrics on the upstream-match config,
further investigation requires either a training-run ablation (not
possible this session) or switching to a non-degenerate training
distribution.

## 2026-04-21 (continued) — Option B landed on main

Three commits, implemented by an Opus subagent per
`/home/igork/.claude/plans/dapper-napping-lark.md`:

- `f6d99185` — `refactor(loss): match upstream DiGress CE parity -- F.cross_entropy on logits, drop epsilon smoothing`
- `59e9593f` — `chore(diffusion): remove dead mask_distributions helper`
- `3760dd9b` — `test(loss): add upstream-parity regression for masked CE helpers`

Files touched:
- `src/tmgg/training/lightning_modules/train_loss_discrete.py` — rewrote
  `masked_node_ce` / `masked_edge_ce` to take raw logits and dispatch to
  `F.cross_entropy` with `label_smoothing: float = 0.0` keyword-only
  parameter (default preserves upstream bit-parity). Removed
  `_epsilon_renormalise`, `_node_mask_fill_uniform`,
  `_edge_mask_fill_uniform`.
- `src/tmgg/training/lightning_modules/diffusion_module.py` — dropped
  `F.softmax` pre-processing in `_compute_loss`; pass raw logits
  directly. Added `target = target.mask()` before the CE helpers to
  zero padding rows (mirrors upstream's `dense_data.mask(node_mask)`
  step). Modified the `X_class` synthesis in `_read_field` to emit
  all-zero rows at padding positions so the `(true != 0).any(-1)`
  predicate drops them for structure-only batches.
- `src/tmgg/diffusion/diffusion_sampling.py` — deleted
  `mask_distributions` (46 lines, zero callers).
- `tests/models/test_train_loss.py` — added `TestUpstreamParity` class
  (three regression tests pinning bit-for-bit equivalence to upstream's
  `F.cross_entropy(reduction='sum') / num_valid_rows` at `atol=1e-6`,
  plus a `label_smoothing` pass-through check). Updated the five
  pre-existing tests to feed raw logits; tolerances tightened from
  `0.01-0.15` to `1e-5`/`1e-6` because the new path has no epsilon
  drift.
- `tests/experiments/test_discrete_diffusion_module.py` — dropped the
  now-obsolete `F.softmax` pre-processing at the `TrainLossDiscrete`
  equivalence assertion (line ~378). `atol=1e-6` still holds because
  both sides share the same `F.cross_entropy` call.

Test result: `1274 passed, 8 skipped, 217 deselected (slow)`. One
`basedpyright` issue in the new test code (`torch.randint(...).item()`
returning `Number` not accepted as an index) was fixed at root cause
with `int(...)` — no silencing.

Two scope-creep items Opus added (both justified): `target.mask()` in
`_compute_loss` and the `_read_field` X_class padding change. Both are
required for the `(true != 0).any(-1)` row predicate to correctly drop
padding positions — upstream's equivalent `dense_data.mask(node_mask)`
was not previously replicated in our path, and without these changes
the new CE would count padding rows as valid no-edge targets. In-scope
for "behavioural upstream parity" as the plan specified.

### Pre-existing issues flagged by Opus, not fixed (out of scope)

- `tests/modal/test_functions.py::test_run_import_preflight_reports_failing_module`
  fails with `SIGILL` on `import ot` on this host.
- `tests/test_config_composition.py` and
  `tests/test_refactoring_regression.py` (6 failures total) —
  `GraphDataModule.__init__() got an unexpected keyword argument
  'eval_meta'` and missing `noise_type` in some single-graph configs.
- 6 basedpyright errors in `src/tmgg/diffusion/`,
  `src/tmgg/training/` — `reportConstantRedefinition` on `M`, `A` and
  `reportAttributeAccessIssue` on `GraphModel` imports. All verified
  pre-existing on `main` before these commits.

### Behavioural-parity claim is now pinned by tests

After these commits, with `label_smoothing=0.0` (the default and what
all current call sites use), `masked_node_ce` and `masked_edge_ce` are
bit-for-bit equivalent to upstream's
`CrossEntropyMetric.update(logits, argmax(target))` → `F.cross_entropy
(logits, argmax(target), reduction='sum')` followed by the per-metric
running mean. The `TestUpstreamParity` regression tests enforce this at
`atol=1e-6`.

### What's still unvalidated

The edge-collapse symptom itself (e≈165 generated edges on the
upstream-match run) has not been empirically verified to resolve —
that requires a training run we did not launch. The parity divergence
is eliminated; whether the divergence was sufficient to cause the
observed collapse (on top of the synthetic-data point-mass structural
issue) remains an empirical question.

### Suggested next step (out of scope for this session)

One of:

- Re-run the upstream-match config on Modal for a few thousand steps
  and inspect the adjacency sample output. If edges collapse again at
  the same density, the data-side pathology dominates; switch to
  SPECTRE fixture or add `p_inter > 0`.
- Skip directly to the SPECTRE fixture end-to-end (matching the
  published DiGress setup) — bypasses both failure modes at once.

Either path is data-side; no further code changes are indicated by
this audit.

## 2026-04-21 (continued) — 15-section per-section review: surviving divergences

After the parity commits landed, a 15-agent per-section review mapped each
spec layer onto our implementation. Reports at
`tmgg/docs/reports/2026-04-21-digress-spec-our-impl-review/`. The full
surviving divergences, with explicit training-failure-impact assessment:

| # | Section | Divergence | Training-failure impact |
|---|---|---|---|
| D1 | §6 Forward noising | `t ∈ {1..T}` not `{0..T}` (we never sample t=0) | **LOW.** 0.1% of training expectation missed at T=1000. Uniform across positions; no edge-bias mechanism. |
| D2 | §6 Forward noising | `forward_sample` returns `GraphData` not a dict with `beta_t/alpha_s_bar` | **NONE.** Plumbing divergence; values re-derived in VLB path from `t_int`. |
| D3 | §11 VLB | Marginalised `kl_diffusion` predicted posterior (`Σ_c p(z_s\|z_t,x_0=c)·p_θ(x_0=c\|z_t)`) vs upstream's direct soft-x₀ Bayes | **NONE.** VLB-reporting only. Does not feed back into training gradient. |
| D4 | §11 VLB | `reconstruction_logp` via `z_1` + marginalised posterior vs upstream's `z_0` + raw softmax | **NONE.** VLB-reporting only. Bounded at ~1e-3 abs. for T=1000. |
| D5 | §11 VLB | `log_pN` silent zero fallback when `_size_distribution is None` | **NONE** (path not taken in normal training). Cleanup item: should be loud-raise. |
| D6 | §13 Evaluator | `compute_sbm_accuracy` uses default `refinement_steps=1000`; upstream's caller passes 100 | **NONE for edge-collapse.** Affects only reported `val/gen/sbm_accuracy` reliability (10× MCMC budget → potentially *better* fits, which could paradoxically reject more samples, but not in a direction that explains e=165 outputs). Evaluator is stateless and correctly decodes our samples; it reports what the sampler emits. |
| D7 | §14 Lightning | LR cosine_warmup scheduler silently inherited; upstream has no scheduler | **UNLIKELY.** Cosine decay slows late-training LR. Could mildly bias convergence but would not produce a stable edge-everywhere attractor. Trainings where this bites would manifest as under-fitting, not mode collapse. |
| D8 | §14 Lightning | `gradient_clip_val=1.0` default (upstream: null) | **UNLIKELY.** Clipping at norm 1 stabilises training; does not introduce directional bias toward any prediction class. Could mask pathological gradients but not create edge-preference. |
| D9 | §15 Hyperparameters | `max_n_nodes=20` baked into model config (inherited from synthetic default) | **NONE for failing run** (synthetic-SBM data is n=20; `max_n_nodes=20` is correct). **Latent trap** if the same model config is reused with SPECTRE data (n ∈ [44,187]): `n/max_n_nodes > 1` would break the size-feature normalisation. Separate pre-run hazard, not the current cause. |
| D10 | §15 Hyperparameters | `diffusion_steps=500` default in static YAML; runs override to 1000 via CLI | **NONE for failing run** (CLI explicitly sets T=1000). Latent misconfiguration risk for future runs. |
| D11 | §1 Problem repr | `X_class=None` synthesised as `dX=2` vs upstream's trivial `dX=1` node-type | **NONE.** No active X-loss on structure-only SBM; the synthesised target is dropped by the row predicate on padding. |

### Summary

**No surviving divergence plausibly causes the observed edge-collapse.**
All VLB-side divergences (D3, D4, D5) affect only reported metrics, not
the training gradient. Forward-sampling and Lightning divergences
(D1, D7, D8) act uniformly on the objective and have no mechanism to
produce an edge-everywhere attractor. Evaluator (D6) faithfully reports
whatever the sampler emits.

The only confirmed gradient-path divergence — `_epsilon_renormalise` in
the CE helpers (§9 candidate #2) — was eliminated in commit `f6d99185`
and is now pinned at `atol=1e-6` by `TestUpstreamParity`.

If a future training run on the fixed code still shows edge-collapse,
the cause is no longer in parity with upstream. It would be:
1. A problem in the learned model behaviour on this specific
   degenerate dataset (training distribution = literally one graph
   repeated 200 times per §deep-synthetic-data audit);
2. Or an unidentified divergence outside the sections audited — but the
   coverage is exhaustive (every layer from data loading through loss
   and sampling to evaluation has a section).

### Cleanup backlog surfaced by the review

Not fixed in this session, but worth tracking:
- D5 (`log_pN` silent fallback): convert to raise per CLAUDE.md fail-loud convention.
- D6 (`refinement_steps` hardcoded): expose as `GraphEvaluator.__init__` kwarg and pass 100 to match upstream's live value.
- D7 (LR scheduler): add `scheduler_config: {type: none}` to `models/digress/*.yaml` for strict-parity runs, or document the scheduler is tmgg's intentional addition.
- D8 (gradient clipping): confirm whether `gradient_clip_val=1.0` is intentional tmgg policy; if so, document in a spec override note, not just inherited.
- D9 / D10 (config traps): add invariant assertions (e.g., `assert max_n_nodes >= dataset.num_nodes_max`; fail loudly if `timesteps` disagrees between data and model configs).
