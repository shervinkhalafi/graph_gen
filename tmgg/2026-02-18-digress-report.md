# DiGress Implementation Audit: TMGG vs. cvignac/DiGress

**Date**: 2026-02-18
**Upstream ref**: https://github.com/cvignac/DiGress (`src/` directory)
**TMGG ref**: `src/tmgg/models/digress/`, `src/tmgg/experiments/discrete_diffusion_generative/`, `src/tmgg/experiments/digress_denoising/`

This report compares every DiGress component in TMGG against the canonical implementation by Vignac et al. (2023). Components are scored as **identical**, **faithful** (same algorithm, minor interface changes), **adapted** (intentional divergences for our research goals), or **missing/stubbed**.

---

## 1. Model Architecture

### 1.1 Core Transformer (`_GraphTransformer` / `GraphTransformer`)

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Input MLPs | `Linear(in → hidden) → ReLU → Linear(hidden → dx) → ReLU` for X, E, y | Identical structure | **Identical** |
| Edge symmetrization (input) | `(mlp_in_E(E) + transpose) / 2` | Same | **Identical** |
| Transformer stack | `n_layers` of `XEyTransformerLayer` | Same | **Identical** |
| Output MLPs | `Linear(dx → hidden) → act_fn → Linear(hidden → out)` for X, E, y | Same | **Identical** |
| Skip connection | `X_out = X_out + X_to_out` (first `out_dim` features of MLP input) | Same | **Identical** |
| Diagonal mask on E | `E = E * diag_mask` (zeros on diagonal) | Same | **Identical** |
| Final edge symmetrization | `(E + E^T) / 2` | Same | **Identical** |
| `act_fn_in` / `act_fn_out` | `nn.ReLU()` default, configurable | Same | **Identical** |

**TMGG addition**: `_GraphTransformer` has an `assume_adjacency_input` parameter (default `True`). When `True` and input X is square, it extracts the diagonal as node features and uses the full matrix as edge features. When `False` (used by `DiscreteGraphTransformer`), X and E are passed through as-is. This does not affect the transformer computation itself — only how inputs are pre-processed.

### 1.2 XEyTransformerLayer

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Constructor params | `dx, de, dy, n_head, dim_ffX=2048, dim_ffE=128, dim_ffy=2048` | Same signature + optional GNN/spectral params | **Faithful** |
| Post-attention residual + LayerNorm | Yes, per stream (X, E, y) | Same | **Identical** |
| Feed-forward | `Linear(d → dim_ff) → ReLU → dropout → Linear(dim_ff → d) → dropout` | Same | **Identical** |
| Post-FFN residual + LayerNorm | Yes | Same | **Identical** |

**TMGG additions**: The TMGG layer accepts optional `use_gnn_q/k/v`, `gnn_num_terms`, `use_spectral_q/k/v`, `spectral_k`, `spectral_num_terms` parameters for GNN and spectral projection modes. These are **not in upstream** and constitute a research extension. When all default to `False`, the layer is identical to upstream.

**Upstream bug note**: `dim_ffy` is present in the config (`configs/model/discrete.yaml`) and the `XEyTransformerLayer` constructor signature, but upstream's `GraphTransformer.__init__` does not extract `dim_ffy` from `hidden_dims` — it passes only `dim_ffX` and `dim_ffE` to the layer constructor, letting `dim_ffy` silently default to 2048. TMGG fixed this by passing all three `dim_ff` values through (commit `facfc17`).

### 1.3 NodeEdgeBlock (Attention)

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Q/K/V projections | `Linear(dx → dx)` | Same (default); also supports GNN/spectral projections | **Faithful** |
| Attention computation | `Q * K / sqrt(df)` (element-wise, not dot product) | Same | **Identical** |
| Edge FiLM on attention | `Y * (e_mul(E) + 1) + e_add(E)` | Same | **Identical** |
| Global → Edge FiLM | `ye_add + (ye_mul + 1) * newE` | Same | **Identical** |
| Global → Node FiLM | `yx_add + (yx_mul + 1) * weighted_V` | Same | **Identical** |
| Edge output projection | `Linear(dx → de)` via `e_out` | Same | **Identical** |
| Global y update | `y_y(y) + x_y(X) + e_y(E)` through 2-layer MLP | Same | **Identical** |
| Xtoy aggregation | `[mean, min, max, std] → Linear(4*dx, dy)` | Same | **Identical** |
| Etoy aggregation | `[mean, min, max, std] → Linear(4*de, dy)` | Same | **Identical** |
| masked_softmax | Sets masked → `-inf`, softmax, NaN → 0 | Same | **Identical** |

**TMGG addition**: When `use_gnn_*` or `use_spectral_*` are enabled, the Q/K/V projections use `BareGraphConvolutionLayer` or `SpectralProjectionLayer` instead of plain `Linear`. These are mutually exclusive per projection. This is a research extension not present in upstream.

### 1.4 Wrappers

| Component | Upstream | TMGG | Verdict |
|-----------|----------|------|---------|
| Denoising wrapper | `GraphTransformer` takes adjacency `(bs, n, n)`, returns edge logits `(bs, n, n)` | Same, via `GraphTransformer(DenoisingModel)` | **Faithful** |
| Generative wrapper | Not a separate class; the `DiscreteDenoisingDiffusion` module calls `GraphTransformer` directly with categorical `(X, E, y, node_mask)` | Separate `DiscreteGraphTransformer` class accepting `(X, E, y, node_mask)`, setting `assume_adjacency_input=False`, returning `GraphFeatures` | **Adapted** |
| Eigenvector embedding | Not in the transformer; handled via `ExtraFeatures` external to the model | `DiscreteGraphTransformer` has `use_eigenvectors` + `TopKEigenLayer` that extracts eigenvectors *inside* the model and concatenates with X | **Adapted** |

The wrapping separation in TMGG is an intentional design choice: the denoising experiment uses `GraphTransformer` (which takes raw adjacency), while the generative experiment uses `DiscreteGraphTransformer` (which takes categorical one-hot). Upstream has one `GraphTransformer` class that always receives categorical features from the diffusion module.

---

## 2. Noise Schedule

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| `PredefinedNoiseScheduleDiscrete` | Stores betas, alphas_bar as buffers | Same | **Identical** |
| Cosine schedule | `cosine_beta_schedule_discrete(T, s=0.008)` | Same function, same formula | **Identical** |
| Custom schedule | `custom_beta_schedule_discrete(T, avg_num_nodes=50, s=0.008, num_edge_classes=2)` with beta floor `1.2 / (p * num_edges)` | Same function, same formula | **Identical** |
| `forward(t_normalized, t_int)` | Returns `betas[t_int]` | Same | **Identical** |
| `get_alpha_bar(...)` | Returns `alphas_bar[t_int]` | Same | **Identical** |
| `clip_noise_schedule` | Clips `alpha_t / alpha_{t-1}` ratio | Present in `diffusion_utils.py` | **Identical** |
| Alpha computation | `alphas = 1 - clamp(betas, 0, 0.9999)`, `alphas_bar = exp(cumsum(log(alphas)))` | Same | **Identical** |

**Note**: `num_edge_classes` is exposed as a constructor parameter in TMGG (default 2), matching upstream's hardcoded default.

---

## 3. Transition Matrices

| Component | Upstream | TMGG | Verdict |
|-----------|----------|------|---------|
| `DiscreteUniformTransition` | `Qt = (1-beta)*I + beta*(1/K)` | Same formula | **Identical** |
| `MarginalUniformTransition` | `Qt = (1-beta)*I + beta*M` where M rows = marginals | Same formula | **Identical** |
| `get_Qt_bar` | `alpha_bar*I + (1-alpha_bar)*U/M` | Same formula | **Identical** |
| `get_limit_dist` | Returns 1-D marginals (or uniform) | Same | **Identical** |
| Beta reshape | Upstream uses `unsqueeze(1)` → `(bs, 1)` which only broadcasts correctly when `bs == K` | TMGG uses `reshape(-1, 1, 1)` → `(bs, 1, 1)`, fixing a broadcasting edge case | **Fixed** |
| `AbsorbingStateTransition` | Present in upstream | **Not implemented** | **Missing** |

The absorbing state transition is only used for absorbing diffusion experiments, which are not part of TMGG's research scope.

---

## 4. Diffusion Utilities

| Utility | Upstream | TMGG | Verdict |
|---------|----------|------|---------|
| `PlaceHolder` | `(X, E, y)` container with `type_as`, `mask(node_mask, collapse)` | Same, including symmetry assertion in `mask()` | **Identical** |
| `sample_discrete_features(probX, probE, node_mask)` | Multinomial sampling, upper-triangle + mirror for symmetry | Same | **Identical** |
| `compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M)` | `z_t @ Qt^T * x_0 @ Qsb / (x_0 @ Qtb @ z_t^T)` | Same | **Identical** |
| `compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb)` | Posterior over all possible x_0 values, denominator clamped to 1e-6 | Same | **Identical** |
| `mask_distributions(true_X/E, pred_X/E, node_mask)` | Sets masked to one-hot class 0, adds 1e-7, renormalises | Same | **Identical** |
| `sample_discrete_feature_noise(limit_dist, node_mask)` | Samples from limit dist, one-hot encodes, symmetrises E | Same | **Identical** |
| `posterior_distributions(...)` | Wrapper calling `compute_posterior_distribution` for X and E | Same | **Identical** |
| `sum_except_batch` | `x.reshape(bs, -1).sum(dim=-1)` | Same | **Identical** |
| `assert_correctly_masked` | Checks masked positions are near-zero | Same | **Identical** |
| Gaussian utilities | `gaussian_KL`, `sigma`, `alpha`, `SNR`, etc. | Present in `diffusion_utils.py` (carried over from upstream, unused in discrete pipeline) | **Identical** |

---

## 5. Training Loss

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Loss class | `TrainLossDiscrete` | Same name and structure | **Faithful** |
| Loss formula | `loss_X + lambda_E * loss_E + lambda_y * loss_y` | `loss_X + lambda_E * loss_E` (no y loss) | **Adapted** |
| `lambda_train` param | List `[5, 0]` (lambda_E=5, lambda_y=0) | Scalar `lambda_E=5.0` | **Adapted** |
| Cross-entropy | `F.cross_entropy(pred_flat, argmax(true_flat))` after filtering masked rows | `-sum(true * log(pred))` per position, with explicit masking | **Faithful** |
| Masking | Filters out all-zero rows (masked positions) | Uses `node_mask` and `diag_mask` to weight loss | **Faithful** |
| Normalisation | Per-graph: divides by number of valid nodes / edges | Same | **Faithful** |

The upstream uses `F.cross_entropy` which internally computes `-sum(target * log(softmax(pred)))`. TMGG receives already-softmaxed predictions and computes `-sum(true * log(pred))` directly. Mathematically equivalent for one-hot targets.

The y loss term with `lambda_y=0` is effectively disabled in upstream's default config. TMGG drops it entirely, which is correct for the synthetic SBM setting where `y` is always empty.

---

## 6. Training Loop

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| `apply_noise` | Sample `t ~ U[0, T]` (train) or `[1, T]` (eval), compute `Qtb`, multiply one-hot by `Qtb`, sample, one-hot encode | Same algorithm, same train/eval distinction | **Identical** |
| `forward` | Concatenate noisy + extra features, append t to y, run transformer, softmax X and E | Same | **Identical** |
| `training_step` | `apply_noise → compute_extra_data → forward → TrainLossDiscrete` | Same, with explicit `.clone()` before loss (to prevent in-place modification of autograd graph) | **Identical** |
| Timestep normalisation | `t_float = t_int / T`, appended to y | Same | **Identical** |
| `compute_extra_data` | Calls `extra_features(X_t, E_t, y_t, node_mask)` | Same | **Identical** |

### 6.1 Data format at training boundary

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Input format | PyG sparse `Data` → `to_dense(x, edge_index, edge_attr, batch)` → `(X, E, y, node_mask)` | Directly receives `(X, E, y, node_mask)` tuple from dataloader with `collate_categorical` | **Adapted** |
| `to_dense` conversion | `utils.to_dense` + `encode_no_edge` | Not needed — data is generated in dense categorical format | **Adapted** |

This is an intentional architectural choice: TMGG generates synthetic graphs as dense adjacency matrices and converts them to categorical format in the datamodule, so the Lightning module never sees PyG sparse format.

---

## 7. Validation Loss (Variational Lower Bound)

| Component | Upstream | TMGG | Verdict |
|-----------|----------|------|---------|
| NLL formula | `kl_prior + loss_all_t - reconstruction_logp` | Same | **Identical** |
| `kl_prior` | `F.kl_div(log(q(z_T|x_0)), limit_dist)`, note: upstream uses reverse KL direction | Same, with explicit comment about the reverse KL direction | **Identical** |
| `compute_Lt` | True posterior `q(z_{t-1}|z_t, x_0)` vs predicted posterior, KL, scaled by T | Same | **Identical** |
| `reconstruction_logp` | Apply one noise step from clean, predict, compute `X * log(pred_X) + E * log(pred_E)` | Same | **Identical** |
| Masked positions in reconstruction | Set to probability 1 (log=0, contributes nothing) | Same | **Identical** |
| `log_pN` term | `DistributionNodes.log_prob(N)` — log probability of node count under empirical distribution | **Not included** | **Missing** |
| VLB accumulators | Mean over validation epoch | Same pattern, using list accumulators | **Faithful** |

The `log_pN` term is missing because TMGG currently uses fixed-size graphs where the node count distribution is degenerate (all graphs have the same size). Once variable-size datasets are integrated via `SizeDistribution`, this term should be added. Its omission introduces a constant offset to the NLL that doesn't affect training or relative model comparison within fixed-size experiments.

---

## 8. Sampling (Reverse Diffusion)

| Aspect | Upstream | TMGG Generative | Verdict |
|--------|----------|-----------------|---------|
| Initial noise | Sample from `limit_dist` via `sample_discrete_feature_noise` | Same | **Identical** |
| Reverse loop | `for s in T-1..0: sample_p_zs_given_zt(s, t, X, E, y, mask)` | Same | **Identical** |
| Reverse step | Predict p(x_0\|z_t), compute `p_s_and_t_given_0` for all x_0, weight by prediction, marginalize, normalise, sample | Same algorithm | **Identical** |
| Normalisation guard | `unnormalized_prob[sum == 0] = 1e-5` | Same | **Identical** |
| Symmetry enforcement | Enforced by `sample_discrete_features` (upper triangle + mirror) | Same | **Identical** |
| Symmetry assertion | `assert E == E^T` at each step | Same (as `RuntimeError` check) | **Identical** |
| Final collapse | `PlaceHolder.mask(node_mask, collapse=True)` — argmax, masked → -1 | Same | **Identical** |
| Node count sampling | `self.node_dist.sample_n(batch_size)` from empirical `DistributionNodes` | Accepts `num_nodes` as int or Tensor; `sample_n_nodes()` available via `SizeDistribution` | **Adapted** |
| Return format | `list[(atom_types, edge_types)]` per graph, trimmed to real node count | Same | **Identical** |

### TMGG Denoising Experiment Sampling

The `DigressDenoisingLightningModule.sample()` uses a **simplified iterative denoising** approach (not true categorical reverse diffusion): starts from random binary graphs, iteratively interpolates `alpha * prediction + (1-alpha) * current`, binarises, and enforces symmetry. This is intentionally different from the generative experiment — it serves as a comparison baseline for the denoising paradigm.

---

## 9. Extra Features

| Component | Upstream | TMGG | Verdict |
|-----------|----------|------|---------|
| `DummyExtraFeatures` | Returns zero-width tensors | Same | **Identical** |
| `ExtraFeatures` | Computes cycles + eigenfeatures from noisy graph | Same class, same modes | **Identical** |
| `extra_features_dims` | `"cycles"→(3,0,5)`, `"eigenvalues"→(3,0,11)`, `"all"→(6,0,11)` | Same | **Identical** |
| Interface | `__call__(noisy_data)` taking a dict | `__call__(X, E, y, node_mask)` taking explicit tensors | **Adapted** |
| `NodeCycleFeatures` / `KNodeCycles` | k=3,4,5 per-node, k=3,4,5,6 per-graph, scaled /10, clamped to 1.0 | Same formulas, same scaling | **Identical** |
| `EigenFeatures` | Laplacian eigendecomposition, masked-diagonal trick, `"eigenvalues"` and `"all"` modes | Same algorithm, same constants (k=5 eigenvalues, k=2 eigenvectors) | **Identical** |
| `get_eigenvalues_features` | Connected component count + first k non-zero eigenvalues | Same | **Identical** |
| `get_eigenvectors_features` | LCC indicator via `torch.mode` + lowest k eigenvectors | Same | **Identical** |
| `compute_laplacian` | Combinatorial L = D - A, or normalised L_sym | Same | **Identical** |
| Normalised node count | `n_valid / max_n_nodes` appended to y | Same | **Identical** |

The interface change (dict → explicit arguments) was an intentional adaptation to TMGG's convention of passing tensors explicitly rather than wrapping them in dictionaries.

### Extra features not used by default

The generative experiment's default config uses `DummyExtraFeatures` (no structural augmentation). The base config has commented-out lines showing how to enable `ExtraFeatures` with the correct `input_dims` adjustments. This matches upstream's SBM experiment config which also uses `null` extra features.

---

## 10. Data Handling

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Data format | PyG sparse `Data` objects, converted to dense at training time | Dense categorical tuples from the start | **Adapted** |
| Datasets | SPECTRE downloads (SBM, planar, comm-20), molecular (QM9, MOSES, GuacaMol) | Synthetic generation (SBM, ER, tree, etc.) via `SyntheticGraphDataset` | **Adapted** |
| Node features | `ones(n, 1)` for SPECTRE; multi-class for molecular | `[0,1]` one-hot (2 classes: absent/present) via `adjacency_to_categorical` | **Faithful** |
| Edge features | 2 classes (no-edge, edge) for SPECTRE | Same: 2 classes via `adjacency_to_categorical` | **Identical** |
| Collation | PyG batching with `to_dense_batch` / `to_dense_adj` at train time | `collate_categorical`: pads variable-size tuples to max node count, class-0 for padding | **Adapted** |
| Marginals | Computed from `DataModule.node_types()` / `edge_counts()` via dataset iteration | Computed in `SyntheticCategoricalDataModule._compute_marginals()` from training data | **Faithful** |
| Node count distribution | `DistributionNodes(histogram)` wrapping `torch.distributions.Categorical` | `SizeDistribution(sizes, counts, max_size)` frozen dataclass with `sample()` | **Adapted** |
| `sample_n_nodes` | Directly on `DistributionNodes.sample_n(batch_size)` | `datamodule.sample_n_nodes(batch_size)` delegating to `SizeDistribution` | **Faithful** |

Key differences:
- Upstream uses PyG's sparse-to-dense conversion at every training step; TMGG stores data in dense format throughout, avoiding repeated conversion overhead.
- TMGG's `SizeDistribution` uses a sparse representation (only stores sizes that actually appear) rather than a dense histogram.
- Upstream downloads pre-generated datasets from the SPECTRE repository; TMGG generates them on-the-fly with configurable parameters.

---

## 11. Evaluation Metrics

| Metric | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Degree MMD | `degree_stats()` → histogram → MMD | `compute_degree_histogram` → `compute_mmd` | **Faithful** |
| Clustering MMD | `clustering_stats()` → 100 bins over [0,1] → MMD | `compute_clustering_histogram` → 100 bins, same range | **Identical** |
| Spectral MMD | `spectral_stats()` → normalised Laplacian eigenvalues, 200 bins, range [-1e-5, 2], PMF | `compute_spectral_histogram` → same parameters | **Identical** |
| MMD kernel | `gaussian_tv`: `exp(-TV^2 / 2σ^2)` default, also `gaussian_emd` and `gaussian` | `gaussian_tv` default, also `gaussian`. No `gaussian_emd` (requires `pyemd`) | **Faithful** |
| MMD estimator | Unbiased: `k11 + k22 - 2*k12` using unique pairs | Same, via `itertools.combinations` | **Identical** |
| Orbit MMD | `orbit_stats_all()` via ORCA (4-node, 15 orbit counts) | **Not implemented** | **Missing** |
| SBM accuracy | `eval_acc_sbm_graph()` via graph-tool blockmodel inference + Wald test | **Not implemented** | **Missing** |
| Planarity check | `eval_acc_planar_graph()` — connected + planar fraction | **Not implemented** | **Missing** |
| Uniqueness / novelty | `eval_fraction_unique()`, `eval_fraction_isomorphic()`, etc. | **Not implemented** | **Missing** |
| Molecular metrics | SMILES validity, QED, etc. via RDKit | **Not implemented** (out of scope) | **Not applicable** |

TMGG evaluates with degree, clustering, and spectral MMD — the three metrics that are universal across all graph types. ORCA orbit counting requires external C++ tools, and SBM accuracy requires `graph-tool`, which adds heavy dependencies. These could be added if needed for paper-ready benchmarking.

---

## 12. Configuration

| Aspect | Upstream | TMGG | Verdict |
|--------|----------|------|---------|
| Config system | Hydra with `config.yaml` root | Hydra with `base_config_discrete_diffusion_generative.yaml` + defaults | **Faithful** |
| Default T | 500 | 500 | **Identical** |
| Default schedule | `cosine` | `cosine` | **Identical** |
| Default transition | `marginal` | `marginal` | **Identical** |
| Default n_layers | 5 | 5 | **Identical** |
| Default hidden_dims | `{dx:256, de:64, dy:64, n_head:8, dim_ffX:256, dim_ffE:128, dim_ffy:128}` | Same in `discrete_default.yaml` | **Identical** |
| Default hidden_mlp_dims | `{X:256, E:128, y:128}` | Same | **Identical** |
| Default lambda_E | 5 | 5.0 | **Identical** |
| Default optimizer | AdamW, lr=0.0002, weight_decay=1e-12, amsgrad=True | Same in `discrete_sbm_official.yaml` | **Identical** |
| Default batch_size | 512 | 32 (configurable) | **Adapted** |
| Training unit | Epochs (`n_epochs=1000`) | Steps (per TMGG convention) | **Adapted** |
| Default extra_features | `"all"` for SBM in paper results | `DummyExtraFeatures` (disabled) | **Adapted** |

The batch size and training unit differences are intentional design choices. TMGG uses step-based training throughout (per project convention), and the smaller default batch size reflects the smaller-scale synthetic experiments. The `discrete_sbm_official.yaml` config matches upstream's optimizer settings exactly for reproduction.

---

## 13. Missing Components (Not in TMGG)

| Component | Upstream location | Reason for omission |
|-----------|-------------------|---------------------|
| `AbsorbingStateTransition` | `noise_schedule.py` | Not used in published DiGress results for SBM/synthetic |
| `LiftedDenoisingDiffusion` (continuous model) | `diffusion_model.py` | Out of scope — TMGG focuses on discrete diffusion |
| ORCA orbit metrics | `analysis/spectre_utils.py` | Requires external C++ binary; can be added for benchmarking |
| SBM / planarity accuracy metrics | `analysis/spectre_utils.py` | Requires `graph-tool` dependency |
| Uniqueness / novelty metrics | `analysis/spectre_utils.py` | Straightforward to add when needed |
| `gaussian_emd` kernel | `analysis/dist_helper.py` | Requires `pyemd` dependency |
| `log_pN` in VLB | `diffusion_model_discrete.py` | Zero for fixed-size graphs; should be added with variable-size support |
| Molecular datasets | `datasets/qm9_dataset.py` etc. | Out of scope |
| Molecular metrics | `metrics/molecular_metrics*.py` | Out of scope |
| EMA | `train_default.yaml` `ema_decay: 0` (disabled by default upstream too) | Not implemented, not needed (disabled in upstream) |
| Visualization tools | `analysis/visualization.py` | Handled differently in TMGG (via base Lightning module) |

---

## 14. TMGG Additions (Not in Upstream)

| Feature | Location | Purpose |
|---------|----------|---------|
| GNN Q/K/V projections | `transformer_model.py`, `BareGraphConvolutionLayer` | Research: polynomial graph convolution as projection in attention |
| Spectral Q/K/V projections | `transformer_model.py`, `SpectralProjectionLayer` | Research: eigenvalue-based filtering as projection in attention |
| `TopKEigenLayer` inside transformer | `discrete_transformer.py` | Embeds eigenvectors as node features within the model, rather than as external extra features |
| `DigressGraphTransformerEmbedding` | `transformer_model.py` | Embedding variant returning node features instead of edge logits |
| `SizeDistribution` dataclass | `size_distribution.py` | Sparse, serializable graph-size distribution for variable-size sampling |
| Denoising experiment with MMD | `digress_denoising/lightning_module.py` | Cross-paradigm comparison between denoising and generative diffusion |
| Evaluate CLI | `evaluate_cli.py` | Command-line tool for checkpoint evaluation with configurable reference datasets |

---

## 15. Summary Scorecard

| Category | Identical | Faithful | Adapted | Missing |
|----------|-----------|----------|---------|---------|
| Core transformer | 12 | 1 | 0 | 0 |
| Attention mechanism | 10 | 1 | 0 | 0 |
| Noise schedule | 7 | 0 | 0 | 0 |
| Transition matrices | 5 | 0 | 0 | 1 (`AbsorbingState`) |
| Diffusion utilities | 10 | 0 | 0 | 0 |
| Training loss | 0 | 3 | 2 | 0 |
| Training loop | 4 | 0 | 2 | 0 |
| VLB | 5 | 1 | 0 | 1 (`log_pN`) |
| Sampling | 8 | 1 | 1 | 0 |
| Extra features | 10 | 0 | 1 | 0 |
| Data handling | 0 | 3 | 5 | 0 |
| Evaluation | 3 | 2 | 0 | 4 |
| Configuration | 8 | 1 | 3 | 0 |
| **Total** | **82** | **13** | **14** | **6** |

The core diffusion algorithm (noise schedule, transition matrices, forward diffusion, posterior computation, ancestral sampling, VLB) is identical to upstream. Adaptations concentrate in data handling (dense vs. sparse, synthetic generation vs. downloads), the training loss interface (scalar `lambda_E` vs. list), and the model wrapper separation (denoising vs. generative). Missing components are either out of scope (molecular, continuous model) or require heavy external dependencies (ORCA, graph-tool, pyemd).
