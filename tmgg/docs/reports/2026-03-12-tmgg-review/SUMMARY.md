# TMGG Multi-Perspective Review — 2026-03-12

**Branch:** `cleanup-2`
**Reviewers:** 5 parallel agents (code quality, PhD student usability, ML implementation, math/literature, research engineer)

---

## Consolidated Critical Findings

Two distinct critical issues surfaced across the five reviews, with strong corroboration.

### CRIT-1: ~~Continuous (Gaussian) diffusion posterior formula is wrong~~ [RESOLVED]

**Reported by:** math/literature review, ML implementation review (independent corroboration)
**File:** `src/tmgg/diffusion/noise_process.py:197-204`

~~`ContinuousNoiseProcess.get_posterior()` uses `mean = alpha_s * adj_t + (1 - alpha_s) * adj_0`, which does not match the DDPM posterior (Ho et al., 2020, Eq. 7). The correct posterior mean is a weighted combination of `x_0` and `x_t` using `beta_t`, `alpha_bar_t`, and `alpha_bar_{t-1}`. The variance formula is also wrong. This affects the `ContinuousSampler` path; the categorical/DiGress pipeline is unaffected.~~

**Fixed:** Replaced with the correct closed-form posterior from Ho et al. 2020, Eq. 6-7 (arXiv:2006.11239).
**Impact:** Gaussian diffusion generation via `ContinuousSampler.sample()` produces incorrect samples. Single-step denoising experiments are not affected.
**Effort:** Low-medium (formula correction, needs testing).

### CRIT-2: `sanity_check.py` calls `.shape` on `GraphData` — will crash

**Reported by:** code quality review
**File:** `src/tmgg/experiments/_shared_utils/orchestration/sanity_check.py:371-374`

`run_experiment_sanity_check` assumes batches are tensors, but production dataloaders return `GraphData` dataclasses. Setting `config.sanity_check = True` crashes immediately. Latent (no config enables it), but blocks anyone trying to debug their setup.

**Impact:** Latent crash, blocks debugging workflow.
**Effort:** Low.

---

## Consolidated Important Findings

Six important issues, grouped by theme.

### Documentation Drift (PhD student review)

The single highest-leverage improvement across all reviews. Docs reference classes, modules, and parameter names that no longer exist after the training-loop unification and cleanup refactors:

| Finding | Description |
|---------|-------------|
| Stale README structure | References `experiment_utils/` (doesn't exist), wrong paths for `exp_configs/` |
| Stale `models.md` | Shows `DenoisingModel` hierarchy that doesn't exist; actual base is `SpectralDenoiser(GraphModel)` |
| Stale `cloud.md` + `extending.md` | Reference `tmgg.runners` module (`LocalRunner`, `ExperimentCoordinator`) that has no source code |
| Stale `data.md` | Parameter names (`dataset_name`, `val_split`) don't match actual constructors (`graph_type`, `train_ratio`) |

### Behavioral Bugs

| Finding | Source | File | Description |
|---------|--------|------|-------------|
| S3 sync registry accumulation | code quality | `logging.py:33` | Module-level `_s3_sync_registry` grows across grid search runs, causing redundant S3 uploads |
| `reconstruction_logp` unclamped `.log()` | math review | `noise_process.py:585` | `0 * -inf = NaN` if predicted probs contain exact zeros |
| `strict` flag dead in exception handler | code quality | `graph_evaluator.py:206-209` | Both branches of `if strict: continue / else: continue` are identical |

### Convention Violations

| Finding | Source | File | Description |
|---------|--------|------|-------------|
| `eval_every_n_epochs` | code quality | `diffusion_module.py:90` | CLAUDE.md mandates step-based, not epoch-based, training configuration |
| `pyright: ignore` density | PhD student, research eng. | Multiple files | 150+ suppression comments; CLAUDE.md says "never silence pyright errors" |

---

## Cross-Review Corroboration Matrix

Issues independently found by multiple reviewers carry higher confidence.

| Issue | Code Quality | PhD Student | ML Impl | Math/Lit | Res. Eng. |
|-------|:---:|:---:|:---:|:---:|:---:|
| Continuous posterior formula wrong | | | X | X | |
| Stale documentation | | X | | | |
| S3 sync registry accumulation | X | | | | |
| `sanity_check.py` GraphData crash | X | | | | |
| `eval_every_n_epochs` convention | X | | | | X |
| `pyright: ignore` density | | X | | | X |
| `assert isinstance` instead of raise | X | | | | |
| `node_mask is not None` dead guard | X | | | | |
| Orphaned `experiment_utils/` dir | | | | | X |
| Duplicate Laplacian implementations | | | X | | |
| Python loop in `ContinuousNoiseProcess.apply` | | | X | | |
| `get_size_distribution` uses padded shape | X | | | | |

---

## What Works Well (consensus across reviewers)

Several reviewers independently praised these aspects:

- **GraphData dataclass** as universal batch type — clean, typed, compositional
- **Lightning module hierarchy** (BaseGraphModule → DiffusionModule → SingleStepDenoisingModule) — well-separated concerns
- **Model factory with registry** — good error messages, discoverable, extensible
- **Categorical diffusion pipeline** — faithful to DiGress (Vignac et al., 2023), mathematically correct
- **Consistent runner pattern** — learning one experiment teaches the pattern for all
- **Tach module boundaries** — enforced dependency constraints between packages
- **Test suite** — good coverage of critical paths (diffusion math, model forward passes, data loading)
- **"Fail loudly" principle** — consistently applied, minimal silent error swallowing
- **Minimal LLM slop** — abstractions are justified, comments explain rationale not restate code

---

## Prioritized Action Plan

Ordered by impact/effort ratio. Items marked with multiple reviewer initials had independent corroboration.

### Tier 1: Fix now (high impact, low effort)

1. ~~**Fix continuous posterior formula** [MATH, IMPL]~~ — **RESOLVED.** Corrected to DDPM Eq. 6-7.
2. **Fix `sanity_check.py` GraphData crash** [CQ] — adapt to `GraphData` batches. Makes debugging workflow usable.
3. **Clear `_s3_sync_registry` per run** [CQ] — one-line fix, prevents redundant S3 uploads in grid search.
4. **Clamp before `.log()` in `reconstruction_logp`** [MATH] — prevents NaN from `0 * -inf`.
5. **Delete orphaned `experiment_utils/` directory** [RE] — 5 minutes, removes confusion.

### Tier 2: Fix soon (high impact, medium effort)

6. **Update stale documentation** [PHD] — highest leverage for onboarding. Touch README, `models.md`, `cloud.md`, `extending.md`, `data.md`.
7. **`eval_every_n_epochs` → `eval_every_n_steps`** [CQ, RE] — aligns with CLAUDE.md step-based convention. Python + YAML + docs.
8. **Fix `compute_sbm_accuracy` `strict` semantics** [CQ] — decide correct behavior for `strict=False` in exception path.

### Tier 3: Clean up (moderate impact, low effort)

9. **Audit `pyright: ignore` suppressions in non-Lightning files** [PHD, RE] — 2-4 hours, many can be resolved.
10. **Replace `assert isinstance` with explicit `raise`** [CQ] — 3 locations, straightforward.
11. **Fix `get_size_distribution` to use `node_mask.sum()`** [CQ] — latent bug for variable-size graphs.
12. **Consolidate duplicate Laplacian implementations** [IMPL] — `extra_features.py` vs `spectral_utils/laplacian.py`.
13. **Vectorize `ContinuousNoiseProcess.apply` loop** [IMPL] — Python `for` loop over batch, should be batched tensor op.

### Tier 4: Polish (low urgency)

14. Standardize docstring style (NumPy everywhere) [CQ]
15. Privatize `sanity_check.py` internal functions [CQ, from M-24]
16. Remove dead `plotting.py` (~840 lines) or move to notebooks [CQ, from M-25]
17. Break up `run_experiment()` monolith [CQ, from M-26]

---

## Statistics by Review

| Review | Critical | Important | Minor | Nitpick | Total |
|--------|----------|-----------|-------|---------|-------|
| Code Quality | 1 | 3 | 6 | 3 | 13 (+6 prior open) |
| PhD Student Usability | 4 | 5 | ~8 | ~4 | ~21 |
| ML Implementation | 1 | 1 | 6 | 3 | 11 |
| Math/Literature | 1 | 2 | 0 | 0 | 3 |
| Research Engineer | 0 | 4 | 9 | 0 | 13 |

After deduplication across reviews, the unique finding count is approximately **35-40 distinct issues**.

---

## Individual Reports

- [`code-quality-review.md`](code-quality-review.md) — bugs, type safety, conventions
- [`phd-student-usability-review.md`](phd-student-usability-review.md) — onboarding, navigation, extensibility
- [`ml-implementation-review.md`](ml-implementation-review.md) — architecture correctness, literature alignment
- [`math-literature-review.md`](math-literature-review.md) — mathematical correctness, numerical stability
- [`research-engineer-review.md`](research-engineer-review.md) — extensibility, config system, tech debt

---

## Consolidated Findings Table

Every finding from every reviewer, verbatim. Where multiple reviewers flagged the same underlying issue, the rows share a **GID** and appear consecutively. Severity is as assessed by the individual reviewer (may differ across reviewers for the same GID).

**Reviewer key:** CQ = Code Quality, PHD = PhD Student, IMPL = ML Implementation, MATH = Math/Literature, RE = Research Engineer.

### Critical

| GID | Reviewer | Original ID | File(s) | Finding | Status |
|-----|----------|-------------|---------|---------|--------|
| G01 | MATH | #1 | `diffusion/noise_process.py:197-204` | Continuous Gaussian posterior mean formula is incorrect. `mean = alpha_s * adj_t + (1.0 - alpha_s) * adj_0` does not match DDPM (Ho et al., 2020, Eq. 7). Correct posterior mean is `mu_tilde = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0 + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t`. Variance formula `raw_var = (1.0 - alpha_s) * (1.0 - alpha_t / alpha_s)` also wrong. Affects `ContinuousSampler` path only; categorical pipeline unaffected. | FIXED |
| G01 | IMPL | CRIT-1 | `diffusion/noise_process.py:196-216` | Continuous posterior formula does not match DDPM. The implemented formula `mean = alpha_s * adj_t + (1 - alpha_s) * adj_0` is a simple linear interpolation between noisy and clean, not the correct Bayesian posterior. Will produce incorrect reverse sampling trajectories in `ContinuousSampler`. Single-step denoising experiments (`SingleStepDenoisingModule`) unaffected (T=1, no reverse sampling). | FIXED |
| G02 | CQ | NEW-C-1 | `experiments/_shared_utils/orchestration/sanity_check.py:371-374` | `run_experiment_sanity_check` calls `sample_batch.shape[1:]` after `sample_batch = next(iter(data_loader))`. Production dataloaders return `GraphData` dataclass, not a tensor. `GraphData` has no `.shape` attribute — raises `AttributeError`. Same assumption in `check_data_loader` (lines 247-269): `batch.shape`, `torch.isnan(batch)`, `batch.transpose(-2,-1)` all assume `Tensor`. Bug is latent: `sanity_check: True` not in any checked-in config. | FIXED |
| G03 | PHD | CRIT-README | `README.md:182-216` | README project structure diagram describes a layout that does not match reality. Lists `experiment_utils/` as top-level source directory (doesn't exist), lists `exp_configs/` as direct child of `src/tmgg/` (actually `src/tmgg/experiments/exp_configs/`), lists `experiment_utils/eigenstructure_study/` and `experiment_utils/embedding_study/` (actually under `experiments/`). A new student following the README will look for files in the wrong places. | FIXED |
| G04 | PHD | CRIT-CLASSES | `docs/experiments.md:78-82`, `docs/how-to-run-experiments.md:107-115`, `docs/configuration.md:199,323` | Docs reference `SpectralDenoisingLightningModule`, `GNNDenoisingLightningModule`, imports from `tmgg.experiments.spectral_arch_denoising.lightning_module` — none exist. Actual class is `SingleStepDenoisingModule` in `tmgg.experiments._shared_utils.lightning_modules.denoising_module`. Student trying checkpoint loading from docs gets ImportError. | FIXED |
| G05 | PHD | CRIT-RUNNERS | `docs/cloud.md:98-103,145-193`, `docs/extending.md:356-431` | Both docs reference imports from `tmgg.runners` (`LocalRunner`, `ExperimentCoordinator`, `StageConfig`, `CloudRunner`, `ExperimentResult`). No `src/tmgg/runners/` directory exists. Actual cloud code is in `src/tmgg/modal/`. Student following "Adding a New Cloud Backend" hits ImportError immediately. | FIXED |

### Important

| GID | Reviewer | Original ID | File(s) | Finding | Status |
|-----|----------|-------------|---------|---------|--------|
| G06 | CQ | NEW-I-1 | `experiments/_shared_utils/logging.py:33,248,265` | `_s3_sync_registry: list[_S3SyncInfo] = []` is module-level mutable list. Each `create_loggers` call appends. `sync_tensorboard_to_s3` iterates entire list. In `grid_search_runner.py` (multiple `run_experiment` calls in one process), registry accumulates entries from all prior runs. Run N triggers N S3 uploads of prior runs' data. Fix: `_s3_sync_registry.clear()` at top of `run_experiment`. | FIXED |
| G07 | CQ | NEW-I-2 | `experiments/_shared_utils/lightning_modules/diffusion_module.py:90,113,141,465` | `eval_every_n_epochs` parameter and gate `current_epoch % self.eval_every_n_epochs != 0` violates CLAUDE.md "always think in steps, NOT in epochs" convention. Epoch-based gating means effective evaluation frequency changes silently when dataset size or batch size changes. Active in YAML configs: `eval_every_n_epochs: 1` and `5`. | FIXED |
| G08 | CQ | NEW-I-3 | `experiments/_shared_utils/evaluation_metrics/graph_evaluator.py:206-209` | `compute_sbm_accuracy` has identical branches: `except ValueError: if strict: continue / else: continue  # skip graph`. Both branches do the same thing. `strict` parameter is semantically dead in the exception path. When `strict=False`, graphs should arguably count as failures (score 0) rather than be excluded, which inflates accuracy. | FIXED |
| G09 | MATH | #2 | `diffusion/noise_process.py:585` | `reconstruction_logp` calls `.log()` without clamping: `loss_X = sum_except_batch(clean.X * pred_probs.X.log())`. If `pred_probs` contains exact zeros, `log(0) = -inf`, and `0 * -inf = NaN`. Docstring says "masked positions should be set to 1 so that log(1) = 0" but this precondition is not enforced internally. Fix: `pred_probs.X.clamp(min=1e-10).log()`. | FIXED |
| G10 | MATH | #3 | `experiments/_shared_utils/evaluation_metrics/mmd_metrics.py:111,171` | `compute_degree_histogram` uses `np.histogram(..., density=True)` (probability density), while `compute_spectral_histogram` uses `density=False` with manual PMF normalization. Numerically equivalent for uniform-width bins (which both use), and `gaussian_tv_kernel` re-normalizes to PMF anyway, so MMD unaffected. Inconsistency could confuse maintainers. | FIXED |
| G10 | IMPL | NITPICK | `experiments/_shared_utils/evaluation_metrics/mmd_metrics.py` | Degree histogram uses `density=True` while spectral histogram uses manual normalization. Both produce valid PMFs; inconsistency is worth noting. The `gaussian_tv_kernel` re-normalizes inputs to PMF anyway (lines 309-314), so the MMD computation is unaffected. | FIXED |
| G11 | PHD | IMP-DATA | `docs/data.md:54-88` | `data.md` parameter names don't match source. Doc describes `GraphDataModule` with `dataset_name`, `dataset_config`, `num_samples_per_graph`, `val_split`, `test_split`. Actual constructor uses `graph_type`, `graph_config`, `samples_per_graph`, `train_ratio`, `val_ratio`. Batch format also wrong: doc shows `clean, noisy = batch` but actual dataloader yields `GraphData` objects, not tuples. | FIXED |
| G12 | PHD | IMP-MODELS | `docs/models.md:7-25` | `docs/models.md` hierarchy shows `BaseModel -> DenoisingModel -> [all models]`. But `base.py` only defines `BaseModel` and `GraphModel` — no `DenoisingModel` class. Actual spectral denoisers inherit from `SpectralDenoiser(GraphModel)`, not `DenoisingModel`. Causes confusion about which base class to use. | FIXED |
| G13 | PHD | IMP-NESTING | `experiments/_shared_utils/lightning_modules/` | Deep nesting of shared utilities: 5 levels to reach key abstractions (e.g., `src/tmgg/experiments/_shared_utils/lightning_modules/denoising_module.py`). `_shared_utils` contains 3 subdirectories (`lightning_modules/`, `evaluation_metrics/`, `orchestration/`) plus standalone files. Not blocking, but new students rely on IDE "go to definition" rather than filesystem browsing. | FIXED — moved to `tmgg.training` (G48), reducing from 5 to 4 levels |
| G14 | PHD | IMP-INIT | `experiments/__init__.py` | `src/tmgg/experiments/__init__.py` is blank (1 line). No module-level docstring, no `__all__`, no indication of what the package contains. Compare to `src/tmgg/data/__init__.py` which exports 30+ names with organized sections. Zero guidance about available experiments. | FIXED |
| G15 | PHD | IMP-PYRIGHT | Multiple files | 150+ `pyright: ignore` comments across source. Worst offenders: `diffusion_module.py` (43), `denoising_module.py` (29), `synthetic_graphs.py` (16). Many suppress `reportUnknownMemberType` from PyTorch/Lightning dynamic typing. Cumulative effect is significant visual noise. Better: type-narrow `self.model` once, or per-file `# pyright: reportUnknownMemberType=false`. | FIXED |
| G15 | RE | IMP-PYRIGHT | 34 files (185 suppressions) | 185 `pyright: ignore` comments across 34 files. Concentration highest in Lightning modules: `diffusion_module.py` (43), `denoising_module.py` (29), `base_graph_module.py` (13). Many are genuine torch/Lightning type-stub limitations, but the volume partially disables type checking in the most critical files. Audit non-Lightning suppressions (e.g., `noise.py` 8, `synthetic_graphs.py` 16) which may indicate real type issues. | FIXED |
| G16 | PHD | IMP-FACTORY | `models/factory.py`, `docs/extending.md` | Adding a new model requires editing `factory.py` (manual `@register_model`). `extending.md` Step 2 says "Choose a Training Module" but the model YAML's `_target_` field (Step 4) actually determines the Lightning module class — this critical detail is buried. | FIXED |
| G17 | RE | IMP-ORPHAN | `src/tmgg/experiment_utils/` | Orphaned `experiment_utils/` directory exists with only `__pycache__/` artifacts and empty `data/` subdirectory. No source files, no imports from it, not in `tach.toml`. Stale `.pyc` files could cause phantom import successes. Newcomer encountering this alongside `experiments/_shared_utils/` assumes two competing utility namespaces. Delete. | FIXED |
| G18 | RE | IMP-EIGEN-CLI | `experiments/eigenstructure_study/cli.py` | 750-line eigenstructure CLI has 10 `except Exception as e` handlers that catch, log, and sometimes continue. Contradicts CLAUDE.md "never do graceful fallback, always fail loudly." Broad `except Exception` catches mask specific failure modes. Fix: catch specific exception types at CLI boundary; let everything else propagate. | FIXED |
| G19 | RE | IMP-FACTORY-DEFAULTS | `models/factory.py` | Factory functions use `config.get("k", 8)`, `config.get("d_k", 64)` — implicit defaults a caller cannot discover without reading factory source. If YAML config omits `k`, model silently uses 8. For research reproducibility, this risks experiments running with unintended defaults. YAMLs specify values explicitly in practice, so this is latent. Fix: `config["k"]` (fail on missing) for architecture params. | FIXED |
| G20 | IMPL | IMP-LOOP | `diffusion/noise_process.py:151-157` | `ContinuousNoiseProcess.apply` applies noise per-sample in a Python `for` loop: `for i in range(bs): noisy_slices.append(self.generator.add_noise(adj[i], eps_i))`. Underlying noise functions support batched input. The per-sample loop exists because each sample may have a different noise level, but broadcasting could handle this. O(bs) slower than vectorized. | FIXED |
| G21 | RE | IMP-SAMPLING-TESTS | tests/ | `CategoricalSampler` and `ContinuousSampler` contain the most mathematically critical code (reverse diffusion). `test_sampler.py` exists, but end-to-end statistical properties (generated distribution matches training distribution for trivially learnable graph class) would benefit from a dedicated statistical test, even if marked `slow`. | DEFERRED |

### Minor

| GID | Reviewer | Original ID | File(s) | Finding | Status |
|-----|----------|-------------|---------|---------|--------|
| G22 | CQ | NEW-M-1 | `data/noising/size_distribution.py:154-156` | `SizeDistribution.from_dict` uses bare `assert isinstance(sizes, list)` etc. Bare `assert` is disabled by `-O` and gives unhelpful `AssertionError`. This is a deserialization boundary where malformed input is real. CLAUDE.md: "fail loudly and informatively with an exception." Fix: explicit `TypeError` raises. | FIXED |
| G23 | CQ | NEW-M-2 | `data/datasets/sbm.py:169-221` | `generate_partitions` inner function redefined on every loop iteration inside `for num_blocks in range(...)`. Creates new function object each iteration. Closure over `num_blocks` is intentional but implicit and confusing. Fix: define once outside loop, pass `num_blocks` as explicit argument. | FIXED |
| G24 | CQ | NEW-M-3 | `experiments/_shared_utils/lightning_modules/diffusion_module.py:311` | `assert isinstance(self.noise_process, CategoricalNoiseProcess)` in `_compute_reconstruction_at_t1`. Disabled under `-O`. Guards production VLB training path. Fix: `if not isinstance: raise TypeError(...)`. | FIXED |
| G25 | CQ | NEW-M-4 | `experiments/_shared_utils/lightning_modules/diffusion_module.py:336` | `node_mask is not None` guard is always True. `GraphData.node_mask` is typed as `Tensor` (non-optional) on a frozen dataclass — cannot be `None`. Dead code adds false uncertainty. Fix: remove guard, de-indent block. | FIXED |
| G26 | CQ | NEW-M-5 | `experiments/discrete_diffusion_generative/datamodule.py:302` | `get_size_distribution` uses `g.node_mask.shape[0]` (padded dimension) instead of `int(g.node_mask.sum().item())` (actual node count). Equal for current fixed-size graphs. Silently wrong for variable-size graphs. Fix: use `.sum().item()`. | FIXED |
| G27 | CQ | NEW-M-6 | `diffusion/noise_process.py:293` | `self._transition_model = model  # type: ignore[assignment]` suppresses pyright error. CLAUDE.md: "never silence pyright errors." Issue is Protocol assignment narrowing. Fix: use explicit `cast(TransitionModel, model)`. | FIXED |
| G28 | IMPL | MIN-CHEB | `models/layers/graph_ops.py:64-89` | `spectral_polynomial` comment references "ChebNet convention" but uses monomial basis (`Lambda^k`), not Chebyshev polynomials (`T_k(Lambda)`). Eigenvalue normalization to [-1,1] mitigates numerical instability. Functionally correct but naming misleading; Chebyshev would be more stable for high polynomial degrees. | FIXED |
| G29 | IMPL | MIN-BILINEAR | `models/spectral_denoisers/bilinear.py:104` | `BilinearDenoiser._spectral_forward` takes `Lambda` parameter but never uses it. Reconstruction `Q K^T / sqrt(d_k)` operates solely on eigenvectors. Cannot distinguish eigenspaces with different eigenvalues. By design for simplicity, but documented as if eigenvalues are available. | FIXED — already documented as "(unused)" in docstrings |
| G30 | IMPL | MIN-GNN-ASYM | `models/gnn/gnn.py:96` | GNN computes `result_adj = torch.bmm(emb_x, emb_y.transpose(1,2))` with separate linear heads — resulting adjacency is not guaranteed symmetric. Works with BCE loss (doesn't require symmetry), but conceptually inconsistent with undirected graph generation. Could benefit from `(A + A.T) / 2`. | FIXED |
| G31 | IMPL | MIN-VLB | `experiments/_shared_utils/lightning_modules/diffusion_module.py:369-412` | VLB computation uses a single random timestep per validation batch, then averages at epoch end. Gives unbiased estimate of `E_t[L_t]` but with high variance. Standard practice for validation monitoring; full VLB too expensive every epoch. | FIXED |
| G32 | IMPL | MIN-LAPLACIAN | `models/digress/extra_features.py:551-586`, `experiments/_shared_utils/spectral_utils/laplacian.py` | Two `compute_laplacian` implementations. `extra_features.py` version symmetrizes result `(L + L.T)/2`; standalone version does not (relies on input symmetry). Both compute the same thing. Should be consolidated. | FIXED — moved `spectral_utils` to `tmgg.utils.spectral`, added `symmetrize` parameter to canonical `compute_laplacian`, deleted duplicate from `extra_features.py` |
| G33 | MATH | MIN-TOPK | `models/layers/topk_eigen.py:183` | `TopKEigenLayer` selects k eigenvectors with largest `|lambda_i|` (magnitude), not k largest eigenvalues (algebraically). Correct for denoising (variance maximization, analogous to PCA), but spectral clustering typically uses smallest Laplacian eigenvalues. Distinction should be documented. | FIXED |
| G34 | MATH | MIN-EIGENGAP | `experiments/_shared_utils/spectral_utils/spectral_deltas.py:81-100` | `compute_eigengap_delta()` computes gap between two largest adjacency eigenvalues. "Eigengap" in spectral graph theory typically refers to the Laplacian gap. The computation is correct for what it claims to measure; naming is slightly non-standard. | FIXED |
| G35 | MATH | MIN-KL-DOC | `diffusion/diffusion_math.py:121` | `gaussian_KL` docstring mentions `p_mu` and `p_sigma` parameters that don't exist. Function only accepts `q_mu` and `q_sigma`, computing `KL(N(mu, sigma^2) || N(0, 1))`. Formula is correct; docstring has phantom parameters. | FIXED |
| G36 | MATH | MIN-MASK-EPS | `diffusion/diffusion_sampling.py:244-252` | `mask_distributions` adds `1e-7` to all entries then renormalizes after setting masked positions to one-hot. Prevents `log(0)` downstream. Correct but rationale not documented in code. | FIXED |
| G37 | PHD | MIN-MODELS-INIT | `models/__init__.py` | Contains only a docstring and no imports. All model access goes through `models.factory.create_model()` or direct submodule imports. Student browsing `from tmgg.models import ...` finds nothing. Fine if you know the factory; opaque otherwise. | FIXED |
| G38 | PHD | MIN-ARCH-PATHS | `docs/architecture.md:7-37` | Architecture doc directory structure uses slightly wrong paths. Shows `_shared_utils/base_graph_module.py` (actual: `_shared_utils/lightning_modules/base_graph_module.py`), `_shared_utils/run_experiment.py` (actual: `_shared_utils/orchestration/run_experiment.py`), `_shared_utils/metrics.py` (actual: `_shared_utils/evaluation_metrics/mmd_metrics.py`). | FIXED |
| G39 | PHD | MIN-HYDRA-DEPTH | `experiments/exp_configs/` | Understanding full config of a spectral denoising experiment requires reading 5+ files: `base_config_spectral_arch.yaml`, `base_config_denoising.yaml`, `_base_infra.yaml`, `task/denoising.yaml`, `models/spectral/linear_pe.yaml`, plus trainer/callbacks/logger defaults. `--cfg job` helps but student must know about it. | FIXED |
| G40 | PHD | MIN-DATAMODULES | `data/data_modules/`, `experiments/discrete_diffusion_generative/datamodule.py` | Two data module hierarchies serve different experiments: `MultiGraphDataModule`/`GraphDataModule` (adjacency-tensor denoising) vs `SyntheticCategoricalDataModule` (one-hot categorical diffusion). No documentation explains when to use which, or why the discrete datamodule lives inside its experiment directory. | OPEN |
| G41 | PHD | MIN-WANDB | `experiments/exp_configs/_base_infra.yaml` | `allow_no_wandb: false` means training crashes without `WANDB_API_KEY`. Quick start docs don't mention this. Student running `uv run tmgg-spectral-arch` first time without W&B setup hits error. Fix is simple (`allow_no_wandb=true`) but error may not point to solution. | FIXED |
| G42 | PHD | MIN-S3 | `experiments/exp_configs/base_config_discrete_diffusion_generative.yaml:71` | Hardcoded S3 TensorBoard logger path `s3://${oc.env:TMGG_S3_BUCKET}/...` requires `TMGG_S3_BUCKET`. Student running discrete diffusion locally without S3 gets interpolation error. Other experiment configs default to local TensorBoard. | FIXED |
| G43 | PHD | MIN-TEMPLATE | — | No script or template to scaffold new experiments. `extending.md` lists 7 steps with 7 artifacts. A `tmgg-scaffold --experiment vae_graph` command would save time and reduce errors. | FIXED |
| G44 | PHD | MIN-CONFIG-VAL | — | Hydra configs validated at runtime when accessed, not at composition time. A typo (e.g., `noise_typ` instead of `noise_type`) only surfaces when code reads that key, possibly deep into training. OmegaConf structured configs could catch this earlier but are not used. | DEFERRED |
| G45 | PHD | MIN-METACLASS | `models/factory.py:21-29` | `_RegistryMeta` metaclass exists solely for `"name" in ModelRegistry` syntax. Same achievable with `"name" in ModelRegistry.keys()`. Metaclass adds complexity for marginal convenience; will confuse students unfamiliar with Python metaclasses. | FIXED |
| G46 | PHD | MIN-DOCSTRINGS | various | Some docstrings restate the obvious: `get_model_name` → "Return model_type from hyperparameters"; `get_model_config` → "Delegate to model.get_config()". Boilerplate quality characteristic of LLM-generated code. Compare to `_spectral_forward`, `_compute_loss`, `training_step` which are genuinely useful. | FIXED — audited, docstrings retained (contain non-obvious info) |
| G47 | RE | MIN-SHARED-ORG | `experiments/_shared_utils/orchestration/progress.py` | `_shared_utils` internal organization: `progress.py` (600+ line Rich progress bar) lives under `orchestration/`. Standalone UI component could live at `_shared_utils` top level. Cosmetic. | FIXED |
| G48 | RE | MIN-EXP-NAMING | `experiments/_shared_utils/` | `_shared_utils` sits under `experiments/` but is really project-wide infrastructure consumed by Modal integration too. `tach.toml` has `tmgg.modal -> tmgg.experiments._shared_utils.orchestration`. The `_` prefix correctly signals "internal" and tach prevents undirected coupling, but could confuse if project grows. | FIXED — moved to `tmgg.training` |
| G49 | RE | MIN-GRID-CONFIG | `experiments/grid_search_runner.py` | `grid_search_base.yaml` exists as both Hydra entry point and base config, uses absolute `CONFIG_PATH` via `Path(__file__)` rather than relative path like all other runners. Config landscape (93 YAMLs) is large but organized. | FIXED |
| G50 | RE | MIN-DUAL-KNOB | `experiments/exp_configs/task/denoising.yaml` | Two config knobs for same setting: `task/denoising.yaml` sets both `model.eval_num_samples: 128` and constructs `GraphEvaluator` with `eval_num_samples: 128`. Independent parameters that happen to share a value. Changing one without other causes confusion. Fix: single source via interpolation. | FIXED |
| G51 | RE | MIN-EIGEN-TESTS | `tests/experiment_utils/test_eigenstructure_study.py` | Eigenstructure study (`eigenstructure_study/`) is substantial (cli.py alone 750 lines) with collection, analysis, storage modules. `test_eigenstructure_study.py` exists but the CLI's many `except Exception` handlers suggest code path less well-tested than core training loop. | OPEN |
| G52 | RE | MIN-EXP-FRICTION | — | Adding a fundamentally new experiment type (not denoising, not diffusion) requires subclassing `BaseGraphModule` or `DiffusionModule`, creating base config YAML, adding CLI entry point in `pyproject.toml`. Reasonable, but main friction is understanding which Hydra config layer to override (`_base_infra` vs `task/X` vs `base_config_X`). | FIXED |
| G53 | RE | MIN-DISCOVERY | `docs/architecture.md` | Extension point discoverability: `__all__` lists define public APIs, factory pattern makes model registration discoverable. What is less discoverable is config composition rules — understanding YAML precedence requires reading `defaults:` lists. | FIXED |
| G54 | RE | MIN-DEPS | `pyproject.toml` | `jupyter>=1.1.1` and `torchshow>=0.5.2` are runtime dependencies, not dev. Only needed for notebooks/visualization, not training. Moving to `[project.optional-dependencies]` reduces Modal container install time. | FIXED |
| G55 | RE | MIN-PLOTTING-EXC | `experiments/_shared_utils/plotting.py:357,366` | Two `except Exception` handlers silently fall back to spring layout when spectral/kamada_kawai layout fails. For visualization this is arguably acceptable, but silent fallback should at least emit `warnings.warn()`. | FIXED — file deleted (G59) |
| G56 | RE | MIN-TOPK-EXC | `models/layers/topk_eigen.py:45` | `except Exception` during SVD in diagnostic `_compute_debugging_metrics`. Inside an error handler already constructing diagnostic info for failed eigendecomposition. Catching broadly here to avoid masking original error is defensible. | FIXED — rationale comment added |
| G57 | RE | MIN-LOSS-MAP | `experiments/_shared_utils/lightning_modules/denoising_module.py` | `_DENOISING_LOSS_MAP` maps `"MSE" -> "mse"`, `"BCEWithLogits" -> "bce_logits"` — legacy bridge converting old config values to new vocabulary. Two different strings refer to same loss type depending on experiment type. Fix: standardize on one set, fail on old names. | FIXED |
| G58 | CQ | M-24 (prior) | `experiments/_shared_utils/orchestration/sanity_check.py` | Five public functions (`check_noise_generator`, `check_data_loader`, etc.) are implementation details reachable only through `maybe_run_sanity_check`; should be private (`_check_...`). | FIXED |
| G59 | CQ | M-25 (prior) | `experiments/_shared_utils/plotting.py` | ~840-line file unreferenced from any production path or runner. May be notebook-only dead code. | FIXED |
| G60 | CQ | M-26 (prior) | `experiments/_shared_utils/orchestration/run_experiment.py` | Single function handles seed, W&B dedup, dirs, config save, datamodule, model, callbacks, checkpoints, training, testing, S3 sync, and W&B cleanup — too many concerns for one function. | OPEN |
| G61 | CQ | M-27 (prior) | `experiments/_shared_utils/lightning_modules/diffusion_module.py` | `_train_loss_discrete` conditionally created (only when `loss_type == "cross_entropy"`) with no class-level type annotation; presence inferred via `isinstance`. | FIXED |
| G62 | CQ | M-28 (prior) | `experiments/_shared_utils/logging.py`, `sanity_check.py`, `plotting.py` | `matplotlib.use("Agg")` as module-level side effect in three files. | FIXED — consolidated to `run_experiment.py` only; `plotting.py` deleted (G59) |
| G63 | CQ | M-35 (prior) | all denoising runners | `main() -> None` discards the `dict` return value from `run_experiment()`, breaking Hydra sweep result collection. | FIXED |

### Nitpick

| GID | Reviewer | Original ID | File(s) | Finding | Status |
|-----|----------|-------------|---------|---------|--------|
| G64 | CQ | NEW-N-1 | `data/datasets/sbm.py` | `generate_block_sizes` uses Google-style docstring (`Args:` / `Returns:` / `Example:`) in a NumPy-style codebase. CLAUDE.md specifies NumPy docstring style. | FIXED |
| G65 | CQ | NEW-N-2 | `data/noising/size_distribution.py` | `SizeDistribution.log_prob` docstring says "Returns log-probability for a given graph size." For unsupported sizes, returns `torch.log(p.clamp(min=1e-30))` ≈ -69, not 0. Should clarify boundary behavior. | FIXED |
| G66 | CQ | NEW-N-3 | `experiments/_shared_utils/lightning_modules/denoising_module.py:191` | `# noqa: ISC003` suppression for implicit string concatenation. Could be rewritten as single string literal to eliminate both the warning and the suppression. | FIXED |
| G67 | IMPL | NIT-MHA | `models/layers/mha_layer.py` | `MultiHeadAttention` `forward` computes Q, K, V all from same input (self-attention only). Class name suggests generality. Could be `MultiHeadSelfAttention` for precision. | FIXED |
| G68 | IMPL | NIT-FROM-ADJ | `data/datasets/graph_types.py:156` | `GraphData.from_adjacency` creates `node_mask = ones(bs, n)`, treating all positions as real. For batches with zero-padded variable-size graphs, caller must handle mask separately. By design, well-documented by `collate` classmethod. | FIXED |
| G69 | MATH | NIT-ROTATION | `data/noising/noise.py:56` | Rotation noise via `V_rot diag(lambda) V_rot^T` preserves spectral structure but produces continuous-valued matrices, not binary adjacency. Differs from other noise types which produce valid adjacency matrices. Docstring does not flag this distinction. | FIXED |
| G70 | PHD | NIT-TESTCMD | `CLAUDE.md`, `README.md` | Test command in CLAUDE.md (`uv run pytest tests/ -x --ignore=tests/modal/... -m "not slow" -v`) differs from README (`uv run pytest`). CLAUDE.md version is more useful (skips slow/Modal tests) but student reading README may hit failures. | FIXED |
| G71 | PHD | NIT-FACTORY-COMPAT | `models/factory.py:122` | Module-level `create_model()` and `register_model()` documented as "backward-compatible module-level API." They are the only way these are used, so describing them as compatibility layer around the "real" class-based API creates impression of more complexity than necessary. | FIXED |

### GID Cross-Reference

| GID | Reviewers | Severity (per reviewer) | Core issue | Status |
|-----|-----------|------------------------|------------|--------|
| G01 | MATH, IMPL | CRITICAL, CRITICAL | Continuous posterior formula wrong | FIXED |
| G10 | MATH, IMPL | IMPORTANT, NITPICK | Histogram normalization inconsistency | FIXED |
| G15 | PHD, RE | IMPORTANT, IMPORTANT | pyright-ignore comment density | FIXED |
