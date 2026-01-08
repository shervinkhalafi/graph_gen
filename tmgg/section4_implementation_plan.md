# Section 4 Implementation Plan

## Answers to your questions

1) The denoising models return logits, not sigmoid outputs. The default training
config uses `loss_type: BCEWithLogits` in `src/tmgg/exp_configs/base_config_training.yaml`,
and `DenoisingLightningModule` wires that to `nn.BCEWithLogitsLoss` in
`src/tmgg/experiment_utils/base_lightningmodule.py`. Sigmoid happens inside the loss,
not inside the model. `DenoisingModel.predict` thresholds logits at zero in
`src/tmgg/models/base.py`, which is equivalent to `sigmoid(logits) > 0.5`, but it
does not return probabilities.

2) Asymmetric X/Y is supported in the GNN and hybrid models, not in the spectral
 denoisers. The asymmetric reconstruction is explicit in
`src/tmgg/models/gnn/gnn.py` and the hybrid wrapper in `src/tmgg/models/hybrid/hybrid.py`.
The spectral denoisers (`linear_pe`, `filter_bank`, `self_attention`) reconstruct with
`V W V^T` and do not expose an X/Y variant, so the section 4.1 asymmetric ablation
would need new spectral model classes and configs.

3) The polynomial graph convolution used in DiGress GNN projections is a spatial
polynomial filter over the normalized adjacency,
`Y = Σ A_norm^i X H[i]`, implemented in `src/tmgg/models/layers/gcn.py`.
The section 4 filter bank is a spectral polynomial over eigenvalues,
`Y = V Σ Λ^k H^(k)`, implemented for reconstruction in
`src/tmgg/models/spectral_denoisers/filter_bank.py`. The former propagates features
across hops; the latter scales eigenmodes in the eigenspace.

4) The current DiGress input features are minimal. With `use_eigenvectors=True`, the
GraphTransformer uses top-k eigenvectors as node features
(`src/tmgg/models/digress/transformer_model.py`). With `use_eigenvectors=False`,
it treats the adjacency as edge features and uses the adjacency diagonal as node
features. There is no implementation of cycle counts or other handcrafted features
from the DiGress paper.

5) The missing extensions are: a spectral asymmetric X/Y variant; a PEARL embedding
implementation and integration as an alternative to eigenvectors; an optional output
nonlinearity ablation that is consistent with the loss; a DiGress variant that replaces
Q/K/V with spectral filter banks (and an option to disable FiLM if the ablation
requires it); and a dataset-level spectral variance analysis plus a synthetic data
generator or selector that targets specific Var(Λ).

## Implementation plan

### 1) Confirm scope and branch
I will confirm we are on a `claude/`, `feat/`, or `igork/` branch before coding,
and I will use atomic commits and tags for milestones, as required by `AGENTS.md`.
I will also confirm which of the Section 4 items you want in the first pass.

### 2) Add output-nonlinearity ablation support
I will add an explicit output transform option for spectral models that can switch
between identity logits and sigmoid probabilities. This will live alongside the
loss selection so that `BCEWithLogits` is only used with logits, and `BCE` is only
used with probabilities. The change will touch
`src/tmgg/experiment_utils/base_lightningmodule.py` and the spectral configs in
`src/tmgg/exp_configs/models/spectral/`. I will add a small unit test to verify
that the chosen transform and loss stay consistent.

### 3) Implement asymmetric spectral models
I will add asymmetric variants of Linear PE and Graph Filter Bank that compute
separate X and Y branches and reconstruct `X Y^T`. These will live in
`src/tmgg/models/spectral_denoisers/` and be wired into
`src/tmgg/experiments/spectral_denoising/lightning_module.py` and new config
groups under `src/tmgg/exp_configs/models/spectral/`. I will add tests that
check output shape and symmetry properties.

### 4) Add PEARL embeddings and integrate them
I will implement PEARL embeddings as a reusable embedding layer in
`src/tmgg/models/embeddings/` or `src/tmgg/models/layers/`, based on the
formula in section 4.1. I will then add a configurable “embedding source”
option for spectral denoisers so they can take eigenvectors or PEARL embeddings.
This requires updating the spectral base class or adding a new shared mixin that
handles embedding extraction before `_spectral_forward`. I will include tests
that verify the embedding shape and determinism for a fixed adjacency.

### 5) Add DiGress filter-bank projections
I will implement a projection module that replaces Q/K/V in `NodeEdgeBlock`
with a spectral filter-bank projection. This will require a design choice:
either compute eigenvectors once per forward pass and apply a spectral filter
to X, or accept precomputed eigenvectors as input features. I will add a config
variant under `src/tmgg/exp_configs/models/digress/` and wire it through
`hidden_dims` in `src/tmgg/models/digress/transformer_model.py`. If the ablation
requires “no FiLM,” I will add a switch to bypass the FiLM edge and global
conditioning.

### 6) Add spectral variance analysis and synthetic targets
I will extend the eigenstructure analysis to compute spectral variance across
graphs, either as a new metric in
`src/tmgg/experiment_utils/eigenstructure_study/analyzer.py` or as a small
post-analysis script. I will also add a synthetic dataset sweep or rejection
sampler that targets specific Var(Λ) ranges using SBM parameters in
`src/tmgg/experiment_utils/data/sbm.py` or
`src/tmgg/experiment_utils/data/synthetic_graphs.py`. The output will include
per-dataset Var(Λ) and the planned correlation study.

### 7) Update configs, docs, and tests
I will add new config groups for the asymmetric and PEARL variants, and I will
document the new options in `docs/models.md` and `docs/experiments.md`. For
tests, I will add minimal checks under `tests/` for each new model class and
for the output transform/loss wiring. Any new tests will include a short
rationale in docstrings, per `AGENTS.md`.

### Open questions
I need decisions on these points before implementation:

- Whether you want the output-nonlinearity ablation to change only evaluation or
  also training loss.
- The exact PEARL embedding definition you want in code, including how many
  powers of A and the normalization conventions.
- Whether the DiGress “filter bank Q/K/V” should be spectral (eigen-based) or a
  spatial polynomial (A_norm-based) that mimics a filter bank.
- Whether you want the “no FiLM” DiGress variant to remove both edge and global
  conditioning, or only edge conditioning.
- Which datasets you want for the spectral variance study and the target ranges
  for Var(Λ).
