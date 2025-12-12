# Models

This document describes the model architectures available in the framework.

## Model Hierarchy

```
BaseModel (src/tmgg/models/base.py)
└── DenoisingModel
    ├── MultiLayerAttention (attention/attention.py)
    ├── GNN (gnn/gnn.py)
    ├── GNNSymmetric (gnn/gnn_sym.py)
    ├── NodeVarGNN (gnn/nvgnn.py)
    ├── SequentialDenoisingModel (hybrid/hybrid.py)
    └── Spectral denoisers (spectral_denoisers/)
```

All models inherit from `DenoisingModel`, which provides domain transformations and configuration utilities.

## Attention Models

### MultiLayerAttention

Multi-layer transformer attention for denoising adjacency matrices. Processes the full adjacency matrix through stacked self-attention layers.

**Location:** `src/tmgg/models/attention/attention.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Model dimension (typically num_nodes) |
| `num_heads` | int | required | Number of attention heads |
| `num_layers` | int | required | Number of transformer layers |
| `d_k` | int | None | Key dimension (defaults to d_model // num_heads) |
| `d_v` | int | None | Value dimension (defaults to d_model // num_heads) |
| `dropout` | float | 0.0 | Dropout rate |
| `bias` | bool | True | Use bias in linear layers |
| `domain` | str | "standard" | Domain transformation |

**Config:** `exp_configs/models/attention/multi_layer_attention.yaml`

**Usage:**

```python
from tmgg.models.attention import MultiLayerAttention

model = MultiLayerAttention(
    d_model=20,
    num_heads=4,
    num_layers=4,
)
output = model(adjacency_matrix)  # (batch, n, n) -> (batch, n, n)
```

## GNN Models

All GNN models use spectral embeddings via eigendecomposition of the adjacency matrix.

### GNN

Standard graph neural network with polynomial graph convolution filters.

**Location:** `src/tmgg/models/gnn/gnn.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | int | required | Number of GCN layers |
| `num_terms` | int | 3 | Polynomial filter terms |
| `feature_dim_in` | int | 10 | Input feature dimension |
| `feature_dim_out` | int | 10 | Output feature dimension |
| `eigenvalue_reg` | float | 0.0 | Eigenvalue regularization |
| `domain` | str | "standard" | Domain transformation |

**Config:** `exp_configs/models/gnn/standard_gnn.yaml`

**Returns:** Tuple of (X, Y) embeddings for reconstruction via X @ Y.T

### GNNSymmetric

Symmetric GNN with shared X and Y embeddings. Uses a single embedding pathway, making the reconstruction inherently symmetric.

**Location:** `src/tmgg/models/gnn/gnn_sym.py`

**Config:** `exp_configs/models/gnn/symmetric_gnn.yaml`

**Returns:** Tuple of (reconstructed_adjacency, X_embeddings)

### NodeVarGNN

Node-variant GNN with per-node filter coefficients. Allows heterogeneous processing across nodes.

**Location:** `src/tmgg/models/gnn/nvgnn.py`

**Config:** `exp_configs/models/gnn/nodevar_gnn.yaml`

## Hybrid Models

### SequentialDenoisingModel

Combines a GNN embedding model with a transformer denoising model. The GNN extracts node embeddings, which are then processed by attention layers.

**Location:** `src/tmgg/models/hybrid/hybrid.py`

**Usage:**

```python
from tmgg.models.hybrid import create_sequential_model

model = create_sequential_model(
    gnn_config={"num_layers": 2, "feature_dim_out": 10},
    transformer_config={"num_heads": 4, "num_layers": 2}
)
```

**Config:** `exp_configs/models/hybrid/hybrid_with_transformer.yaml`

## Spectral Denoisers

Models operating in the spectral domain, using the eigendecomposition of the noisy adjacency matrix. All output raw logits; use `model.predict(logits)` for [0,1] probabilities.

**Location:** `src/tmgg/models/spectral_denoisers/`

### LinearPE

Linear transformation in eigenspace with optional bias correction.

**Formula:**
```
Â = V W V^T + 1 b^T + b 1^T
```

where V ∈ R^{n×k} are top-k eigenvectors, W ∈ R^{k×k} is learnable, and b ∈ R^{max_n} is a bias vector capturing node-specific degree corrections.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors |
| `max_nodes` | int | 200 | Maximum graph size (for bias) |
| `use_bias` | bool | True | Enable node-specific bias term |

**Config:** `exp_configs/models/spectral/linear_pe.yaml`

### GraphFilterBank

Spectral polynomial filter with learnable coefficient matrices.

**Formula:**
```
W = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H^{(ℓ)}
Â = V W V^T
```

where H^{(ℓ)} ∈ R^{k×k} are learnable coefficient matrices. The polynomial filter allows learning frequency-dependent transformations: different eigenvalue magnitudes can be amplified or attenuated differently.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors |
| `polynomial_degree` | int | 5 | Polynomial degree K |

**Config:** `exp_configs/models/spectral/filter_bank.yaml`

### SelfAttentionDenoiser

Scaled dot-product attention on eigenvector embeddings.

**Formula:**
```
Q = V W_Q,  K = V W_K
Â = Q K^T / √d_k
```

where W_Q, W_K ∈ R^{k×d_k} are learnable projections. The 1/√d_k scaling stabilizes gradients following transformer practice.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors (input dim) |
| `d_k` | int | 64 | Key/query dimension |

**Config:** `exp_configs/models/spectral/self_attention.yaml`

## DiGress Models

Diffusion-based transformer for graph denoising. Used as baseline for comparing spectral architectures.

**Location:** `src/tmgg/models/digress/`

### DiGress Transformer

Score-based model operating on graph structure. Two variants are used in experiments:

**Official settings** (`digress_sbm_small.yaml`):
- LR=0.0002, AdamW + amsgrad, weight_decay=1e-12
- No LR scheduling

**High LR variant** (`digress_sbm_small_highlr.yaml`):
- LR=1e-2 (matching spectral models)
- For fair comparison with spectral architectures

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_layers` | int | 4 | Number of transformer layers |
| `node_feature_dim` | int | 50 | Node feature dimension |
| `use_eigenvectors` | bool | True | Use eigenvector features |
| `hidden_dims.dx` | int | 128 | Node hidden dimension |
| `hidden_dims.de` | int | 32 | Edge hidden dimension |
| `hidden_dims.n_head` | int | 4 | Number of attention heads |

**Config:** `exp_configs/models/digress/digress_sbm_small.yaml`

## Baseline Models

Simple baselines for comparison.

**Location:** `src/tmgg/experiments/baselines/`

### Linear Baseline

Linear transformation initialized at identity.

**Formula:**
```
A_pred = W @ A @ W^T + b
```

**Config:** `exp_configs/models/baselines/linear.yaml`

### MLP Baseline

Multi-layer perceptron applied to flattened adjacency.

**Config:** `exp_configs/models/baselines/mlp.yaml`

## Layers

Shared layers used by the models:

### EigenEmbedding

Computes eigenvector embeddings from adjacency matrices.

**Location:** `src/tmgg/models/layers/eigen_embedding.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eigenvalue_reg` | float | 0.0 | Diagonal regularization for gradient stability |

Eigendecomposition gradients can be unstable when eigenvalues are close together. Setting `eigenvalue_reg` to a small value (e.g., 1e-3) adds a diagonal perturbation that spreads eigenvalues apart.

### GraphConvolutionLayer

Polynomial graph convolution layer.

**Location:** `src/tmgg/models/layers/gcn.py`

### MultiHeadAttention

Standard multi-head attention layer.

**Location:** `src/tmgg/models/layers/mha_layer.py`

## Domain Transformations

Models support two domain transformations, configured via the `domain` parameter:

### Standard Domain

- Input: Adjacency matrix used directly
- Output: Sigmoid applied to produce probabilities in [0, 1]

### Inv-Sigmoid Domain

- Input: Logit transform applied (numerically stabilized)
- Output: In training mode, raw logits are returned for BCEWithLogitsLoss; in eval mode, sigmoid is applied

The inv-sigmoid domain can improve numerical stability for sparse graphs.

```python
# Using inv-sigmoid domain
model = GNN(num_layers=2, domain="inv-sigmoid")
```

## Eigenvalue Regularization

GNN models can experience gradient instability due to eigendecomposition. When eigenvalues are close together, gradients involve terms like 1/(λ_i - λ_j) that can explode.

To mitigate this, set `eigenvalue_reg` to a small positive value:

```python
model = GNN(num_layers=2, eigenvalue_reg=1e-3)
```

This adds `eigenvalue_reg * I` to the adjacency matrix before eigendecomposition, spreading eigenvalues apart. Use values between 1e-4 and 1e-2 if you observe NaN gradients or unstable training.
