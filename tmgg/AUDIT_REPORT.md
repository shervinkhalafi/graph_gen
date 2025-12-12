# TMGG Mathematical Audit Report

## Executive Summary

The tmgg codebase is a research framework for graph denoising using spectral and neural network methods. The core task is recovering clean adjacency matrices from noisy observations. This audit examined all model implementations, loss computations, metrics, and data transformations. I identified **14 issues** categorized by severity based on their impact on model correctness and training stability.

---

## Codebase Overview

**Purpose:** Learn to denoise adjacency matrices via various architectures:
1. **Attention models** - treat adjacency as sequence, apply multi-head attention
2. **GNN models** - eigendecomposition → polynomial graph convolution → outer product reconstruction
3. **Spectral denoisers** - operate directly on top-k eigenvectors with learnable transformations
4. **Hybrid models** - GNN embeddings + attention refinement

**Data flow:**
```
A_clean → add_noise(A, eps) → A_noisy → model(A_noisy) → logits → sigmoid() → A_pred
                                                               ↓
                                                    BCEWithLogitsLoss(logits, A_clean)
```

---

## Issue 1 [CRITICAL]: Missing Attention Scaling Factor

**File:** `tmgg/src/tmgg/models/layers/mha_layer.py`
**Line:** 59

### Code
```python
# Scaling factor for dot product attention
self.scale = 1
```

### Mathematical Background

The standard scaled dot-product attention formula from "Attention Is All You Need" (Vaswani et al., 2017):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The scaling factor $1/\sqrt{d_k}$ is not optional decoration. The dot product $q \cdot k$ for random unit vectors in $d_k$ dimensions has variance proportional to $d_k$. Without scaling:

- For $d_k = 64$ (typical), dot products have ~8× larger magnitude than intended
- Pre-softmax logits become large (e.g., ±50 instead of ±6)
- $\text{softmax}(50) ≈ 1.0$, $\text{softmax}(-50) ≈ 0.0$ (near one-hot)
- Gradient $\partial \text{softmax}/\partial x = \text{softmax}(x)(1 - \text{softmax}(x)) ≈ 0$ when saturated

### Verification

Line 100 shows the scores computation:
```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # scale=1, no effect
```

The default `d_k = d_model // num_heads`. For the config `d_model=200, num_heads=4`, we have `d_k=50`. Without scaling, scores are ~7× too large.

### Impact

- **Training:** Vanishing gradients through attention make the model effectively random for attention-based refinement. The model may still train via other pathways (residual connections, output projection) but attention mechanism contributes minimally.
- **Inference:** Attention weights collapse to near-uniform or near-one-hot distributions, losing the ability to attend selectively.

### Correct Implementation
```python
self.scale = 1.0 / math.sqrt(self.d_k)
```

---

## Issue 2 [CRITICAL]: Eigendecomposition Without Symmetry Enforcement

**File:** `tmgg/src/tmgg/models/layers/eigen_embedding.py`
**Line:** 116

### Code
```python
_, V = torch.linalg.eigh(A[i])
```

### Mathematical Background

`torch.linalg.eigh()` computes eigendecomposition for Hermitian (self-adjoint) matrices. For real matrices, this means symmetric: $A = A^T$. Given a non-symmetric matrix $B$, `eigh()` silently operates on $(B + B^T)/2$, the symmetric part.

For undirected graphs, the adjacency matrix should be symmetric by construction. However:

1. **Floating-point accumulation:** Operations like `V @ Λ @ V^T` produce numerical asymmetry (typically $10^{-7}$ to $10^{-5}$)
2. **Batch operations:** `torch.bmm` may accumulate errors differently across batch dimension
3. **Gradient flow:** Backprop through eigh is sensitive to eigenvalue gaps; near-degenerate eigenvalues cause numerical issues

### Verification

The `add_rotation_noise` function in `noise.py` (lines 48-52, 55-72) explicitly uses `eigh` on input A, then reconstructs via `V_rot @ diag(λ) @ V_rot^T`. This reconstruction can introduce asymmetry through floating-point errors.

The `TopKEigenLayer` (line 75) also uses `eigh` but doesn't symmetrize:
```python
eigenvalues, eigenvectors = torch.linalg.eigh(A)
```

### Impact

- Silent loss of the antisymmetric component if present
- Inconsistent eigenvector orientations across training iterations
- Potential for subtle training instabilities that are hard to diagnose

### Correct Implementation
```python
A_sym = (A + A.transpose(-1, -2)) / 2
_, V = torch.linalg.eigh(A_sym)
```

---

## Issue 3 [HIGH]: Matrix Power Overflow in Polynomial GCN

**File:** `tmgg/src/tmgg/models/layers/gcn.py`
**Lines:** 42-44

### Code
```python
Y_hat = X @ self.H[0]
for i in range(1, self.num_terms + 1):
    A_power_i = torch.matrix_power(A, i)
    Y_hat += torch.bmm(A_power_i, X) @ self.H[i]
```

### Mathematical Background

For adjacency matrix $A$ with spectral radius $\rho(A) = \max_i |\lambda_i|$:
- If $\rho(A) > 1$: $\|A^i\| \to \infty$ exponentially
- If $\rho(A) < 1$: $\|A^i\| \to 0$ exponentially
- If $\rho(A) = 1$: $\|A^i\|$ bounded but may oscillate

For binary adjacency matrices of graphs with $n$ nodes and average degree $d$:
- Expected spectral radius $\approx d$ (for Erdős-Rényi graphs)
- Dense graphs can have $\rho(A) > 10$

With `num_terms=3` (default), $A^3$ can have entries in the thousands for dense graphs.

### Verification

No normalization is applied. The layer initializes `H` with Xavier uniform (line 20-21), expecting inputs of moderate scale. When $A^3$ has large values, the effective input scale is wrong for the learned weights.

### Impact

- **Dense/large graphs:** Gradient explosion, NaN loss
- **Sparse graphs:** May work but polynomial terms contribute unevenly
- **Training instability:** Learning rate tuning becomes graph-density dependent

### Mitigation Options

1. **Symmetric normalization:** $\tilde{A} = D^{-1/2} A D^{-1/2}$ guarantees $\rho(\tilde{A}) \leq 1$
2. **Random walk normalization:** $\tilde{A} = D^{-1} A$ ensures row-stochastic property
3. **Chebyshev polynomials:** Bounded on [-1, 1], numerically stable

---

## Issue 4 [HIGH]: Dynamic Layer Recreation in Forward Pass

**File:** `tmgg/src/tmgg/models/gnn/nvgnn.py`
**Lines:** 77-84

### Code
```python
# Dynamically create layers if needed with correct num_nodes
if len(self.layers) == 0 or self.layers[0].num_nodes != Z.shape[1]:
    self.layers = nn.ModuleList()
    for _ in range(self.num_layers):
        self.layers.append(
            NodeVarGraphConvolutionLayer(
                self.num_terms, self.feature_dim, Z.shape[1]
            )
        )
```

### Mathematical Background

PyTorch's autograd tracks parameters through the module hierarchy. When you assign a new `nn.ModuleList` to `self.layers` during forward:

1. The new layers get fresh random initialization
2. The optimizer (created at training start) holds references to the **old** parameters
3. Gradients flow to the new layers but optimizer updates the old (orphaned) parameters
4. Checkpoint loading fails due to state dict key mismatch

### Verification

This triggers when graph size changes between batches. The `__init__` creates initial layers (lines 40-44), but forward replaces them if node count differs.

### Impact

- **Training:** Model appears to train but weights are randomly reinitialized each forward pass when graph size varies
- **Checkpoints:** Loading fails with "unexpected key" or "missing key" errors
- **Memory:** Creates new tensors each forward, potential memory leak

### Correct Approach

Use size-agnostic architectures (global pooling, message passing) or enforce fixed graph size. If variable size is required, design the architecture around it from the start rather than dynamic recreation.

---

## Issue 5 [HIGH]: Parameter Truncation in NodeVarGCN Layer

**File:** `tmgg/src/tmgg/models/layers/nvgcn_layer.py`
**Lines:** 45-53, 69-76

### Code
```python
if self.h.shape[2] != num_nodes:
    h_vals = self.h[0, c]
    if h_vals.shape[0] > num_nodes:
        h_vals = h_vals[:num_nodes]  # TRUNCATION: discards learned weights
    else:
        h_vals = torch.nn.functional.pad(
            h_vals, (0, num_nodes - h_vals.shape[0])  # PADDING: dead gradients
        )
```

### Mathematical Background

The parameter tensor `self.h` has shape `(num_terms+1, num_channels, num_nodes)`. This means weights are node-specific, which is unusual and creates the variable-size problem.

- **Truncation:** When current graph is smaller than stored `num_nodes`, learned weights for nodes > current size are simply ignored. The model learned these weights, but they're discarded.
- **Zero-padding:** When graph is larger, new node positions get zero weights. Gradient at zero-padded positions is zero (dead neurons).

### Verification

The padding/truncation happens for every polynomial term (outer loop lines 65-86) and every channel (inner loop line 42). This is executed during forward, not initialization.

### Impact

- Training on variable-size graphs causes learned parameters to be randomly discarded or dead
- No warning or error is raised
- Model capacity is effectively limited to smallest graph seen

### Design Issue

Node-variant graph convolution fundamentally conflicts with variable graph sizes. Either:
1. Enforce fixed graph size
2. Use node-agnostic parameterization (shared weights across nodes)

---

## Issue 6 [MEDIUM-HIGH]: Eigenvalue Power Accumulation in Spectral Filter Bank

**File:** `tmgg/src/tmgg/models/spectral_denoisers/filter_bank.py`
**Lines:** 104-112

### Code
```python
Lambda_power = torch.ones_like(Lambda)  # (batch, k), Λ^0 = 1
for ell in range(self.polynomial_degree):
    Lambda_matrix = Lambda_power.unsqueeze(-1).expand(-1, -1, k)  # (batch, k, k)
    W = W + Lambda_matrix * self.H[ell].unsqueeze(0)
    Lambda_power = Lambda_power * Lambda  # Accumulates Λ^ℓ
```

### Mathematical Background

Computes spectral polynomial filter:
$$W = \sum_{\ell=0}^{K-1} \Lambda^\ell \odot H^{(\ell)}$$

where $\Lambda = \text{diag}(\lambda_1, ..., \lambda_k)$ contains eigenvalues.

For symmetric adjacency matrices of undirected graphs, eigenvalues can be positive or negative with $|\lambda| \leq n-1$ (complete graph bound). Typical values for random graphs with $n=50$ nodes might have $|\lambda_1| \approx 10$.

With `polynomial_degree=5`:
- $\lambda^5$ for $\lambda=10$ gives $10^5 = 100,000$
- $\lambda^5$ for $\lambda=0.5$ gives $0.03$
- Ratio: $3 \times 10^6$ scale difference within same batch

### Verification

No eigenvalue normalization or clamping is applied. The `TopKEigenLayer` (which provides eigenvalues) returns raw eigenvalues from `eigh`.

### Impact

- $H^{(\ell)}$ for higher polynomial orders must learn to compensate for huge scale differences
- Gradient magnitudes vary by orders of magnitude across polynomial terms
- Training may be dominated by a single eigenvalue magnitude

### Mitigation

Normalize eigenvalues to [-1, 1] before polynomial computation:
```python
Lambda_normalized = Lambda / Lambda.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
```

---

## Issue 7 [MEDIUM]: Metrics `eigsh` Without Convergence Handling

**File:** `tmgg/src/tmgg/experiment_utils/metrics.py`
**Lines:** 40-43, 78-79

### Code
```python
l_true, _ = eigsh(A_true, k=k, which="LM", maxiter=10000)
l_pred, _ = eigsh(A_pred, k=k, which="LM", maxiter=10000)
```

### Mathematical Background

`scipy.sparse.linalg.eigsh` uses ARPACK's implicitly restarted Lanczos method, which is iterative and can fail to converge for:
- Ill-conditioned matrices
- Near-degenerate eigenvalues
- Very sparse matrices with disconnected components

### Verification

The `maxiter=10000` helps but doesn't guarantee convergence. No try/except wraps these calls. A `scipy.sparse.linalg.ArpackNoConvergence` exception crashes metric computation.

### Impact

- Validation/test metrics can crash training
- Especially problematic for very sparse graphs or graphs with many zero eigenvalues

### Mitigation
```python
try:
    l_true, _ = eigsh(A_true, k=k, which="LM", maxiter=10000)
except ArpackNoConvergence:
    # Fallback to dense eigendecomposition for small matrices
    l_true = np.linalg.eigvalsh(A_true)[-k:]
```

---

## Issue 8 [MEDIUM]: Inconsistent Sigmoid Application Across Models

**Files:**
- `models/attention/attention.py:82` - applies sigmoid
- `models/gnn/gnn_sym.py:73` - applies sigmoid
- `models/gnn/nvgnn.py:92` - applies sigmoid
- `models/hybrid/hybrid.py:74` - applies sigmoid
- `models/spectral_denoisers/*.py` - returns raw logits

**Base class contract** (`models/base.py:88-104, 115-119`):
```python
def predict(self, logits: torch.Tensor) -> torch.Tensor:
    """Convert model output (logits) to predictions (probabilities)."""
    return torch.sigmoid(logits)

def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple]:
    """Returns: Raw logits (unbounded). Use predict() for [0, 1] probabilities."""
    pass
```

### Analysis

The base class documentation states `forward()` should return logits, and `predict()` applies sigmoid. But several models apply sigmoid internally in `forward()`:

```python
# attention.py:82
return torch.sigmoid(x)

# gnn_sym.py:73
return torch.sigmoid(outer), X

# nvgnn.py:92
return torch.sigmoid(outer)

# hybrid.py:74
return torch.sigmoid(A_recon)
```

While spectral denoisers (filter_bank, linear_pe) correctly return raw logits.

### Impact

- If training code uses `BCEWithLogitsLoss` (expects logits) but model returns probabilities: mathematically wrong loss
- Currently, `base_lightningmodule.py:326-327` uses `transform_for_loss()` which returns output unchanged, assuming logits
- Models applying sigmoid internally break this assumption

### Verification

Looking at `base_lightningmodule.py:82-88`:
```python
if loss_type == "MSE":
    self.criterion = nn.MSELoss()
elif loss_type == "BCEWithLogits":
    self.criterion = nn.BCEWithLogitsLoss()
```

If `loss_type="BCEWithLogits"` is used with attention/GNN models that return sigmoid(output), the loss computes:
$$-[y \log(\sigma(\sigma(x))) + (1-y)\log(1-\sigma(\sigma(x)))]$$

This is wrong. The double-sigmoid compresses the output range further.

### Recommendation

Standardize all models to return logits. Remove sigmoid from forward() in attention.py, gnn_sym.py, nvgnn.py, hybrid.py.

---

## Issue 9 [MEDIUM]: Division by Small Values in diffusion_utils

**File:** `tmgg/src/tmgg/models/digress/diffusion_utils.py`
**Line:** 356

### Code
```python
denominator = prod.unsqueeze(-1)  # bs, N, d0, 1
denominator[denominator == 0] = 1e-6
out = numerator / denominator
```

### Analysis

The exact equality check `denominator == 0` only catches exact floating-point zeros. Values like `1e-30` pass through and cause:
- Division result ~$10^{30}$
- Potential overflow to inf
- NaN propagation through subsequent operations

### Verification

The surrounding context (lines 353-359) shows this is computing a posterior distribution. Division by near-zero values corrupts the posterior.

### Mitigation

Use a threshold-based clamp:
```python
denominator = denominator.clamp(min=1e-6)
```

Or use the numerically stable form with log-space computation.

---

## Issue 10 [MEDIUM]: Residual Connection Shape Assumptions

**File:** `tmgg/src/tmgg/models/hybrid/hybrid.py`
**Lines:** 61-65

### Code
```python
if self.denoising_model is not None:
    Z_denoised = self.denoising_model(Z)
    Z_pred = Z + Z_denoised  # Direct addition assumes matching shapes
```

### Analysis

The residual assumes `denoising_model` returns same shape as input. The `MultiLayerAttention` model does preserve shape, but:

1. No explicit assertion/check
2. If attention model's d_model differs from `2 * feature_dim`, silent broadcasting or error
3. The factory function `create_sequential_model()` (lines 111-149) correctly computes `d_model = 2 * feature_dim_out`, but direct instantiation could break this

### Impact

Low in practice since the factory function is used, but fragile architecture.

---

## Issue 11 [LOW-MEDIUM]: Sign Normalization Edge Case

**File:** `tmgg/src/tmgg/models/spectral_denoisers/topk_eigen.py`
**Lines:** 118-149

### Code
```python
eps = 1e-10
abs_V = V.abs()
nonzero_mask = abs_V > eps  # (batch, n, k)

# Find first nonzero via cumsum
cumsum_mask = nonzero_mask.cumsum(dim=1)
first_nonzero_mask = (cumsum_mask == 1) & nonzero_mask
```

### Analysis

This correctly handles the standard case. The edge case handling (lines 141-145):
```python
sign_multipliers = torch.where(
    sign_multipliers == 0,
    torch.ones_like(sign_multipliers),
    sign_multipliers
)
```

Sets sign to +1 if eigenvector is all-zero. This is mathematically degenerate (zero eigenvector indicates a bug in eigh or k > rank(A)).

### Impact

Handles the edge case gracefully but silently. A warning would be useful for debugging.

---

## Issue 12 [LOW]: Masked Softmax with -inf

**File:** `tmgg/src/tmgg/models/layers/masked_softmax.py`
**Lines:** 37-39

### Code
```python
x_masked = x.clone()
x_masked[mask == 0] = -float("inf")
return torch.softmax(x_masked, **kwargs)
```

### Analysis

There's already a guard for all-masked (line 34-35):
```python
if mask.sum() == 0:
    return torch.zeros_like(x)
```

But partial masking where an entire row is masked could still produce NaN from `softmax([-inf, -inf, ...])`. PyTorch's softmax handles this (returns 0/0 which is NaN), but the function doesn't check for row-wise all-masked.

### Impact

Low - requires specific masking pattern that's unlikely in practice.

---

## Issue 13 [LOW]: Division by Small Eigenvalue Norm in Metrics

**File:** `tmgg/src/tmgg/experiment_utils/metrics.py`
**Line:** 43

### Code
```python
eigval_error = np.linalg.norm(l_pred - l_true) / np.linalg.norm(l_true)
```

### Analysis

For graphs with near-zero eigenvalues (very sparse, near-empty), `norm(l_true) → 0`, causing division instability. This is unlikely for typical adjacency matrices but possible for:
- Graphs with mostly isolated nodes
- Very low density random graphs

---

## Issue 14 [LOW]: Tanh After LayerNorm Redundancy

**File:** `tmgg/src/tmgg/models/layers/gcn.py`
**Lines:** 45-46

### Code
```python
Y_hat = self.layer_norm(Y_hat)
Y_hat = self.activation(Y_hat)  # nn.Tanh()
```

### Analysis

LayerNorm normalizes to approximately $\mathcal{N}(0, 1)$. For standard normal inputs, $\tanh(x) \approx x$ for $|x| < 1$ (linear regime). The nonlinearity is mostly wasted.

### Impact

Not incorrect, but Tanh provides minimal nonlinearity benefit after LayerNorm. Consider ReLU/GELU or remove one of the two.

---

## Summary Table

| # | Severity | File | Lines | Issue | Impact |
|---|----------|------|-------|-------|--------|
| 1 | CRITICAL | `mha_layer.py` | 59 | Missing 1/√d_k scaling | Softmax saturation, dead attention |
| 2 | CRITICAL | `eigen_embedding.py` | 116 | No symmetry enforcement | Silent corruption for non-symmetric A |
| 3 | HIGH | `gcn.py` | 42-44 | Unbounded matrix powers | Overflow for dense graphs |
| 4 | HIGH | `nvgnn.py` | 77-84 | Dynamic layer recreation | Breaks training/checkpoints |
| 5 | HIGH | `nvgcn_layer.py` | 45-76 | Parameter truncation/padding | Loses learned weights |
| 6 | MED-HIGH | `filter_bank.py` | 104-112 | Eigenvalue power accumulation | Scale imbalance in gradients |
| 7 | MEDIUM | `metrics.py` | 40-43 | No eigsh convergence handling | Can crash validation |
| 8 | MEDIUM | Multiple | - | Inconsistent sigmoid application | Wrong loss for BCEWithLogits |
| 9 | MEDIUM | `diffusion_utils.py` | 356 | Exact zero check for division | Near-zero division unhandled |
| 10 | MEDIUM | `hybrid.py` | 61-65 | Residual shape assumption | Fragile architecture |
| 11 | LOW-MED | `topk_eigen.py` | 118-149 | Zero eigenvector edge case | Silent handling of degenerate case |
| 12 | LOW | `masked_softmax.py` | 37-39 | Row-wise all-masked not handled | Potential NaN |
| 13 | LOW | `metrics.py` | 43 | Division by small norm | Instability for sparse graphs |
| 14 | LOW | `gcn.py` | 45-46 | Tanh after LayerNorm | Redundant nonlinearity |

---

## Files Affected

```
tmgg/src/tmgg/models/
├── layers/
│   ├── mha_layer.py          [#1: CRITICAL]
│   ├── eigen_embedding.py    [#2: CRITICAL]
│   ├── gcn.py                [#3: HIGH, #14: LOW]
│   ├── nvgcn_layer.py        [#5: HIGH]
│   └── masked_softmax.py     [#12: LOW]
├── gnn/
│   ├── nvgnn.py              [#4: HIGH]
│   └── gnn_sym.py            [#8: MEDIUM]
├── attention/
│   └── attention.py          [#8: MEDIUM]
├── hybrid/
│   └── hybrid.py             [#8, #10: MEDIUM]
├── spectral_denoisers/
│   ├── topk_eigen.py         [#11: LOW-MED]
│   └── filter_bank.py        [#6: MED-HIGH]
└── digress/
    └── diffusion_utils.py    [#9: MEDIUM]

tmgg/src/tmgg/experiment_utils/
└── metrics.py                [#7, #13: MEDIUM/LOW]
```

---

## Fixes Applied

All 14 issues have been addressed. Below is a summary of each fix.

### Issue 1 [CRITICAL]: Missing Attention Scaling Factor
**Fix:** Added `math.sqrt(self.d_k)` scaling factor in `mha_layer.py:62`.
```python
self.scale = 1.0 / math.sqrt(self.d_k)
```
The scaling prevents softmax saturation when d_k is large (per Vaswani et al. 2017).

### Issue 2 [CRITICAL]: Eigendecomposition Without Symmetry Enforcement
**Fix:** Symmetrize input before `eigh` in both `eigen_embedding.py:119` and `topk_eigen.py:77`.
```python
A_sym = (A + A.transpose(-1, -2)) / 2
eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
```
Enforces symmetry to handle floating-point asymmetries from prior operations.

### Issue 3 [HIGH]: Matrix Power Overflow in Polynomial GCN
**Fix:** Added symmetric normalization `D^{-1/2} A D^{-1/2}` in `gcn.py:43-53` before computing matrix powers. This bounds the spectral radius to [-1, 1], preventing overflow for dense graphs.

### Issue 4 [HIGH]: Dynamic Layer Recreation in NodeVarGNN
**Fix:** Removed dynamic layer recreation logic from `nvgnn.py`. The redesigned `NodeVarGraphConvolutionLayer` now uses node-agnostic parameters, supporting any graph size without recreation.

### Issue 5 [HIGH]: Parameter Truncation in NodeVarGCN Layer
**Fix:** Complete architectural redesign of `nvgcn_layer.py`. Changed parameter tensor from `(num_terms+1, num_channels, num_nodes)` to `(num_terms+1, num_channels_in, num_channels_out)`, making weights node-agnostic. Added symmetric normalization for stability.

### Issue 6 [MEDIUM-HIGH]: Eigenvalue Power Accumulation
**Fix:** Normalize eigenvalues to [-1, 1] in `filter_bank.py:100-104` before polynomial computation:
```python
Lambda_max = Lambda.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
Lambda_normalized = Lambda / Lambda_max
```
Prevents gradient imbalance across polynomial terms.

### Issue 7 [MEDIUM]: Metrics `eigsh` Without Convergence Handling
**Fix:** Added `_safe_eigsh()` wrapper in `metrics.py` with `ArpackNoConvergence` exception handling and fallback to dense `np.linalg.eigh`.

### Issue 8 [MEDIUM]: Inconsistent Sigmoid Application
**Fix:** Removed `torch.sigmoid()` from `forward()` in:
- `attention.py:82`
- `gnn_sym.py:73`
- `nvgnn.py:86`
- `hybrid.py:79`

All models now return raw logits per base class contract; `predict()` applies sigmoid.

### Issue 9 [MEDIUM]: Division by Small Values in diffusion_utils
**Fix:** Replaced exact zero check with `clamp(min=1e-6)` in `diffusion_utils.py:359`:
```python
denominator = denominator.clamp(min=1e-6)
```
Catches near-zero values that the exact equality check missed.

### Issue 10 [MEDIUM]: Residual Connection Shape Assumptions
**Fix:** Added explicit shape assertion in `hybrid.py:64-67`:
```python
assert Z_denoised.shape == Z.shape, (
    f"Denoising model output shape {Z_denoised.shape} != input shape {Z.shape}."
)
```

### Issue 11 [LOW-MEDIUM]: Sign Normalization Edge Case
**Fix:** Added warning when zero eigenvector detected in `topk_eigen.py:148-156`:
```python
if zero_eigenvector_mask.any():
    warnings.warn("Zero eigenvector detected; may indicate k > rank(A)")
```

### Issue 12 [LOW]: Masked Softmax with -inf
**Fix:** Handle NaN from row-wise all-masked case in `masked_softmax.py:41-45`:
```python
nan_mask = torch.isnan(result)
if nan_mask.any():
    result = torch.where(nan_mask, torch.zeros_like(result), result)
```

### Issue 13 [LOW]: Division by Small Eigenvalue Norm
**Fix:** Guard denominator in `metrics.py:73-74`:
```python
eigval_error = np.linalg.norm(l_pred - l_true) / max(norm_true, 1e-10)
```

### Issue 14 [LOW]: Tanh After LayerNorm Redundancy
**Fix:** Replaced `nn.Tanh()` with `nn.GELU()` in `gcn.py:25` and `nvgcn_layer.py:77`. GELU provides better nonlinearity after LayerNorm since Tanh is nearly linear in the normalized range.
