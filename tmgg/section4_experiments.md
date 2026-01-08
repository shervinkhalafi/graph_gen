# Section 4: Experiments

## 4.1 Simple Architectures

### Architecture 1: Graph Filter Bank
- **Model**:
  - \( Y = \sum_{k=0}^{K-1} S^k V H^{(k)} = V \sum_{k=0}^{K-1} \Lambda^k H^{(k)} \)
  - Reconstruction: \( \hat A = YY^\top \)

- **Ablations**:
  1. **Non-linearity**: \( \hat A = \sigma(YY^\top) \)
  2. **Asymmetry**:
     - \( X = \sum_{k} S^k V H_1^{(k)}, \; Y = \sum_{k} S^k V H_2^{(k)} \)
     - \( \hat A = XY^\top \)
  3. **PEARL embeddings instead of eigenvectors**:
     - Embeddings: \( E \in \mathbb{R}^{n \times c} \), with
       \( e_i = \mathrm{diag}(\sum_{k=0}^{K-1} h_i^{(k)} A^k) \)
     - Filter: \( Y = \sum_{k=0}^{K-1} S^k E H^{(k)} \)

- **Total variants**: 8

---

### Architecture 2: Linear Positional Encoding (PE)
- **Model**:
  - \( Y = V W + \mathbf{1} b^\top \)
  - Reconstruction: \( \hat A = YY^\top \)

- **Variants**:
  - Same ablations as Graph Filter Bank
  - **Total variants**: 8

---

## 4.2 Graph Self-Attention

- **Baseline**:
  - Graph self-attention model (DiGress-style), without FiLM.

- **Modified variant**:
  - Replace linear projection blocks (Q/K/V) with graph filter banks.
  - Two attention heads with outputs \( Y_1, Y_2 \)
  - Reconstructions:
    - \( \hat A = Y_1 Y_2^\top \)
    - or \( \hat A = \sigma(Y_1 Y_2^\top) \)

- **Inputs**:
  - Eigenvectors **or**
  - DiGress embeddings

- **Total variants**:
  - 2 input types × 2 linear ops = 4
  - (8 if output non-linearity is included)

---

## 4.3 DiGress Variants

- **Architecture 1**: Full DiGress
- **Architecture 2**: DiGress with linear blocks replaced by graph filter banks

- **Inputs**:
  - Eigenvectors
  - PEARL embeddings

- **Total variants**: 4

---

## 4.4 Dataset-Level Analysis

- Compute **spectral variance**:
  - \( \mathrm{Var}(\Lambda) \) across graphs in each dataset

- **Hypothesis**:
  - Larger spectral variance ⇒ larger performance gap favoring graph filters

- **Synthetic benchmarks**:
  - Multiple SBM datasets
  - Controlled levels of \( \mathrm{Var}(\Lambda) \)

