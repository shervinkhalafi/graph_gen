# Pickup: SBM Eigenstructure Covariance Study (2026-01-12)

## Research Question

How do axes of variation in SBM parameters affect the covariance structure of the graph eigenspectrum?

**Core observation**: Fixed SBM parameters yield nearly zero eigenvalue covariance across samples (all graphs are structurally similar). Allowing variation along certain axes introduces meaningful covariance structure that may be exploitable for denoising or generation.

## Axes of Variation to Study

### 1. Number of Communities (k)
- Range: 2-8 blocks
- Hypothesis: More communities → more distinct eigenvalue clusters, higher inter-cluster covariance

### 2. Community Sizes
- **Balanced**: All blocks have equal size
- **Imbalanced**: Power-law or random size distribution
- Hypothesis: Imbalanced sizes create heterogeneous eigenvalue spacing, introducing covariance between "large block" and "small block" eigenvalues

### 3. Connectivity Patterns
- **p_intra** (within-block edge probability): Controls block cohesion
- **q_inter** (between-block edge probability): Controls block separation
- **Ratio p/q**: Determines community detectability
- Hypothesis: As p/q → 1, eigenstructure transitions from block-diagonal to random, with maximum covariance near the phase transition

### 4. Block Connectivity Structure
- **Assortative**: High p_intra, low q_inter (standard community structure)
- **Disassortative**: Low p_intra, high q_inter (bipartite-like)
- **Core-periphery**: Dense core connected to sparse periphery
- **Hierarchical**: Nested block structure

## Experimental Design

### Dataset Generation

```python
# Pseudocode for covariance study dataset
sbm_configs = []
for k in [2, 3, 4, 5, 6, 8]:  # number of communities
    for size_dist in ['balanced', 'power_law', 'random']:
        for p_intra in [0.3, 0.5, 0.7, 0.9, 1.0]:
            for q_ratio in [0.0, 0.1, 0.3, 0.5]:  # q = q_ratio * p_intra
                sbm_configs.append({
                    'num_blocks': k,
                    'size_distribution': size_dist,
                    'p_intra': p_intra,
                    'q_inter': q_ratio * p_intra,
                    'num_nodes': 50,  # or varying
                })
```

### Metrics to Compute

For each configuration, generate N graphs and compute:

1. **Eigenvalue covariance matrix**: Cov(λ_i, λ_j) across samples
2. **Eigenvector alignment variance**: How stable are eigenvector orientations?
3. **Spectral gap statistics**: Distribution of λ_k - λ_{k+1}
4. **Frobenius norm of covariance**: ||Cov(Λ)||_F as summary statistic

### Analysis Questions

1. Which axis of variation contributes most to eigenvalue covariance?
2. Is there interaction between axes (e.g., k × size_dist)?
3. Does the covariance structure predict denoising difficulty?
4. Can we identify "eigenvalue modes" that correspond to structural features?

## Connection to Denoising

The eigenstructure covariance may inform:

1. **Prior design**: If certain eigenvalue patterns co-vary, the denoiser can exploit this
2. **Noise model**: Understanding natural variation helps distinguish it from added noise
3. **Shrinkage targets**: Covariance structure suggests where to shrink eigenvalues

## Implementation Notes

### Existing Infrastructure

- `tmgg.experiment_utils.data.GraphDataModule` supports SBM generation
- `TopKEigenLayer` extracts eigenvalues/eigenvectors
- Modal infrastructure can run large parameter sweeps

### New Code Needed

1. **Covariance computation module**: Given a dataset, compute eigenvalue covariance
2. **Parameter sweep generator**: Generate SBM configs covering the variation axes
3. **Visualization**: Heatmaps of covariance matrices, PCA of eigenvalue distributions

### Potential Location

```
src/tmgg/analysis/
    eigenstructure_covariance.py  # Core covariance computation
    sbm_parameter_sweep.py        # Config generation for sweep
    covariance_visualization.py   # Plotting utilities
```

## Next Steps

1. [ ] Implement eigenvalue covariance computation for a single SBM config
2. [ ] Generate small pilot sweep (2-3 values per axis) to validate approach
3. [ ] Visualize covariance matrices and identify patterns
4. [ ] Scale up to full parameter sweep on Modal
5. [ ] Analyze which axes contribute most to covariance
6. [ ] Connect findings to denoising model design

## References

- Eigenvalue distribution of SBM: Related to Wigner semicircle with block structure corrections
- Phase transition in community detection: Decelle et al., spectral detectability threshold
- Shrinkage estimation: Ledoit-Wolf, eigenvalue shrinkage under covariance estimation
