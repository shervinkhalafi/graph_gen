# Eigenstructure Experiment Analysis Summary

**Data**: 2013 runs from `graph_denoise_team`, 1955 finished, 1870 with valid test_mse > 0

## Key Findings

### 1. Model Type
| Model Type | N | Mean test_mse | Min | Max |
|------------|---|---------------|-----|-----|
| DiGress | 832 | **0.0871** | 0.0007 | 0.3377 |
| Spectral | 1038 | 0.1871 | 0.0056 | 0.4983 |

**Conclusion**: DiGress significantly outperforms Spectral denoising.

### 2. Stage Progression
| Stage | N | Mean test_mse | Min | Best Config |
|-------|---|---------------|-----|-------------|
| **stage2c** | 384 | **0.0752** | 0.0371 | digress_transformer |
| stage1c | 144 | 0.1063 | 0.1007 | digress_transformer |
| stage1f | 144 | 0.1065 | 0.1007 | digress_transformer |
| stage2 | 576 | 0.1158 | 0.0371 | mixed |
| stage1 | 269 | 0.2414 | 0.0007 | spectral_linear (on sbm_small) |
| stage1d | 144 | 0.3626 | 0.3025 | asymmetric (worst) |

**Conclusion**: Stage2c achieves best results. Stage1d (asymmetric) is significantly worse.

### 3. Architecture Variants (DiGress)
| Architecture | N | Mean test_mse | Min |
|--------------|---|---------------|-----|
| gnn_all | 132 | 0.0836 | 0.0372 |
| gnn_v | 132 | 0.0837 | 0.0371 |
| digress_default | 384 | 0.0868 | 0.0371 |
| gnn_qk | 168 | 0.0886 | 0.0371 |

**Conclusion**: All DiGress variants perform essentially identically (<3% difference).

### 4. K Value (Number of Eigenvectors)
| K | N | Mean test_mse | Min |
|---|---|---------------|-----|
| **32** | 528 | **0.0988** | 0.0371 |
| 16 | 772 | 0.1432 | 0.0056 |
| 8 | 458 | 0.1822 | 0.0272 |
| 50 | 96 | 0.1901 | 0.1488 |

**Conclusion**: k=32 is optimal overall. Higher k (50) doesn't help.

### 5. Hyperparameters (Stage2c)

**Learning Rate x Weight Decay (mean test_mse)**:
| LR \ WD | 0.001 | 0.01 |
|---------|-------|------|
| 0.0005 | 0.0750 | 0.0752 |
| 0.0010 | 0.0752 | 0.0752 |

**Conclusion**: Hyperparameters have minimal effect within the tested range. All combinations achieve ~0.075.

### 6. Asymmetric Attention
| Asymmetric | N | Mean test_mse |
|------------|---|---------------|
| False | 1798 | **0.1338** |
| True | 72 | 0.3626 |

**Conclusion**: Asymmetric attention performs significantly worse (2.7x higher error).

## Top 10 Runs Overall

| Rank | test_mse | Project | Stage | Architecture | K | LR |
|------|----------|---------|-------|--------------|---|-----|
| 1 | 0.0007 | initial_widening | stage1 | spectral_linear | 50 | 0.0002 |
| 2 | 0.0056 | stage2_validation | other | other | 16 | 0.01 |
| 3 | 0.0371 | spectral_denoising | stage2 | digress_default | 32 | 0.0005 |
| 4 | 0.0371 | spectral_denoising | stage2c | gnn_v | 32 | 0.0005 |
| 5 | 0.0371 | spectral_denoising | stage2c | gnn_qk | 16 | 0.001 |

**Note**: The 0.0007 result is from early POC on sbm_small dataset, not comparable to main experiments.

## Hyperparameter Importance (Random Forest Permutation Importance)

### For test_mse:
| Feature | Permutation Importance |
|---------|------------------------|
| **stage** | **0.758** |
| **model_type** | **0.496** |
| lr | 0.136 |
| wd | 0.050 |
| arch | 0.040 |
| k | 0.013 |
| asymmetric | 0.000 |

### For test_subspace:
| Feature | Permutation Importance |
|---------|------------------------|
| **stage** | **0.911** |
| **model_type** | **0.360** |
| lr | 0.141 |
| arch | 0.072 |
| k | 0.026 |
| wd | 0.002 |
| asymmetric | 0.000 |

**Key Insight**: Stage and model_type dominate performance. Within a given stage/model, hyperparameters (lr, wd, k, arch) have relatively minor effects.

## Seed-Averaged Performance (Best Configurations)

Within Stage2c+DiGress, configurations were run with 3 seeds. The spread between configurations (0.0012) is smaller than the within-configuration variance (std ~0.028), confirming architecture choice is inconsequential.

### Best DiGress Configurations (Stage2c, seed-averaged)
| Architecture | K | LR | WD | N | MSE mean±std | MSE min |
|--------------|---|-----|------|---|--------------|---------|
| gnn_v | 16 | 0.0005 | 0.001 | 12 | 0.0745±0.028 | 0.0376 |
| gnn_all | 16 | 0.0005 | 0.001 | 12 | 0.0745±0.028 | 0.0376 |
| gnn_qk | 16 | 0.0005 | 0.001 | 12 | 0.0745±0.028 | 0.0376 |
| digress_default | 16 | 0.0005 | 0.001 | 12 | 0.0747±0.028 | 0.0376 |

### Spectral Model (best configurations)
| Architecture | Stage | N | MSE mean±std | MSE min |
|--------------|-------|---|--------------|---------|
| self_attention | stage2 | 288 | 0.1274±0.107 | 0.0406 |
| linear_pe | stage2 | 144 | 0.1335±0.120 | 0.0392 |

### By Noise Level (single-noise training)
| Noise Level | N | MSE mean±std | MSE min |
|-------------|---|--------------|---------|
| [0.1] | 104 | 0.1189±0.064 | 0.0272 |
| [0.2] | 94 | 0.1234±0.059 | 0.0388 |
| [0.01] | 99 | 0.1281±0.067 | 0.0007 |
| [0.01,...,0.3] | 1512 | 0.1452±0.118 | 0.0371 |

## Recommendations

1. **Use stage2c** with any DiGress architecture variant
2. **k=16 or k=32** - both achieve ~0.075 MSE in Stage2c (k=16 slightly faster)
3. **Hyperparameters**: lr=5e-4, wd=1e-3 marginally best, but all tested values work
4. **Avoid asymmetric attention** - significantly degrades performance
5. **Architecture choice doesn't matter** - gnn_v, gnn_all, gnn_qk, default all equivalent
6. **Noise level [0.1]** achieves best single-noise performance, but mixed training is robust

## Scripts

- `scripts/analyze_experiments.py` - Download and analyze all W&B data
  ```bash
  uv run scripts/analyze_experiments.py                    # Full download + analysis
  uv run scripts/analyze_experiments.py --skip-download    # Use cached data
  uv run scripts/analyze_experiments.py --importance-only  # Only importance analysis
  ```

- Data files:
  - `eigenstructure_results_full/all_runs.parquet` - All run data (2013 runs, 344 columns)
  - `eigenstructure_results_full/importance.csv` - Hyperparameter importance scores
  - `eigenstructure_results_full/seed_averaged_summary.csv` - Seed-averaged performance by configuration (230 unique configs)
