# Exact 1:1 Mathematical Replication Guide

This directory contains scripts and configurations for exact mathematical replication of the original experiments from `denoising/` scripts and `new_denoiser.ipynb`.

## Critical Fixes Applied

### 1. Attention Model d_k/d_v Mismatch (CRITICAL)
- **Original Issue**: Current config used `d_k=d_v=2.5` (d_model//num_heads)
- **Fix**: Created `denoising-script-match.yaml` with `d_k=d_v=20` (matches original k=20)
- **Impact**: This completely changes model capacity and mathematical behavior

### 2. Data Generation Inconsistency  
- **Original Issue**: Mixed fixed vs random block sizes
- **Fix**: 
  - `denoising-script-match.yaml`: Fixed `[10,5,3,2]` (matches train_attention.sh, train_gnn.sh)
  - `notebook-match.yaml`: Random partitions (matches new_denoiser.ipynb)
- **Impact**: Ensures exact same graph structures as original experiments

### 3. Training Parameter Alignment
- **Fix**: All hyperparameters exactly match originals (epochs, batch sizes, learning rates, etc.)

## Replication Scripts

### Denoising Scripts Replication
```bash
# Exact replication of train_attention.sh and train_gnn.sh
./scripts/replicate_denoising_scripts_exact.sh

# Quick validation (sanity check only)
./scripts/replicate_denoising_scripts_exact.sh -c
```

**Replicates**:
- `denoising/train_attention.sh`: MultiLayerAttention with 3 noise types
- `denoising/train_gnn.sh`: GNN with 3 noise types

**Key Parameters**:
- Fixed block sizes: `[10,5,3,2]`
- Attention: `d_k=20, d_v=20, num_heads=8, num_layers=8`
- GNN: `num_layers=1, num_terms=4, feature_dim=20`
- Training: 1000 epochs, 128 samples/epoch, batch_size=32
- Loss: MSE, Adam(lr=0.001)

### Notebook Replication
```bash
# Exact replication of new_denoiser.ipynb
./scripts/replicate_notebook_exact.sh

# Quick validation (sanity check only)  
./scripts/replicate_notebook_exact.sh -c
```

**Replicates**:
- `new_denoiser.ipynb`: Hybrid GNN+Transformer model

**Key Parameters**:
- Random partitions: `generate_block_sizes(20, min_blocks=2, max_blocks=4)`
- GNN: `layers=2, terms=2, dim_in=20, dim_out=5`
- Transformer: `layers=4, heads=4, d_k=10, d_v=10`
- Training: 200 epochs, 1000 samples, batch_size=100
- Loss: BCE, Adam(lr=0.005), CosineAnnealingWarmRestarts

## Validation

### Quick Validation
```bash
# Comprehensive validation of all configurations
./scripts/validate_replication.sh
```

This checks:
- ✅ All model configurations load successfully
- ✅ Critical parameters match originals
- ✅ Sanity checks pass for all experiments
- ✅ Data generation works correctly

### Manual Verification Checklist

#### Attention Model (denoising scripts)
- [ ] `d_k = 20` (NOT 2.5)
- [ ] `d_v = 20` (NOT 2.5) 
- [ ] `num_heads = 8`
- [ ] `num_layers = 8`
- [ ] Fixed block sizes `[10,5,3,2]`
- [ ] 1000 epochs, MSE loss, Adam(lr=0.001)

#### GNN Model (denoising scripts)
- [ ] `num_layers = 1`
- [ ] `num_terms = 4` (original t=4)
- [ ] `feature_dim = 20` (original k=20)
- [ ] Fixed block sizes `[10,5,3,2]`
- [ ] 1000 epochs, MSE loss, Adam(lr=0.001)

#### Hybrid Model (notebook)
- [ ] GNN: `layers=2, terms=2, dim_in=20, dim_out=5`
- [ ] Transformer: `layers=4, heads=4, d_k=10, d_v=10`
- [ ] Random partitions (not fixed)
- [ ] 200 epochs, BCE loss, Adam(lr=0.005)
- [ ] CosineAnnealingWarmRestarts scheduler

## Configuration Files

### Model Configurations
- `attention_denoising/config/model/denoising-script-match.yaml`
- `gnn_denoising/config/model/denoising-script-match.yaml`  
- `hybrid_denoising/config/model/notebook-match.yaml`

### Data Configurations
- `*/config/data/denoising-script-match.yaml` (fixed block sizes)
- `hybrid_denoising/config/data/notebook-match.yaml` (random partitions)

### Experiment Configurations
- `*/config/experiment/denoising-script-match.yaml`
- `hybrid_denoising/config/experiment/notebook-match.yaml`

## Usage Examples

### Test Single Configuration
```bash
# Test attention model exact match
tmgg-attention --config-name=denoising-script-match sanity_check=true

# Test with specific noise type
tmgg-attention --config-name=denoising-script-match \
    data.noise_type=gaussian data.noise_levels="[0.3]" \
    sanity_check=true
```

### Run Full Experiment
```bash
# Run attention experiment exactly matching train_attention.sh
tmgg-attention --config-name=denoising-script-match \
    data.noise_type=digress data.noise_levels="[0.3]" \
    experiment_name="attention_digress_exact"
```

## Mathematical Equivalence Verification

The configurations ensure:

1. **Identical Model Architectures**: Every parameter matches the original implementations
2. **Identical Data Generation**: Same block sizes, partitions, and noise procedures  
3. **Identical Training**: Same optimizers, learning rates, schedulers, and loss functions
4. **Identical Random Seeds**: Reproducible results with seed=42

Any numerical differences should be due to:
- PyTorch version differences
- Hardware floating-point precision
- CUDA vs CPU execution

But the mathematical operations should be identical.