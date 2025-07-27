#!/bin/bash
# Script to run sanity checks for all experiment types

# Exit on error
set -e

echo "Running Sanity Checks for Graph Denoising Experiments"
echo "====================================================="
echo

# Configuration
SANITY_CHECK_DIR="outputs/sanity_checks"
SEED=42

# Function to run sanity check for a specific experiment
run_sanity_check() {
    local experiment_type=$1
    local model_config=$2
    local noise_type=$3
    local command=$4
    
    echo "Running sanity check: $experiment_type with $model_config and $noise_type noise"
    echo "--------------------------------------------------------------------------------"
    
    output_dir="$SANITY_CHECK_DIR/${experiment_type}/${model_config}_${noise_type}"
    
    # Run the command with sanity_check=true
    $command \
        hydra.run.dir="$output_dir" \
        seed=$SEED \
        data.noise_type="$noise_type" \
        data.noise_levels="[0.1,0.3]" \
        data.num_samples_per_graph=50 \
        data.batch_size=16 \
        model=$model_config \
        sanity_check=true \
        trainer.max_epochs=1 \
        wandb=null
    
    echo "✓ Sanity check completed for $experiment_type/$model_config/$noise_type"
    echo
}

# Create output directory
mkdir -p "$SANITY_CHECK_DIR"

echo "1. Testing Attention-based models"
echo "================================="
for noise_type in "gaussian" "digress" "rotation"; do
    run_sanity_check "attention" "multi_layer_attention" "$noise_type" "tmgg-attention"
done

echo
echo "2. Testing GNN-based models"
echo "==========================="
for model in "standard_gnn" "nodevar_gnn" "symmetric_gnn"; do
    for noise_type in "gaussian" "digress" "rotation"; do
        run_sanity_check "gnn" "$model" "$noise_type" "tmgg-gnn"
    done
done

echo
echo "3. Testing Hybrid models"
echo "========================"
for model in "hybrid_with_transformer" "hybrid_gnn_only"; do
    for noise_type in "gaussian" "digress" "rotation"; do
        run_sanity_check "hybrid" "$model" "$noise_type" "tmgg-hybrid"
    done
done

echo
echo "All sanity checks completed!"
echo "============================"
echo
echo "Check the results in: $SANITY_CHECK_DIR"
echo "Each experiment directory contains:"
echo "  - config.yaml: Full configuration used"
echo "  - sanity_check_plots/: Diagnostic visualizations"
echo "  - Console output with detailed check results"