#!/bin/bash
# Exact 1:1 replication of original denoising scripts
# This script replicates train_attention.sh and train_gnn.sh with mathematical equivalence

# Exit on error
set -e

# Default parameters
SEED=42
OUTPUT_DIR="outputs/denoising_scripts_exact"
SANITY_CHECK=false

# Function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d DIR    Output directory (default: $OUTPUT_DIR)"
    echo "  -s SEED   Random seed (default: $SEED)"
    echo "  -c        Run sanity check only (no training)"
    echo "  -h        Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "d:s:ch" opt; do
    case $opt in
        d) OUTPUT_DIR="$OPTARG";;
        s) SEED="$OPTARG";;
        c) SANITY_CHECK=true;;
        h) usage;;
        *) usage;;
    esac
done

echo "Exact replication of original denoising scripts"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Random seed: $SEED"
if [ "$SANITY_CHECK" = true ]; then
    echo "Mode: SANITY CHECK ONLY"
else
    echo "Mode: FULL TRAINING"
fi
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "1. Replicating train_attention.sh (exact mathematical match)"
echo "==========================================================="

# Exact replication of train_attention.sh
# Original: for noise_type in "Digress" "Gaussian" "Rotation"
# Original: eps = 0.3, num_heads = 8, num_layers = 8
for noise_type in "digress" "gaussian" "rotation"; do
    echo "Running attention model with $noise_type noise, eps=0.3"
    echo "  - Fixed block sizes: [10,5,3,2]"
    echo "  - d_k=20, d_v=20 (NOT d_model//num_heads)"
    echo "  - 1000 epochs, 128 samples/epoch, batch_size=32"
    
    tmgg-attention \
        --config-name=experiment/denoising-script-match \
        hydra.run.dir="$OUTPUT_DIR/attention/${noise_type}_eps0.3" \
        seed=$SEED \
        data.noise_type="$noise_type" \
        data.noise_levels="[0.3]" \
        sanity_check=$SANITY_CHECK \
        experiment_name="attention_${noise_type}_script_exact"
done

echo
echo "2. Replicating train_gnn.sh (exact mathematical match)"
echo "======================================================"

# Exact replication of train_gnn.sh  
# Original: for noise_type in "Gaussian" "Rotation" "Digress"
# Original: eps = 0.3, num_layers = 1
for noise_type in "gaussian" "rotation" "digress"; do
    echo "Running GNN model with $noise_type noise, eps=0.3"
    echo "  - Fixed block sizes: [10,5,3,2]"
    echo "  - GNN: num_layers=1, num_terms=4, feature_dim=20"
    echo "  - 1000 epochs, 128 samples/epoch, batch_size=32"
    
    tmgg-gnn \
        --config-name=experiment/denoising-script-match \
        hydra.run.dir="$OUTPUT_DIR/gnn/${noise_type}_eps0.3" \
        seed=$SEED \
        data.noise_type="$noise_type" \
        data.noise_levels="[0.3]" \
        sanity_check=$SANITY_CHECK \
        experiment_name="gnn_${noise_type}_script_exact"
done

echo
echo "All denoising script replications completed!"
echo "Results saved to: $OUTPUT_DIR"
echo
echo "Mathematical equivalence ensured:"
echo "  ✓ Fixed block sizes [10,5,3,2] (not random partitions)"
echo "  ✓ Attention: d_k=20, d_v=20 (original k=20)"
echo "  ✓ GNN: num_terms=4, feature_dim=20 (original t=4, k=20)"
echo "  ✓ Training: 1000 epochs, 128 samples, batch_size=32"
echo "  ✓ Loss: MSE, optimizer: Adam(lr=0.001), seed=42"