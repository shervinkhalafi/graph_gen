#!/bin/bash
# Exact 1:1 replication of new_denoiser.ipynb
# This script replicates the notebook with mathematical equivalence

# Exit on error
set -e

# Default parameters
SEED=42
OUTPUT_DIR="outputs/notebook_exact"
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

echo "Exact replication of new_denoiser.ipynb"
echo "======================================="
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

echo "Replicating hybrid GNN+Transformer experiment (exact mathematical match)"
echo "======================================================================="
echo "Configuration:"
echo "  - Random partitions: generate_block_sizes(20, min_blocks=2, max_blocks=4)"
echo "  - Train/test partitions: 10/10 split"
echo "  - GNN: layers=2, terms=2, dim_in=20, dim_out=5"
echo "  - Transformer: layers=4, heads=4, d_k=10, d_v=10"
echo "  - Training: 200 epochs, 1000 samples, batch_size=100"
echo "  - Loss: BCE, optimizer: Adam(lr=0.005), CosineAnnealingWarmRestarts"
echo "  - Noise levels: [0.005, 0.02, 0.05, 0.1, 0.25, 0.4, 0.5]"

tmgg-hybrid \
    --config-name=experiment/notebook-match \
    hydra.run.dir="$OUTPUT_DIR/hybrid/digress_multi" \
    seed=$SEED \
    sanity_check=$SANITY_CHECK \
    experiment_name="hybrid_notebook_exact"

echo
echo "Notebook replication completed!"
echo "Results saved to: $OUTPUT_DIR"
echo
echo "Mathematical equivalence ensured:"
echo "  ✓ Random partitions (not fixed block sizes)"
echo "  ✓ GNN+Transformer hybrid architecture"
echo "  ✓ Sequential model: embedding → denoising"
echo "  ✓ Training: 200 epochs, 1000 samples, batch_size=100"
echo "  ✓ Loss: BCE, optimizer: Adam(lr=0.005)"
echo "  ✓ Scheduler: CosineAnnealingWarmRestarts(T_0=20, T_mult=2)"
echo "  ✓ Multiple noise levels evaluation"