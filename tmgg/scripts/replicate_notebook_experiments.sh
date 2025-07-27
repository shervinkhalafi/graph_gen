#!/bin/bash
# Script to replicate the exact experiments from denoising notebooks

# Exit on error
set -e

# Default configuration
SEED=42
OUTPUT_DIR="outputs/notebook_replication"
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

echo "Replicating experiments from denoising notebooks"
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

# 1. Replicate train_attention.sh experiments
echo "1. Replicating attention experiments (train_attention.sh)"
echo "---------------------------------------------------------"

# Adjust parameters for sanity check mode
if [ "$SANITY_CHECK" = true ]; then
    max_epochs=1
    num_samples=32
    batch_size=16
    echo "(Sanity check mode - reduced epochs and data)"
else
    max_epochs=1000
    num_samples=128
    batch_size=32
fi

# Single configuration as in train_attention.sh
for noise_type in "digress" "gaussian" "rotation"; do
    echo "Running attention model with $noise_type noise, eps=0.3"
    
    tmgg-attention \
        hydra.run.dir="$OUTPUT_DIR/attention/${noise_type}_eps0.3" \
        seed=$SEED \
        data.noise_type="$noise_type" \
        data.noise_levels="[0.3]" \
        data.dataset_config.block_sizes="[10,5,3,2]" \
        data.num_samples_per_graph=$num_samples \
        data.batch_size=$batch_size \
        model.num_heads=8 \
        model.num_layers=8 \
        trainer.max_epochs=$max_epochs \
        sanity_check=$SANITY_CHECK \
        experiment_name="attention_${noise_type}_0.3"
done

echo
echo "2. Replicating GNN experiments (train_gnn.sh)"
echo "---------------------------------------------"

# Use same parameters as attention section
if [ "$SANITY_CHECK" = true ]; then
    echo "(Sanity check mode - reduced epochs and data)"
fi

# Single configuration as in train_gnn.sh
for noise_type in "gaussian" "rotation" "digress"; do
    echo "Running GNN model with $noise_type noise, eps=0.3"
    
    tmgg-gnn \
        hydra.run.dir="$OUTPUT_DIR/gnn/${noise_type}_eps0.3" \
        seed=$SEED \
        data.noise_type="$noise_type" \
        data.noise_levels="[0.3]" \
        data.dataset_config.block_sizes="[10,5,3,2]" \
        data.num_samples_per_graph=$num_samples \
        data.batch_size=$batch_size \
        model=standard_gnn \
        model.num_layers=1 \
        trainer.max_epochs=$max_epochs \
        sanity_check=$SANITY_CHECK \
        experiment_name="gnn_${noise_type}_0.3"
done

echo
echo "3. Replicating hybrid experiment (new_denoiser.ipynb)"
echo "----------------------------------------------------"

# Adjust parameters for sanity check mode
if [ "$SANITY_CHECK" = true ]; then
    hybrid_epochs=1
    hybrid_samples=100
    hybrid_batch_size=50
    noise_levels="[0.1,0.3]"
    echo "(Sanity check mode - reduced epochs, data, and noise levels)"
else
    hybrid_epochs=200
    hybrid_samples=1000
    hybrid_batch_size=100
    noise_levels="[0.005,0.02,0.05,0.1,0.25,0.4,0.5]"
fi

# Exact configuration from new_denoiser.ipynb
echo "Running hybrid model with digress noise at multiple levels"

tmgg-hybrid \
    hydra.run.dir="$OUTPUT_DIR/hybrid/digress_multi" \
    seed=$SEED \
    data.noise_type="digress" \
    data.noise_levels="$noise_levels" \
    data.dataset_config.num_nodes=20 \
    data.dataset_config.num_train_partitions=10 \
    data.dataset_config.num_test_partitions=10 \
    data.dataset_config.p_intra=1.0 \
    data.dataset_config.q_inter=0.0 \
    data.num_samples_per_graph=$hybrid_samples \
    data.batch_size=$hybrid_batch_size \
    model=hybrid_with_transformer \
    model.gnn_num_layers=2 \
    model.gnn_num_terms=2 \
    model.gnn_feature_dim_in=20 \
    model.gnn_feature_dim_out=5 \
    model.transformer_num_layers=4 \
    model.transformer_num_heads=4 \
    model.transformer_d_k=10 \
    model.transformer_d_v=10 \
    model.learning_rate=0.005 \
    model.loss_type="BCE" \
    trainer.max_epochs=$hybrid_epochs \
    sanity_check=$SANITY_CHECK \
    experiment_name="hybrid_digress_multi" \
    wandb.project="graph-denoising" \
    wandb.name="training_run"

echo
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo
echo "To visualize results:"
echo "  - Check the $OUTPUT_DIR directory for saved plots"
echo "  - View metrics on Weights & Biases (if configured)"
echo "  - Use tensorboard: tensorboard --logdir=$OUTPUT_DIR"