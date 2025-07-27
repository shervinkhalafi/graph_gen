#!/bin/bash
# Script to replicate graph denoising experiments from notebooks using tmgg

# Exit on error
set -e

# Default parameters
EXPERIMENT_DIR="outputs/denoising_experiments"
SEED=42
SANITY_CHECK=false

# Function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d DIR    Output directory (default: $EXPERIMENT_DIR)"
    echo "  -s SEED   Random seed (default: $SEED)"
    echo "  -c        Run sanity check only (no training)"
    echo "  -h        Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "d:s:ch" opt; do
    case $opt in
        d) EXPERIMENT_DIR="$OPTARG";;
        s) SEED="$OPTARG";;
        c) SANITY_CHECK=true;;
        h) usage;;
        *) usage;;
    esac
done

echo "Running graph denoising experiments"
echo "Output directory: $EXPERIMENT_DIR"
echo "Random seed: $SEED"
if [ "$SANITY_CHECK" = true ]; then
    echo "Mode: SANITY CHECK ONLY"
else
    echo "Mode: FULL TRAINING"
fi
echo

# Create output directory
mkdir -p "$EXPERIMENT_DIR"

# Function to run attention experiments
run_attention_experiments() {
    echo "=== Running Attention-based Denoising Experiments ==="
    
    # Fixed parameters from notebooks
    local num_heads=8
    local num_layers=8
    local d_model=20
    local max_epochs=1000
    
    # Adjust for sanity check mode
    if [ "$SANITY_CHECK" = true ]; then
        max_epochs=1
        echo "(Sanity check mode - epochs set to 1)"
    fi
    
    for noise_type in "digress" "gaussian" "rotation"; do
        for eps in 0.1 0.3 0.5; do
            echo "Running Attention model with $noise_type noise at level $eps"
            
            output_dir="$EXPERIMENT_DIR/attention/${noise_type}_eps${eps}"
            
            tmgg-attention \
                hydra.run.dir="$output_dir" \
                seed=$SEED \
                data.noise_type="$noise_type" \
                data.noise_levels="[$eps]" \
                model.num_heads=$num_heads \
                model.num_layers=$num_layers \
                model.d_model=$d_model \
                trainer.max_epochs=$max_epochs \
                sanity_check=$SANITY_CHECK \
                experiment_name="attention_${noise_type}_${eps}" \
                wandb.project="tmgg-replication" \
                wandb.name="attention_${noise_type}_eps${eps}"
        done
    done
}

# Function to run GNN experiments
run_gnn_experiments() {
    echo "=== Running GNN-based Denoising Experiments ==="
    
    # Fixed parameters from notebooks
    local num_layers=1
    local num_terms=4
    local feature_dim=20
    local max_epochs=1000
    
    # Adjust for sanity check mode
    if [ "$SANITY_CHECK" = true ]; then
        max_epochs=1
        echo "(Sanity check mode - epochs set to 1)"
    fi
    
    for noise_type in "digress" "gaussian" "rotation"; do
        for eps in 0.1 0.3 0.5; do
            echo "Running GNN model with $noise_type noise at level $eps"
            
            output_dir="$EXPERIMENT_DIR/gnn/${noise_type}_eps${eps}"
            
            tmgg-gnn \
                hydra.run.dir="$output_dir" \
                seed=$SEED \
                data.noise_type="$noise_type" \
                data.noise_levels="[$eps]" \
                model=nodevar_gnn \
                model.num_layers=$num_layers \
                model.num_terms=$num_terms \
                model.feature_dim_out=$feature_dim \
                trainer.max_epochs=$max_epochs \
                sanity_check=$SANITY_CHECK \
                experiment_name="gnn_${noise_type}_${eps}" \
                wandb.project="tmgg-replication" \
                wandb.name="gnn_${noise_type}_eps${eps}"
        done
    done
}

# Function to run hybrid experiments (matching notebook setup)
run_hybrid_experiments() {
    echo "=== Running Hybrid GNN+Transformer Denoising Experiments ==="
    
    # Fixed parameters from notebooks
    local gnn_num_layers=2
    local gnn_num_terms=2
    local gnn_feature_dim_in=20
    local gnn_feature_dim_out=5
    local transformer_num_layers=4
    local transformer_num_heads=4
    local learning_rate=0.005
    local max_epochs=200
    
    # Adjust for sanity check mode
    if [ "$SANITY_CHECK" = true ]; then
        max_epochs=1
        noise_levels="[0.1,0.3]"
        echo "(Sanity check mode - epochs set to 1, reduced noise levels)"
    else
        # Multiple noise levels as in notebook
        noise_levels="[0.005,0.02,0.05,0.1,0.25,0.4,0.5]"
    fi
    
    echo "Running Hybrid model with digress noise at multiple levels"
    
    output_dir="$EXPERIMENT_DIR/hybrid/multi_noise"
    
    tmgg-hybrid \
        hydra.run.dir="$output_dir" \
        seed=$SEED \
        data.noise_type="digress" \
        data.noise_levels="$noise_levels" \
        model=hybrid_with_transformer \
        model.gnn_num_layers=$gnn_num_layers \
        model.gnn_num_terms=$gnn_num_terms \
        model.gnn_feature_dim_in=$gnn_feature_dim_in \
        model.gnn_feature_dim_out=$gnn_feature_dim_out \
        model.transformer_num_layers=$transformer_num_layers \
        model.transformer_num_heads=$transformer_num_heads \
        model.learning_rate=$learning_rate \
        trainer.max_epochs=$max_epochs \
        sanity_check=$SANITY_CHECK \
        experiment_name="hybrid_multi_noise" \
        wandb.project="tmgg-replication" \
        wandb.name="hybrid_digress_multi"
}

# Function to run a specific experiment type
run_experiment() {
    case "$1" in
        attention)
            run_attention_experiments
            ;;
        gnn)
            run_gnn_experiments
            ;;
        hybrid)
            run_hybrid_experiments
            ;;
        all)
            run_attention_experiments
            echo
            run_gnn_experiments
            echo
            run_hybrid_experiments
            ;;
        *)
            echo "Unknown experiment type: $1"
            echo "Valid types: attention, gnn, hybrid, all"
            exit 1
            ;;
    esac
}

# Check if experiment type was provided
if [ $# -lt 1 ]; then
    echo "Running all experiments..."
    run_experiment all
else
    # Get the last argument as experiment type
    experiment_type="${@: -1}"
    run_experiment "$experiment_type"
fi

echo
echo "All experiments completed!"
echo "Results saved to: $EXPERIMENT_DIR"