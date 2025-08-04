#!/bin/bash
# Script to run hybrid GNN+Transformer denoising experiments
#
# Usage modes:
#   1. Single experiment: ./train_hybrid.sh [options]
#   2. Component batch:   ./train_hybrid.sh --component <config-noise> (e.g., transformer-gaussian)
#   3. Full replication:  ./train_hybrid.sh --replicate

# Default values
USE_TRANSFORMER=true
GNN_LAYERS=2
TRANSFORMER_LAYERS=4
TRANSFORMER_HEADS=4
NOISE_TYPE="digress"
NOISE_LEVEL=0.3
NUM_EPOCHS=200
EXPERIMENT_NAME="hybrid_denoising"
COMPONENT=""
REPLICATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --component)
            COMPONENT="$2"
            shift 2
            ;;
        --replicate)
            REPLICATE=true
            shift
            ;;
        --no-transformer)
            USE_TRANSFORMER=false
            shift
            ;;
        --gnn-layers)
            GNN_LAYERS="$2"
            shift 2
            ;;
        --transformer-layers)
            TRANSFORMER_LAYERS="$2"
            shift 2
            ;;
        --transformer-heads)
            TRANSFORMER_HEADS="$2"
            shift 2
            ;;
        --noise-type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        --noise-level)
            NOISE_LEVEL="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to run a single experiment
run_single_experiment() {
    local config_type=$1
    local noise_type=$2
    local noise_level=$3
    local exp_name="${4:-hybrid_${config_type}_${noise_type}_eps${noise_level}}"
    
    echo "================================================"
    echo "Running hybrid experiment:"
    echo "  Config: $config_type"
    echo "  Noise type: $noise_type"
    echo "  Noise level: $noise_level"
    
    if [ "$config_type" = "hybrid_with_transformer" ]; then
        echo "  GNN layers: $GNN_LAYERS"
        echo "  Transformer layers: $TRANSFORMER_LAYERS, heads: $TRANSFORMER_HEADS"
    else
        echo "  GNN layers: $GNN_LAYERS (GNN only mode)"
    fi
    echo "================================================"
    
    tmgg-hybrid \
        experiment.name="$exp_name" \
        model="$config_type" \
        model.gnn_num_layers=$GNN_LAYERS \
        model.transformer_num_layers=$TRANSFORMER_LAYERS \
        model.transformer_num_heads=$TRANSFORMER_HEADS \
        trainer.max_epochs=$NUM_EPOCHS \
        data.noise_type="$noise_type" \
        evaluation.noise_levels="[$noise_level]" \
        ++tags="[model_${config_type},noise_${noise_type},eps_${noise_level},gnn_${GNN_LAYERS}layers,trans_${TRANSFORMER_LAYERS}layers]"
}

# Component experiments for each configuration-noise combination
run_transformer_gaussian() {
    echo "Running hybrid with transformer - Gaussian noise..."
    local noise_levels=(0.05 0.1 0.2 0.3 0.4 0.5)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "hybrid_with_transformer" "gaussian" "$eps"
    done
}

run_transformer_rotation() {
    echo "Running hybrid with transformer - Rotation noise..."
    local noise_levels=(0.05 0.1 0.15 0.2 0.25 0.3)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "hybrid_with_transformer" "rotation" "$eps"
    done
}

run_transformer_digress() {
    echo "Running hybrid with transformer - Digress noise..."
    local noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "hybrid_with_transformer" "digress" "$eps"
    done
}

run_gnn_only_gaussian() {
    echo "Running hybrid GNN only - Gaussian noise..."
    local noise_levels=(0.05 0.1 0.2 0.3 0.4 0.5)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "hybrid_gnn_only" "gaussian" "$eps"
    done
}

run_gnn_only_rotation() {
    echo "Running hybrid GNN only - Rotation noise..."
    local noise_levels=(0.05 0.1 0.15 0.2 0.25 0.3)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "hybrid_gnn_only" "rotation" "$eps"
    done
}

run_gnn_only_digress() {
    echo "Running hybrid GNN only - Digress noise..."
    local noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "hybrid_gnn_only" "digress" "$eps"
    done
}

# Variation experiments with different layer configurations
run_transformer_variations() {
    echo "Running transformer configuration variations..."
    local noise_type="digress"
    local noise_level=0.3
    
    # Different GNN layer counts
    for gnn_l in 1 2 3 4; do
        GNN_LAYERS=$gnn_l
        run_single_experiment "hybrid_with_transformer" "$noise_type" "$noise_level" \
            "hybrid_transformer_gnn${gnn_l}_trans${TRANSFORMER_LAYERS}_${noise_type}_eps${noise_level}"
    done
    GNN_LAYERS=2  # Reset to default
    
    # Different transformer layer counts
    for trans_l in 2 4 6 8; do
        TRANSFORMER_LAYERS=$trans_l
        run_single_experiment "hybrid_with_transformer" "$noise_type" "$noise_level" \
            "hybrid_transformer_gnn${GNN_LAYERS}_trans${trans_l}_${noise_type}_eps${noise_level}"
    done
    TRANSFORMER_LAYERS=4  # Reset to default
}

# Full replication of all experiments
run_replication() {
    echo "Running full hybrid replication experiments..."
    
    # Hybrid with transformer
    run_transformer_gaussian
    run_transformer_rotation
    run_transformer_digress
    
    # Hybrid GNN only
    run_gnn_only_gaussian
    run_gnn_only_rotation
    run_gnn_only_digress
    
    # Configuration variations
    run_transformer_variations
}

# Main execution logic
if [ "$REPLICATE" = true ]; then
    run_replication
elif [ -n "$COMPONENT" ]; then
    case $COMPONENT in
        transformer-gaussian)
            run_transformer_gaussian
            ;;
        transformer-rotation)
            run_transformer_rotation
            ;;
        transformer-digress)
            run_transformer_digress
            ;;
        gnn-only-gaussian)
            run_gnn_only_gaussian
            ;;
        gnn-only-rotation)
            run_gnn_only_rotation
            ;;
        gnn-only-digress)
            run_gnn_only_digress
            ;;
        all-transformer)
            run_transformer_gaussian
            run_transformer_rotation
            run_transformer_digress
            ;;
        all-gnn-only)
            run_gnn_only_gaussian
            run_gnn_only_rotation
            run_gnn_only_digress
            ;;
        variations)
            run_transformer_variations
            ;;
        all)
            run_replication
            ;;
        *)
            echo "Unknown component: $COMPONENT"
            echo "Valid components:"
            echo "  Single: transformer-gaussian, transformer-rotation, transformer-digress"
            echo "         gnn-only-gaussian, gnn-only-rotation, gnn-only-digress"
            echo "  Groups: all-transformer, all-gnn-only, variations, all"
            exit 1
            ;;
    esac
else
    # Run single experiment with provided parameters
    if [ "$USE_TRANSFORMER" = true ]; then
        MODEL_CONFIG="hybrid_with_transformer"
    else
        MODEL_CONFIG="hybrid_gnn_only"
    fi
    run_single_experiment "$MODEL_CONFIG" "$NOISE_TYPE" "$NOISE_LEVEL" "$EXPERIMENT_NAME"
fi