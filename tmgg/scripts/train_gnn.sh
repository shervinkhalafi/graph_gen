#!/bin/bash
# Script to run GNN-based denoising experiments
#
# Usage modes:
#   1. Single experiment: ./train_gnn.sh [options]
#   2. Component batch:   ./train_gnn.sh --component <model-noise> (e.g., standard-gaussian)
#   3. Full replication:  ./train_gnn.sh --replicate

# Default values
MODEL_TYPE="standard_gnn"  # Can be standard_gnn, symmetric_gnn, or nodevar_gnn
NOISE_TYPE="digress"
NOISE_LEVEL=0.3
NUM_LAYERS=1
NUM_EPOCHS=400
EXPERIMENT_NAME="gnn_denoising"
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
        --model-type)
            MODEL_TYPE="$2"
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
        --num-layers)
            NUM_LAYERS="$2"
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
    local model_type=$1
    local noise_type=$2
    local noise_level=$3
    local exp_name="${4:-gnn_${model_type}_${noise_type}_eps${noise_level}}"
    
    echo "================================================"
    echo "Running GNN experiment:"
    echo "  Model: $model_type"
    echo "  Noise type: $noise_type"
    echo "  Noise level: $noise_level"
    echo "  Layers: $NUM_LAYERS"
    echo "================================================"
    
    tmgg-gnn \
        experiment.name="$exp_name" \
        model="$model_type" \
        model.num_layers=$NUM_LAYERS \
        trainer.max_epochs=$NUM_EPOCHS \
        data.noise_type="$noise_type" \
        evaluation.noise_levels="[$noise_level]" \
        ++tags="[model_${model_type},noise_${noise_type},eps_${noise_level},layers_${NUM_LAYERS}]"
}

# Component experiments for each model-noise combination
run_standard_gaussian() {
    echo "Running standard GNN with Gaussian noise..."
    local noise_levels=(0.05 0.1 0.2 0.3 0.4 0.5)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "standard_gnn" "gaussian" "$eps"
    done
}

run_standard_rotation() {
    echo "Running standard GNN with Rotation noise..."
    local noise_levels=(0.05 0.1 0.15 0.2 0.25 0.3)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "standard_gnn" "rotation" "$eps"
    done
}

run_standard_digress() {
    echo "Running standard GNN with Digress noise..."
    local noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "standard_gnn" "digress" "$eps"
    done
}

run_symmetric_gaussian() {
    echo "Running symmetric GNN with Gaussian noise..."
    local noise_levels=(0.05 0.1 0.2 0.3 0.4 0.5)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "symmetric_gnn" "gaussian" "$eps"
    done
}

run_symmetric_rotation() {
    echo "Running symmetric GNN with Rotation noise..."
    local noise_levels=(0.05 0.1 0.15 0.2 0.25 0.3)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "symmetric_gnn" "rotation" "$eps"
    done
}

run_symmetric_digress() {
    echo "Running symmetric GNN with Digress noise..."
    local noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "symmetric_gnn" "digress" "$eps"
    done
}

run_nodevar_gaussian() {
    echo "Running NodeVar GNN with Gaussian noise..."
    local noise_levels=(0.05 0.1 0.2 0.3 0.4 0.5)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "nodevar_gnn" "gaussian" "$eps"
    done
}

run_nodevar_rotation() {
    echo "Running NodeVar GNN with Rotation noise..."
    local noise_levels=(0.05 0.1 0.15 0.2 0.25 0.3)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "nodevar_gnn" "rotation" "$eps"
    done
}

run_nodevar_digress() {
    echo "Running NodeVar GNN with Digress noise..."
    local noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "nodevar_gnn" "digress" "$eps"
    done
}

# Full replication of all experiments
run_replication() {
    echo "Running full GNN replication experiments..."
    
    # Standard GNN
    run_standard_gaussian
    run_standard_rotation
    run_standard_digress
    
    # Symmetric GNN
    run_symmetric_gaussian
    run_symmetric_rotation
    run_symmetric_digress
    
    # NodeVar GNN
    run_nodevar_gaussian
    run_nodevar_rotation
    run_nodevar_digress
}

# Main execution logic
if [ "$REPLICATE" = true ]; then
    run_replication
elif [ -n "$COMPONENT" ]; then
    case $COMPONENT in
        standard-gaussian)
            run_standard_gaussian
            ;;
        standard-rotation)
            run_standard_rotation
            ;;
        standard-digress)
            run_standard_digress
            ;;
        symmetric-gaussian)
            run_symmetric_gaussian
            ;;
        symmetric-rotation)
            run_symmetric_rotation
            ;;
        symmetric-digress)
            run_symmetric_digress
            ;;
        nodevar-gaussian)
            run_nodevar_gaussian
            ;;
        nodevar-rotation)
            run_nodevar_rotation
            ;;
        nodevar-digress)
            run_nodevar_digress
            ;;
        all-standard)
            run_standard_gaussian
            run_standard_rotation
            run_standard_digress
            ;;
        all-symmetric)
            run_symmetric_gaussian
            run_symmetric_rotation
            run_symmetric_digress
            ;;
        all-nodevar)
            run_nodevar_gaussian
            run_nodevar_rotation
            run_nodevar_digress
            ;;
        all)
            run_replication
            ;;
        *)
            echo "Unknown component: $COMPONENT"
            echo "Valid components:"
            echo "  Single: standard-gaussian, standard-rotation, standard-digress"
            echo "         symmetric-gaussian, symmetric-rotation, symmetric-digress"
            echo "         nodevar-gaussian, nodevar-rotation, nodevar-digress"
            echo "  Groups: all-standard, all-symmetric, all-nodevar, all"
            exit 1
            ;;
    esac
else
    # Run single experiment with provided parameters
    run_single_experiment "$MODEL_TYPE" "$NOISE_TYPE" "$NOISE_LEVEL" "$EXPERIMENT_NAME"
fi