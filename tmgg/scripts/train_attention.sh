#!/bin/bash
# Script to run attention-based denoising experiments
#
# Usage modes:
#   1. Single experiment: ./train_attention.sh [options]
#   2. Component batch:   ./train_attention.sh --component <gaussian|rotation|digress|all>
#   3. Full replication:  ./train_attention.sh --replicate

# Default values
NOISE_TYPE="digress"
NOISE_LEVEL=0.3
NUM_LAYERS=8
NUM_HEADS=8
NUM_EPOCHS=400
EXPERIMENT_NAME="attention_denoising"
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
  --num-heads)
    NUM_HEADS="$2"
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
    local noise_type=$1
    local noise_level=$2
    local exp_name="${3:-attention_${noise_type}_eps${noise_level}}"
    
    echo "================================================"
    echo "Running attention experiment:"
    echo "  Noise type: $noise_type"
    echo "  Noise level: $noise_level"
    echo "  Layers: $NUM_LAYERS, Heads: $NUM_HEADS"
    echo "================================================"
    
    tmgg-attention \
      +experiment.name="$exp_name" \
      model.num_layers=$NUM_LAYERS \
      model.num_heads=$NUM_HEADS \
      trainer.max_epochs=$NUM_EPOCHS \
      data.noise_type="$noise_type" \
      evaluation.noise_levels="[$noise_level]" \
      ++tags="[noise_${noise_type},eps_${noise_level},layers_${NUM_LAYERS},heads_${NUM_HEADS}]"
}

# Component experiments for each noise type
run_gaussian_experiments() {
    echo "Running Gaussian noise experiments..."
    local noise_levels=(0.05 0.1 0.2 0.3 0.4 0.5)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "gaussian" "$eps"
    done
}

run_rotation_experiments() {
    echo "Running Rotation noise experiments..."
    local noise_levels=(0.05 0.1 0.15 0.2 0.25 0.3)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "rotation" "$eps"
    done
}

run_digress_experiments() {
    echo "Running Digress noise experiments..."
    local noise_levels=(0.1 0.2 0.3 0.4 0.5 0.6)
    for eps in "${noise_levels[@]}"; do
        run_single_experiment "digress" "$eps"
    done
}

# Full replication of all experiments
run_replication() {
    echo "Running full attention replication experiments..."
    run_gaussian_experiments
    run_rotation_experiments
    run_digress_experiments
}

# Main execution logic
if [ "$REPLICATE" = true ]; then
    run_replication
elif [ -n "$COMPONENT" ]; then
    case $COMPONENT in
        gaussian)
            run_gaussian_experiments
            ;;
        rotation)
            run_rotation_experiments
            ;;
        digress)
            run_digress_experiments
            ;;
        all)
            run_replication
            ;;
        *)
            echo "Unknown component: $COMPONENT"
            echo "Valid components: gaussian, rotation, digress, all"
            exit 1
            ;;
    esac
else
    # Run single experiment with provided parameters
    run_single_experiment "$NOISE_TYPE" "$NOISE_LEVEL" "$EXPERIMENT_NAME"
fi

