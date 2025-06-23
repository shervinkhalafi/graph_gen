#!/bin/bash
# Script to run GNN-based denoising experiments

# Default values
MODEL_TYPE="standard_gnn"  # Can be standard_gnn, symmetric_gnn, or nodevar_gnn
NOISE_TYPE="digress"
NOISE_LEVEL=0.3
NUM_LAYERS=1
NUM_EPOCHS=400
EXPERIMENT_NAME="gnn_denoising"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

echo "Running GNN denoising experiment..."
echo "Model type: $MODEL_TYPE"
echo "Noise type: $NOISE_TYPE"
echo "Noise level: $NOISE_LEVEL"
echo "Number of layers: $NUM_LAYERS"
echo "Number of epochs: $NUM_EPOCHS"

# Run the experiment
tmgg-gnn \
    experiment.name="$EXPERIMENT_NAME" \
    model="$MODEL_TYPE" \
    model.num_layers=$NUM_LAYERS \
    trainer.max_epochs=$NUM_EPOCHS \
    data.noise_type="$NOISE_TYPE" \
    evaluation.noise_levels="[$NOISE_LEVEL]" \
    ++tags="[model_${MODEL_TYPE},noise_${NOISE_TYPE},eps_${NOISE_LEVEL},layers_${NUM_LAYERS}]"