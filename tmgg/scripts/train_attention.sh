#!/bin/bash
# Script to run attention-based denoising experiments

# Default values
NOISE_TYPE="digress"
NOISE_LEVEL=0.3
NUM_LAYERS=8
NUM_HEADS=8
NUM_EPOCHS=400
EXPERIMENT_NAME="attention_denoising"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

echo "Running attention denoising experiment..."
echo "Noise type: $NOISE_TYPE"
echo "Noise level: $NOISE_LEVEL"
echo "Number of layers: $NUM_LAYERS"
echo "Number of heads: $NUM_HEADS"
echo "Number of epochs: $NUM_EPOCHS"

# Run the experiment
tmgg-attention \
    experiment.name="$EXPERIMENT_NAME" \
    model.num_layers=$NUM_LAYERS \
    model.num_heads=$NUM_HEADS \
    trainer.max_epochs=$NUM_EPOCHS \
    data.noise_type="$NOISE_TYPE" \
    evaluation.noise_levels="[$NOISE_LEVEL]" \
    ++tags="[noise_${NOISE_TYPE},eps_${NOISE_LEVEL},layers_${NUM_LAYERS},heads_${NUM_HEADS}]"