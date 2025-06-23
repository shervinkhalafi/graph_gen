#!/bin/bash
# Script to run hybrid GNN+Transformer denoising experiments

# Default values
USE_TRANSFORMER=true
GNN_LAYERS=2
TRANSFORMER_LAYERS=4
TRANSFORMER_HEADS=4
NOISE_TYPE="digress"
NOISE_LEVEL=0.3
NUM_EPOCHS=200
EXPERIMENT_NAME="hybrid_denoising"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

echo "Running hybrid denoising experiment..."
echo "Use transformer: $USE_TRANSFORMER"
echo "GNN layers: $GNN_LAYERS"
if [ "$USE_TRANSFORMER" = true ]; then
    echo "Transformer layers: $TRANSFORMER_LAYERS"
    echo "Transformer heads: $TRANSFORMER_HEADS"
fi
echo "Noise type: $NOISE_TYPE"
echo "Noise level: $NOISE_LEVEL"
echo "Number of epochs: $NUM_EPOCHS"

# Select model configuration based on whether transformer is used
if [ "$USE_TRANSFORMER" = true ]; then
    MODEL_CONFIG="hybrid_with_transformer"
else
    MODEL_CONFIG="hybrid_gnn_only"
fi

# Run the experiment
tmgg-hybrid \
    experiment.name="$EXPERIMENT_NAME" \
    model="$MODEL_CONFIG" \
    model.gnn_num_layers=$GNN_LAYERS \
    model.transformer_num_layers=$TRANSFORMER_LAYERS \
    model.transformer_num_heads=$TRANSFORMER_HEADS \
    trainer.max_epochs=$NUM_EPOCHS \
    data.noise_type="$NOISE_TYPE" \
    evaluation.noise_levels="[$NOISE_LEVEL]" \
    ++tags="[model_${MODEL_CONFIG},noise_${NOISE_TYPE},eps_${NOISE_LEVEL},gnn_${GNN_LAYERS}layers,trans_${TRANSFORMER_LAYERS}layers]"