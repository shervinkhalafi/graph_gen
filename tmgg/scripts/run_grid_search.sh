#!/usr/bin/env zsh
# Grid search script for comparing architectures with equal parameter counts (~4000)
#
# This script runs a comprehensive grid search across:
# - 3 architectures (Attention, GNN, Hybrid)
# - 3 noise types (Gaussian, Rotation, Digress)
# - 5 noise levels (0.05, 0.1, 0.2, 0.3, 0.4)
# - 3 learning rates (0.0001, 0.001, 0.01)
#
# Total: 135 experiments

set -e  # Exit on error

echo "=================================="
echo "TMGG Grid Search - Equal Parameters (~4000)"
echo "=================================="
echo ""

# Configuration (zsh arrays)
NOISE_TYPES=(gaussian rotation digress)
NOISE_LEVELS=(0.05 0.1 0.2 0.3 0.4)
LEARNING_RATES=(0.0001 0.001 0.01)
MAX_EPOCHS=200
BATCH_SIZE=32

# Function to run single experiment
function run_experiment() {
    model=$1
    noise_type=$2
    noise_level=$3
    lr=$4
    
    echo "Running: $model | $noise_type | eps=$noise_level | lr=$lr"
    
    # Build command based on model type
    case $model in
        "attention")
            model_config="grid_attention_4k"
            model_name="attention"
            ;;
        "gnn")
            model_config="grid_gnn_4k"
            model_name="gnn"
            ;;
        "hybrid")
            model_config="grid_hybrid_4k"
            model_name="hybrid"
            ;;
        *)
            echo "Unknown model: $model"
            exit 1
            ;;
    esac
    
    # Run the experiment using grid search runner
    uv run tmgg-grid-search \
        model=$model_config \
        data=grid_$noise_type \
        learning_rate=$lr \
        "data.noise_levels=[$noise_level]" \
        trainer.max_epochs=$MAX_EPOCHS \
        data.batch_size=$BATCH_SIZE \
        model_name=$model_name \
        noise_level=$noise_level \
        experiment_name="${model_name}_${noise_type}_lr${lr}_eps${noise_level}" \
        hydra.run.dir="./outputs/grid_search/${model_name}/${noise_type}/lr${lr}/eps${noise_level}"
}

# Function to run experiments for a specific model
function run_model_experiments() {
    model=$1
    
    echo ""
    echo "======================================"
    echo "Starting experiments for: $model"
    echo "======================================"
    
    for noise_type in ${NOISE_TYPES[@]}; do
        for noise_level in ${NOISE_LEVELS[@]}; do
            for lr in ${LEARNING_RATES[@]}; do
                run_experiment "$model" "$noise_type" "$noise_level" "$lr"
                echo "---"
            done
        done
    done
    
    echo "Completed $model experiments"
}

# Main execution
function main() {
    # Parse command line arguments
    if [[ "$#" -eq 0 ]]; then
        echo "Usage: $0 [attention|gnn|hybrid|all] [--parallel]"
        echo ""
        echo "Options:"
        echo "  attention    Run only attention experiments"
        echo "  gnn         Run only GNN experiments"
        echo "  hybrid      Run only hybrid experiments"
        echo "  all         Run all experiments"
        echo "  --parallel  Run experiments in parallel (use with caution)"
        exit 1
    fi
    
    models_to_run=$1
    parallel=${2:-""}
    
    # Start timer
    start_time=$(date +%s)
    
    case $models_to_run in
        "attention")
            run_model_experiments "attention"
            ;;
        "gnn")
            run_model_experiments "gnn"
            ;;
        "hybrid")
            run_model_experiments "hybrid"
            ;;
        "all")
            if [[ "$parallel" == "--parallel" ]]; then
                echo "Running all models in parallel..."
                run_model_experiments "attention" &
                run_model_experiments "gnn" &
                run_model_experiments "hybrid" &
                wait
            else
                run_model_experiments "attention"
                run_model_experiments "gnn"
                run_model_experiments "hybrid"
            fi
            ;;
        *)
            echo "Unknown option: $models_to_run"
            exit 1
            ;;
    esac
    
    # End timer
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    echo ""
    echo "======================================"
    echo "Grid Search Complete!"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo "Results saved to: ./outputs/grid_search/"
    echo "======================================"
}

# Run main function
main "$@"