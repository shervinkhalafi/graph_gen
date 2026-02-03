#!/usr/bin/env zsh
# evaluate_all_digress.zsh
# Evaluate all checkpoints for DiGress diffusion runs on Modal
#
# Usage:
#   ./scripts/evaluate_all_digress.zsh              # Full evaluation
#   ./scripts/evaluate_all_digress.zsh --dry-run    # Show what would run
#   ./scripts/evaluate_all_digress.zsh --debug      # Use debug GPU (T4)

set -e

# Configuration
CONFIG_DIR="configs/stage2/2026-01-07"
GPU_TIER="standard"  # A10G - good balance of cost/speed
NUM_SAMPLES=500
NUM_STEPS=100
SPLITS=(train val test)  # Array for proper argument passing
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --debug)
            GPU_TIER="debug"
            ;;
        --fast)
            GPU_TIER="fast"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run] [--debug] [--fast]"
            exit 1
            ;;
    esac
done

echo "=== DiGress MMD Evaluation ==="
echo "Config dir: $CONFIG_DIR"
echo "GPU tier: $GPU_TIER"
echo "Samples: $NUM_SAMPLES, Steps: $NUM_STEPS"
echo "Splits: ${SPLITS[*]}"
echo ""

# Step 1: Deploy Modal app (ensures latest code is deployed)
if [[ "$DRY_RUN" == "false" ]]; then
    echo "Deploying Modal app..."
    doppler run -- uv run modal deploy -m tmgg.modal.runner
    echo ""
fi

# Step 2: Extract run_ids from config files
RUN_IDS=($(ls ${CONFIG_DIR}/*.json 2>/dev/null | xargs -I{} basename {} .json))

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "ERROR: No config files found in $CONFIG_DIR"
    exit 1
fi

echo "Found ${#RUN_IDS[@]} runs to evaluate"
echo ""

# Step 3: Spawn evaluations for all runs (fire-and-forget)
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN - would evaluate these runs:"
    for run_id in "${RUN_IDS[@]}"; do
        echo "  - $run_id"
    done
    echo ""
    echo "Command that would be run for each:"
    echo "  doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \\"
    echo "      --run-id <run_id> \\"
    echo "      --all-checkpoints \\"
    echo "      --splits ${SPLITS[*]} \\"
    echo "      --num-samples $NUM_SAMPLES \\"
    echo "      --num-steps $NUM_STEPS \\"
    echo "      --gpu $GPU_TIER \\"
    echo "      --no-wait"
else
    echo "Spawning evaluations..."
    SPAWNED=0
    FAILED=0

    for run_id in "${RUN_IDS[@]}"; do
        echo "  -> $run_id"
        doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \
            --run-id "$run_id" \
            --all-checkpoints \
            --splits "${SPLITS[@]}" \
            --num-samples $NUM_SAMPLES \
            --num-steps $NUM_STEPS \
            --gpu $GPU_TIER \
            --no-wait
        ((SPAWNED++))
        echo ""
    done

    echo ""
    echo "=== Summary ==="
    echo "Spawned: $SPAWNED"
    echo "Failed: $FAILED"
    echo ""
    echo "Check Modal dashboard for progress: https://modal.com/apps"
    echo "Results will be saved to /data/outputs/{run_id}/mmd_evaluation_{checkpoint}.json"
fi
