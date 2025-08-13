#!/bin/bash
# Quick test script to verify noise generators are working correctly

# Exit on error
set -e

echo "Testing noise generators with different configurations"
echo "===================================================="
echo

# Test directory
TEST_DIR="outputs/noise_generator_tests"
mkdir -p "$TEST_DIR"

# Quick test with reduced epochs
MAX_EPOCHS=10
BATCH_SIZE=16
NUM_SAMPLES=50

echo "1. Testing Gaussian noise"
echo "------------------------"
tmgg-attention \
    hydra.run.dir="$TEST_DIR/gaussian" \
    seed=42 \
    data.noise_type="gaussian" \
    data.noise_levels="[0.1,0.3]" \
    data.num_samples_per_graph=$NUM_SAMPLES \
    data.batch_size=$BATCH_SIZE \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.check_val_every_n_epoch=5 \
    experiment_name="test_gaussian" \
    wandb=null  # Disable wandb for testing

echo
echo "2. Testing Digress noise"
echo "-----------------------"
tmgg-attention \
    hydra.run.dir="$TEST_DIR/digress" \
    seed=42 \
    data.noise_type="digress" \
    data.noise_levels="[0.1,0.3]" \
    data.num_samples_per_graph=$NUM_SAMPLES \
    data.batch_size=$BATCH_SIZE \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.check_val_every_n_epoch=5 \
    experiment_name="test_digress" \
    wandb=null

echo
echo "3. Testing Rotation noise"
echo "------------------------"
tmgg-attention \
    hydra.run.dir="$TEST_DIR/rotation" \
    seed=42 \
    data.noise_type="rotation" \
    data.noise_levels="[0.1,0.3]" \
    data.num_samples_per_graph=$NUM_SAMPLES \
    data.batch_size=$BATCH_SIZE \
    model.rotation_k=20 \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.check_val_every_n_epoch=5 \
    experiment_name="test_rotation" \
    wandb=null

echo
echo "4. Testing with different rotation_k values"
echo "------------------------------------------"
for k in 10 15 20; do
    echo "Testing rotation noise with k=$k"
    tmgg-gnn \
        hydra.run.dir="$TEST_DIR/rotation_k${k}" \
        seed=42 \
        data.noise_type="rotation" \
        data.noise_levels="[0.2]" \
        data.num_samples_per_graph=$NUM_SAMPLES \
        data.batch_size=$BATCH_SIZE \
        model.rotation_k=$k \
        trainer.max_epochs=$MAX_EPOCHS \
        experiment_name="test_rotation_k${k}" \
        wandb=null
done

echo
echo "All tests completed!"
echo "Check logs in: $TEST_DIR"