#!/bin/bash
# Test script to validate TMGG legacy compatibility
# Runs minimal experiments to ensure parameter mapping works correctly

echo "=== Testing TMGG Legacy Compatibility ==="
echo ""

echo "Testing attention denoising with minimal parameters..."
uv run tmgg-attention \
    data=legacy_match \
    '+data.noise_type=digress' \
    '+data.noise_levels=[0.3]' \
    'model.noise_type=digress' \
    'model.noise_levels=[0.3]' \
    model.num_heads=8 \
    model.num_layers=8 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=2 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    seed=42 \
    hydra.run.dir="./outputs/test_compatibility/attention_test"

if [ $? -eq 0 ]; then
    echo "✓ Attention denoising test PASSED"
else
    echo "✗ Attention denoising test FAILED"
    exit 1
fi

echo ""
echo "Testing GNN denoising with minimal parameters..."
uv run tmgg-gnn \
    data=legacy_match \
    '+data.noise_type=digress' \
    '+data.noise_levels=[0.3]' \
    'model.noise_type=digress' \
    'model.noise_levels=[0.3]' \
    model.num_layers=1 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=2 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    seed=42 \
    hydra.run.dir="./outputs/test_compatibility/gnn_test"

if [ $? -eq 0 ]; then
    echo "✓ GNN denoising test PASSED"
else
    echo "✗ GNN denoising test FAILED"  
    exit 1
fi

echo ""
echo "=== All compatibility tests PASSED ==="
echo "Legacy parameter mapping works correctly!"
echo "Ready to run full experiments with:"
echo "  ./scripts/tmgg_train_attention.sh"  
echo "  ./scripts/tmgg_train_gnn.sh"