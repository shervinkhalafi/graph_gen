#!/bin/bash
# TMGG-based replica of ../denoising/train_attention.sh
# Replicates exact same experimental setup using clean tmgg infrastructure

echo "Starting TMGG attention denoising experiments..."
echo "Replicating parameters from ../denoising/train_attention.sh"

for noise_type in "digress" "gaussian" "rotation"; do
    echo ""
    echo "=== Running Attention Denoising: $noise_type noise, eps=0.3 ==="
    
    uv run tmgg-attention \
        data=legacy_match \
        "+data.noise_type=$noise_type" \
        '+data.noise_levels=[0.3]' \
        "model.noise_type=$noise_type" \
        'model.noise_levels=[0.3]' \
        model.num_heads=8 \
        model.num_layers=8 \
        trainer.max_epochs=1000 \
        seed=42 \
        hydra.run.dir="./outputs/legacy_replication/attention_${noise_type}_eps0.3"
        
    echo "Completed: Attention with $noise_type noise"
done

echo ""
echo "=== All TMGG attention experiments completed ==="
echo "Results saved to ./outputs/legacy_replication/"