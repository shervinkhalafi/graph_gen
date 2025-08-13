#!/bin/bash
# TMGG-based replica of ../denoising/train_gnn.sh  
# Replicates exact same experimental setup using clean tmgg infrastructure

echo "Starting TMGG GNN denoising experiments..."
echo "Replicating parameters from ../denoising/train_gnn.sh"

# Note: order matches legacy script (Gaussian, Rotation, Digress)
for noise_type in "gaussian" "rotation" "digress"; do
    echo ""
    echo "=== Running GNN Denoising: $noise_type noise, eps=0.3 ==="
    
    uv run tmgg-gnn \
        data=legacy_match \
        "+data.noise_type=$noise_type" \
        '+data.noise_levels=[0.3]' \
        "model.noise_type=$noise_type" \
        'model.noise_levels=[0.3]' \
        model.num_layers=1 \
        trainer.max_epochs=1000 \
        seed=42 \
        hydra.run.dir="./outputs/legacy_replication/gnn_${noise_type}_eps0.3"
        
    echo "Completed: GNN with $noise_type noise"
done

echo ""
echo "=== All TMGG GNN experiments completed ==="
echo "Results saved to ./outputs/legacy_replication/"