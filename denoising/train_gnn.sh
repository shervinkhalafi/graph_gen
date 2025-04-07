#!/bin/bash

for noise_type in "Gaussian" "Rotation" "Digress"; do

    for eps in 0.3; do

        # Run with custom parameters
        python main.py \
            --seed 42 \
            --model_type 'GNN' \
            --noise_type $noise_type \
            --eps $eps \
            --num_layers 1 \
            --block_sizes "[10, 5, 3, 2]" \
            --num_epochs 1000 \
            --num_samples_per_epoch 128 \
            --batch_size 32
    done
done

