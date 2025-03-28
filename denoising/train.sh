#!/bin/bash

for noise_type in "Gaussian" "Rotation" "Digress"; do

    for eps in 0.1 0.2 0.3 0.4 0.5; do

        for num_heads in 4 8 16 32; do

            for num_layers in 1 2 3 4; do

                # Run with custom parameters
                python main.py \
                    --seed 42 \
                    --noise_type $noise_type \
                    --eps $eps \
                    --num_heads $num_heads \
                    --num_layers $num_layers \
                    --block_sizes "[10, 5, 3, 2]" \
                    --num_epochs 400 \
                    --num_samples_per_epoch 128 \
                    --batch_size 32
            done
        done
    done
done

