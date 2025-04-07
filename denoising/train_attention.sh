#!/bin/bash

for noise_type in "Digress" "Gaussian" "Rotation"; do

    for eps in 0.3; do

        #for num_heads in 4 8 16 32; do

            # for num_layers in 2 4 8; do

                python main.py \
                    --seed 42 \
                    --model_type 'MultiLayerAttention' \
                    --noise_type $noise_type \
                    --eps $eps \
                    --num_heads 8 \
                    --num_layers 8 \
                    --block_sizes "[10, 5, 3, 2]" \
                    --num_epochs 1000 \
                    --num_samples_per_epoch 128 \
                    --batch_size 32
            #done
        #done
    done
done

