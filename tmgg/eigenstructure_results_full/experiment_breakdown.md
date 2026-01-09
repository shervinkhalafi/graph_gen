# Experiment Breakdown

**Total finished runs**: 1955

## Model Type × Architecture

| model_type   | architecture   |   N |
|--------------|----------------|-----|
| DiGress      | default        | 411 |
| DiGress      | gnn_qk         | 168 |
| DiGress      | gnn_all        | 132 |
| DiGress      | gnn_v          | 132 |
| Spectral     | self_attention | 446 |
| Spectral     | linear_pe      | 437 |
| Spectral     | filter_bank    | 228 |
| unknown      | unknown        |   1 |


## Dataset

| dataset         |    N |
|-----------------|------|
| sbm             | 1032 |
| tree            |  275 |
| erdos_renyi     |  269 |
| d_regular       |  240 |
| ring_of_cliques |   35 |
| regular         |   32 |
| pyg_qm9         |   26 |
| pyg_enzymes     |   18 |
| pyg_proteins    |   14 |
| lfr             |   14 |


## Model Type × Dataset Coverage

| model_type   |   d_regular |   erdos_renyi |   lfr |   pyg_enzymes |   pyg_proteins |   pyg_qm9 |   regular |   ring_of_cliques |   sbm |   tree |
|:-------------|------------:|--------------:|------:|--------------:|---------------:|----------:|----------:|------------------:|------:|-------:|
| DiGress      |         132 |           132 |     3 |             0 |              1 |         0 |         0 |                 0 |   439 |    136 |
| Spectral     |         108 |           137 |    11 |            18 |             13 |        26 |        32 |                35 |   592 |    139 |
| unknown      |           0 |             0 |     0 |             0 |              0 |         0 |         0 |                 0 |     1 |      0 |


## Architecture × Dataset Coverage

|                                |   d_regular |   erdos_renyi |   lfr |   pyg_enzymes |   pyg_proteins |   pyg_qm9 |   regular |   ring_of_cliques |   sbm |   tree |
|:-------------------------------|------------:|--------------:|------:|--------------:|---------------:|----------:|----------:|------------------:|------:|-------:|
| ('DiGress', 'default')         |          60 |            60 |     3 |             0 |              1 |         0 |         0 |                 0 |   223 |     64 |
| ('DiGress', 'gnn_all')         |          24 |            24 |     0 |             0 |              0 |         0 |         0 |                 0 |    60 |     24 |
| ('DiGress', 'gnn_qk')          |          24 |            24 |     0 |             0 |              0 |         0 |         0 |                 0 |    96 |     24 |
| ('DiGress', 'gnn_v')           |          24 |            24 |     0 |             0 |              0 |         0 |         0 |                 0 |    60 |     24 |
| ('Spectral', 'filter_bank')    |           0 |             7 |     0 |             4 |              3 |         9 |         7 |                 9 |   181 |      8 |
| ('Spectral', 'linear_pe')      |          36 |            51 |     8 |             7 |              5 |         9 |        16 |                17 |   235 |     53 |
| ('Spectral', 'self_attention') |          72 |            79 |     3 |             7 |              5 |         8 |         9 |                 9 |   176 |     78 |
| ('unknown', 'unknown')         |           0 |             0 |     0 |             0 |              0 |         0 |         0 |                 0 |     1 |      0 |


## Other Experimental Dimensions


### Asymmetric Attention

| asymmetric   |    N |
|--------------|------|
| no           | 1883 |
| yes          |   72 |


### Input Embedding

| input_embedding   |    N |
|-------------------|------|
| default           | 1470 |
| spectral_pe       |  485 |


### Noise Level

| noise_level   |    N |
|---------------|------|
| multi         | 1514 |
| 0.1           |  126 |
| 0.01          |  116 |
| 0.2           |  113 |
| 0.05          |   45 |
| 0.3           |   41 |


## Hyperparameters


### K (eigenvectors)

|   k |   N |
|-----|-----|
|   8 | 469 |
|  16 | 779 |
|  32 | 528 |
|  50 | 158 |


### Learning Rate

|     lr |   N |
|--------|-----|
| 0.0001 | 184 |
| 0.0002 |  14 |
| 0.0005 | 660 |
| 0.001  | 768 |
| 0.01   | 329 |


### Weight Decay

|     wd |   N |
|--------|-----|
| 0      | 226 |
| 1e-12  | 215 |
| 0.0001 |   6 |
| 0.001  | 751 |
| 0.01   | 757 |


## Performance Summary: Model × Architecture × Dataset

(mean test_acc ± std, N runs)

| Model    | Architecture   | planar   | sbm                 | sbm_small   |
|----------|----------------|----------|---------------------|-------------|
| DiGress  | default        | -        | 0.892±0.028 (n=220) | -           |
| DiGress  | gnn_all        | -        | 0.894±0.003 (n=60)  | -           |
| DiGress  | gnn_qk         | -        | 0.894±0.002 (n=96)  | -           |
| DiGress  | gnn_v          | -        | 0.894±0.003 (n=60)  | -           |
| Spectral | filter_bank    | -        | 0.683±0.110 (n=160) | -           |
| Spectral | linear_pe      | -        | 0.701±0.076 (n=210) | -           |
| Spectral | self_attention | -        | 0.731±0.069 (n=156) | -           |
