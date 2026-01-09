# Model × Dataset Performance Table

## Best test_acc by Model/Ablation/Dataset

| Model    | Ablation       |   default | sbm    | sbm_small   |
|----------|----------------|-----------|--------|-------------|
| DiGress  | default        |    0.9629 | 0.8993 | 0.9993      |
| DiGress  | gnn_all        |    0.9628 | 0.8991 | -           |
| DiGress  | gnn_qk         |    0.9629 | 0.8991 | -           |
| DiGress  | gnn_v          |    0.9629 | 0.8993 | -           |
| Spectral | default        |    1      | -      | -           |
| Spectral | filter_bank    |    0.6798 | -      | -           |
| Spectral | linear_pe      |    0.9608 | 0.6975 | -           |
| Spectral | self_attention |    0.9594 | 0.7031 | -           |

## Mean±Std test_acc by Model/Ablation/Dataset

| Model    | Ablation       | default     | sbm         | sbm_small   |
|----------|----------------|-------------|-------------|-------------|
| DiGress  | default        | 0.914±0.034 | 0.895±0.002 | 0.912±0.085 |
| DiGress  | gnn_all        | 0.921±0.028 | 0.894±0.002 | -           |
| DiGress  | gnn_qk         | 0.914±0.027 | 0.894±0.003 | -           |
| DiGress  | gnn_v          | 0.921±0.028 | 0.894±0.003 | -           |
| Spectral | default        | 0.903±0.051 | -           | -           |
| Spectral | filter_bank    | 0.615±0.047 | -           | -           |
| Spectral | linear_pe      | 0.805±0.122 | 0.664±0.028 | -           |
| Spectral | self_attention | 0.887±0.090 | 0.692±0.006 | -           |

## Run Counts (N)

| Model    | Ablation       |   default | sbm   | sbm_small   |
|----------|----------------|-----------|-------|-------------|
| DiGress  | default        |       331 | 60    | 9           |
| DiGress  | gnn_all        |       108 | 24    | -           |
| DiGress  | gnn_qk         |       144 | 24    | -           |
| DiGress  | gnn_v          |       108 | 24    | -           |
| Spectral | default        |       195 | -     | -           |
| Spectral | filter_bank    |       111 | -     | -           |
| Spectral | linear_pe      |       334 | 36    | -           |
| Spectral | self_attention |       292 | 72    | -           |

## Notes
- **default**: Main planar graphs dataset
- **sbm**: Stochastic Block Model graphs  
- **sbm_small**: Small SBM graphs (early POC, limited runs)
- Spectral model has limited coverage on sbm datasets
