# Semantic Grouping Analysis

**Metric**: test_acc (higher is better)

**Total valid runs**: 1872

## Summary Overview

| Grouping              | Best Group   |   Best test_acc | vs 2nd Best         | Significant?   | Effect Size   |
|-----------------------|--------------|-----------------|---------------------|----------------|---------------|
| Model Type            | spectral     |          1      | Δ=0.0996 (p=0.0000) | YES            | large         |
| Dataset               | pyg_qm9      |          1      | Δ=0.1660 (p=0.0000) | YES            | large         |
| Asymmetric Attention  | no           |          1      | Δ=0.2290 (p=0.0000) | YES            | large         |
| Noise Level           | single_0.1   |          1      | Δ=0.0033 (p=0.7262) | NO             | negligible    |
| Input Embeddings      | default      |          1      | Δ=0.0778 (p=0.0000) | YES            | medium        |
| DiGress Ablations     | default      |          0.9629 | Δ=0.0031 (p=0.2505) | NO             | negligible    |
| Spectral Architecture | linear_pe    |          0.9608 | Δ=0.0572 (p=0.0000) | YES            | small         |
| K Value               | 16.0         |          1      | -                   | -              | -             |
| Learning Rate         | 0.01         |          1      | -                   | -              | -             |
| Weight Decay          | 1e-12        |          1      | -                   | -              | -             |

### Model Type

| Group    |    N |   test_acc BEST | test_acc mean±std   | Range            |
|----------|------|-----------------|---------------------|------------------|
| spectral | 1040 |          1      | 0.8134±0.1297       | [0.5017, 1.0000] |
| digress  |  832 |          0.9993 | 0.9129±0.0306       | [0.6623, 0.9993] |


**Statistical comparisons (vs next best):**

- spectral < digress: Δ=0.0996, p=0.0000 ✓, d=-1.01 (large)

### Dataset

| Group           |   N |   test_acc BEST | test_acc mean±std   | Range            |
|-----------------|-----|-----------------|---------------------|------------------|
| pyg_qm9         |  26 |          1      | 0.9558±0.0410       | [0.8512, 1.0000] |
| sbm             | 962 |          0.9993 | 0.7898±0.1150       | [0.5017, 0.9993] |
| tree            | 271 |          0.9629 | 0.9603±0.0026       | [0.9423, 0.9629] |
| lfr             |   6 |          0.9554 | 0.9494±0.0063       | [0.9424, 0.9554] |
| d_regular       | 240 |          0.9407 | 0.9400±0.0001       | [0.9400, 0.9407] |
| regular         |  32 |          0.94   | 0.9389±0.0025       | [0.9314, 0.9400] |
| ring_of_cliques |  35 |          0.932  | 0.8461±0.0499       | [0.7724, 0.9320] |
| pyg_proteins    |  13 |          0.9084 | 0.8940±0.0194       | [0.8652, 0.9084] |
| erdos_renyi     | 269 |          0.9053 | 0.9010±0.0057       | [0.8580, 0.9053] |
| pyg_enzymes     |  18 |          0.8943 | 0.8673±0.0304       | [0.7909, 0.8943] |


**Statistical comparisons (vs next best):**

- pyg_qm9 > sbm: Δ=0.1660, p=0.0000 ✓, d=1.46 (large)
- sbm < tree: Δ=0.1706, p=0.0000 ✓, d=-1.68 (large)
- tree > lfr: Δ=0.0110, p=0.0000 ✓, d=4.06 (large)
- lfr > d_regular: Δ=0.0093, p=0.0000 ✓, d=10.27 (large)
- d_regular > regular: Δ=0.0011, p=0.0000 ✓, d=1.33 (large)
- regular > ring_of_cliques: Δ=0.0927, p=0.0000 ✓, d=2.57 (large)
- ring_of_cliques < pyg_proteins: Δ=0.0479, p=0.0016 ✓, d=-1.09 (large)
- pyg_proteins < erdos_renyi: Δ=0.0070, p=0.0004 ✓, d=-1.01 (large)
- erdos_renyi > pyg_enzymes: Δ=0.0337, p=0.0000 ✓, d=3.63 (large)

### Asymmetric Attention

| Group   |    N |   test_acc BEST | test_acc mean±std   | Range            |
|---------|------|-----------------|---------------------|------------------|
| no      | 1800 |          1      | 0.8664±0.1029       | [0.5017, 1.0000] |
| yes     |   72 |          0.6975 | 0.6374±0.0455       | [0.5017, 0.6975] |


**Statistical comparisons (vs next best):**

- no > yes: Δ=0.2290, p=0.0000 ✓, d=2.26 (large)

### Noise Level

| Group            |    N |   test_acc BEST | test_acc mean±std   | Range            |
|------------------|------|-----------------|---------------------|------------------|
| single_0.1       |  107 |          1      | 0.8824±0.0720       | [0.6287, 1.0000] |
| single_0.2       |   96 |          1      | 0.8791±0.0610       | [0.7357, 1.0000] |
| single_0.01      |   99 |          0.9993 | 0.8719±0.0671       | [0.7438, 0.9993] |
| single_0.05      |   32 |          0.9768 | 0.8485±0.0629       | [0.7398, 0.9768] |
| multi_[0.01-0.3] | 1509 |          0.9629 | 0.8547±0.1182       | [0.5017, 0.9629] |
| single_0.3       |   29 |          0.9608 | 0.8077±0.0575       | [0.7347, 0.9608] |


**Statistical comparisons (vs next best):**

- single_0.1 > single_0.2: Δ=0.0033, p=0.7262 ✗, d=0.05 (negligible)
- single_0.2 > single_0.01: Δ=0.0072, p=0.4314 ✗, d=0.11 (negligible)
- single_0.01 > single_0.05: Δ=0.0234, p=0.0839 ✗, d=0.35 (small)
- single_0.05 < multi_[0.01-0.3]: Δ=0.0062, p=0.7673 ✗, d=-0.05 (negligible)
- multi_[0.01-0.3] > single_0.3: Δ=0.0470, p=0.0330 ✓, d=0.40 (small)

### Input Embeddings

| Group       |    N |   test_acc BEST | test_acc mean±std   | Range            |
|-------------|------|-----------------|---------------------|------------------|
| default     | 1453 |          1      | 0.8750±0.1016       | [0.5017, 1.0000] |
| spectral_pe |  419 |          0.9993 | 0.7972±0.1185       | [0.6145, 0.9993] |


**Statistical comparisons (vs next best):**

- default > spectral_pe: Δ=0.0778, p=0.0000 ✓, d=0.74 (medium)

### DiGress Ablations

| Group   |   N |   test_acc BEST | test_acc mean±std   | Range            |
|---------|-----|-----------------|---------------------|------------------|
| default | 384 |          0.9629 | 0.9132±0.0266       | [0.8884, 0.9629] |
| gnn_v   | 132 |          0.9629 | 0.9163±0.0275       | [0.8886, 0.9629] |
| gnn_qk  | 168 |          0.9629 | 0.9114±0.0260       | [0.8884, 0.9629] |
| gnn_all | 132 |          0.9628 | 0.9164±0.0274       | [0.8884, 0.9628] |


**Statistical comparisons (vs next best):**

- default < gnn_v: Δ=0.0031, p=0.2505 ✗, d=-0.12 (negligible)
- gnn_v > gnn_qk: Δ=0.0049, p=0.1192 ✗, d=0.18 (negligible)
- gnn_qk < gnn_all: Δ=0.0049, p=0.1112 ✗, d=-0.19 (negligible)

### Spectral Architecture

| Group          |   N |   test_acc BEST | test_acc mean±std   | Range            |
|----------------|-----|-----------------|---------------------|------------------|
| linear_pe      | 370 |          0.9608 | 0.7911±0.1235       | [0.6145, 0.9608] |
| self_attention | 364 |          0.9594 | 0.8484±0.1119       | [0.6335, 0.9594] |
| filter_bank    | 111 |          0.6798 | 0.6153±0.0469       | [0.5017, 0.6798] |


**Statistical comparisons (vs next best):**

- linear_pe < self_attention: Δ=0.0572, p=0.0000 ✓, d=-0.49 (small)
- self_attention > filter_bank: Δ=0.2331, p=0.0000 ✓, d=2.32 (large)

### K Value

|   Group |   N |   test_acc BEST | test_acc mean±std   | Range            |
|---------|-----|-----------------|---------------------|------------------|
|      16 | 777 |          1      | 0.8569±0.1158       | [0.5369, 1.0000] |
|       8 | 455 |          0.9728 | 0.8181±0.1321       | [0.5017, 0.9728] |
|      32 | 528 |          0.9629 | 0.9012±0.0655       | [0.6151, 0.9629] |
|      50 |  96 |          0.8512 | 0.8099±0.0398       | [0.7347, 0.8512] |


### Learning Rate

|   Group |   N |   test_acc BEST | test_acc mean±std   | Range            |
|---------|-----|-----------------|---------------------|------------------|
|  0.01   | 283 |          1      | 0.8871±0.0594       | [0.6287, 1.0000] |
|  0.0002 |  14 |          0.9993 | 0.8836±0.0948       | [0.6623, 0.9993] |
|  0.0005 | 660 |          0.9629 | 0.8670±0.1103       | [0.6038, 0.9629] |
|  0.001  | 735 |          0.9629 | 0.8608±0.1040       | [0.6273, 0.9629] |
|  0.0001 | 180 |          0.896  | 0.7616±0.1454       | [0.5017, 0.8960] |


### Weight Decay

|   Group |   N |   test_acc BEST | test_acc mean±std   | Range            |
|---------|-----|-----------------|---------------------|------------------|
|  1e-12  | 209 |          1      | 0.9016±0.0552       | [0.6623, 1.0000] |
|  0.01   | 752 |          0.9629 | 0.8554±0.1178       | [0.5019, 0.9629] |
|  0.001  | 751 |          0.9629 | 0.8557±0.1176       | [0.5017, 0.9629] |
|  0      | 154 |          0.9608 | 0.8265±0.0613       | [0.6287, 0.9608] |
|  0.0001 |   6 |          0.6764 | 0.6455±0.0211       | [0.6273, 0.6764] |


## Conclusions

- **Model Type**: spectral significantly better than digress (d=-1.01, large effect)
- **Dataset**: pyg_qm9 significantly better than sbm (d=1.46, large effect)
- **Asymmetric Attention**: no significantly better than yes (d=2.26, large effect)
- **Noise Level**: No meaningful difference between groups (p=0.726, d=0.05)
- **Input Embeddings**: default significantly better than spectral_pe (d=0.74, medium effect)
- **DiGress Ablations**: No meaningful difference between groups (p=0.251, d=-0.12)