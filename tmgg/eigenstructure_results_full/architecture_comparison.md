# Architecture Comparison by Dataset

Highlights best architecture per dataset. **Bold** = winner, (n) = run count.
Format: best/mean (n). B=best peak, M=best mean.

## Spectral Architectures

| Dataset | self_attn | linear_pe | filter_bank | Winner |
|---------|-----------|-----------|-------------|--------|
| d_regular | **0.940/0.940** (72) BM | 0.940/0.940 (36) | - | self_attn/self_attn |
| erdos_renyi | 0.902/0.900 (79) | **0.902/0.900** (51) BM | 0.888/0.883 (7) | lin_pe/lin_pe |
| lfr | - | **0.955/0.949** (6) BM | - | lin_pe/lin_pe |
| pyg_enzymes | 0.877/0.852 (7) | 0.879/0.869 (7) | **0.894/0.891** (4) BM | FB/FB |
| pyg_proteins | 0.908/0.891 (5) | **0.908/0.892** (5) B | **0.908/0.902** (3) M | lin_pe/FB |
| pyg_qm9 | 0.941/0.907 (8) | **1.000/0.996** (9) BM | 0.973/0.958 (9) | lin_pe/lin_pe |
| regular | **0.940/0.940** (9) B | **0.940/0.940** (16) M | 0.940/0.935 (7) | self_attn/lin_pe |
| ring_of_cliques | 0.852/0.810 (9) | 0.879/0.840 (17) | **0.932/0.894** (9) BM | FB/FB |
| sbm | **0.865/0.731** (156) M | **0.950/0.701** (210) B | 0.870/0.683 (160) | lin_pe/self_attn |
| tree | **0.961/0.958** (78) B | **0.961/0.961** (53) M | 0.960/0.955 (8) | self_attn/lin_pe |

**Filter bank wins**: pyg_enzymes (B+M), ring_of_cliques (B+M), pyg_proteins (M only)

## DiGress Architectures

| Dataset | default | gnn_all | gnn_qk | gnn_v | Winner |
|---------|---------|---------|--------|-------|--------|
| d_regular | **0.941/0.940** (60) BM | 0.940/0.940 (24) | 0.940/0.940 (24) | 0.940/0.940 (24) | default/default |
| erdos_renyi | **0.905/0.903** (60) B | 0.904/0.903 (24) | **0.904/0.903** (24) M | 0.904/0.903 (24) | default/gnn_qk |
| sbm | **0.999/0.892** (220) B | **0.899/0.894** (60) M | 0.899/0.894 (96) | 0.899/0.894 (60) | default/gnn_all |
| tree | **0.963/0.962** (60) B | 0.963/0.962 (24) | 0.963/0.962 (24) | **0.963/0.962** (24) M | default/gnn_v |

**Default wins or ties best** in all datasets. Mean differences are <0.3%.