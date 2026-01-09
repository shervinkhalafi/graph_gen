# Compact Experiment Configuration Table

(Aggregated over hyperparameter settings - shows best HP result per config)

**Total finished runs**: 1955

## Configurations (aggregated over HP)

| Model | Arch | Dataset | Asym | Embed | Noise | N | Acc (mean±std) | Best |
|-------|------|---------|------|-------|-------|---|----------------|------|
| DiGress | default | d_regular | no | default | multi | 60 | 0.940±0.000 | 0.941 |
|  |  | erdos_renyi | no | default | multi | 60 | 0.903±0.001 | 0.905 |
|  |  | lfr | no | default | 0.01 | 1 | - | - |
|  |  | lfr | no | default | 0.1 | 1 | - | - |
|  |  | lfr | no | default | 0.2 | 1 | - | - |
|  |  | pyg_proteins | no | default | 0.2 | 1 | - | - |
|  |  | sbm | no | default | 0.01 | 1 | - | - |
|  |  | sbm | no | default | 0.1 | 8 | 0.804±0.095 | 0.920 |
|  |  | sbm | no | default | 0.2 | 1 | - | - |
|  |  | sbm | no | default | multi | 204 | 0.894±0.003 | 0.899 |
|  |  | sbm | no | spectral_pe | 0.01 | 2 | 0.999±0.000 | 0.999 |
|  |  | sbm | no | spectral_pe | 0.05 | 2 | 0.977±0.000 | 0.977 |
|  |  | sbm | no | spectral_pe | 0.1 | 2 | 0.904±0.000 | 0.904 |
|  |  | sbm | no | spectral_pe | 0.2 | 2 | 0.846±0.000 | 0.846 |
|  |  | sbm | no | spectral_pe | 0.3 | 1 | 0.753 | 0.753 |
|  |  | tree | no | default | 0.01 | 1 | - | - |
|  |  | tree | no | default | 0.1 | 2 | - | - |
|  |  | tree | no | default | 0.2 | 1 | - | - |
|  |  | tree | no | default | multi | 60 | 0.962±0.000 | 0.963 |
|  | gnn_all | d_regular | no | default | multi | 24 | 0.940±0.000 | 0.940 |
|  |  | erdos_renyi | no | default | multi | 24 | 0.903±0.001 | 0.904 |
|  |  | sbm | no | default | multi | 60 | 0.894±0.003 | 0.899 |
|  |  | tree | no | default | multi | 24 | 0.962±0.000 | 0.963 |
|  | gnn_qk | d_regular | no | default | multi | 24 | 0.940±0.000 | 0.940 |
|  |  | erdos_renyi | no | default | multi | 24 | 0.903±0.000 | 0.904 |
|  |  | sbm | no | default | multi | 96 | 0.894±0.002 | 0.899 |
|  |  | tree | no | default | multi | 24 | 0.962±0.001 | 0.963 |
|  | gnn_v | d_regular | no | default | multi | 24 | 0.940±0.000 | 0.940 |
|  |  | erdos_renyi | no | default | multi | 24 | 0.903±0.001 | 0.904 |
|  |  | sbm | no | default | multi | 60 | 0.894±0.003 | 0.899 |
|  |  | tree | no | default | multi | 24 | 0.962±0.001 | 0.963 |
| Spectral | filter_bank | erdos_renyi | no | default | 0.01 | 3 | 0.877±0.001 | 0.878 |
|  |  | erdos_renyi | no | default | 0.1 | 1 | 0.885 | 0.885 |
|  |  | erdos_renyi | no | default | 0.2 | 3 | 0.888±0.001 | 0.888 |
|  |  | pyg_enzymes | no | default | 0.1 | 2 | 0.892±0.003 | 0.894 |
|  |  | pyg_enzymes | no | default | 0.2 | 2 | 0.890±0.003 | 0.892 |
|  |  | pyg_proteins | no | default | 0.01 | 1 | 0.892 | 0.892 |
|  |  | pyg_proteins | no | default | 0.1 | 1 | 0.908 | 0.908 |
|  |  | pyg_proteins | no | default | 0.2 | 1 | 0.906 | 0.906 |
|  |  | pyg_qm9 | no | default | 0.01 | 3 | 0.950±0.010 | 0.960 |
|  |  | pyg_qm9 | no | default | 0.1 | 3 | 0.967±0.005 | 0.973 |
|  |  | pyg_qm9 | no | default | 0.2 | 3 | 0.958±0.006 | 0.961 |
|  |  | regular | no | default | 0.01 | 2 | 0.940±0.000 | 0.940 |
|  |  | regular | no | default | 0.1 | 3 | 0.935±0.000 | 0.935 |
|  |  | regular | no | default | 0.2 | 2 | 0.931±0.000 | 0.932 |
|  |  | ring_of_cliques | no | default | 0.01 | 3 | 0.822±0.006 | 0.827 |
|  |  | ring_of_cliques | no | default | 0.1 | 3 | 0.928±0.002 | 0.930 |
|  |  | ring_of_cliques | no | default | 0.2 | 3 | 0.931±0.001 | 0.932 |
|  |  | sbm | no | default | 0.01 | 3 | 0.832±0.001 | 0.833 |
|  |  | sbm | no | default | 0.1 | 3 | 0.868±0.001 | 0.869 |
|  |  | sbm | no | default | 0.2 | 3 | 0.869±0.001 | 0.870 |
|  |  | sbm | no | default | multi | 76 | 0.616±0.047 | 0.680 |
|  |  | sbm | no | spectral_pe | 0.01 | 12 | 0.821±0.014 | 0.835 |
|  |  | sbm | no | spectral_pe | 0.05 | 12 | 0.835±0.015 | 0.861 |
|  |  | sbm | no | spectral_pe | 0.1 | 12 | 0.839±0.018 | 0.869 |
|  |  | sbm | no | spectral_pe | 0.2 | 12 | 0.838±0.020 | 0.869 |
|  |  | sbm | no | spectral_pe | 0.3 | 12 | 0.829±0.031 | 0.868 |
|  |  | sbm | yes | default | multi | 36 | 0.614±0.048 | 0.680 |
|  |  | tree | no | default | 0.01 | 3 | 0.960±0.000 | 0.960 |
|  |  | tree | no | default | 0.1 | 3 | 0.954±0.000 | 0.954 |
|  |  | tree | no | default | 0.2 | 2 | 0.949±0.000 | 0.949 |
|  | linear_pe | d_regular | no | spectral_pe | multi | 36 | 0.940±0.000 | 0.940 |
|  |  | erdos_renyi | no | default | 0.01 | 4 | 0.889±0.001 | 0.890 |
|  |  | erdos_renyi | no | default | 0.1 | 3 | 0.901±0.000 | 0.901 |
|  |  | erdos_renyi | no | default | 0.2 | 3 | 0.901±0.000 | 0.901 |
|  |  | erdos_renyi | no | spectral_pe | 0.01 | 1 | 0.889 | 0.889 |
|  |  | erdos_renyi | no | spectral_pe | 0.05 | 1 | 0.901 | 0.901 |
|  |  | erdos_renyi | no | spectral_pe | 0.1 | 1 | 0.901 | 0.901 |
|  |  | erdos_renyi | no | spectral_pe | 0.2 | 1 | 0.901 | 0.901 |
|  |  | erdos_renyi | no | spectral_pe | 0.3 | 1 | 0.901 | 0.901 |
|  |  | erdos_renyi | no | spectral_pe | multi | 36 | 0.902±0.000 | 0.902 |
|  |  | lfr | no | default | 0.01 | 1 | 0.942 | 0.942 |
|  |  | lfr | no | default | 0.1 | 2 | 0.955 | 0.955 |
|  |  | lfr | no | default | 0.2 | 2 | 0.954 | 0.954 |
|  |  | lfr | no | spectral_pe | 0.01 | 1 | 0.942 | 0.942 |
|  |  | lfr | no | spectral_pe | 0.05 | 1 | 0.946 | 0.946 |
|  |  | lfr | no | spectral_pe | 0.1 | 1 | 0.955 | 0.955 |
|  |  | pyg_enzymes | no | default | 0.01 | 2 | 0.846±0.001 | 0.847 |
|  |  | pyg_enzymes | no | default | 0.1 | 2 | 0.879±0.000 | 0.879 |
|  |  | pyg_enzymes | no | default | 0.2 | 3 | 0.877±0.000 | 0.877 |
|  |  | pyg_proteins | no | default | 0.01 | 2 | 0.868±0.001 | 0.868 |
|  |  | pyg_proteins | no | default | 0.1 | 2 | 0.908±0.000 | 0.908 |
|  |  | pyg_proteins | no | default | 0.2 | 1 | 0.908 | 0.908 |
|  |  | pyg_qm9 | no | default | 0.01 | 4 | 0.992±0.003 | 0.994 |
|  |  | pyg_qm9 | no | default | 0.1 | 3 | 1.000±0.000 | 1.000 |
|  |  | pyg_qm9 | no | default | 0.2 | 2 | 1.000±0.000 | 1.000 |
|  |  | regular | no | default | 0.01 | 3 | 0.940±0.000 | 0.940 |
|  |  | regular | no | default | 0.1 | 4 | 0.940±0.000 | 0.940 |
|  |  | regular | no | default | 0.2 | 4 | 0.940±0.000 | 0.940 |
|  |  | regular | no | spectral_pe | 0.01 | 1 | 0.940 | 0.940 |
|  |  | regular | no | spectral_pe | 0.05 | 1 | 0.940 | 0.940 |
|  |  | regular | no | spectral_pe | 0.1 | 1 | 0.940 | 0.940 |
|  |  | regular | no | spectral_pe | 0.2 | 1 | 0.940 | 0.940 |
|  |  | regular | no | spectral_pe | 0.3 | 1 | 0.940 | 0.940 |
|  |  | ring_of_cliques | no | default | 0.01 | 4 | 0.850±0.007 | 0.855 |
|  |  | ring_of_cliques | no | default | 0.1 | 4 | 0.877±0.002 | 0.879 |
|  |  | ring_of_cliques | no | default | 0.2 | 4 | 0.799±0.001 | 0.800 |
|  |  | ring_of_cliques | no | spectral_pe | 0.01 | 1 | 0.855 | 0.855 |
|  |  | ring_of_cliques | no | spectral_pe | 0.05 | 1 | 0.867 | 0.867 |
|  |  | ring_of_cliques | no | spectral_pe | 0.1 | 1 | 0.876 | 0.876 |
|  |  | ring_of_cliques | no | spectral_pe | 0.2 | 1 | 0.798 | 0.798 |
|  |  | ring_of_cliques | no | spectral_pe | 0.3 | 1 | 0.780 | 0.780 |
|  |  | sbm | no | default | 0.01 | 3 | 0.872±0.001 | 0.874 |
|  |  | sbm | no | default | 0.1 | 4 | 0.882±0.000 | 0.883 |
|  |  | sbm | no | default | 0.2 | 4 | 0.867±0.001 | 0.867 |
|  |  | sbm | no | spectral_pe | 0.01 | 16 | 0.784±0.049 | 0.872 |
|  |  | sbm | no | spectral_pe | 0.05 | 14 | 0.795±0.057 | 0.880 |
|  |  | sbm | no | spectral_pe | 0.1 | 16 | 0.780±0.097 | 0.950 |
|  |  | sbm | no | spectral_pe | 0.2 | 13 | 0.791±0.058 | 0.866 |
|  |  | sbm | no | spectral_pe | 0.3 | 12 | 0.769±0.039 | 0.815 |
|  |  | sbm | no | spectral_pe | multi | 117 | 0.661±0.028 | 0.697 |
|  |  | sbm | yes | spectral_pe | multi | 36 | 0.661±0.028 | 0.697 |
|  |  | tree | no | default | 0.01 | 4 | 0.960±0.000 | 0.960 |
|  |  | tree | no | default | 0.1 | 4 | 0.961±0.000 | 0.961 |
|  |  | tree | no | default | 0.2 | 4 | 0.961±0.000 | 0.961 |
|  |  | tree | no | spectral_pe | 0.01 | 1 | 0.960 | 0.960 |
|  |  | tree | no | spectral_pe | 0.05 | 1 | 0.961 | 0.961 |
|  |  | tree | no | spectral_pe | 0.1 | 1 | 0.961 | 0.961 |
|  |  | tree | no | spectral_pe | 0.2 | 1 | 0.961 | 0.961 |
|  |  | tree | no | spectral_pe | 0.3 | 1 | 0.961 | 0.961 |
|  |  | tree | no | spectral_pe | multi | 36 | 0.961±0.000 | 0.961 |
|  | self_attention | d_regular | no | default | multi | 72 | 0.940±0.000 | 0.940 |
|  |  | erdos_renyi | no | default | 0.01 | 3 | 0.860±0.002 | 0.863 |
|  |  | erdos_renyi | no | default | 0.1 | 2 | 0.899±0.000 | 0.899 |
|  |  | erdos_renyi | no | default | 0.2 | 2 | 0.900±0.000 | 0.900 |
|  |  | erdos_renyi | no | default | multi | 72 | 0.901±0.000 | 0.902 |
|  |  | lfr | no | default | 0.01 | 1 | - | - |
|  |  | lfr | no | default | 0.1 | 2 | - | - |
|  |  | pyg_enzymes | no | default | 0.01 | 2 | 0.791±0.001 | 0.792 |
|  |  | pyg_enzymes | no | default | 0.1 | 2 | 0.876±0.001 | 0.876 |
|  |  | pyg_enzymes | no | default | 0.2 | 3 | 0.877±0.000 | 0.877 |
|  |  | pyg_proteins | no | default | 0.01 | 2 | 0.866±0.001 | 0.867 |
|  |  | pyg_proteins | no | default | 0.1 | 1 | 0.908 | 0.908 |
|  |  | pyg_proteins | no | default | 0.2 | 2 | 0.908±0.000 | 0.908 |
|  |  | pyg_qm9 | no | default | 0.01 | 3 | 0.916±0.015 | 0.934 |
|  |  | pyg_qm9 | no | default | 0.1 | 3 | 0.931±0.012 | 0.941 |
|  |  | pyg_qm9 | no | default | 0.2 | 2 | 0.859±0.011 | 0.867 |
|  |  | regular | no | default | 0.01 | 3 | 0.939±0.000 | 0.940 |
|  |  | regular | no | default | 0.1 | 3 | 0.940±0.000 | 0.940 |
|  |  | regular | no | default | 0.2 | 3 | 0.940±0.000 | 0.940 |
|  |  | ring_of_cliques | no | default | 0.01 | 3 | 0.780±0.007 | 0.787 |
|  |  | ring_of_cliques | no | default | 0.1 | 3 | 0.851±0.001 | 0.852 |
|  |  | ring_of_cliques | no | default | 0.2 | 3 | 0.800±0.002 | 0.801 |
|  |  | sbm | no | default | 0.01 | 3 | 0.837±0.002 | 0.840 |
|  |  | sbm | no | default | 0.1 | 3 | 0.863±0.002 | 0.865 |
|  |  | sbm | no | default | 0.2 | 2 | 0.844±0.001 | 0.845 |
|  |  | sbm | no | default | multi | 108 | 0.687±0.015 | 0.703 |
|  |  | sbm | no | spectral_pe | 0.01 | 12 | 0.827±0.004 | 0.835 |
|  |  | sbm | no | spectral_pe | 0.05 | 12 | 0.844±0.010 | 0.859 |
|  |  | sbm | no | spectral_pe | 0.1 | 12 | 0.843±0.012 | 0.863 |
|  |  | sbm | no | spectral_pe | 0.2 | 12 | 0.835±0.004 | 0.842 |
|  |  | sbm | no | spectral_pe | 0.3 | 12 | 0.788±0.023 | 0.802 |
|  |  | tree | no | default | 0.01 | 1 | 0.942 | 0.942 |
|  |  | tree | no | default | 0.1 | 2 | 0.961±0.000 | 0.961 |
|  |  | tree | no | default | 0.2 | 3 | 0.961±0.000 | 0.961 |
|  |  | tree | no | default | multi | 72 | 0.958±0.001 | 0.959 |
| unknown | unknown | sbm | no | spectral_pe | multi | 1 | - | - |

**Total unique semantic configurations**: 157