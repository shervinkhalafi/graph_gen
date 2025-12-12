#!/usr/bin/env zsh
# Run DiGress variants for stage 1 comparison
# Compares eigenvector vs adjacency input modes across LR settings

set -e

echo "=== EIGENVECTOR MODE ==="
echo ""

echo "--- DiGress eigenvec + official LR (0.0002) ---"
uv run tmgg-digress \
    '+models/digress@model=digress_sbm_small' \
    data=sbm_single_graph \
    data.noise_levels='[0.1]'

echo ""
echo "--- DiGress eigenvec + high LR (1e-2) ---"
uv run tmgg-digress \
    '+models/digress@model=digress_sbm_small_highlr' \
    data=sbm_single_graph \
    data.noise_levels='[0.1]'

echo ""
echo "=== ADJACENCY MODE ==="
echo ""

echo "--- DiGress adj + official LR (0.0002) ---"
uv run tmgg-digress \
    '+models/digress@model=digress_sbm_small_adj' \
    data=sbm_single_graph \
    data.noise_levels='[0.1]'

echo ""
echo "--- DiGress adj + high LR (1e-2) ---"
uv run tmgg-digress \
    '+models/digress@model=digress_sbm_small_highlr_adj' \
    data=sbm_single_graph \
    data.noise_levels='[0.1]'

echo ""
echo "=== Done ==="
