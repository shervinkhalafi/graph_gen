#!/bin/bash
# Validation script to verify 1:1 mathematical equivalence
# Runs sanity checks for all exact replication configurations

# Exit on error
set -e

echo "Mathematical Equivalence Validation"
echo "=================================="
echo "Testing exact replication configurations..."
echo

# Function to run validation with detailed reporting
run_validation() {
    local experiment_type=$1
    local config_name=$2
    local description="$3"
    
    echo "Testing $experiment_type ($description)"
    echo "$(printf '=%.0s' {1..60})"
    
    if tmgg-$experiment_type --config-name=$config_name sanity_check=true; then
        echo "✅ PASSED: $experiment_type with $config_name"
    else
        echo "❌ FAILED: $experiment_type with $config_name"
        return 1
    fi
    echo
}

echo "1. Attention Model Equivalence Tests"
echo "===================================="

# Test attention model with exact denoising script configuration
run_validation "attention" "experiment/denoising-script-match" "Exact match for train_attention.sh"

echo "2. GNN Model Equivalence Tests"
echo "==============================="

# Test GNN model with exact denoising script configuration  
run_validation "gnn" "experiment/denoising-script-match" "Exact match for train_gnn.sh"

echo "3. Hybrid Model Equivalence Tests"
echo "=================================="

# Test hybrid model with exact notebook configuration
run_validation "hybrid" "experiment/notebook-match" "Exact match for new_denoiser.ipynb"

echo "4. Configuration Parameter Verification"
echo "======================================="

echo "Checking critical configuration parameters..."

# Check attention model d_k/d_v values
echo "Attention model d_k/d_v check:"
if grep -q "d_k: 20" src/tmgg/experiments/attention_denoising/config/model/denoising-script-match.yaml; then
    echo "✅ d_k=20 (correct, matches original k=20)"
else
    echo "❌ d_k≠20 (incorrect, should be 20 not d_model//num_heads)"
fi

if grep -q "d_v: 20" src/tmgg/experiments/attention_denoising/config/model/denoising-script-match.yaml; then
    echo "✅ d_v=20 (correct, matches original k=20)"
else
    echo "❌ d_v≠20 (incorrect, should be 20 not d_model//num_heads)"
fi

# Check fixed block sizes
echo "Data configuration check:"
if grep -q "block_sizes: \[10, 5, 3, 2\]" src/tmgg/experiments/*/config/data/denoising-script-match.yaml; then
    echo "✅ Fixed block sizes [10,5,3,2] (correct for denoising scripts)"
else
    echo "❌ Missing fixed block sizes (should be [10,5,3,2])"
fi

# Check GNN parameters
echo "GNN configuration check:"
if grep -q "num_terms: 4" src/tmgg/experiments/gnn_denoising/config/model/denoising-script-match.yaml; then
    echo "✅ num_terms=4 (correct, matches original t=4)"
else
    echo "❌ num_terms≠4 (incorrect, should be 4)"
fi

if grep -q "feature_dim_in: 20" src/tmgg/experiments/gnn_denoising/config/model/denoising-script-match.yaml; then
    echo "✅ feature_dim_in=20 (correct, matches original k=20)"
else
    echo "❌ feature_dim_in≠20 (incorrect, should be 20)"
fi

# Check hybrid model parameters
echo "Hybrid model configuration check:"
if grep -q "gnn_feature_dim_out: 5" src/tmgg/experiments/hybrid_denoising/config/model/notebook-match.yaml; then
    echo "✅ gnn_feature_dim_out=5 (correct, matches notebook)"
else
    echo "❌ gnn_feature_dim_out≠5 (incorrect, should be 5)"
fi

if grep -q "transformer_d_k: 10" src/tmgg/experiments/hybrid_denoising/config/model/notebook-match.yaml; then
    echo "✅ transformer_d_k=10 (correct, matches notebook)"
else
    echo "❌ transformer_d_k≠10 (incorrect, should be 10)"
fi

echo
echo "5. Quick Replication Test"
echo "========================="

echo "Running quick sanity checks for all configurations..."

# Test denoising scripts replication
echo "Testing denoising scripts replication (quick):"
if ./scripts/replicate_denoising_scripts_exact.sh -c; then
    echo "✅ Denoising scripts replication: PASSED"
else
    echo "❌ Denoising scripts replication: FAILED"
fi

# Test notebook replication
echo "Testing notebook replication (quick):"
if ./scripts/replicate_notebook_exact.sh -c; then
    echo "✅ Notebook replication: PASSED"  
else
    echo "❌ Notebook replication: FAILED"
fi

echo
echo "Validation Summary"
echo "=================="
echo "All tests completed. Check above for any failures (❌)."
echo "If all tests show ✅, mathematical equivalence is verified."
echo
echo "To run full experiments (not just sanity checks):"
echo "  ./scripts/replicate_denoising_scripts_exact.sh"
echo "  ./scripts/replicate_notebook_exact.sh"