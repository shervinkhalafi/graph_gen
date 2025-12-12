#!/usr/bin/env python3
"""Test script to verify parameter counting functionality."""

import torch
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.gnn import GNN
from tmgg.models.hybrid import create_sequential_model


def test_attention_model():
    """Test parameter counting for attention model."""
    print("\n" + "="*60)
    print("Testing Attention Model Parameter Count")
    print("="*60)
    
    model = MultiLayerAttention(
        d_model=20,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=True
    )
    
    counts = model.parameter_count()
    print(f"Total parameters: {counts['total']:,}")
    print(f"Self parameters: {counts['self']:,}")
    
    for key, value in counts.items():
        if key not in ['total', 'self']:
            if isinstance(value, dict) and 'total' in value:
                print(f"  {key}: {value['total']:,}")
    
    return counts


def test_gnn_model():
    """Test parameter counting for GNN model."""
    print("\n" + "="*60)
    print("Testing GNN Model Parameter Count")
    print("="*60)
    
    model = GNN(
        num_layers=2,
        num_terms=3,
        feature_dim_in=10,
        feature_dim_out=10
    )
    
    counts = model.parameter_count()
    print(f"Total parameters: {counts['total']:,}")
    print(f"Self parameters: {counts['self']:,}")
    
    for key, value in counts.items():
        if key not in ['total', 'self']:
            if isinstance(value, dict) and 'total' in value:
                print(f"  {key}: {value['total']:,}")
    
    return counts


def test_hybrid_model():
    """Test parameter counting for hybrid model."""
    print("\n" + "="*60)
    print("Testing Hybrid Model Parameter Count")
    print("="*60)
    
    # Create hybrid model with GNN + Transformer
    gnn_config = {
        "num_layers": 2,
        "num_terms": 2,
        "feature_dim_in": 20,
        "feature_dim_out": 5,
    }
    
    transformer_config = {
        "num_layers": 2,
        "num_heads": 4,
        "d_k": None,
        "d_v": None,
        "dropout": 0.0,
        "bias": True,
    }
    
    model = create_sequential_model(gnn_config, transformer_config)
    
    counts = model.parameter_count()
    print(f"Total parameters: {counts['total']:,}")
    print(f"Self parameters: {counts['self']:,}")
    
    # Print hierarchical breakdown
    def print_hierarchy(counts_dict, indent=0):
        for key, value in counts_dict.items():
            if key not in ['total', 'self']:
                if isinstance(value, dict):
                    if 'total' in value:
                        print(f"{'  ' * (indent+1)}{key}: {value['total']:,}")
                        # Print sub-components
                        sub_dict = {k: v for k, v in value.items() if k not in ['total', 'self']}
                        if sub_dict:
                            print_hierarchy(sub_dict, indent+1)
    
    print_hierarchy(counts)
    
    return counts


def test_lightning_module():
    """Test parameter counting in Lightning module."""
    print("\n" + "="*60)
    print("Testing Lightning Module Parameter Count")
    print("="*60)
    
    from tmgg.experiments.attention_denoising.lightning_module import AttentionDenoisingLightningModule
    
    module = AttentionDenoisingLightningModule(
        d_model=20,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=True
    )
    
    # Manually call log_parameter_count to test it
    module.log_parameter_count()
    
    return module


if __name__ == "__main__":
    print("Testing Parameter Counting Functionality")
    print("="*60)
    
    # Test individual models
    attention_counts = test_attention_model()
    gnn_counts = test_gnn_model()
    hybrid_counts = test_hybrid_model()
    
    # Test Lightning module integration
    lightning_module = test_lightning_module()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)