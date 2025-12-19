#!/usr/bin/env python3
"""Test parameter counting for all Lightning modules."""

from tmgg.experiments.attention_denoising.lightning_module import (
    AttentionDenoisingLightningModule,
)
from tmgg.experiments.digress_denoising.lightning_module import (
    DigressDenoisingLightningModule,
)
from tmgg.experiments.gnn_denoising.lightning_module import GNNDenoisingLightningModule
from tmgg.experiments.hybrid_denoising.lightning_module import (
    HybridDenoisingLightningModule,
)


def test_all_modules():
    """Test parameter counting for all Lightning modules."""

    print("\n" + "=" * 70)
    print("Testing Parameter Counting for All Lightning Modules")
    print("=" * 70)

    # Test Attention module
    print("\n1. Attention Denoising Module:")
    print("-" * 40)
    attention_module = AttentionDenoisingLightningModule(
        d_model=20, num_heads=4, num_layers=4, dropout=0.0, bias=True
    )
    attention_module.log_parameter_count()

    # Test GNN module
    print("\n2. GNN Denoising Module:")
    print("-" * 40)
    gnn_module = GNNDenoisingLightningModule(
        model_type="GNN",
        num_layers=2,
        num_terms=3,
        feature_dim_in=20,
        feature_dim_out=20,
    )
    gnn_module.log_parameter_count()

    # Test Hybrid module with transformer
    print("\n3. Hybrid Denoising Module (with Transformer):")
    print("-" * 40)
    hybrid_module = HybridDenoisingLightningModule(
        gnn_num_layers=2,
        gnn_num_terms=2,
        gnn_feature_dim_in=20,
        gnn_feature_dim_out=5,
        use_transformer=True,
        transformer_num_layers=4,
        transformer_num_heads=4,
    )
    hybrid_module.log_parameter_count()

    # Test Hybrid module without transformer (GNN only)
    print("\n4. Hybrid Denoising Module (GNN only):")
    print("-" * 40)
    hybrid_gnn_only = HybridDenoisingLightningModule(
        gnn_num_layers=2,
        gnn_num_terms=2,
        gnn_feature_dim_in=20,
        gnn_feature_dim_out=5,
        use_transformer=False,
    )
    hybrid_gnn_only.log_parameter_count()

    # Test Digress module
    print("\n5. Digress Denoising Module:")
    print("-" * 40)
    digress_module = DigressDenoisingLightningModule(
        n_layers=4,
        node_feature_dim=20,
        use_eigenvectors=False,
    )
    digress_module.log_parameter_count()

    print("\n" + "=" * 70)
    print("Summary of Parameter Counts:")
    print("=" * 70)

    # Collect parameter counts
    modules = [
        ("Attention", attention_module),
        ("GNN", gnn_module),
        ("Hybrid (with Transformer)", hybrid_module),
        ("Hybrid (GNN only)", hybrid_gnn_only),
        ("Digress", digress_module),
    ]

    for name, module in modules:
        if hasattr(module.model, "parameter_count"):
            counts = module.model.parameter_count()
            total = counts["total"]
        else:
            total = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
        print(f"{name:30s}: {total:>10,} parameters")

    print("=" * 70)


if __name__ == "__main__":
    test_all_modules()
