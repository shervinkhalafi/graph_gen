#!/usr/bin/env python3
"""Calculate hyperparameter configurations for equal parameter counts across models."""

import itertools
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.gnn import GNN
from tmgg.models.hybrid import create_sequential_model


def calculate_attention_params(d_model, num_heads, num_layers):
    """Calculate parameter count for attention model."""
    model = MultiLayerAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0,
        bias=True
    )
    return model.parameter_count()['total']


def calculate_gnn_params(num_layers, num_terms, feature_dim_in, feature_dim_out):
    """Calculate parameter count for GNN model."""
    model = GNN(
        num_layers=num_layers,
        num_terms=num_terms,
        feature_dim_in=feature_dim_in,
        feature_dim_out=feature_dim_out
    )
    return model.parameter_count()['total']


def calculate_hybrid_params(gnn_layers, gnn_terms, gnn_feat_in, gnn_feat_out, 
                           trans_layers, trans_heads):
    """Calculate parameter count for hybrid model."""
    gnn_config = {
        "num_layers": gnn_layers,
        "num_terms": gnn_terms,
        "feature_dim_in": gnn_feat_in,
        "feature_dim_out": gnn_feat_out,
    }
    transformer_config = {
        "num_layers": trans_layers,
        "num_heads": trans_heads,
        "d_k": None,
        "d_v": None,
        "dropout": 0.0,
        "bias": True,
    }
    model = create_sequential_model(gnn_config, transformer_config)
    return model.parameter_count()['total']


def find_matching_configs(target_params, tolerance=100):
    """Find configurations that match target parameter count."""
    configs = {
        'attention': [],
        'gnn': [],
        'hybrid': []
    }
    
    # Search attention configs
    for d_model in [10, 15, 20, 25, 30]:
        for num_heads in [2, 4, 5, 8, 10]:
            if d_model % num_heads != 0:
                continue
            for num_layers in range(1, 15):
                params = calculate_attention_params(d_model, num_heads, num_layers)
                if abs(params - target_params) <= tolerance:
                    configs['attention'].append({
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'params': params
                    })
    
    # Search GNN configs
    for num_layers in range(1, 8):
        for num_terms in range(2, 8):
            for feature_dim in [5, 10, 15, 20, 25, 30]:
                params = calculate_gnn_params(num_layers, num_terms, feature_dim, feature_dim)
                if abs(params - target_params) <= tolerance:
                    configs['gnn'].append({
                        'num_layers': num_layers,
                        'num_terms': num_terms,
                        'feature_dim_in': feature_dim,
                        'feature_dim_out': feature_dim,
                        'params': params
                    })
    
    # Search hybrid configs
    for gnn_layers in [1, 2, 3]:
        for gnn_terms in [2, 3, 4]:
            for gnn_feat_in in [10, 15, 20]:
                for gnn_feat_out in [5, 10]:
                    for trans_layers in [1, 2, 3, 4]:
                        for trans_heads in [2, 4, 5]:
                            if (gnn_feat_out * 2) % trans_heads != 0:
                                continue
                            params = calculate_hybrid_params(
                                gnn_layers, gnn_terms, gnn_feat_in, gnn_feat_out,
                                trans_layers, trans_heads
                            )
                            if abs(params - target_params) <= tolerance:
                                configs['hybrid'].append({
                                    'gnn_layers': gnn_layers,
                                    'gnn_terms': gnn_terms,
                                    'gnn_feat_in': gnn_feat_in,
                                    'gnn_feat_out': gnn_feat_out,
                                    'trans_layers': trans_layers,
                                    'trans_heads': trans_heads,
                                    'params': params
                                })
    
    return configs


def find_best_target_params():
    """Find parameter counts that work well for all architectures."""
    print("Finding parameter counts achievable by all architectures...")
    
    # Try different target parameter counts
    good_targets = []
    for target in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        configs = find_matching_configs(target, tolerance=500)
        
        if configs['attention'] and configs['gnn'] and configs['hybrid']:
            good_targets.append({
                'target': target,
                'num_attention': len(configs['attention']),
                'num_gnn': len(configs['gnn']),
                'num_hybrid': len(configs['hybrid']),
                'configs': configs
            })
    
    return good_targets


def main():
    # Find good target parameter counts
    targets = find_best_target_params()
    
    print("\n" + "="*70)
    print("PARAMETER COUNT ANALYSIS")
    print("="*70)
    
    for target_info in targets:
        target = target_info['target']
        print(f"\nTarget: ~{target} parameters")
        print(f"  Attention configs found: {target_info['num_attention']}")
        print(f"  GNN configs found: {target_info['num_gnn']}")
        print(f"  Hybrid configs found: {target_info['num_hybrid']}")
    
    # Pick a good target (around 4000 parameters seems reasonable)
    target_params = 4000
    configs = find_matching_configs(target_params, tolerance=200)
    
    print("\n" + "="*70)
    print(f"RECOMMENDED CONFIGURATIONS (Target: {target_params} ± 200)")
    print("="*70)
    
    # Pick best configs for each architecture
    print("\n1. ATTENTION MODELS:")
    attention_configs = sorted(configs['attention'], key=lambda x: abs(x['params'] - target_params))[:3]
    for cfg in attention_configs:
        print(f"   d_model={cfg['d_model']}, heads={cfg['num_heads']}, layers={cfg['num_layers']}")
        print(f"   → {cfg['params']:,} parameters")
    
    print("\n2. GNN MODELS:")
    gnn_configs = sorted(configs['gnn'], key=lambda x: abs(x['params'] - target_params))[:3]
    for cfg in gnn_configs:
        print(f"   layers={cfg['num_layers']}, terms={cfg['num_terms']}, dim={cfg['feature_dim_in']}")
        print(f"   → {cfg['params']:,} parameters")
    
    print("\n3. HYBRID MODELS:")
    hybrid_configs = sorted(configs['hybrid'], key=lambda x: abs(x['params'] - target_params))[:3]
    for cfg in hybrid_configs:
        print(f"   GNN: layers={cfg['gnn_layers']}, terms={cfg['gnn_terms']}, dim_in={cfg['gnn_feat_in']}, dim_out={cfg['gnn_feat_out']}")
        print(f"   Transformer: layers={cfg['trans_layers']}, heads={cfg['trans_heads']}")
        print(f"   → {cfg['params']:,} parameters")
    
    # Select final configs for grid search
    print("\n" + "="*70)
    print("FINAL SELECTED CONFIGURATIONS FOR GRID SEARCH")
    print("="*70)
    
    final_configs = {
        'attention': attention_configs[0] if attention_configs else None,
        'gnn': gnn_configs[0] if gnn_configs else None,
        'hybrid': hybrid_configs[0] if hybrid_configs else None
    }
    
    for arch, cfg in final_configs.items():
        if cfg:
            print(f"\n{arch.upper()}: {cfg['params']:,} parameters")
            print(f"  Config: {cfg}")
    
    return final_configs


if __name__ == "__main__":
    final_configs = main()