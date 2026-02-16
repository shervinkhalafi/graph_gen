"""Shared model factory for graph architectures.

Centralizes model construction that was previously duplicated across Lightning
modules. Each module's ``_make_model`` becomes a thin wrapper that packs its
parameters into a config dict and calls :func:`create_model`.

Models register themselves via the :func:`register_model` decorator; the old
if-chain is replaced by a dict lookup in :data:`MODEL_REGISTRY`.
"""

from collections.abc import Callable
from typing import Any

import torch.nn as nn

MODEL_REGISTRY: dict[str, Callable[[dict[str, Any]], nn.Module]] = {}


def register_model(*names: str) -> Callable:
    """Decorator that registers a model factory function under one or more names.

    Parameters
    ----------
    *names
        One or more string keys that will map to the decorated factory in
        :data:`MODEL_REGISTRY`.

    Raises
    ------
    ValueError
        If any *name* is already registered (prevents silent overwrites).
    """

    def decorator(
        fn: Callable[[dict[str, Any]], nn.Module],
    ) -> Callable[[dict[str, Any]], nn.Module]:
        for name in names:
            if name in MODEL_REGISTRY:
                raise ValueError(f"Model type '{name}' already registered")
            MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def create_model(model_type: str, config: dict[str, Any]) -> nn.Module:
    """Instantiate a denoising model from a type string and config dict.

    Parameters
    ----------
    model_type
        Model architecture identifier. Supported types:

        *Spectral denoisers* -- ``linear_pe``, ``filter_bank``,
        ``self_attention`` (SelfAttentionDenoiser),
        ``bilinear`` (BilinearDenoiser),
        ``self_attention_mlp`` / ``bilinear_mlp`` (BilinearDenoiserWithMLP),
        ``multilayer_self_attention`` / ``multilayer_attention`` /
        ``multilayer_bilinear`` (MultiLayerBilinearDenoiser),
        ``self_attention_strict_shrinkage``, ``self_attention_relaxed_shrinkage``

        *GNN models* -- ``gnn`` / ``GNN``, ``gnn_sym`` / ``GNNSymmetric``,
        ``NodeVarGNN``

        *Hybrid* -- ``hybrid`` (EigenEmbedding + BilinearDenoiser),
        ``hybrid_sequential`` (GNN + optional Transformer)

        *Baselines* -- ``linear``, ``mlp``

    config
        Configuration dictionary.  Uses ``config.get("param", default)`` for
        all lookups; unrecognised keys are silently ignored.

    Returns
    -------
    nn.Module
        Instantiated model (always a ``DenoisingModel`` subclass at runtime).

    Raises
    ------
    ValueError
        If *model_type* is not recognised.
    """
    factory_fn = MODEL_REGISTRY.get(model_type)
    if factory_fn is None:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Registered types: {sorted(MODEL_REGISTRY.keys())}"
        )
    return factory_fn(config)


# -----------------------------------------------------------------------
# Spectral denoisers
# -----------------------------------------------------------------------


@register_model("linear_pe")
def _make_linear_pe(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers import LinearPE

    return LinearPE(
        k=config.get("k", 8),
        max_nodes=config.get("max_nodes", 200),
        use_bias=config.get("use_bias", True),
    )


@register_model("filter_bank")
def _make_filter_bank(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers import GraphFilterBank

    return GraphFilterBank(
        k=config.get("k", 8),
        polynomial_degree=config.get("polynomial_degree", 5),
    )


@register_model("self_attention")
def _make_self_attention(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser

    return SelfAttentionDenoiser(
        k=config.get("k", 8),
        d_k=config.get("d_k", 64),
    )


@register_model("bilinear")
def _make_bilinear(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers.bilinear import BilinearDenoiser

    return BilinearDenoiser(
        k=config.get("k", 8),
        d_k=config.get("d_k", 64),
    )


@register_model("self_attention_mlp", "bilinear_mlp")
def _make_bilinear_mlp(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers.bilinear import BilinearDenoiserWithMLP

    return BilinearDenoiserWithMLP(
        k=config.get("k", 8),
        d_k=config.get("d_k", 64),
        mlp_hidden_dim=config.get("mlp_hidden_dim", 128),
        mlp_num_layers=config.get("mlp_num_layers", 2),
    )


@register_model(
    "multilayer_self_attention", "multilayer_attention", "multilayer_bilinear"
)
def _make_multilayer_bilinear(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers.bilinear import (
        MultiLayerBilinearDenoiser as MultiLayerSelfAttentionDenoiser,
    )

    # The spectral module stores the transformer MLP dim separately as
    # "transformer_mlp_hidden_dim"; the generative module reuses
    # "mlp_hidden_dim".  Prefer the explicit key, fall back to generic.
    mlp_dim = config.get("transformer_mlp_hidden_dim", config.get("mlp_hidden_dim"))
    return MultiLayerSelfAttentionDenoiser(
        k=config.get("k", 8),
        d_model=config.get("d_model", 64),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        use_mlp=config.get("use_mlp", True),
        mlp_hidden_dim=mlp_dim,
        dropout=config.get("dropout", 0.0),
    )


@register_model("self_attention_strict_shrinkage")
def _make_strict_shrinkage(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers import StrictShrinkageWrapper
    from tmgg.models.spectral_denoisers.bilinear import BilinearDenoiser

    inner = BilinearDenoiser(k=config.get("k", 8), d_k=config.get("d_k", 64))
    return StrictShrinkageWrapper(
        inner_model=inner,
        max_rank=config.get("shrinkage_max_rank", 50),
        aggregation=config.get("shrinkage_aggregation", "mean"),
        hidden_dim=config.get("shrinkage_hidden_dim", 128),
        mlp_layers=config.get("shrinkage_mlp_layers", 2),
    )


@register_model("self_attention_relaxed_shrinkage")
def _make_relaxed_shrinkage(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers import RelaxedShrinkageWrapper
    from tmgg.models.spectral_denoisers.bilinear import BilinearDenoiser

    inner = BilinearDenoiser(k=config.get("k", 8), d_k=config.get("d_k", 64))
    return RelaxedShrinkageWrapper(
        inner_model=inner,
        max_rank=config.get("shrinkage_max_rank", 50),
        aggregation=config.get("shrinkage_aggregation", "mean"),
        hidden_dim=config.get("shrinkage_hidden_dim", 128),
        mlp_layers=config.get("shrinkage_mlp_layers", 2),
    )


# -----------------------------------------------------------------------
# GNN models
# -----------------------------------------------------------------------


@register_model("gnn", "GNN")
def _make_gnn(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.gnn import GNN

    return GNN(
        num_layers=config.get("num_layers", 4),
        num_terms=config.get("num_terms", 3),
        feature_dim_in=config.get("feature_dim_in", 10),
        feature_dim_out=config.get("feature_dim_out", 10),
    )


@register_model("gnn_sym", "GNNSymmetric")
def _make_gnn_symmetric(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.gnn import GNNSymmetric

    return GNNSymmetric(
        num_layers=config.get("num_layers", 4),
        num_terms=config.get("num_terms", 3),
        feature_dim_in=config.get("feature_dim_in", 10),
        feature_dim_out=config.get("feature_dim_out", 10),
    )


@register_model("NodeVarGNN")
def _make_node_var_gnn(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.gnn import NodeVarGNN

    return NodeVarGNN(
        num_layers=config.get("num_layers", 4),
        num_terms=config.get("num_terms", 3),
        feature_dim=config.get("feature_dim", config.get("feature_dim_out", 10)),
    )


# -----------------------------------------------------------------------
# Hybrid models
# -----------------------------------------------------------------------


@register_model("hybrid")
def _make_hybrid(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.hybrid import SequentialDenoisingModel
    from tmgg.models.layers import EigenEmbedding
    from tmgg.models.spectral_denoisers.bilinear import BilinearDenoiser

    embedding = EigenEmbedding(
        eigenvalue_reg=config.get("eigenvalue_reg", 0.0),
    )
    denoiser = BilinearDenoiser(
        k=config.get("k", 8),
        d_k=config.get("d_k", 64),
    )
    return SequentialDenoisingModel(
        embedding_model=embedding,
        denoising_model=denoiser,  # pyright: ignore[reportArgumentType]  # cross-subpackage subtype
    )


@register_model("hybrid_sequential")
def _make_hybrid_sequential(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.hybrid import create_sequential_model

    gnn_cfg: dict[str, Any] = {
        "num_layers": config.get("gnn_num_layers", 2),
        "num_terms": config.get("gnn_num_terms", 2),
        "feature_dim_in": config.get("gnn_feature_dim_in", 20),
        "feature_dim_out": config.get("gnn_feature_dim_out", 5),
    }
    transformer_cfg: dict[str, Any] | None = None
    if config.get("use_transformer", True):
        transformer_cfg = {
            "num_layers": config.get("transformer_num_layers", 4),
            "num_heads": config.get("transformer_num_heads", 4),
            "d_k": config.get("transformer_d_k"),
            "d_v": config.get("transformer_d_v"),
            "dropout": config.get("transformer_dropout", 0.0),
            "bias": config.get("transformer_bias", True),
        }
    return create_sequential_model(gnn_cfg, transformer_cfg)


# -----------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------


@register_model("linear")
def _make_linear(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.baselines import LinearBaseline

    return LinearBaseline(max_nodes=config.get("max_nodes", 200))


@register_model("mlp")
def _make_mlp(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.baselines import MLPBaseline

    return MLPBaseline(
        max_nodes=config.get("max_nodes", 200),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 2),
    )
