"""Model registry and factory for graph architectures.

:class:`ModelRegistry` is the singleton that maps model-type strings to factory
callables. :func:`create_model` and the :func:`register_model` decorator are
backward-compatible convenience wrappers around the class API.

Every factory must return a :class:`~tmgg.models.base.BaseModel` instance;
``ModelRegistry.create`` enforces this at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, KeysView
from typing import Any, ClassVar

import torch.nn as nn

from tmgg.models.base import BaseModel


class _RegistryMeta(type):
    """Metaclass enabling ``"name" in ModelRegistry`` and ``for name in ModelRegistry``."""

    def __contains__(cls, name: object) -> bool:
        return name in cls._factories  # pyright: ignore[reportAttributeAccessIssue]

    def __iter__(cls):  # pyright: ignore[reportSelfClsParameterName]
        return iter(cls._factories)  # pyright: ignore[reportAttributeAccessIssue]


class ModelRegistry(metaclass=_RegistryMeta):
    """Singleton registry mapping model-type strings to factory callables.

    Factories are callables ``(dict[str, Any]) -> nn.Module`` whose return
    value must be a :class:`~tmgg.models.base.BaseModel` subclass.
    The contract is enforced by :meth:`create` at instantiation time.

    Supports ``"name" in ModelRegistry`` via metaclass.
    """

    _factories: ClassVar[dict[str, Callable[[dict[str, Any]], nn.Module]]] = {}

    @classmethod
    def register(
        cls, name: str, factory_fn: Callable[[dict[str, Any]], nn.Module]
    ) -> None:
        """Register *factory_fn* under *name*.

        Parameters
        ----------
        name
            Model-type key (must be unique).
        factory_fn
            Callable that accepts a config dict and returns an ``nn.Module``.

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        if name in cls._factories:
            raise ValueError(f"Model type '{name}' already registered")
        cls._factories[name] = factory_fn

    @classmethod
    def deregister(cls, name: str) -> None:
        """Remove *name* from the registry.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in cls._factories:
            raise KeyError(f"Model type '{name}' not registered")
        del cls._factories[name]

    @classmethod
    def create(cls, model_type: str, config: dict[str, Any]) -> BaseModel:
        """Instantiate a model by type string and config dict.

        Parameters
        ----------
        model_type
            Registered model-type key.
        config
            Configuration dictionary passed to the factory callable.

        Returns
        -------
        BaseModel
            The instantiated model (``GraphModel`` subclass).

        Raises
        ------
        ValueError
            If *model_type* is not registered.
        TypeError
            If the factory returns something other than a ``BaseModel``.
        """
        factory_fn = cls._factories.get(model_type)
        if factory_fn is None:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Registered types: {sorted(cls._factories.keys())}"
            )
        model = factory_fn(config)
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"Factory for '{model_type}' returned {type(model).__name__}, "
                f"expected BaseModel subclass"
            )
        return model

    @classmethod
    def keys(cls) -> KeysView[str]:
        """Return a view of all registered model-type names."""
        return cls._factories.keys()


# -----------------------------------------------------------------------
# Backward-compatible module-level API
# -----------------------------------------------------------------------


def create_model(model_type: str, config: dict[str, Any]) -> BaseModel:
    """Convenience wrapper around :meth:`ModelRegistry.create`."""
    return ModelRegistry.create(model_type, config)


def register_model(*names: str) -> Callable:
    """Decorator that registers a factory function under one or more names.

    Parameters
    ----------
    *names
        One or more string keys that will map to the decorated factory in
        :class:`ModelRegistry`.

    Raises
    ------
    ValueError
        If any *name* is already registered.
    """

    def decorator(
        fn: Callable[[dict[str, Any]], nn.Module],
    ) -> Callable[[dict[str, Any]], nn.Module]:
        for name in names:
            ModelRegistry.register(name, fn)
        return fn

    return decorator


# -----------------------------------------------------------------------
# Spectral denoisers
# -----------------------------------------------------------------------


@register_model("linear_pe")
def _make_linear_pe(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers import LinearPE

    return LinearPE(
        k=config.get("k", 8),
        max_nodes=config.get(
            "max_nodes", 200
        ),  # hard ceiling; override for graphs > 200 nodes
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


@register_model("bilinear_mlp")
def _make_bilinear_mlp(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers.bilinear import BilinearDenoiserWithMLP

    return BilinearDenoiserWithMLP(
        k=config.get("k", 8),
        d_k=config.get("d_k", 64),
        mlp_hidden_dim=config.get("mlp_hidden_dim", 128),
        mlp_num_layers=config.get("mlp_num_layers", 2),
    )


@register_model("multilayer_bilinear")
def _make_multilayer_bilinear(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.spectral_denoisers.bilinear import MultiLayerBilinearDenoiser

    # The spectral module stores the transformer MLP dim separately as
    # "transformer_mlp_hidden_dim"; the generative module reuses
    # "mlp_hidden_dim".  Prefer the explicit key, fall back to generic.
    mlp_dim = config.get("transformer_mlp_hidden_dim", config.get("mlp_hidden_dim"))
    return MultiLayerBilinearDenoiser(
        k=config.get("k", 8),
        d_model=config.get("d_model", 64),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        use_mlp=config.get("use_mlp", True),
        mlp_hidden_dim=mlp_dim,
        dropout=config.get("dropout", 0.0),
    )


@register_model("attention")
def _make_attention(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.attention.attention import MultiLayerAttention

    return MultiLayerAttention(
        d_model=config.get("d_model", 64),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        d_k=config.get("d_k"),
        d_v=config.get("d_v"),
        dropout=config.get("dropout", 0.0),
        bias=config.get("bias", False),
        use_residual=config.get("use_residual", True),
    )


# -----------------------------------------------------------------------
# GNN models
# -----------------------------------------------------------------------


@register_model("gnn")
def _make_gnn(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.gnn import GNN

    return GNN(
        num_layers=config.get("num_layers", 4),
        num_terms=config.get("num_terms", 3),
        feature_dim_in=config.get("feature_dim_in", 10),
        feature_dim_out=config.get("feature_dim_out", 10),
    )


@register_model("gnn_sym")
def _make_gnn_symmetric(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.gnn import GNNSymmetric

    return GNNSymmetric(
        num_layers=config.get("num_layers", 4),
        num_terms=config.get("num_terms", 3),
        feature_dim_in=config.get("feature_dim_in", 10),
        feature_dim_out=config.get("feature_dim_out", 10),
    )


@register_model("node_var_gnn")
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

    from .gnn import GNN

    embedding = GNN(
        num_layers=config.get("num_layers", 2),
        num_terms=config.get("num_terms", 2),
        feature_dim_in=config.get("feature_dim_in", 20),
        feature_dim_out=config.get("feature_dim_out", 5),
        eigenvalue_reg=config.get("eigenvalue_reg", 0.0),
    )
    return SequentialDenoisingModel(
        embedding_model=embedding,
        denoising_model=None,
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


# -----------------------------------------------------------------------
# Discrete diffusion models
# -----------------------------------------------------------------------


@register_model("graph_transformer")
def _make_graph_transformer(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.digress.extra_features import ExtraFeaturesProvider
    from tmgg.models.digress.transformer_model import GraphTransformer

    extra_features: ExtraFeaturesProvider | None = None
    extra_features_type = config.get("extra_features_type")
    use_eigenvectors = config.get("use_eigenvectors", False)

    if extra_features_type:
        from tmgg.models.digress.extra_features import ExtraFeatures

        extra_features = ExtraFeatures(
            extra_features_type,
            max_n_nodes=config.get("max_n_nodes", 200),
        )
    elif use_eigenvectors:
        from tmgg.models.digress.extra_features import EigenvectorAugmentation

        k = config.get("k")
        if k is None:
            raise ValueError("k must be specified when use_eigenvectors=True")
        extra_features = EigenvectorAugmentation(k=k)

    # Split hidden_dims: integer dimensions stay, boolean/projection flags
    # move to projection_config for type safety.
    _PROJECTION_KEYS = {
        "use_gnn_q",
        "use_gnn_k",
        "use_gnn_v",
        "gnn_num_terms",
        "use_spectral_q",
        "use_spectral_k",
        "use_spectral_v",
        "spectral_k",
        "spectral_num_terms",
    }
    raw_hidden = config.get("hidden_dims", {"dx": 32, "de": 16, "dy": 32, "n_head": 2})
    hidden_dims = {k: v for k, v in raw_hidden.items() if k not in _PROJECTION_KEYS}
    proj_from_hidden = {k: v for k, v in raw_hidden.items() if k in _PROJECTION_KEYS}

    # Prefer explicit projection_config; fall back to flags extracted from hidden_dims
    projection_config = config.get("projection_config") or proj_from_hidden or None

    return GraphTransformer(
        n_layers=config.get("n_layers", 2),
        input_dims=config.get("input_dims", {"X": 2, "E": 2, "y": 0}),
        hidden_mlp_dims=config.get("hidden_mlp_dims", {"X": 32, "E": 16, "y": 32}),
        hidden_dims=hidden_dims,
        output_dims=config.get("output_dims", {"X": 0, "E": 2, "y": 0}),
        extra_features=extra_features,
        use_timestep=config.get("use_timestep", False),
        projection_config=projection_config,
    )


@register_model("linear")
def _make_linear(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.baselines import LinearBaseline

    return LinearBaseline(max_nodes=config.get("max_nodes", 200))


@register_model("mlp")
def _make_mlp(config: dict[str, Any]) -> nn.Module:
    from tmgg.models.baselines import MLPBaseline

    return MLPBaseline(
        max_nodes=config.get(
            "max_nodes", 200
        ),  # hard ceiling; override for graphs > 200 nodes
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 2),
    )
