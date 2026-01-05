"""Lightning module for spectral denoising experiments.

This module provides a unified interface for training all spectral denoising
architectures (Linear PE, Graph Filter Bank, Self-Attention) through a single
configurable Lightning module.
"""

from typing import Any

import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.spectral_denoisers import (
    GraphFilterBank,
    LinearPE,
    MultiLayerSelfAttentionDenoiser,
    SelfAttentionDenoiser,
    SelfAttentionDenoiserWithMLP,
)


class SpectralDenoisingLightningModule(DenoisingLightningModule):
    """Unified Lightning module for spectral denoising architectures.

    This module dispatches to different spectral denoising models based on
    the `model_type` configuration parameter.

    All models output raw logits. Use model.predict() for probabilities.

    Parameters
    ----------
    model_type : str
        Architecture to use. One of:
        - "linear_pe": Linear positional encoding
        - "filter_bank": Graph filter bank with spectral polynomial
        - "self_attention": Query-key attention on eigenvectors
        - "self_attention_mlp": Self-attention with MLP post-processing
        - "multilayer_self_attention": Stacked transformer blocks on eigenvectors
    k : int
        Number of eigenvectors to use.
    max_nodes : int, optional
        Maximum nodes (for Linear PE bias). Default 200.
    use_bias : bool, optional
        Use bias in Linear PE. Default True.
    polynomial_degree : int, optional
        Polynomial degree for filter bank. Default 5.
    d_k : int, optional
        Key dimension for self-attention. Default 64.
    mlp_hidden_dim : int, optional
        Hidden dimension for MLP (self_attention_mlp only). Default 128.
    mlp_num_layers : int, optional
        Number of MLP layers (self_attention_mlp only). Default 2.
    d_model : int, optional
        Hidden dimension for multilayer_self_attention. Default 64.
    num_heads : int, optional
        Attention heads for multilayer_self_attention. Default 4.
    num_layers : int, optional
        Transformer blocks for multilayer_self_attention. Default 2.
    use_mlp : bool, optional
        Include MLP in transformer blocks. Default True.
    transformer_mlp_hidden_dim : int, optional
        MLP hidden dim in transformer blocks. Defaults to 4*d_model.
    dropout : float, optional
        Dropout probability for multilayer_self_attention. Default 0.0.
    learning_rate : float, optional
        Learning rate. Default 1e-4.
    weight_decay : float, optional
        Weight decay for AdamW. Default 1e-2.
    optimizer_type : str, optional
        Optimizer: "adam" or "adamw". Default "adamw".
    **kwargs
        Additional arguments passed to base class.

    Examples
    --------
    >>> module = SpectralDenoisingLightningModule(
    ...     model_type="linear_pe",
    ...     k=8,
    ...     learning_rate=1e-4,
    ...     weight_decay=1e-2,
    ...     optimizer_type="adamw",
    ... )
    """

    VALID_MODEL_TYPES = {
        "linear_pe",
        "filter_bank",
        "self_attention",
        "self_attention_mlp",
        "multilayer_self_attention",
    }

    def __init__(
        self,
        model_type: str = "linear_pe",
        k: int = 8,
        max_nodes: int = 200,
        use_bias: bool = True,
        polynomial_degree: int = 5,
        d_k: int = 64,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 2,
        # MultiLayerSelfAttention parameters
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        use_mlp: bool = True,
        transformer_mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        # Optimizer parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        **kwargs: Any,
    ):
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.VALID_MODEL_TYPES}, "
                f"got '{model_type}'"
            )

        self._model_type = model_type
        self._k = k
        self._max_nodes = max_nodes
        self._use_bias = use_bias
        self._polynomial_degree = polynomial_degree
        self._d_k = d_k
        self._mlp_hidden_dim = mlp_hidden_dim
        self._mlp_num_layers = mlp_num_layers
        # MultiLayerSelfAttention
        self._d_model = d_model
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._use_mlp = use_mlp
        self._transformer_mlp_hidden_dim = transformer_mlp_hidden_dim
        self._dropout = dropout

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            **kwargs,
        )

    def _make_model(self, *args: Any, **kwargs: Any) -> nn.Module:
        """Instantiate the spectral denoising model based on config.

        Returns
        -------
        nn.Module
            Configured spectral denoiser model.
        """
        if self._model_type == "linear_pe":
            return LinearPE(
                k=self._k,
                max_nodes=self._max_nodes,
                use_bias=self._use_bias,
            )
        elif self._model_type == "filter_bank":
            return GraphFilterBank(
                k=self._k,
                polynomial_degree=self._polynomial_degree,
            )
        elif self._model_type == "self_attention":
            return SelfAttentionDenoiser(
                k=self._k,
                d_k=self._d_k,
            )
        elif self._model_type == "self_attention_mlp":
            return SelfAttentionDenoiserWithMLP(
                k=self._k,
                d_k=self._d_k,
                mlp_hidden_dim=self._mlp_hidden_dim,
                mlp_num_layers=self._mlp_num_layers,
            )
        elif self._model_type == "multilayer_self_attention":
            return MultiLayerSelfAttentionDenoiser(
                k=self._k,
                d_model=self._d_model,
                num_heads=self._num_heads,
                num_layers=self._num_layers,
                use_mlp=self._use_mlp,
                mlp_hidden_dim=self._transformer_mlp_hidden_dim,
                dropout=self._dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def get_model_name(self) -> str:
        """Get model name for visualization.

        Returns
        -------
        str
            Human-readable model name.
        """
        name_map = {
            "linear_pe": "Linear PE",
            "filter_bank": f"Filter Bank (K={self._polynomial_degree})",
            "self_attention": f"Self-Attention (d_k={self._d_k})",
            "self_attention_mlp": f"Self-Attention+MLP (d_k={self._d_k}, h={self._mlp_hidden_dim})",
            "multilayer_self_attention": (
                f"MultiLayer-SA (L={self._num_layers}, d={self._d_model}, "
                f"h={self._num_heads}, mlp={self._use_mlp})"
            ),
        }
        return name_map.get(self._model_type, self._model_type)

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging.

        Returns
        -------
        dict
            Model configuration dictionary.
        """
        config = self.model.get_config()
        config["model_type"] = self._model_type
        return config
