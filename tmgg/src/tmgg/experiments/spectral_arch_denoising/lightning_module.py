"""Lightning module for spectral denoising experiments.

This module provides a unified interface for training all spectral denoising
architectures (Linear PE, Graph Filter Bank, Self-Attention) through a single
configurable Lightning module.
"""

from typing import Any

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.base import DenoisingModel
from tmgg.models.factory import create_model


class SpectralDenoisingLightningModule(DenoisingLightningModule):
    """Unified Lightning module for spectral denoising architectures.

    This module dispatches to different spectral denoising models based on
    the `model_type` configuration parameter.

    All models output raw logits. Use model.predict() for probabilities.

    Parameters
    ----------
    model_type : str
        Architecture to use. Must be a key in ``MODEL_REGISTRY`` (see
        ``tmgg.models.factory``). Typical spectral choices:
        "linear_pe", "filter_bank", "self_attention", "self_attention_mlp",
        "multilayer_self_attention".
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
        # Shrinkage wrapper parameters
        shrinkage_max_rank: int = 50,
        shrinkage_aggregation: str = "mean",
        shrinkage_hidden_dim: int = 128,
        shrinkage_mlp_layers: int = 2,
        # Optimizer parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        **kwargs: Any,
    ):
        # Populate self.hparams before super().__init__() because
        # _make_model (called during super().__init__) reads from it.
        # See the Template Method contract on _make_model.
        self.save_hyperparameters()

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            **kwargs,
        )

    def _make_model(self, *args: Any, **kwargs: Any) -> DenoisingModel:
        """Instantiate the spectral denoising model via the shared factory.

        Reads all architecture parameters from ``self.hparams``, which the
        subclass ``__init__`` populates via ``save_hyperparameters()`` before
        calling ``super().__init__()``.

        Returns
        -------
        DenoisingModel
            Configured spectral denoiser model.
        """
        hp = self.hparams
        config: dict[str, Any] = {
            "k": hp["k"],
            "max_nodes": hp["max_nodes"],
            "use_bias": hp["use_bias"],
            "polynomial_degree": hp["polynomial_degree"],
            "d_k": hp["d_k"],
            "mlp_hidden_dim": hp["mlp_hidden_dim"],
            "mlp_num_layers": hp["mlp_num_layers"],
            "d_model": hp["d_model"],
            "num_heads": hp["num_heads"],
            "num_layers": hp["num_layers"],
            "use_mlp": hp["use_mlp"],
            "transformer_mlp_hidden_dim": hp["transformer_mlp_hidden_dim"],
            "dropout": hp["dropout"],
            "shrinkage_max_rank": hp["shrinkage_max_rank"],
            "shrinkage_aggregation": hp["shrinkage_aggregation"],
            "shrinkage_hidden_dim": hp["shrinkage_hidden_dim"],
            "shrinkage_mlp_layers": hp["shrinkage_mlp_layers"],
        }
        model = create_model(hp["model_type"], config)
        assert isinstance(model, DenoisingModel)
        return model

    def get_model_name(self) -> str:
        """Get model name for visualization.

        Returns
        -------
        str
            Human-readable model name.
        """
        hp = self.hparams
        name_map = {
            "linear_pe": "Linear PE",
            "filter_bank": f"Filter Bank (K={hp['polynomial_degree']})",
            "self_attention": f"Self-Attention (d_k={hp['d_k']})",
            "self_attention_mlp": f"Self-Attention+MLP (d_k={hp['d_k']}, h={hp['mlp_hidden_dim']})",
            "multilayer_self_attention": (
                f"MultiLayer-SA (L={hp['num_layers']}, d={hp['d_model']}, "
                f"h={hp['num_heads']}, mlp={hp['use_mlp']})"
            ),
        }
        return name_map.get(hp["model_type"], hp["model_type"])

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging.

        Returns
        -------
        dict
            Model configuration dictionary.
        """
        config = self.model.get_config()
        config["model_type"] = self.hparams["model_type"]
        return config
