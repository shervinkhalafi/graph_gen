"""Lightning module for baseline denoising experiments.

Baseline models serve as sanity checks: if these simple models cannot learn,
the training pipeline itself has issues. If they learn but spectral models
don't, the issue lies in the spectral architecture.
"""

from typing import Any, Dict

import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.baselines import LinearBaseline, MLPBaseline


class BaselineLightningModule(DenoisingLightningModule):
    """Lightning module for baseline denoising models.

    Supports two baseline architectures for sanity checking:
    - linear: Simple W @ A @ W.T + b transformation
    - mlp: Flatten -> MLP -> reshape

    Parameters
    ----------
    model_type : str
        Architecture to use: "linear" or "mlp".
    max_nodes : int
        Maximum number of nodes in graphs.
    hidden_dim : int, optional
        Hidden layer size for MLP. Default 256.
    num_layers : int, optional
        Number of hidden layers for MLP. Default 2.
    learning_rate : float, optional
        Learning rate. Default 1e-3.
    weight_decay : float, optional
        Weight decay for AdamW. Default 1e-2.
    optimizer_type : str, optional
        Optimizer: "adam" or "adamw". Default "adamw".
    **kwargs
        Additional arguments passed to base class.

    Examples
    --------
    >>> module = BaselineLightningModule(
    ...     model_type="mlp",
    ...     max_nodes=32,
    ...     hidden_dim=256,
    ...     learning_rate=1e-3,
    ... )
    """

    VALID_MODEL_TYPES = {"linear", "mlp"}

    def __init__(
        self,
        model_type: str = "linear",
        max_nodes: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
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
        self._max_nodes = max_nodes
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            **kwargs,
        )

    def _make_model(self, *args: Any, **kwargs: Any) -> nn.Module:
        """Instantiate the baseline model based on config.

        Returns
        -------
        nn.Module
            Configured baseline model.
        """
        if self._model_type == "linear":
            return LinearBaseline(max_nodes=self._max_nodes)
        elif self._model_type == "mlp":
            return MLPBaseline(
                max_nodes=self._max_nodes,
                hidden_dim=self._hidden_dim,
                num_layers=self._num_layers,
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
            "linear": "Linear Baseline",
            "mlp": f"MLP Baseline (h={self._hidden_dim})",
        }
        return name_map.get(self._model_type, self._model_type)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for logging.

        Returns
        -------
        dict
            Model configuration dictionary.
        """
        config = self.model.get_config()
        config["model_type"] = self._model_type
        return config
