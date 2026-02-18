"""Lightning module for baseline denoising experiments.

Baseline models serve as sanity checks: if these simple models cannot learn,
the training pipeline itself has issues. If they learn but spectral models
don't, the issue lies in the spectral architecture.
"""

from typing import Any

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.base import DenoisingModel
from tmgg.models.factory import create_model


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
        """Instantiate the baseline model via the shared factory.

        Reads architecture parameters from ``self.hparams``.

        Returns
        -------
        DenoisingModel
            Configured baseline model.
        """
        hp = self.hparams
        config: dict[str, Any] = {
            "max_nodes": hp["max_nodes"],
            "hidden_dim": hp["hidden_dim"],
            "num_layers": hp["num_layers"],
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
            "linear": "Linear Baseline",
            "mlp": f"MLP Baseline (h={hp['hidden_dim']})",
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
