"""PyTorch Lightning module for GNN-based denoising."""

from typing import Any

import torch

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.base import DenoisingModel
from tmgg.models.factory import create_model


class GNNDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for GNN-based graph denoising."""

    def __init__(
        self,
        model_type: str = "GNN",
        num_layers: int = 1,
        num_terms: int = 4,
        feature_dim_in: int = 20,
        feature_dim_out: int = 20,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: dict[str, Any] | None = None,
        eval_noise_levels: list[float] | None = None,
        noise_type: str = "digress",
        rotation_k: int = 20,
        seed: int | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ):
        """
        Initialize the Lightning module.

        Args:
            model_type: Type of GNN model ("GNN", "NodeVarGNN", "GNNSymmetric")
            num_layers: Number of GNN layers
            num_terms: Number of polynomial terms
            feature_dim_in: Input feature dimension
            feature_dim_out: Output feature dimension
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCEWithLogits")
            scheduler_config: Optional scheduler configuration
            eval_noise_levels: Noise levels for evaluation (overrides datamodule's)
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
            rotation_k: Dimension for rotation noise skew matrix
            seed: Random seed for reproducible noise generation
        """
        # Populate self.hparams before super().__init__() so that subclass
        # params are captured. See the Template Method contract on _make_model.
        self.save_hyperparameters()

        super().__init__(
            model_type=model_type,
            num_layers=num_layers,
            num_terms=num_terms,
            feature_dim_in=feature_dim_in,
            feature_dim_out=feature_dim_out,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            eval_noise_levels=eval_noise_levels,
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed,
            **kwargs,
        )

    def _make_model(
        self,
        *args: Any,
        model_type: str = "GNN",
        num_layers: int = 1,
        num_terms: int = 4,
        feature_dim_in: int = 20,
        feature_dim_out: int = 20,
        **kwargs: Any,
    ) -> DenoisingModel:
        config: dict[str, Any] = {
            "num_layers": num_layers,
            "num_terms": num_terms,
            "feature_dim_in": feature_dim_in,
            "feature_dim_out": feature_dim_out,
        }
        model = create_model(model_type, config)
        assert isinstance(model, DenoisingModel)
        return model

    def get_model_name(self) -> str:
        return "GNN"

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Compute adjacency logits from input.

        Parameters
        ----------
        x
            Input adjacency matrix of shape (batch, n, n).
        t
            Diffusion timestep tensor (unused by GNN models).

        Returns
        -------
        torch.Tensor
            Adjacency logits (pre-sigmoid) of shape (batch, n, n).
        """
        # GNN models are not timestep-conditioned. The t parameter is
        # accepted for interface compatibility but intentionally discarded.
        return self.model(x)
