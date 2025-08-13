"""PyTorch Lightning module for Digress GraphTransformer-based denoising."""

from typing import Any, Dict, List, Optional

import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.digress.transformer_model import GraphTransformer


class DigressDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for Digress GraphTransformer-based graph denoising."""

    def __init__(
        self,
        use_eigenvectors: bool = False,
        node_feature_dim: int = 20,
        n_layers: int = 4,
        hidden_mlp_dims: Optional[Dict[str, int]] = None,
        hidden_dims: Optional[Dict[str, int]] = None,
        output_dims: Optional[Dict[str, int]] = None,
        learning_rate: float = 0.001,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
    ):
        super().__init__(
            use_eigenvectors=use_eigenvectors,
            node_feature_dim=node_feature_dim,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            learning_rate=learning_rate,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            noise_levels=noise_levels,
            noise_type=noise_type,
        )

    def _make_model(
        self,
        *args,
        n_layers: int = 4,
        hidden_mlp_dims: Optional[Dict[str, int]] = None,
        hidden_dims: Optional[Dict[str, int]] = None,
        output_dims: Optional[Dict[str, int]] = None,
        node_feature_dim: int = 20,
        use_eigenvectors: bool = False,
        **kwargs,
    ) -> nn.Module:
        """

        Defaults:
            hidden_mlp_dims:
            X: 256
            E: 64
            y: 256
            hidden_dims:
            dx: 128
            de: 32
            dy: 128
            n_head: 4
            output_dims:
            X: 0
            E: 1
            y: 0
        """
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {"X": 256, "E": 64, "y": 256}
        if hidden_dims is None:
            hidden_dims = {"dx": 128, "de": 32, "dy": 128, "n_head": 4}
        if output_dims is None:
            output_dims = {"X": 0, "E": 1, "y": 0}
        input_dims = {
            "X": node_feature_dim if use_eigenvectors else 1,
            "E": 1,
            "y": 0,
        }

        return GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )
