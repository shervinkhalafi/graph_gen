"""PyTorch Lightning module for Digress GraphTransformer-based denoising."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.digress.transformer_model import GraphTransformer
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer


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
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: Optional[int] = None,
        # Config-only params (used for experiment naming, not model behavior)
        digress_arch: Optional[str] = None,
        digress_mode: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        # Store before super().__init__ since _make_model is called there
        self._use_eigenvectors = use_eigenvectors
        self._node_feature_dim = node_feature_dim

        super().__init__(
            use_eigenvectors=use_eigenvectors,
            node_feature_dim=node_feature_dim,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            noise_levels=noise_levels,
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed,
        )

        # Initialize eigenvector layer if using eigenvector features
        if use_eigenvectors:
            self.eigen_layer = TopKEigenLayer(k=node_feature_dim)
        else:
            self.eigen_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, optionally using eigenvector features.

        Parameters
        ----------
        x
            Noisy adjacency matrix of shape (batch, n, n).

        Returns
        -------
        torch.Tensor
            Denoised adjacency matrix.
        """
        if self._use_eigenvectors and self.eigen_layer is not None:
            # Extract eigenvectors as node features
            V, _ = self.eigen_layer(x)  # V: (batch, n, k)
            return self.model(V)
        return self.model(x)

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
        """Create the DiGress GraphTransformer model.

        Parameters
        ----------
        n_layers
            Number of transformer layers.
        hidden_mlp_dims
            Hidden MLP dimensions. Defaults to {"X": 256, "E": 64, "y": 256}.
        hidden_dims
            Transformer dimensions. Defaults to {"dx": 128, "de": 32, "dy": 128, "n_head": 4}.
        output_dims
            Output dimensions. Defaults to {"X": 0, "E": 1, "y": 0}.
        node_feature_dim
            Dimension of node features when use_eigenvectors=True.
        use_eigenvectors
            If True, model receives eigenvector features (bs, n, k) from forward().
            If False, model receives adjacency matrix (bs, n, n) directly.
            This sets assume_adjacency_input=not use_eigenvectors on the model.
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
            assume_adjacency_input=not use_eigenvectors,
        )

    def get_model_name(self) -> str:
        """Get the name of the model for visualization purposes."""
        return "Digress"
