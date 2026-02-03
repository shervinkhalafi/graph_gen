"""PyTorch Lightning module for Digress GraphTransformer-based denoising."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tmgg.experiment_utils.base_lightningmodule import DenoisingLightningModule
from tmgg.models.digress.transformer_model import GraphTransformer


class DigressDenoisingLightningModule(DenoisingLightningModule):
    """PyTorch Lightning module for Digress GraphTransformer-based graph denoising.

    Eigenvector extraction (when use_eigenvectors=True) is handled internally
    by the GraphTransformer model, matching the architecture of spectral denoisers.
    """

    def __init__(
        self,
        use_eigenvectors: bool = False,
        k: int = 20,
        n_layers: int = 4,
        hidden_mlp_dims: dict[str, int] | None = None,
        hidden_dims: dict[str, int] | None = None,
        output_dims: dict[str, int] | None = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: dict[str, Any] | None = None,
        noise_levels: list[float] | None = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: int | None = None,
        # Config-only params (used for experiment naming, not model behavior)
        digress_arch: str | None = None,
        digress_mode: str | None = None,
        model_type: str | None = None,
        # Legacy parameter name (deprecated, use k instead)
        node_feature_dim: int | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ):
        # Handle legacy parameter name
        if node_feature_dim is not None:
            k = node_feature_dim

        # Store before super().__init__ since _make_model is called there
        self._use_eigenvectors = use_eigenvectors
        self._k = k

        super().__init__(
            use_eigenvectors=use_eigenvectors,
            k=k,
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
            **kwargs,
        )

    # forward() is inherited from base class - model handles eigenvector extraction

    def _make_model(
        self,
        *args,
        n_layers: int = 4,
        hidden_mlp_dims: dict[str, int] | None = None,
        hidden_dims: dict[str, int] | None = None,
        output_dims: dict[str, int] | None = None,
        k: int = 20,
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
        k
            Number of eigenvectors to extract when use_eigenvectors=True.
        use_eigenvectors
            If True, model extracts eigenvectors from adjacency internally.
            If False, model receives adjacency matrix directly.
        """
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {"X": 256, "E": 64, "y": 256}
        if hidden_dims is None:
            hidden_dims = {"dx": 128, "de": 32, "dy": 128, "n_head": 4}
        if output_dims is None:
            output_dims = {"X": 0, "E": 1, "y": 0}
        input_dims = {
            "X": k if use_eigenvectors else 1,
            "E": 1,
            "y": 0,
        }

        return GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            use_eigenvectors=use_eigenvectors,
            k=k if use_eigenvectors else None,
        )

    def get_model_name(self) -> str:
        """Get the name of the model for visualization purposes."""
        return "Digress"

    @torch.no_grad()
    def sample(
        self,
        num_graphs: int,
        num_nodes: int,
        num_steps: int = 100,
        noise_schedule: str = "cosine",
    ) -> list[torch.Tensor]:
        """Generate graphs by iterative denoising from noise.

        Implements discrete denoising diffusion sampling: starts from random
        binary graphs and iteratively denoises using the learned model.

        Parameters
        ----------
        num_graphs
            Number of graphs to generate.
        num_nodes
            Number of nodes per graph.
        num_steps
            Number of denoising steps.
        noise_schedule
            Noise schedule type: "linear", "cosine", or "quadratic".

        Returns
        -------
        list[torch.Tensor]
            List of generated adjacency matrices, each (num_nodes, num_nodes).
        """
        self.eval()
        device = next(self.parameters()).device

        # Build noise schedule
        t = np.linspace(0, 1, num_steps)
        if noise_schedule == "linear":
            schedule = t
        elif noise_schedule == "cosine":
            schedule = 1 - np.cos(t * np.pi / 2)
        elif noise_schedule == "quadratic":
            schedule = t**2
        else:
            raise ValueError(f"Unknown noise_schedule: {noise_schedule}")

        # Start from random binary graphs (50% edge probability)
        z_t = (
            torch.rand(num_graphs, num_nodes, num_nodes, device=device) > 0.5
        ).float()
        # Make symmetric
        z_t = (z_t + z_t.transpose(-2, -1)) / 2
        z_t = (z_t > 0.5).float()
        # Zero diagonal (no self-loops)
        z_t = z_t * (1 - torch.eye(num_nodes, device=device))

        # Iterative denoising
        for step in reversed(range(num_steps)):
            eps = schedule[step]

            # Forward pass to get model prediction
            output = self.forward(z_t)
            predictions = self.model.predict(output)

            # Interpolate between prediction and current state based on noise level
            # High noise (step near num_steps) → trust model less
            # Low noise (step near 0) → trust model more
            alpha = 1.0 - eps
            z_t = alpha * predictions + (1 - alpha) * z_t

            # Binarize for discrete graph
            z_t = (z_t > 0.5).float()
            # Enforce symmetry
            z_t = (z_t + z_t.transpose(-2, -1)) / 2
            z_t = (z_t > 0.5).float()
            # Enforce zero diagonal
            z_t = z_t * (1 - torch.eye(num_nodes, device=device))

        # Final prediction
        output = self.forward(z_t)
        final = self.model.logits_to_graph(output)
        # Enforce symmetry (DiGress-style averaging)
        final = (final + final.transpose(-2, -1)) / 2
        final = (final > 0.5).float()
        # Enforce zero diagonal
        final = final * (1 - torch.eye(num_nodes, device=device))

        return [final[i] for i in range(num_graphs)]
