"""Lightning module for generative graph modeling with MMD evaluation.

This module implements discrete denoising diffusion for graph generation:
- Training: Sample timestep, add noise, predict clean graph
- Sampling: Iterative denoising from noise to clean graphs
- Evaluation: MMD metrics on graph-theoretic properties
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, override

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from tmgg.experiment_utils.data.noise_generators import (
    NoiseGenerator,
    create_noise_generator,
)
from tmgg.experiment_utils.mmd_metrics import (
    adjacency_to_networkx,
    compute_mmd_metrics,
)
from tmgg.models import GNN, GNNSymmetric, SequentialDenoisingModel
from tmgg.models.base import DenoisingModel
from tmgg.models.spectral_denoisers import (
    GraphFilterBank,
    LinearPE,
    MultiLayerSelfAttentionDenoiser,
    SelfAttentionDenoiser,
    SelfAttentionDenoiserWithMLP,
)

if TYPE_CHECKING:
    pass


def get_noise_schedule(
    schedule: Literal["linear", "cosine", "quadratic"],
    num_steps: int,
) -> np.ndarray:
    """Get noise levels for each timestep.

    Parameters
    ----------
    schedule
        Schedule type.
    num_steps
        Number of diffusion steps.

    Returns
    -------
    np.ndarray
        Noise levels from 0 to 1 for each timestep.
    """
    t = np.linspace(0, 1, num_steps)

    if schedule == "linear":
        return t
    elif schedule == "cosine":
        return 1 - np.cos(t * np.pi / 2)
    elif schedule == "quadratic":
        return t**2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class GenerativeLightningModule(pl.LightningModule):
    """Generative graph model with discrete denoising diffusion and MMD evaluation.

    This module supports all existing denoising architectures for ablation studies
    on generative graph modeling. Training uses a diffusion-style denoising
    objective, and evaluation computes MMD metrics on graph-theoretic properties.

    Parameters
    ----------
    model_type
        Architecture to use. One of:
        - "linear_pe": Linear positional encoding
        - "filter_bank": Graph filter bank with spectral polynomial
        - "self_attention": Query-key attention on eigenvectors
        - "self_attention_mlp": Self-attention with MLP post-processing
        - "multilayer_attention": Stacked transformer blocks
        - "gnn": Graph neural network
        - "gnn_sym": Symmetric GNN
        - "hybrid": Sequential denoising model
    model_config
        Configuration dictionary for the model architecture.
    num_diffusion_steps
        Number of diffusion timesteps.
    noise_schedule
        Noise schedule type: "linear", "cosine", or "quadratic".
    noise_type
        Type of noise to use: "digress", "gaussian", or "rotation".
    loss_type
        Loss function: "MSE" or "BCEWithLogits".
    mmd_kernel
        Kernel for MMD computation: "gaussian" (L2) or "gaussian_tv" (DiGress).
    mmd_sigma
        Sigma for Gaussian kernel.
    eval_num_samples
        Number of graphs to generate for MMD evaluation.
    learning_rate
        Learning rate for optimizer.
    weight_decay
        Weight decay for AdamW.
    optimizer_type
        Optimizer type: "adam" or "adamw".
    """

    SUPPORTED_ARCHITECTURES = {
        "linear_pe",
        "filter_bank",
        "self_attention",
        "self_attention_mlp",
        "multilayer_attention",
        "gnn",
        "gnn_sym",
        "hybrid",
    }

    def __init__(
        self,
        model_type: str = "self_attention",
        model_config: dict[str, Any] | None = None,
        num_diffusion_steps: int = 100,
        noise_schedule: Literal["linear", "cosine", "quadratic"] = "cosine",
        noise_type: str = "digress",
        loss_type: str = "MSE",
        mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
        mmd_sigma: float = 1.0,
        eval_num_samples: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_type not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_ARCHITECTURES}, "
                f"got '{model_type}'"
            )

        self.model_type = model_type
        self.model_config = model_config or {}
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_schedule_type = noise_schedule
        self.noise_type = noise_type
        self.mmd_kernel: Literal["gaussian", "gaussian_tv"] = mmd_kernel
        self.mmd_sigma = mmd_sigma
        self.eval_num_samples = eval_num_samples
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type.lower()

        # Build noise schedule
        self.noise_schedule = get_noise_schedule(noise_schedule, num_diffusion_steps)

        # Create noise generator
        self.noise_generator: NoiseGenerator = create_noise_generator(
            noise_type=noise_type,
            rotation_k=self.model_config.get("k", 20),
        )

        # Create model (typed as DenoisingModel for method access)
        self.model: DenoisingModel = self._make_model()

        # Loss function
        if loss_type == "MSE":
            self.criterion: nn.Module = nn.MSELoss()
        elif loss_type == "BCEWithLogits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Storage for validation/test graphs
        self._ref_graphs: list[nx.Graph[Any]] = []
        self._num_nodes: int | None = None

    def _make_model(self) -> DenoisingModel:
        """Instantiate the model based on configuration.

        Returns
        -------
        DenoisingModel
            Configured denoising model.
        """
        cfg = self.model_config

        if self.model_type == "linear_pe":
            return LinearPE(
                k=cfg.get("k", 8),
                max_nodes=cfg.get("max_nodes", 200),
                use_bias=cfg.get("use_bias", True),
            )
        elif self.model_type == "filter_bank":
            return GraphFilterBank(
                k=cfg.get("k", 8),
                polynomial_degree=cfg.get("polynomial_degree", 5),
            )
        elif self.model_type == "self_attention":
            return SelfAttentionDenoiser(
                k=cfg.get("k", 8),
                d_k=cfg.get("d_k", 64),
            )
        elif self.model_type == "self_attention_mlp":
            return SelfAttentionDenoiserWithMLP(
                k=cfg.get("k", 8),
                d_k=cfg.get("d_k", 64),
                mlp_hidden_dim=cfg.get("mlp_hidden_dim", 128),
                mlp_num_layers=cfg.get("mlp_num_layers", 2),
            )
        elif self.model_type == "multilayer_attention":
            return MultiLayerSelfAttentionDenoiser(
                k=cfg.get("k", 8),
                d_model=cfg.get("d_model", 64),
                num_heads=cfg.get("num_heads", 4),
                num_layers=cfg.get("num_layers", 2),
                use_mlp=cfg.get("use_mlp", True),
                mlp_hidden_dim=cfg.get("mlp_hidden_dim"),
                dropout=cfg.get("dropout", 0.0),
            )
        elif self.model_type == "gnn":
            return GNN(
                num_layers=cfg.get("num_layers", 4),
                num_terms=cfg.get("num_terms", 3),
                feature_dim_in=cfg.get("feature_dim_in", 10),
                feature_dim_out=cfg.get("feature_dim_out", 10),
            )
        elif self.model_type == "gnn_sym":
            return GNNSymmetric(
                num_layers=cfg.get("num_layers", 4),
                num_terms=cfg.get("num_terms", 3),
                feature_dim_in=cfg.get("feature_dim_in", 10),
                feature_dim_out=cfg.get("feature_dim_out", 10),
            )
        elif self.model_type == "hybrid":
            # Hybrid requires an embedding model and denoising model
            # EigenEmbedding extracts eigenvectors, SelfAttentionDenoiser uses k of them
            from tmgg.models.layers import EigenEmbedding

            embedding = EigenEmbedding(
                eigenvalue_reg=cfg.get("eigenvalue_reg", 0.0),
            )
            denoiser = SelfAttentionDenoiser(
                k=cfg.get("k", 8),
                d_k=cfg.get("d_k", 64),
            )
            return SequentialDenoisingModel(
                embedding_model=embedding,
                denoising_model=denoiser,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x
            Noisy adjacency matrix.

        Returns
        -------
        torch.Tensor
            Model output (logits).
        """
        return cast(torch.Tensor, self.model(x))

    def _sample_timestep(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random timesteps and corresponding noise levels.

        Parameters
        ----------
        batch_size
            Number of samples in batch.

        Returns
        -------
        tuple
            (timesteps, noise_levels) as tensors.
        """
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,))
        noise_levels = torch.tensor(
            [self.noise_schedule[int(t_i.item())] for t_i in t],
            dtype=torch.float32,
        )
        return t, noise_levels

    @override
    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Training step: sample timestep, add noise, predict clean graph.

        Parameters
        ----------
        batch
            Batch of clean adjacency matrices.
        batch_idx
            Batch index.

        Returns
        -------
        dict
            Dictionary with loss.
        """
        target = batch
        batch_size = batch.shape[0]

        # Sample timestep and noise level
        _, noise_levels = self._sample_timestep(batch_size)

        # Add noise at the sampled level (use mean noise level for batch)
        eps = float(noise_levels.mean().item())
        noisy = self.noise_generator.add_noise(batch, eps)

        # Forward pass
        output = self.forward(noisy)

        # Compute loss
        output_for_loss, target_for_loss = self.model.transform_for_loss(output, target)
        loss = self.criterion(output_for_loss, target_for_loss)

        # Compute accuracy
        with torch.no_grad():
            predictions = self.model.logits_to_graph(output)
            accuracy = (predictions == target).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/noise_level", eps, on_step=False, on_epoch=True)

        return {"loss": loss}

    @override
    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Validation step: accumulate reference graphs for MMD evaluation.

        Parameters
        ----------
        batch
            Batch of clean adjacency matrices.
        batch_idx
            Batch index.

        Returns
        -------
        dict
            Dictionary with validation loss.
        """
        target = batch

        # Store number of nodes for sampling
        if self._num_nodes is None:
            self._num_nodes = batch.shape[1]

        # Accumulate reference graphs for MMD computation
        for i in range(batch.shape[0]):
            if len(self._ref_graphs) < self.eval_num_samples:
                self._ref_graphs.append(adjacency_to_networkx(batch[i]))

        # Compute validation loss at a fixed noise level (mid-schedule)
        mid_eps = float(self.noise_schedule[self.num_diffusion_steps // 2])
        noisy = self.noise_generator.add_noise(batch, mid_eps)
        output = self.forward(noisy)

        output_for_loss, target_for_loss = self.model.transform_for_loss(output, target)
        loss = self.criterion(output_for_loss, target_for_loss)

        with torch.no_grad():
            predictions = self.model.logits_to_graph(output)
            accuracy = (predictions == target).float().mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True)

        return {"val_loss": loss}

    @override
    def on_validation_epoch_end(self) -> None:
        """Generate graphs and compute MMD metrics at end of validation."""
        if len(self._ref_graphs) < 2 or self._num_nodes is None:
            self._ref_graphs.clear()
            return

        # Generate graphs
        num_to_generate = min(len(self._ref_graphs), self.eval_num_samples)
        generated = self.sample(
            num_graphs=num_to_generate,
            num_nodes=self._num_nodes,
        )

        # Convert to NetworkX graphs
        gen_graphs = [adjacency_to_networkx(g) for g in generated]

        # Compute MMD metrics
        mmd_results = compute_mmd_metrics(
            self._ref_graphs[:num_to_generate],
            gen_graphs,
            kernel=self.mmd_kernel,
            sigma=self.mmd_sigma,
        )

        # Log MMD metrics
        self.log("val/degree_mmd", mmd_results.degree_mmd, prog_bar=True)
        self.log("val/clustering_mmd", mmd_results.clustering_mmd)
        self.log("val/spectral_mmd", mmd_results.spectral_mmd)

        # Clear reference graphs for next epoch
        self._ref_graphs.clear()

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        """Test step: same as validation."""
        return self.validation_step(batch, batch_idx)

    @override
    def on_test_epoch_end(self) -> None:
        """Compute MMD metrics at end of test epoch."""
        self.on_validation_epoch_end()

    @torch.no_grad()
    def sample(
        self,
        num_graphs: int,
        num_nodes: int,
        num_steps: int | None = None,
    ) -> list[torch.Tensor]:
        """Generate graphs by iterative denoising.

        Parameters
        ----------
        num_graphs
            Number of graphs to generate.
        num_nodes
            Number of nodes per graph.
        num_steps
            Number of denoising steps. Defaults to num_diffusion_steps.

        Returns
        -------
        list
            List of generated adjacency matrices.
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps

        self.eval()
        device = next(self.parameters()).device

        # Start from noise (random edges with 0.5 probability)
        z_t = (
            torch.rand(num_graphs, num_nodes, num_nodes, device=device) > 0.5
        ).float()
        # Make symmetric
        z_t = (z_t + z_t.transpose(-2, -1)) / 2
        z_t = (z_t > 0.5).float()
        # Zero diagonal
        z_t = z_t * (1 - torch.eye(num_nodes, device=device))

        # Iterative denoising
        for t in reversed(range(num_steps)):
            eps = self.noise_schedule[t]

            # Forward pass
            logits = self.forward(z_t)
            predictions = self.model.predict(logits)

            # Interpolate between prediction and current based on noise level
            # At high noise (t near num_steps), trust model less
            # At low noise (t near 0), trust model more
            alpha = 1.0 - eps  # Trust model more as noise decreases
            z_t = alpha * predictions + (1 - alpha) * z_t

            # Binarize for discrete graph
            z_t = (z_t > 0.5).float()
            # Ensure symmetric and zero diagonal
            z_t = (z_t + z_t.transpose(-2, -1)) / 2
            z_t = (z_t > 0.5).float()
            z_t = z_t * (1 - torch.eye(num_nodes, device=device))

        # Final prediction
        logits = self.forward(z_t)
        final = self.model.logits_to_graph(logits)
        # Enforce symmetry (DiGress-style averaging)
        final = (final + final.transpose(-2, -1)) / 2
        final = (final > 0.5).float()
        # Enforce zero diagonal
        final = final * (1 - torch.eye(num_nodes, device=device))

        return [final[i] for i in range(num_graphs)]

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        if self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            return torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
            )

    def get_model_name(self) -> str:
        """Get human-readable model name."""
        name_map = {
            "linear_pe": "Linear PE",
            "filter_bank": "Filter Bank",
            "self_attention": "Self-Attention",
            "self_attention_mlp": "Self-Attention+MLP",
            "multilayer_attention": "MultiLayer Attention",
            "gnn": "GNN",
            "gnn_sym": "Symmetric GNN",
            "hybrid": "Hybrid",
        }
        return name_map.get(self.model_type, self.model_type)

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging."""
        return {
            "model_type": self.model_type,
            **self.model_config,
        }
