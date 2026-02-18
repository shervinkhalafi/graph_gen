"""Lightning module for generative graph modeling with MMD evaluation.

This module implements discrete denoising diffusion for graph generation:
- Training: Sample timestep, add noise, predict clean graph
- Sampling: Iterative denoising from noise to clean graphs
- Evaluation: MMD metrics on graph-theoretic properties
"""

from __future__ import annotations

from typing import Any, Literal, cast, override

import networkx as nx
import pytorch_lightning as pl
import torch
import torch.nn as nn

from tmgg.experiment_utils.data.noise_generators import (
    NoiseGenerator,
    create_noise_generator,
)
from tmgg.experiment_utils.mmd_evaluator import MMDEvaluator
from tmgg.experiment_utils.mmd_metrics import adjacency_to_networkx
from tmgg.experiment_utils.model_logging import log_parameter_count
from tmgg.experiment_utils.optimizer_config import (
    OptimizerLRSchedulerConfig,
    SchedulerInfo,
    configure_optimizers_from_config,
)
from tmgg.experiment_utils.sampling import get_noise_schedule
from tmgg.models.base import DenoisingModel
from tmgg.models.factory import create_model


class GenerativeLightningModule(pl.LightningModule):
    """Generative graph model with discrete denoising diffusion and MMD evaluation.

    This module supports all existing denoising architectures for ablation studies
    on generative graph modeling. Training uses a diffusion-style denoising
    objective, and evaluation computes MMD metrics on graph-theoretic properties.

    Parameters
    ----------
    model_type
        Architecture to use. Must be a key in ``MODEL_REGISTRY`` (see
        ``tmgg.models.factory``). Common choices: "self_attention",
        "multilayer_attention", "gnn", "gnn_sym", "hybrid", "bilinear".
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
    amsgrad
        Whether to enable AMSGrad variant.
    scheduler_config
        Scheduler configuration dict, or None to disable scheduling.
        Supports types: "cosine", "cosine_warmup", "step".
    visualization_interval
        Steps between logged visualizations.
    """

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
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,
        visualization_interval: int = 100,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.model_config = model_config or {}
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_schedule_type = noise_schedule
        self.noise_type = noise_type
        self.mmd_kernel = mmd_kernel
        self.mmd_sigma = mmd_sigma
        self.eval_num_samples = eval_num_samples
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type.lower()

        self.amsgrad = amsgrad
        self.scheduler_config = scheduler_config
        self.visualization_interval = visualization_interval

        # Build noise schedule
        self.noise_schedule = get_noise_schedule(noise_schedule, num_diffusion_steps)

        # Create noise generator
        self.noise_generator: NoiseGenerator = create_noise_generator(
            noise_type=noise_type,
            rotation_k=self.model_config.get("k", 20),
        )

        self.model: DenoisingModel = self._make_model()

        # Loss function
        if loss_type == "MSE":
            self.criterion: nn.Module = nn.MSELoss()
        elif loss_type == "BCEWithLogits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # MMD evaluation
        self.mmd_evaluator = MMDEvaluator(
            eval_num_samples=eval_num_samples,
            kernel=mmd_kernel,
            sigma=mmd_sigma,
        )

        # Scheduler info for logging (set by configure_optimizers)
        self._scheduler_info: SchedulerInfo | None = None

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        """Configure optimizers and learning rate schedulers."""
        result, scheduler_info = configure_optimizers_from_config(
            self,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
            amsgrad=self.amsgrad,
            scheduler_config=self.scheduler_config,
        )
        self._scheduler_info = scheduler_info
        return result

    def _make_model(self) -> DenoisingModel:
        """Instantiate the model via the shared factory.

        Returns
        -------
        DenoisingModel
            Configured denoising model.
        """
        model = create_model(self.model_type, self.model_config)
        assert isinstance(model, DenoisingModel)
        return model

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

        # Sample per-element timesteps and noise levels
        _, noise_levels = self._sample_timestep(batch_size)

        # Apply noise per element — NoiseGenerator.add_noise takes a scalar eps,
        # so we loop over the batch to give each graph its own noise level.
        noisy_parts = [
            self.noise_generator.add_noise(
                batch[i : i + 1], float(noise_levels[i].item())
            )
            for i in range(batch_size)
        ]
        noisy = torch.cat(noisy_parts, dim=0)

        # Forward pass
        output = self.forward(noisy)

        # Compute loss (DenoisingModel methods are resolved via Lightning's __getattr__)
        output_for_loss, target_for_loss = self.model.transform_for_loss(output, target)  # pyright: ignore[reportCallIssue]
        loss = self.criterion(output_for_loss, target_for_loss)

        # Compute accuracy
        with torch.no_grad():
            predictions = self.model.logits_to_graph(output)  # pyright: ignore[reportCallIssue]
            accuracy = (predictions == target).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/noise_level",
            float(noise_levels.mean().item()),
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss}

    def _eval_step(self, batch: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        """Shared evaluation logic for validation and test steps.

        Parameters
        ----------
        batch
            Batch of clean adjacency matrices.
        prefix
            Logging prefix, e.g. ``"val"`` or ``"test"``.

        Returns
        -------
        dict
            Dictionary with ``{prefix}_loss`` key.
        """
        target = batch

        # Accumulate reference graphs for MMD evaluation
        self.mmd_evaluator.set_num_nodes(batch.shape[1])
        for i in range(batch.shape[0]):
            self.mmd_evaluator.accumulate(adjacency_to_networkx(batch[i]))

        # Compute loss at a fixed noise level (mid-schedule)
        mid_eps = float(self.noise_schedule[self.num_diffusion_steps // 2])
        noisy = self.noise_generator.add_noise(batch, mid_eps)
        output = self.forward(noisy)

        output_for_loss, target_for_loss = self.model.transform_for_loss(output, target)  # pyright: ignore[reportCallIssue]
        loss = self.criterion(output_for_loss, target_for_loss)

        with torch.no_grad():
            predictions = self.model.logits_to_graph(output)  # pyright: ignore[reportCallIssue]
            accuracy = (predictions == target).float().mean()

        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/accuracy", accuracy, on_step=False, on_epoch=True)

        return {f"{prefix}_loss": loss}

    def _eval_epoch_end(self, prefix: str) -> None:
        """Shared epoch-end logic: generate graphs and compute MMD metrics.

        Parameters
        ----------
        prefix
            Logging prefix, e.g. ``"val"`` or ``"test"``.
        """
        num_nodes = self.mmd_evaluator.num_nodes
        if self.mmd_evaluator.num_ref_graphs < 2 or num_nodes is None:
            self.mmd_evaluator.clear()
            return

        num_to_generate = min(self.mmd_evaluator.num_ref_graphs, self.eval_num_samples)
        gen_graphs = self._sample_graphs_for_eval(num_to_generate, num_nodes)
        results = self.mmd_evaluator.evaluate(gen_graphs)

        if results is not None:
            self.log(f"{prefix}/degree_mmd", results.degree_mmd, prog_bar=True)
            self.log(f"{prefix}/clustering_mmd", results.clustering_mmd)
            self.log(f"{prefix}/spectral_mmd", results.spectral_mmd)

    @override
    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Validation step: accumulate reference graphs for MMD evaluation."""
        return self._eval_step(batch, "val")

    @override
    def on_validation_epoch_end(self) -> None:
        """Generate graphs and compute MMD metrics at end of validation."""
        self._eval_epoch_end("val")

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        """Test step: accumulate reference graphs for MMD evaluation."""
        return self._eval_step(batch, "test")

    @override
    def on_test_epoch_end(self) -> None:
        """Compute MMD metrics at end of test epoch."""
        self._eval_epoch_end("test")

    def _sample_graphs_for_eval(
        self,
        num_graphs: int,
        num_nodes: int,
    ) -> list[nx.Graph[Any]]:
        """Generate graphs via heuristic denoising and convert to NetworkX."""
        generated = self.sample(num_graphs=num_graphs, num_nodes=num_nodes)
        return [adjacency_to_networkx(g) for g in generated]

    @torch.no_grad()
    def sample(
        self,
        num_graphs: int,
        num_nodes: int,
        num_steps: int | None = None,
    ) -> list[torch.Tensor]:
        """Generate graphs by iterative denoising.

        .. warning::
            This method uses a heuristic interpolation ``z_t = alpha * pred +
            (1-alpha) * z_t`` that has **no theoretical grounding** in the
            diffusion framework. For principled ancestral sampling, use
            ``DiscreteDiffusionLightningModule.sample_batch()`` which
            implements proper posterior-based reverse diffusion (Vignac et al.
            2023, Ho et al. 2020). Results from this sampler should **not** be
            directly compared against the discrete diffusion sampler.

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
            predictions = self.model.predict(logits)  # pyright: ignore[reportCallIssue]

            # HEURISTIC: Linear interpolation, NOT proper diffusion posterior.
            # See docstring warning. For correct sampling, use
            # DiscreteDiffusionLightningModule.sample_batch().
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
        final = self.model.logits_to_graph(logits)  # pyright: ignore[reportCallIssue]
        # Enforce symmetry (DiGress-style averaging)
        final = (final + final.transpose(-2, -1)) / 2
        final = (final > 0.5).float()
        # Enforce zero diagonal
        final = final * (1 - torch.eye(num_nodes, device=device))

        return [final[i] for i in range(num_graphs)]

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Log model parameter count at training start."""
        log_parameter_count(self.model, self.get_model_name(), self.logger)

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

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
            "bilinear": "Bilinear",
        }
        return name_map.get(self.model_type, self.model_type)

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging."""
        return {
            "model_type": self.model_type,
            **self.model_config,
        }
