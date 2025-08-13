# TODO: add the criterion, training setup etc. as an inheritable thing
# then just change model (define setup hook?) since we want to study architectures
"""Base Lightning module for denoising experiments."""

import abc
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch as pt
import torch.nn as nn

from tmgg.experiment_utils import (
    compute_batch_metrics,
    create_graph_denoising_figure,
    create_noise_generator,
)
from tmgg.experiment_utils.logging import log_figure


class DenoisingLightningModule(pl.LightningModule, abc.ABC):
    """PyTorch Lightning module for attention-based graph denoising."""

    def __init__(
        self,
        *args,
        dropout: float = 0.0,
        bias: bool = True,
        learning_rate: float = 0.001,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: Optional[int] = None,
        domain: str = "standard",
        **kwargs,
    ):
        """
        Initialize the Lightning module.

        Args:
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration
            noise_levels: List of noise levels to sample from during training
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
            rotation_k: Dimension for rotation noise skew matrix (number of eigenvectors)
            seed: Random seed for reproducible noise generation
            domain: Domain for adjacency matrix processing ("standard" or "inv-sigmoid")
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model: nn.Module = self._make_model(
            *args,
            bias=bias,
            dropout=dropout,
            learning_rate=learning_rate,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            noise_levels=noise_levels,
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed,
            domain=domain,
            **kwargs,
        )
        # Loss function
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "BCE":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Store configuration
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2, 0.3]
        self.noise_type = noise_type

        # Create noise generator
        self.noise_generator = create_noise_generator(
            noise_type=noise_type, rotation_k=rotation_k, seed=seed
        )

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

    @abstractmethod
    def _make_model(self, *args: Any, **kwargs: Any) -> nn.Module:  # pyright: ignore[reportExplicitAny, reportAny]
        """
        Instantiate the model based on the config
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention model.

        Args:
            x: Input adjacency matrix

        Returns:
            Reconstructed adjacency matrix
        """
        # Model returns reconstructed adjacency matrix directly
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index

        Returns:
            Training loss
        """
        # Sample random noise level for this batch
        eps = np.random.choice(self.noise_levels)

        # Add noise using noise generator
        batch_noisy = self.noise_generator.add_noise(batch, eps)

        # Forward pass using noisy adjacency matrix
        output = self.forward(batch_noisy)

        # Transform target to match output domain and compute loss
        # Note: During training with inv-sigmoid domain, the model returns raw logits
        # and _apply_target_transform converts the target to logit space for consistent loss computation
        target = self.model._apply_target_transform(batch)
        loss = self.criterion(output, target)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_noise_level", eps, on_step=False, on_epoch=True)

        return loss

    def _val_or_test(self, mode: str, batch):
        # Evaluate  across all noise levels
        mode_loss_mean = 0.0
        batch_metrics_mean = defaultdict(lambda: 0.0)
        N = len(self.noise_levels)
        for eps in self.noise_levels:
            # Add noise using noise generator
            batch_noisy = self.noise_generator.add_noise(batch, eps)

            # Forward pass
            output = self.forward(batch_noisy)

            # For loss computation, ensure both output and target are in the same domain as training
            # This is crucial for comparable loss values between training and validation
            if self.model.domain == "inv-sigmoid":
                # During validation, the model returns probabilities (sigmoid applied)
                # but we need to compute loss in logit space to match training loss scale
                output_for_loss = torch.logit(output)
                target_for_loss = torch.logit(batch)
                mode_loss = self.criterion(output_for_loss, target_for_loss)
            else:
                # Standard domain: use output and target as-is
                mode_loss = self.criterion(output, batch)

            # Compute reconstruction metrics (output already has appropriate transformations applied in forward)
            batch_metrics = compute_batch_metrics(batch, output)

            # Log metrics
            self.log(
                f"{mode}_{eps}/loss",
                mode_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            for metric_name, value in batch_metrics.items():
                self.log(
                    f"{mode}_{eps}/{metric_name}", value, on_step=False, on_epoch=True
                )
            mode_loss_mean += mode_loss / N
            for k, v in batch_metrics.items():
                batch_metrics_mean[k] += v / N

        # Log metrics
        self.log(
            f"{mode}/loss", mode_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        for metric_name, value in batch_metrics_mean.items():
            self.log(f"{mode}/{metric_name}", value, on_step=False, on_epoch=True)
        mode_loss_mean: pt.Tensor
        batch_metrics_mean: dict[str, pt.Tensor]
        return {f"{mode}_loss": mode_loss_mean, **batch_metrics_mean}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.

        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index

        Returns:
            Dictionary of test metrics
        """
        return self._val_or_test(mode="test", batch=batch)

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        return self._val_or_test(mode="val", batch=batch)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch for visualization."""
        if self.current_epoch % 10 == 0:  # Log visualizations every 10 epochs
            self._log_visualizations("val")

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch for visualization."""
        self._log_visualizations("test")

    def _log_visualizations(self, stage: str) -> None:
        """
        Log visualizations to configured loggers.

        Args:
            stage: Stage name ("val" or "test")
        """
        if not self.logger or self.trainer.sanity_checking:
            return

        # Get a sample from the appropriate dataloader
        if stage == "val":
            dataloader = self.trainer.datamodule.val_dataloader()
        elif stage == "test":
            dataloader = self.trainer.datamodule.test_dataloader()
        else:
            return  # Should not happen

        # Get a single batch and take the first sample for visualization
        batch = next(iter(dataloader))
        A_sample = batch[0]

        # Use the new flexible visualization
        noise_levels = self.trainer.datamodule.noise_levels
        noise_type = getattr(self.trainer.datamodule, "noise_type", "Digress")

        # Create denoise function that properly handles the model output
        def denoise_fn(A_noisy):
            with torch.no_grad():
                self.eval()
                A_input = A_noisy.to(self.device)
                if A_input.ndim == 2:
                    A_input = A_input.unsqueeze(0)
                output = self.forward(A_input)
                return output.squeeze(0)

        # Log visualization for each noise level
        for eps in noise_levels:
            try:
                fig = create_graph_denoising_figure(
                    A_clean=A_sample,
                    noise_fn=self.noise_generator.add_noise,
                    denoise_fn=denoise_fn,
                    noise_level=eps,
                    noise_type=noise_type,
                    title_prefix="Attention - ",
                )
                plot_name = f"{stage}_denoising_{noise_type}_eps_{eps:.3f}"
                log_figure(
                    self.logger,
                    plot_name,
                    fig,
                    global_step=self.current_epoch,
                )
                print(f"Logged individual plot: {plot_name}")
            except Exception as e:
                print(f"Failed to create/log plot for eps={eps}: {e}")
                import traceback
                traceback.print_exc()

        # Also create multi-noise visualization
        try:
            from tmgg.experiment_utils.plotting import create_multi_noise_visualization

            multi_fig = create_multi_noise_visualization(
                A_sample,
                self,
                self.noise_generator.add_noise,
                noise_levels,
                self.device,
            )
            overview_name = f"{stage}_multi_noise_{noise_type}_overview"
            log_figure(
                self.logger,
                overview_name,
                multi_fig,
                global_step=self.current_epoch,
            )
            print(f"Logged multi-noise overview: {overview_name}")
        except Exception as e:
            print(f"Failed to create/log multi-noise overview: {e}")
            import traceback
            traceback.print_exc()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.scheduler_config is None:
            return optimizer

        scheduler_type = self.scheduler_config.get("type", "cosine")

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_config.get("T_0", 20),
                T_mult=self.scheduler_config.get("T_mult", 2),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get("step_size", 50),
                gamma=self.scheduler_config.get("gamma", 0.1),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for logging."""
        return self.model.get_config()
