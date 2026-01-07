# TODO: add the criterion, training setup etc. as an inheritable thing
# then just change model (define setup hook?) since we want to study architectures
"""Base Lightning module for denoising experiments."""

from __future__ import annotations

import abc
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Protocol, cast, override, runtime_checkable

import matplotlib.figure
import numpy as np
import pytorch_lightning as pl
import torch
import torch as pt
import torch.nn as nn
from pytorch_lightning.loggers import Logger

from tmgg.experiment_utils import (
    compute_batch_metrics,
    create_graph_denoising_figure,
    create_network_denoising_figure,
    create_noise_generator,
)
from tmgg.experiment_utils.exceptions import ConfigurationError
from tmgg.experiment_utils.logging import log_figure


@runtime_checkable
class _DenoisingModelProtocol(Protocol):
    """Protocol defining the interface expected from denoising models.

    This protocol is used for type checking within experiment_utils without
    creating a dependency on tmgg.models, respecting module boundaries.
    """

    def parameter_count(self) -> dict[str, Any]:
        """Return hierarchical parameter counts."""
        ...

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        ...

    def transform_for_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform output and target for loss computation."""
        ...

    def predict(self, logits: torch.Tensor, zero_diag: bool = True) -> torch.Tensor:
        """Convert logits to predictions."""
        ...

    def logits_to_graph(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to binary graph predictions."""
        ...

    def parameters(self, recurse: bool = True) -> Any:  # pyright: ignore[reportExplicitAny]
        """Return an iterator over module parameters (from nn.Module)."""
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ...


class DenoisingLightningModule(pl.LightningModule, abc.ABC):
    """PyTorch Lightning module for attention-based graph denoising."""

    def __init__(
        self,
        *args,
        dropout: float = 0.0,
        bias: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: dict[str, Any] | None = None,
        eval_noise_levels: list[float] | None = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: int | None = None,
        visualization_interval: int = 5000,
        spectral_k: int = 4,
        log_spectral_deltas: bool = False,
        log_rotation_angles: bool = False,
        **kwargs,
    ):
        """
        Initialize the Lightning module.

        Args:
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization) coefficient for AdamW
            optimizer_type: Optimizer to use ("adam" or "adamw")
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration. Supports:
                - type: "cosine", "cosine_warmup", or "step"
                - T_warmup: Warmup steps (for cosine_warmup)
                - T_0, T_mult: CosineAnnealingWarmRestarts params
                - step_size, gamma: StepLR params
            eval_noise_levels: List of noise levels for evaluation. If None, uses
                the datamodule's noise_levels (same as training).
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
            rotation_k: Dimension for rotation noise skew matrix (number of eigenvectors)
            seed: Random seed for reproducible noise generation
            visualization_interval: Steps between logging visualizations (default: 5000)
            spectral_k: Number of top eigenvectors for subspace comparison in spectral
                delta metrics. Only used when log_spectral_deltas=True.
            log_spectral_deltas: If True, compute and log spectral delta metrics during
                validation/test. Tracks eigengap delta, algebraic connectivity delta,
                eigenvalue drift, and subspace distance for noisy→clean and denoised→clean.
            log_rotation_angles: If True (and log_spectral_deltas=True), also compute
                Procrustes rotation angle and residual metrics.
        """
        super().__init__()
        _ = self.save_hyperparameters()

        # Model (typed as Protocol for methods, but actual impl is nn.Module subclass)
        self.model: _DenoisingModelProtocol = self._make_model(
            *args,
            bias=bias,
            dropout=dropout,
            learning_rate=learning_rate,
            loss_type=loss_type,
            scheduler_config=scheduler_config,
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed,
            **kwargs,
        )
        # Loss function (models output logits, use BCEWithLogitsLoss for stability)
        self.criterion: nn.Module
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "BCEWithLogits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ConfigurationError(
                f"Unknown loss_type: {loss_type}. Use 'MSE' or 'BCEWithLogits'."
            )

        # Store configuration
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.optimizer_type: str = optimizer_type.lower()
        self.amsgrad: bool = amsgrad
        self.scheduler_config: dict[str, Any] | None = scheduler_config
        self._eval_noise_levels_override: list[float] | None = eval_noise_levels
        self.noise_type: str = noise_type
        self.visualization_interval: int = visualization_interval

        # Create noise generator
        from tmgg.experiment_utils.data.noise_generators import NoiseGenerator

        self.noise_generator: NoiseGenerator = create_noise_generator(
            noise_type=noise_type, rotation_k=rotation_k, seed=seed
        )

        # Metrics tracking
        self.train_metrics: list[Any] = []
        self.val_metrics: list[Any] = []

        # Spectral delta logging configuration
        self.spectral_k: int = spectral_k
        self.log_spectral_deltas: bool = log_spectral_deltas
        self.log_rotation_angles: bool = log_rotation_angles

    @property
    def noise_levels(self) -> list[float]:
        """Get noise levels from the datamodule (authoritative source).

        Raises
        ------
        RuntimeError
            If not attached to a trainer with a datamodule that has noise_levels.
        """
        dm = self.datamodule
        if dm is None:
            raise RuntimeError(
                "Cannot access noise_levels: not attached to trainer with datamodule. "
                "Ensure the module is attached to a trainer before accessing noise_levels."
            )
        levels = getattr(dm, "noise_levels", None)
        if levels is None:
            raise RuntimeError(
                f"Datamodule {type(dm).__name__} does not have noise_levels attribute. "
                "Ensure your DataModule provides 'noise_levels'."
            )
        return cast(list[float], levels)

    @property
    def eval_noise_levels(self) -> list[float]:
        """Get noise levels for evaluation.

        Returns eval_noise_levels if explicitly set, otherwise falls back to
        noise_levels (same levels for training and evaluation).
        """
        if self._eval_noise_levels_override is not None:
            return self._eval_noise_levels_override
        return self.noise_levels  # Default: same as training

    @property
    def datamodule(self) -> pl.LightningDataModule | None:
        """Access the trainer's datamodule, if attached.

        Returns None if no trainer is attached or if the trainer has no datamodule.
        Uses getattr to avoid pyright errors with Lightning's incomplete type stubs.
        """
        if self.trainer is None:
            return None
        return cast(
            pl.LightningDataModule | None, getattr(self.trainer, "datamodule", None)
        )

    def _get_datamodule(self) -> pl.LightningDataModule | None:
        """Safely get the datamodule from the trainer.

        Deprecated: Use the `datamodule` property instead.
        """
        return self.datamodule

    @abstractmethod
    def _make_model(self, *args: Any, **kwargs: Any) -> _DenoisingModelProtocol:  # pyright: ignore[reportExplicitAny, reportAny]
        """
        Instantiate the model based on the config
        """
        pass

    def get_model_name(self) -> str:
        """Get the name of the model for visualization purposes.

        Returns:
            Model name to use in plot titles and logging.
        """
        return "Base"

    def log_parameter_count(self) -> None:
        """Log the parameter count of the model in a formatted way."""
        if not hasattr(self.model, "parameter_count"):
            # Fallback for models without parameter_count method
            total_params: int = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            _ = print(f"\n{'=' * 50}")
            _ = print(f"Model: {self.get_model_name()}")
            _ = print(f"Total Trainable Parameters: {total_params:,}")
            _ = print(f"{'=' * 50}\n")

            # Log to logger if available
            if self.logger:
                _ = self.logger.log_hyperparams({"total_parameters": total_params})
            return

        # Get hierarchical parameter counts
        param_counts: dict[str, Any] = self.model.parameter_count()

        # Format and print parameter counts
        def format_counts(counts: dict[str, Any], indent: int = 0) -> list[str]:
            """Recursively format parameter counts."""
            lines: list[str] = []
            prefix: str = "  " * indent

            # Skip printing "self" and "total" at sub-levels, just show them at top level
            for key, value in counts.items():
                if key == "total" and indent == 0:
                    continue  # Will be printed separately at the top
                elif key == "self":
                    if value > 0:
                        _ = lines.append(f"{prefix}├─ {key}: {value:,}")
                elif isinstance(value, dict):
                    if "total" in value:
                        _ = lines.append(f"{prefix}├─ {key}: {value['total']:,}")
                        # Recursively format sub-counts if they exist
                        sub_items: dict[str, Any] = {
                            k: v for k, v in value.items() if k != "total"
                        }
                        if sub_items:
                            sub_lines: list[str] = format_counts(sub_items, indent + 1)
                            _ = lines.extend(sub_lines)
                    else:
                        _ = lines.append(f"{prefix}├─ {key}:")
                        sub_lines = format_counts(value, indent + 1)
                        _ = lines.extend(sub_lines)
                else:
                    _ = lines.append(f"{prefix}├─ {key}: {value:,}")

            return lines

        # Print formatted output
        _ = print(f"\n{'=' * 60}")
        _ = print(f"Model Parameter Count: {self.get_model_name()}")
        _ = print(f"{'=' * 60}")
        _ = print(f"Total Trainable Parameters: {param_counts['total']:,}")
        _ = print("-" * 60)

        formatted_lines: list[str] = format_counts(param_counts)
        for line in formatted_lines:
            _ = print(line)

        _ = print(f"{'=' * 60}\n")

        # Log to logger if available
        if self.logger:
            _ = self.logger.log_hyperparams(
                {
                    "total_parameters": param_counts["total"],
                    "parameter_breakdown": param_counts,
                }
            )

    @override
    def setup(self, stage: str) -> None:
        """Validate configuration before training/testing begins.

        Parameters
        ----------
        stage
            Either 'fit', 'validate', 'test', or 'predict'.
        """
        _ = super().setup(stage)

        # Validate datamodule provides required attributes
        dm = self.datamodule
        if dm is None:
            return  # Skip validation if no trainer/datamodule

        required_attrs: list[str] = ["noise_levels"]
        missing: list[str] = [attr for attr in required_attrs if not hasattr(dm, attr)]

        if missing:
            raise ValueError(
                f"DataModule {type(dm).__name__} missing required attributes: {missing}. "
                "Ensure your DataModule provides 'noise_levels' attribute."
            )

    @override
    def on_fit_start(self) -> None:
        """Called at the beginning of training."""
        # Log parameter count at the start of training
        _ = self.log_parameter_count()

        # Log scheduler configuration for cosine_warmup
        if (
            self.scheduler_config
            and self.scheduler_config.get("type") == "cosine_warmup"
            and hasattr(self, "_scheduler_T_warmup")
        ):
            T_warmup: int = self._scheduler_T_warmup
            T_max: int = self._scheduler_T_max
            total_steps: int = self._scheduler_estimated_total_steps

            # Estimate steps per epoch for readable output
            steps_per_epoch: int = 1
            dm = self.datamodule
            if dm is not None:
                try:
                    train_loader = dm.train_dataloader()
                    dataset_size: int = len(train_loader.dataset)
                    batch_size: int = getattr(dm, "batch_size", train_loader.batch_size)
                    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
                except Exception:
                    pass

            warmup_epochs: float = (
                T_warmup / steps_per_epoch if steps_per_epoch > 0 else 0
            )
            decay_epochs: float = T_max / steps_per_epoch if steps_per_epoch > 0 else 0
            total_epochs: float = (
                total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
            )

            _ = print(f"\n{'=' * 60}")
            _ = print("Scheduler Configuration (cosine_warmup)")
            _ = print(f"{'=' * 60}")
            _ = print(f"  Warmup:     {T_warmup:,} steps ({warmup_epochs:.1f} epochs)")
            _ = print(f"  LR at zero: {T_max:,} steps ({decay_epochs:.1f} epochs)")
            _ = print(
                f"  Total est:  {total_steps:,} steps ({total_epochs:.0f} epochs)"
            )
            _ = print(f"{'=' * 60}\n")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention model.

        Args:
            x: Input adjacency matrix

        Returns:
            Reconstructed adjacency matrix
        """
        # Model returns reconstructed adjacency matrix directly
        output: torch.Tensor = self.model(x)
        return output

    def _apply_noise(self, batch: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply noise to batch.

        Parameters
        ----------
        batch
            Clean adjacency matrices, shape (B, N, N).
        eps
            Noise level to apply.

        Returns
        -------
        torch.Tensor
            Noisy adjacency matrices, same shape as input.
        """
        return self.noise_generator.add_noise(batch, eps)

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, Any]:
        """Execute single training step with noise application.

        Parameters
        ----------
        batch
            Batch of clean adjacency matrices, shape (B, N, N).
        batch_idx
            Index of current batch.

        Returns
        -------
        dict
            Dictionary with 'loss' (required by Lightning) and 'logits' for debugging.
        """
        # Sample noise level randomly from training noise levels
        eps: float = float(np.random.choice(self.noise_levels))
        target: torch.Tensor = batch

        # Add noise
        batch_noisy: torch.Tensor = self._apply_noise(batch, eps)

        # Forward pass using noisy adjacency matrix
        output: torch.Tensor = self.forward(batch_noisy)

        # Transform output and target to match for loss computation
        # This handles domain transformations (inv-sigmoid vs standard) automatically
        output_for_loss: torch.Tensor
        target_for_loss: torch.Tensor
        output_for_loss, target_for_loss = self.model.transform_for_loss(output, target)
        loss: torch.Tensor = self.criterion(output_for_loss, target_for_loss)

        # Compute train accuracy from logits using model's thresholding
        with torch.no_grad():
            predictions: torch.Tensor = self.model.logits_to_graph(output)
            train_acc: torch.Tensor = (predictions == target).float().mean()

        # Log metrics
        _ = self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        _ = self.log(
            "train/accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        _ = self.log("train/noise_level", eps, on_step=False, on_epoch=True)

        # Return dict with loss and logits for debugging callbacks
        return {"loss": loss, "logits": output}

    def _val_or_test(self, mode: str, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        target: torch.Tensor = batch

        # Evaluate across all eval noise levels
        mode_loss_mean: float = 0.0
        batch_metrics_mean: defaultdict[str, float] = defaultdict(lambda: 0.0)
        N: int = len(self.eval_noise_levels)
        for eps in self.eval_noise_levels:
            # Add noise
            batch_noisy: torch.Tensor = self._apply_noise(batch, eps)

            # Forward pass
            output: torch.Tensor = self.forward(batch_noisy)

            # Transform output and target to match for loss computation
            # This ensures comparable loss values between training and validation
            output_for_loss: torch.Tensor
            target_for_loss: torch.Tensor
            output_for_loss, target_for_loss = self.model.transform_for_loss(
                output, target
            )
            mode_loss: torch.Tensor = self.criterion(output_for_loss, target_for_loss)

            # Compute reconstruction metrics on predictions (post-sigmoid)
            predictions: torch.Tensor = self.model.predict(output)
            batch_metrics: dict[str, float] = compute_batch_metrics(target, predictions)

            # Log metrics
            _ = self.log(
                f"{mode}_{eps}/loss",
                mode_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            for metric_name, value in batch_metrics.items():
                _ = self.log(
                    f"{mode}_{eps}/{metric_name}", value, on_step=False, on_epoch=True
                )

            # Log spectral delta metrics if enabled
            if self.log_spectral_deltas:
                self._log_spectral_deltas(mode, eps, target, batch_noisy, predictions)

            mode_loss_mean += mode_loss.item() / N
            for k, v in batch_metrics.items():
                batch_metrics_mean[k] += v / N

        # Log metrics
        _ = self.log(
            f"{mode}/loss", mode_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        for metric_name, value in batch_metrics_mean.items():
            _ = self.log(f"{mode}/{metric_name}", value, on_step=False, on_epoch=True)
        mode_loss_mean_tensor: pt.Tensor = torch.tensor(mode_loss_mean)
        batch_metrics_mean_dict: dict[str, pt.Tensor] = {
            k: torch.tensor(v) for k, v in batch_metrics_mean.items()
        }
        return {f"{mode}_loss": mode_loss_mean_tensor, **batch_metrics_mean_dict}

    def _log_spectral_deltas(
        self,
        mode: str,
        eps: float,
        A_clean: torch.Tensor,
        A_noisy: torch.Tensor,
        A_denoised: torch.Tensor,
    ) -> None:
        """Log spectral delta metrics for noisy→clean and denoised→clean comparisons.

        Computes four spectral delta metrics:
        - eigengap_delta: Relative change in spectral gap
        - alg_conn_delta: Relative change in algebraic connectivity
        - eigenvalue_drift: Relative L2 distance of eigenvalues
        - subspace_distance: Frobenius norm of projection difference

        Parameters
        ----------
        mode
            Logging mode ("val" or "test").
        eps
            Noise level.
        A_clean
            Clean adjacency matrices, shape (batch, n, n).
        A_noisy
            Noisy adjacency matrices, same shape.
        A_denoised
            Denoised (predicted) adjacency matrices, same shape.
        """
        from tmgg.experiment_utils.spectral_deltas import compute_spectral_deltas

        with torch.no_grad():
            # Compute spectral deltas: noisy vs clean
            deltas_noisy = compute_spectral_deltas(
                A_clean,
                A_noisy,
                k=self.spectral_k,
                compute_rotation=self.log_rotation_angles,
            )
            for name, values in deltas_noisy.items():
                _ = self.log(
                    f"{mode}_{eps}/noisy_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

            # Compute spectral deltas: denoised vs clean
            deltas_denoised = compute_spectral_deltas(
                A_clean,
                A_denoised,
                k=self.spectral_k,
                compute_rotation=self.log_rotation_angles,
            )
            for name, values in deltas_denoised.items():
                _ = self.log(
                    f"{mode}_{eps}/denoised_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Test step.

        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index

        Returns:
            Dictionary of test metrics
        """
        return self._val_or_test(mode="test", batch=batch)

    @override
    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        return self._val_or_test(mode="val", batch=batch)

    @override
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation for visualization (step-based)."""
        if self.global_step % self.visualization_interval == 0:
            _ = self._log_visualizations("val")

    @override
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch for visualization."""
        _ = self._log_visualizations("test")

    def _log_visualizations(self, stage: str) -> None:
        """
        Log visualizations to configured loggers.

        Args:
            stage: Stage name ("val" or "test")
        """
        trainer = self.trainer
        if not self.logger or trainer is None or trainer.sanity_checking:
            return

        # Require datamodule for visualization
        dm = self.datamodule
        if dm is None:
            return

        # Get a sample from the appropriate dataloader
        dataloader: Any
        if stage == "val":
            dataloader = dm.val_dataloader()
        elif stage == "test":
            dataloader = dm.test_dataloader()
        else:
            return  # Should not happen

        # Get a single batch and take the first sample for visualization
        batch: torch.Tensor = next(iter(dataloader))
        A_sample: torch.Tensor = batch[0]

        # Use consistent noise_levels from property (which uses datamodule's when available)
        noise_levels: list[float] = self.noise_levels
        noise_type: str = getattr(dm, "noise_type", self.noise_type)

        # Create denoise function that returns (predictions, logits) tuple
        def denoise_fn(
            A_noisy: np.ndarray | torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                _ = self.eval()
                A_input: torch.Tensor = (
                    A_noisy.to(self.device)
                    if isinstance(A_noisy, torch.Tensor)
                    else torch.from_numpy(A_noisy).to(self.device)
                )
                if A_input.ndim == 2:
                    A_input = A_input.unsqueeze(0)
                logits: torch.Tensor = self.forward(A_input)
                predictions: torch.Tensor = self.model.predict(logits)
                return predictions.squeeze(0), logits.squeeze(0)

        # Log visualization for each noise level
        for eps in noise_levels:
            try:
                fig: matplotlib.figure.Figure = create_graph_denoising_figure(
                    A_clean=A_sample,
                    noise_fn=cast(Any, self.noise_generator.add_noise),
                    denoise_fn=denoise_fn,
                    noise_level=eps,
                    noise_type=noise_type,
                    title_prefix=f"{self.get_model_name()} - ",
                )
                plot_name: str = f"{stage}_denoising_{noise_type}_eps_{eps:.3f}"
                _ = log_figure(
                    cast(list[Logger], self.loggers),
                    plot_name,
                    fig,
                    global_step=self.global_step,
                )
                _ = print(f"Logged individual plot: {plot_name}")
            except Exception as e:
                _ = print(f"Failed to create/log plot for eps={eps}: {e}")
                import traceback

                _ = traceback.print_exc()

            # Log node-link network visualization for small graphs
            n_nodes: int = (
                A_sample.shape[0] if A_sample.ndim == 2 else A_sample.shape[-1]
            )
            if n_nodes <= 50:
                try:
                    network_fig = create_network_denoising_figure(
                        A_clean=A_sample,
                        noise_fn=cast(Any, self.noise_generator.add_noise),
                        denoise_fn=denoise_fn,
                        noise_level=eps,
                        noise_type=noise_type,
                        title_prefix=f"{self.get_model_name()} - ",
                        layout="spring",
                        show_edge_diff=True,
                    )
                    if network_fig is not None:
                        network_plot_name: str = (
                            f"{stage}_network_{noise_type}_eps_{eps:.3f}"
                        )
                        _ = log_figure(
                            cast(list[Logger], self.loggers),
                            network_plot_name,
                            network_fig,
                            global_step=self.global_step,
                        )
                        _ = print(f"Logged network plot: {network_plot_name}")
                except Exception as e:
                    _ = print(f"Failed to create/log network plot for eps={eps}: {e}")

        # Also create multi-noise visualization
        try:
            from tmgg.experiment_utils.plotting import create_multi_noise_visualization

            multi_fig: matplotlib.figure.Figure = create_multi_noise_visualization(
                A_sample,
                self,
                self.noise_generator.add_noise,
                noise_levels,
                str(self.device),
            )
            overview_name: str = f"{stage}_multi_noise_{noise_type}_overview"
            _ = log_figure(
                cast(list[Logger], self.loggers),
                overview_name,
                multi_fig,
                global_step=self.global_step,
            )
            _ = print(f"Logged multi-noise overview: {overview_name}")
        except Exception as e:
            _ = print(f"Failed to create/log multi-noise overview: {e}")
            import traceback

            _ = traceback.print_exc()

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Select optimizer based on type
        optimizer: torch.optim.Optimizer
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        else:  # default to adam
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                amsgrad=self.amsgrad,
            )

        if self.scheduler_config is None:
            return optimizer

        scheduler_type: str = self.scheduler_config.get("type", "cosine")

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
        elif scheduler_type == "cosine_warmup":
            # Cosine schedule with linear warmup
            #
            # Supports two configuration modes:
            # 1. Fraction-based (recommended): warmup_fraction, decay_fraction
            #    - Automatically scales with batch_size, dataset size, and max_epochs
            #    - warmup_fraction: fraction of total training for linear warmup (e.g., 0.02)
            #    - decay_fraction: fraction of training at which LR reaches 0 (e.g., 0.8)
            #
            # 2. Legacy step-based: T_warmup, T_max
            #    - Requires manual recalculation when batch_size or epochs change
            #    - T_max = (num_samples / batch_size) × expected_epochs
            #
            # After T_max/decay point, progress is clamped at 1.0 to keep LR
            # at minimum instead of oscillating (cos would cycle if progress > 1).
            import warnings

            # Estimate total training steps from trainer context
            # Prefer max_steps if set (step-based training), otherwise compute from epochs
            estimated_total_steps = None
            trainer = self.trainer
            if trainer is not None:
                # First check if max_steps is explicitly configured
                if trainer.max_steps and trainer.max_steps > 0:
                    estimated_total_steps = trainer.max_steps
                else:
                    dm = self.datamodule
                    if dm is not None:
                        # Fallback: compute from epochs (legacy compatibility)
                        try:
                            train_loader = dm.train_dataloader()
                            dataset_size = len(train_loader.dataset)
                            batch_size = getattr(
                                dm, "batch_size", train_loader.batch_size
                            )
                            steps_per_epoch = (
                                dataset_size + batch_size - 1
                            ) // batch_size
                            max_epochs_val = trainer.max_epochs
                            max_epochs = (
                                max_epochs_val
                                if max_epochs_val is not None and max_epochs_val > 0
                                else 100
                            )
                            estimated_total_steps = steps_per_epoch * max_epochs
                        except Exception:
                            pass  # Fall back to defaults below

            if estimated_total_steps is None:
                # Fallback when trainer context unavailable (e.g., during testing)
                estimated_total_steps = 10000
                if "warmup_fraction" in self.scheduler_config:
                    warnings.warn(
                        "Could not estimate total training steps from trainer. "
                        f"Using fallback of {estimated_total_steps} steps. "
                        "Scheduler fractions may not work as expected.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Compute T_warmup and T_max from fractions or legacy values
            if "warmup_fraction" in self.scheduler_config:
                # Fraction-based configuration (preferred)
                warmup_fraction = self.scheduler_config.get("warmup_fraction", 0.02)
                decay_fraction = self.scheduler_config.get("decay_fraction", 0.8)
                T_warmup = int(warmup_fraction * estimated_total_steps)
                T_max = int(decay_fraction * estimated_total_steps)
            else:
                # Legacy step-based configuration (backward compatible)
                T_warmup = self.scheduler_config.get("T_warmup", 100)
                T_max = self.scheduler_config.get("T_max", 1000)

            # Validation
            if T_max <= T_warmup:
                raise ValueError(
                    f"Scheduler T_max ({T_max}) must be > T_warmup ({T_warmup}). "
                    f"Check scheduler_config: decay_fraction must be > warmup_fraction."
                )

            if T_max < estimated_total_steps * 0.5:
                warnings.warn(
                    f"Scheduler T_max={T_max} is less than 50% of estimated training "
                    f"({estimated_total_steps} steps). LR will reach minimum early and "
                    f"stay there for the remaining training. Consider increasing "
                    f"decay_fraction or T_max.",
                    UserWarning,
                    stacklevel=2,
                )

            # Store computed values for logging in on_fit_start
            self._scheduler_T_warmup = T_warmup
            self._scheduler_T_max = T_max
            self._scheduler_estimated_total_steps = estimated_total_steps

            def lr_lambda(step: int) -> float:
                if step < T_warmup:
                    # Linear warmup: scale from 0 to 1
                    return step / max(1, T_warmup)
                else:
                    # Cosine decay: scale from 1 to 0
                    # Clamp progress at 1.0 to prevent LR oscillation if step > T_max
                    progress = (step - T_warmup) / max(1, T_max - T_warmup)
                    progress = min(1.0, progress)  # Clamp to prevent oscillation
                    return 0.5 * (1 + np.cos(np.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
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

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging."""
        return self.model.get_config()
