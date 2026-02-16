"""Base Lightning module for denoising experiments."""

from __future__ import annotations

import abc
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Any, cast, override

import matplotlib.figure
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import Logger

from tmgg.experiment_utils import (
    compute_batch_metrics,
    create_graph_denoising_figure,
    create_network_denoising_figure,  # pyright: ignore[reportAttributeAccessIssue]
    create_noise_generator,
)
from tmgg.experiment_utils.exceptions import ConfigurationError
from tmgg.experiment_utils.logging import log_figure
from tmgg.experiment_utils.model_logging import (
    log_parameter_count as _log_parameter_count,
)
from tmgg.experiment_utils.optimizer_config import (
    OptimizerLRSchedulerConfig,
    SchedulerInfo,
    configure_optimizers_from_config,
)
from tmgg.models.base import DenoisingModel


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
        noise_type: str = "digress",
        rotation_k: int = 20,
        seed: int | None = None,
        visualization_interval: int = 5000,
        spectral_k: int = 4,
        log_spectral_deltas: bool = False,
        log_rotation_angles: bool = False,
        **kwargs,
    ):
        """Initialize the denoising Lightning module.

        Parameters
        ----------
        dropout : float
            Dropout probability.
        bias : bool
            Whether to use bias in linear layers.
        learning_rate : float
            Learning rate for optimizer.
        weight_decay : float
            Weight decay (L2 regularization) coefficient for AdamW.
        optimizer_type : str
            Optimizer to use (``"adam"`` or ``"adamw"``).
        loss_type : str
            Loss function type (``"MSE"`` or ``"BCEWithLogits"``).
        scheduler_config : dict or None
            Optional scheduler configuration with keys ``type``
            (``"cosine"``, ``"cosine_warmup"``, ``"step"``),
            ``T_warmup``, ``T_0``, ``T_mult``, ``step_size``, ``gamma``.
        eval_noise_levels : list[float] or None
            Noise levels for evaluation. Falls back to the datamodule's
            ``noise_levels`` when None.
        noise_type : str
            Noise type (``"gaussian"``, ``"rotation"``, ``"digress"``).
        rotation_k : int
            Dimension for rotation noise skew matrix.
        seed : int or None
            Random seed for reproducible noise generation.
        visualization_interval : int
            Global steps between visualization logging.
        spectral_k : int
            Number of top eigenvectors for spectral delta metrics.
            Only used when *log_spectral_deltas* is True.
        log_spectral_deltas : bool
            If True, compute eigengap delta, algebraic connectivity delta,
            eigenvalue drift, and subspace distance during validation/test.
        log_rotation_angles : bool
            If True (and *log_spectral_deltas* is True), also compute
            Procrustes rotation angle and residual metrics.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model: DenoisingModel = self._make_model(
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

        # Spectral delta logging configuration
        self.spectral_k: int = spectral_k
        self.log_spectral_deltas: bool = log_spectral_deltas
        self.log_rotation_angles: bool = log_rotation_angles

        # Seeded RNG for noise level sampling (avoids global numpy state)
        self._noise_rng: np.random.Generator = np.random.default_rng(seed)

        # Populated by configure_optimizers when cosine_warmup scheduler is used
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

    def get_model_name(self) -> str:
        """Return the model name for display in logs and plots.

        Subclasses should override to return a meaningful name.
        """
        return "Base"

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging.

        Returns
        -------
        dict[str, Any]
            Configuration dict as reported by the underlying model.
        """
        return self.model.get_config()  # pyright: ignore[reportAny]

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

    @abstractmethod
    def _make_model(self, *args: Any, **kwargs: Any) -> DenoisingModel:  # pyright: ignore[reportExplicitAny]
        """Create and return the denoising model for this experiment."""
        pass

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
        _log_parameter_count(self.model, self.get_model_name(), self.logger)

        # Log scheduler configuration for cosine_warmup
        if (
            self.scheduler_config
            and self.scheduler_config.get("type") == "cosine_warmup"
            and self._scheduler_info is not None
        ):
            T_warmup: int = self._scheduler_info.T_warmup
            T_max: int = self._scheduler_info.T_max
            total_steps: int = self._scheduler_info.estimated_total_steps

            # Estimate steps per epoch for readable output
            steps_per_epoch: int = 1
            dm = self.datamodule
            if dm is not None:
                try:
                    train_loader = dm.train_dataloader()
                    dataset_size: int = len(train_loader.dataset)
                    batch_size: int = getattr(dm, "batch_size", train_loader.batch_size)
                    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
                except (TypeError, AttributeError, RuntimeError) as exc:
                    warnings.warn(
                        f"Could not estimate steps_per_epoch for scheduler logging: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )

            warmup_epochs: float = (
                T_warmup / steps_per_epoch if steps_per_epoch > 0 else 0
            )
            decay_epochs: float = T_max / steps_per_epoch if steps_per_epoch > 0 else 0
            total_epochs: float = (
                total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
            )

            print(f"\n{'=' * 60}")
            print("Scheduler Configuration (cosine_warmup)")
            print(f"{'=' * 60}")
            print(f"  Warmup:     {T_warmup:,} steps ({warmup_epochs:.1f} epochs)")
            print(f"  LR at zero: {T_max:,} steps ({decay_epochs:.1f} epochs)")
            print(f"  Total est:  {total_steps:,} steps ({total_epochs:.0f} epochs)")
            print(f"{'=' * 60}\n")

    @override
    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the denoising model.

        Parameters
        ----------
        x
            Input adjacency matrix.
        t
            Diffusion timestep tensor, or None for unconditional denoising.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency matrix.
        """
        # Model returns reconstructed adjacency matrix directly
        output: torch.Tensor = self.model(x, t=t)
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
        eps: float = float(self._noise_rng.choice(self.noise_levels))
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
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/noise_level", eps, on_step=False, on_epoch=True)

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

            # Log spectral delta metrics if enabled
            if self.log_spectral_deltas:
                self._log_spectral_deltas(mode, eps, target, batch_noisy, predictions)

            mode_loss_mean += mode_loss.item() / N
            for k, v in batch_metrics.items():
                batch_metrics_mean[k] += v / N

        # Log metrics
        self.log(
            f"{mode}/loss", mode_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        for metric_name, value in batch_metrics_mean.items():
            self.log(f"{mode}/{metric_name}", value, on_step=False, on_epoch=True)
        mode_loss_mean_tensor: torch.Tensor = torch.tensor(mode_loss_mean)
        batch_metrics_mean_dict: dict[str, torch.Tensor] = {
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
                self.log(
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
                self.log(
                    f"{mode}_{eps}/denoised_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        return self._val_or_test(mode="test", batch=batch)

    @override
    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
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
        """Log denoising visualizations to configured loggers.

        Parameters
        ----------
        stage : str
            Either ``"val"`` or ``"test"``.
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
            fig: matplotlib.figure.Figure = create_graph_denoising_figure(
                A_clean=A_sample,
                noise_fn=cast(Any, self.noise_generator.add_noise),
                denoise_fn=cast(Any, denoise_fn),
                noise_level=eps,
                noise_type=noise_type,
                title_prefix=f"{self.get_model_name()} - ",
            )
            plot_name: str = f"{stage}_denoising_{noise_type}_eps_{eps:.3f}"
            log_figure(
                cast(list[Logger], self.loggers),
                plot_name,
                fig,
                global_step=self.global_step,
            )
            print(f"Logged individual plot: {plot_name}")

            # Log node-link network visualization for small graphs
            n_nodes: int = (
                A_sample.shape[0] if A_sample.ndim == 2 else A_sample.shape[-1]
            )
            if n_nodes <= 50:
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
                    log_figure(
                        cast(list[Logger], self.loggers),
                        network_plot_name,
                        network_fig,
                        global_step=self.global_step,
                    )
                    print(f"Logged network plot: {network_plot_name}")

        # Also create multi-noise visualization
        from tmgg.experiment_utils.plotting import create_multi_noise_visualization

        multi_fig: matplotlib.figure.Figure = create_multi_noise_visualization(
            A_sample,
            self,
            self.noise_generator.add_noise,
            noise_levels,
            str(self.device),
        )
        overview_name: str = f"{stage}_multi_noise_{noise_type}_overview"
        log_figure(
            cast(list[Logger], self.loggers),
            overview_name,
            multi_fig,
            global_step=self.global_step,
        )
        print(f"Logged multi-noise overview: {overview_name}")
