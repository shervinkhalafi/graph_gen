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
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        eval_noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
        rotation_k: int = 20,
        seed: Optional[int] = None,
        visualization_interval: int = 5000,
        fixed_noise_seed: Optional[int] = None,
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
            noise_levels: List of noise levels to sample from during training
            eval_noise_levels: List of noise levels for evaluation. If None, uses
                noise_levels (same as training).
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
            rotation_k: Dimension for rotation noise skew matrix (number of eigenvectors)
            seed: Random seed for reproducible noise generation
            visualization_interval: Steps between logging visualizations (default: 5000)
            fixed_noise_seed: If set, pre-computes and caches noise patterns for each
                (batch_idx, eps) pair. The cached noisy batch is reused across epochs,
                enabling constant noise memorization tests. Set to None for fresh noise.
        """
        super().__init__()
        # Exclude noise_levels to avoid hparam conflict with datamodule
        # (datamodule is the authoritative source for noise_levels)
        self.save_hyperparameters(ignore=["noise_levels"])

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
            **kwargs,
        )
        # Loss function (models output logits, use BCEWithLogitsLoss for stability)
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "BCEWithLogits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Use 'MSE' or 'BCEWithLogits'.")

        # Store configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type.lower()
        self.amsgrad = amsgrad
        self.scheduler_config = scheduler_config
        self._noise_levels_override = noise_levels  # May be None if using datamodule's
        self._eval_noise_levels_override = eval_noise_levels  # May be None to use noise_levels
        self.noise_type = noise_type
        self.visualization_interval = visualization_interval

        # Create noise generator
        self.noise_generator = create_noise_generator(
            noise_type=noise_type, rotation_k=rotation_k, seed=seed
        )

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

        # Fixed noise mode for sanity check (constant noise memorization)
        self.fixed_noise_seed = fixed_noise_seed
        self._noise_cache: Dict[tuple, torch.Tensor] = {}  # Cache for (N, eps) -> noisy graph
        self._clean_cache: Dict[int, torch.Tensor] = {}  # Cache for N -> clean graph

    @property
    def noise_levels(self) -> List[float]:
        """Get noise levels from datamodule or override.

        Returns the noise levels to use for training and evaluation. Prefers
        the datamodule's noise_levels when attached to a trainer, but falls
        back to the override value (or default) during standalone usage.
        """
        # Try to get noise_levels from trainer's datamodule
        # Note: Lightning's trainer property raises RuntimeError when not attached
        try:
            trainer = self.trainer
            if (
                trainer is not None
                and hasattr(trainer, "datamodule")
                and trainer.datamodule is not None
                and hasattr(trainer.datamodule, "noise_levels")
            ):
                return trainer.datamodule.noise_levels
        except RuntimeError:
            pass  # Not attached to trainer yet

        # Fall back to override or default
        if self._noise_levels_override is not None:
            return self._noise_levels_override
        return [0.01, 0.05, 0.1, 0.2, 0.3]  # Default fallback

    @property
    def eval_noise_levels(self) -> List[float]:
        """Get noise levels for evaluation.

        Returns eval_noise_levels if explicitly set, otherwise falls back to
        noise_levels (same levels for training and evaluation).
        """
        if self._eval_noise_levels_override is not None:
            return self._eval_noise_levels_override
        return self.noise_levels  # Default: same as training

    @abstractmethod
    def _make_model(self, *args: Any, **kwargs: Any) -> nn.Module:  # pyright: ignore[reportExplicitAny, reportAny]
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
        if not hasattr(self.model, 'parameter_count'):
            # Fallback for models without parameter_count method
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\n{'='*50}")
            print(f"Model: {self.get_model_name()}")
            print(f"Total Trainable Parameters: {total_params:,}")
            print(f"{'='*50}\n")
            
            # Log to logger if available
            if self.logger:
                self.logger.log_hyperparams({"total_parameters": total_params})
            return
        
        # Get hierarchical parameter counts
        param_counts = self.model.parameter_count()
        
        # Format and print parameter counts
        def format_counts(counts: Dict[str, Any], indent: int = 0) -> List[str]:
            """Recursively format parameter counts."""
            lines = []
            prefix = "  " * indent
            
            # Skip printing "self" and "total" at sub-levels, just show them at top level
            for key, value in counts.items():
                if key == "total" and indent == 0:
                    continue  # Will be printed separately at the top
                elif key == "self":
                    if value > 0:
                        lines.append(f"{prefix}├─ {key}: {value:,}")
                elif isinstance(value, dict):
                    if "total" in value:
                        lines.append(f"{prefix}├─ {key}: {value['total']:,}")
                        # Recursively format sub-counts if they exist
                        sub_items = {k: v for k, v in value.items() if k != 'total'}
                        if sub_items:
                            sub_lines = format_counts(sub_items, indent + 1)
                            lines.extend(sub_lines)
                    else:
                        lines.append(f"{prefix}├─ {key}:")
                        sub_lines = format_counts(value, indent + 1)
                        lines.extend(sub_lines)
                else:
                    lines.append(f"{prefix}├─ {key}: {value:,}")
            
            return lines
        
        # Print formatted output
        print(f"\n{'='*60}")
        print(f"Model Parameter Count: {self.get_model_name()}")
        print(f"{'='*60}")
        print(f"Total Trainable Parameters: {param_counts['total']:,}")
        print("-" * 60)
        
        formatted_lines = format_counts(param_counts)
        for line in formatted_lines:
            print(line)
        
        print(f"{'='*60}\n")
        
        # Log to logger if available
        if self.logger:
            self.logger.log_hyperparams({
                "total_parameters": param_counts['total'],
                "parameter_breakdown": param_counts
            })

    def setup(self, stage: str) -> None:
        """Validate configuration before training/testing begins.

        Parameters
        ----------
        stage
            Either 'fit', 'validate', 'test', or 'predict'.
        """
        super().setup(stage)

        # Validate datamodule provides required attributes
        if self.trainer is None or self.trainer.datamodule is None:
            return  # Skip validation if no trainer/datamodule

        dm = self.trainer.datamodule
        required_attrs = ["noise_levels"]
        missing = [attr for attr in required_attrs if not hasattr(dm, attr)]

        if missing:
            raise ValueError(
                f"DataModule {type(dm).__name__} missing required attributes: {missing}. "
                "Ensure your DataModule provides 'noise_levels' attribute."
            )

    def on_fit_start(self) -> None:
        """Called at the beginning of training."""
        # Log parameter count at the start of training
        self.log_parameter_count()

        # Log scheduler configuration for cosine_warmup
        if (
            self.scheduler_config
            and self.scheduler_config.get("type") == "cosine_warmup"
            and hasattr(self, "_scheduler_T_warmup")
        ):
            T_warmup = self._scheduler_T_warmup
            T_max = self._scheduler_T_max
            total_steps = self._scheduler_estimated_total_steps

            # Estimate steps per epoch for readable output
            steps_per_epoch = 1
            if self.trainer and self.trainer.datamodule:
                try:
                    train_loader = self.trainer.datamodule.train_dataloader()
                    dataset_size = len(train_loader.dataset)
                    batch_size = getattr(
                        self.trainer.datamodule, "batch_size", train_loader.batch_size
                    )
                    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
                except Exception:
                    pass

            warmup_epochs = T_warmup / steps_per_epoch if steps_per_epoch > 0 else 0
            decay_epochs = T_max / steps_per_epoch if steps_per_epoch > 0 else 0
            total_epochs = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0

            print(f"\n{'='*60}")
            print("Scheduler Configuration (cosine_warmup)")
            print(f"{'='*60}")
            print(f"  Warmup:     {T_warmup:,} steps ({warmup_epochs:.1f} epochs)")
            print(f"  LR at zero: {T_max:,} steps ({decay_epochs:.1f} epochs)")
            print(f"  Total est:  {total_steps:,} steps ({total_epochs:.0f} epochs)")
            print(f"{'='*60}\n")

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

    def _apply_noise(self, batch: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply noise to batch, with optional caching for sanity check mode.

        In normal mode (fixed_noise_seed=None), generates fresh noise each call.
        In sanity check mode (fixed_noise_seed set), generates ONE noise pattern
        and broadcasts it to all graphs in all batches.

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
        if self.fixed_noise_seed is None:
            return self.noise_generator.add_noise(batch, eps)

        # Sanity check mode: use SINGLE noise pattern for all batches
        B, N, _ = batch.shape
        cache_key = (N, eps)  # Key by graph size and eps only

        if cache_key not in self._noise_cache:
            # Generate noise for ONE graph with fixed seed
            # Use cached clean graph to ensure consistent (clean, noisy) pair
            clean_graph = self._get_cached_clean(batch)
            torch.manual_seed(self.fixed_noise_seed)
            np.random.seed(self.fixed_noise_seed)
            self._noise_cache[cache_key] = self.noise_generator.add_noise(
                clean_graph, eps
            ).detach()

        # Broadcast to batch size
        cached = self._noise_cache[cache_key].to(batch.device)
        return cached.expand(B, -1, -1)

    def _get_cached_clean(self, batch: torch.Tensor) -> torch.Tensor:
        """Get cached clean graph for fixed noise mode.

        Caches the first clean graph seen for each graph size N.
        All subsequent calls return this same graph, ensuring the model
        always trains on the same (noisy, clean) pair.
        """
        B, N, _ = batch.shape
        if N not in self._clean_cache:
            self._clean_cache[N] = batch[:1].clone().detach()
        return self._clean_cache[N]

    def _get_fixed_target(self, batch: torch.Tensor) -> torch.Tensor:
        """Get target for fixed noise mode.

        Returns the cached clean graph broadcasted to batch size.
        This ensures the target matches the cached noisy input.
        """
        B, N, _ = batch.shape
        cached = self._get_cached_clean(batch).to(batch.device)
        return cached.expand(B, -1, -1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Execute single training step with noise application.

        In normal mode: noise is sampled fresh for each batch.
        In sanity check mode (fixed_noise_seed set): uses cached noise patterns
        so the model sees identical noisy inputs across epochs.

        Parameters
        ----------
        batch
            Batch of clean adjacency matrices, shape (B, N, N).
        batch_idx
            Index of current batch. Used as cache key in fixed noise mode.

        Returns
        -------
        dict
            Dictionary with 'loss' (required by Lightning) and 'logits' for debugging.
        """
        # In fixed noise mode, use first noise level; else sample randomly
        if self.fixed_noise_seed is not None:
            eps = self.noise_levels[0]
            # Use cached clean graph as target (matches cached noisy input)
            target = self._get_fixed_target(batch)
        else:
            eps = np.random.choice(self.noise_levels)
            target = batch

        # Add noise (cached in fixed noise mode, fresh otherwise)
        batch_noisy = self._apply_noise(batch, eps)

        # Forward pass using noisy adjacency matrix
        output = self.forward(batch_noisy)

        # Transform output and target to match for loss computation
        # This handles domain transformations (inv-sigmoid vs standard) automatically
        output_for_loss, target_for_loss = self.model.transform_for_loss(output, target)
        loss = self.criterion(output_for_loss, target_for_loss)

        # Compute train accuracy from logits using model's thresholding
        with torch.no_grad():
            predictions = self.model.logits_to_graph(output)
            train_acc = (predictions == target).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_noise_level", eps, on_step=False, on_epoch=True)

        # Return dict with loss and logits for debugging callbacks
        return {"loss": loss, "logits": output}

    def _val_or_test(self, mode: str, batch):
        # In fixed noise mode, use cached clean graph as target
        if self.fixed_noise_seed is not None:
            target = self._get_fixed_target(batch)
        else:
            target = batch

        # Evaluate across all eval noise levels
        mode_loss_mean = 0.0
        batch_metrics_mean = defaultdict(lambda: 0.0)
        N = len(self.eval_noise_levels)
        for eps in self.eval_noise_levels:
            # Add noise (cached in fixed noise mode, fresh otherwise)
            batch_noisy = self._apply_noise(batch, eps)

            # Forward pass
            output = self.forward(batch_noisy)

            # Transform output and target to match for loss computation
            # This ensures comparable loss values between training and validation
            output_for_loss, target_for_loss = self.model.transform_for_loss(
                output, target
            )
            mode_loss = self.criterion(output_for_loss, target_for_loss)

            # Compute reconstruction metrics on predictions (post-sigmoid)
            predictions = self.model.predict(output)
            batch_metrics = compute_batch_metrics(target, predictions)

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
        """Called at the end of validation for visualization (step-based)."""
        if self.global_step % self.visualization_interval == 0:
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

        # Use consistent noise_levels from property (which uses datamodule's when available)
        noise_levels = self.noise_levels
        noise_type = getattr(self.trainer.datamodule, "noise_type", self.noise_type)

        # Create denoise function that returns (predictions, logits) tuple
        def denoise_fn(A_noisy):
            with torch.no_grad():
                self.eval()
                A_input = A_noisy.to(self.device)
                if A_input.ndim == 2:
                    A_input = A_input.unsqueeze(0)
                logits = self.forward(A_input)
                predictions = self.model.predict(logits)
                return predictions.squeeze(0), logits.squeeze(0)

        # Log visualization for each noise level
        for eps in noise_levels:
            try:
                fig = create_graph_denoising_figure(
                    A_clean=A_sample,
                    noise_fn=self.noise_generator.add_noise,
                    denoise_fn=denoise_fn,
                    noise_level=eps,
                    noise_type=noise_type,
                    title_prefix=f"{self.get_model_name()} - ",
                )
                plot_name = f"{stage}_denoising_{noise_type}_eps_{eps:.3f}"
                log_figure(
                    self.loggers,  # Use all loggers, not just first
                    plot_name,
                    fig,
                    global_step=self.global_step,
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
                self.loggers,  # Use all loggers, not just first
                overview_name,
                multi_fig,
                global_step=self.global_step,
            )
            print(f"Logged multi-noise overview: {overview_name}")
        except Exception as e:
            print(f"Failed to create/log multi-noise overview: {e}")
            import traceback
            traceback.print_exc()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Select optimizer based on type
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
            if self.trainer is not None:
                # First check if max_steps is explicitly configured
                if self.trainer.max_steps and self.trainer.max_steps > 0:
                    estimated_total_steps = self.trainer.max_steps
                elif self.trainer.datamodule is not None:
                    # Fallback: compute from epochs (legacy compatibility)
                    try:
                        train_loader = self.trainer.datamodule.train_dataloader()
                        dataset_size = len(train_loader.dataset)
                        batch_size = getattr(
                            self.trainer.datamodule, "batch_size", train_loader.batch_size
                        )
                        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
                        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs > 0 else 100
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

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for logging."""
        return self.model.get_config()
