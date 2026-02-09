"""Optimizer configuration mixin for Lightning modules.

Extracted from DenoisingLightningModule.configure_optimizers to allow reuse
in both the denoising and future diffusion Lightning modules.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch


class OptimizerMixin:
    """Mixin providing optimizer and LR scheduler configuration.

    Intended for use with ``pl.LightningModule`` subclasses. The host class
    must set the following attributes before ``configure_optimizers`` is called:

    Attributes
    ----------
    learning_rate : float
        Base learning rate.
    weight_decay : float
        Weight decay coefficient (used only with AdamW).
    optimizer_type : str
        One of ``"adam"`` or ``"adamw"``.
    amsgrad : bool
        Whether to enable AMSGrad variant.
    scheduler_config : dict[str, Any] | None
        Scheduler configuration dict, or None to disable scheduling.
        Supports types: ``"cosine"``, ``"cosine_warmup"``, ``"step"``.
    """

    # Provided by the host LightningModule.__init__; declared here for type
    # checking.  The "uninitialized" warnings are inherent to the mixin pattern.
    learning_rate: float  # pyright: ignore[reportUninitializedInstanceVariable]
    weight_decay: float  # pyright: ignore[reportUninitializedInstanceVariable]
    optimizer_type: str  # pyright: ignore[reportUninitializedInstanceVariable]
    amsgrad: bool  # pyright: ignore[reportUninitializedInstanceVariable]
    scheduler_config: dict[str, Any] | None  # pyright: ignore[reportUninitializedInstanceVariable]

    # Set by _configure_cosine_warmup, read by on_fit_start in the host.
    _scheduler_T_warmup: int  # pyright: ignore[reportUninitializedInstanceVariable]
    _scheduler_T_max: int  # pyright: ignore[reportUninitializedInstanceVariable]
    _scheduler_estimated_total_steps: int  # pyright: ignore[reportUninitializedInstanceVariable]

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        torch.optim.Optimizer | dict[str, Any]
            Plain optimizer when no scheduler is configured, otherwise a dict
            with ``"optimizer"`` and ``"lr_scheduler"`` keys following the
            Lightning convention.
        """
        # Select optimizer based on type
        optimizer: torch.optim.Optimizer
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),  # pyright: ignore[reportAttributeAccessIssue]  # from LightningModule
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )
        else:  # default to adam
            optimizer = torch.optim.Adam(
                self.parameters(),  # pyright: ignore[reportAttributeAccessIssue]  # from LightningModule
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
            return self._configure_cosine_warmup(optimizer)
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

    # ------------------------------------------------------------------
    # cosine_warmup helper (kept as a method so subclasses can override)
    # ------------------------------------------------------------------

    def _configure_cosine_warmup(
        self, optimizer: torch.optim.Optimizer
    ) -> dict[str, Any]:
        """Build a cosine-with-linear-warmup LR scheduler.

        Supports two configuration modes:

        1. **Fraction-based** (recommended): ``warmup_fraction``,
           ``decay_fraction`` — automatically scales with batch size,
           dataset size, and ``max_epochs``.
        2. **Legacy step-based**: ``T_warmup``, ``T_max`` — requires
           manual recalculation when batch size or epochs change.

        After the decay point, progress is clamped at 1.0 so the LR stays
        at its minimum instead of oscillating.

        Parameters
        ----------
        optimizer
            Already-constructed optimizer instance.

        Returns
        -------
        dict[str, Any]
            Lightning-format dict with optimizer and step-level scheduler.
        """
        assert self.scheduler_config is not None  # caller guarantees this

        # Estimate total training steps from trainer context.
        # Prefer max_steps if set (step-based training), otherwise compute
        # from epochs.
        estimated_total_steps: int | None = None
        trainer: pl.Trainer | None = getattr(self, "trainer", None)
        if trainer is not None:
            if trainer.max_steps and trainer.max_steps > 0:
                estimated_total_steps = trainer.max_steps
            else:
                dm: pl.LightningDataModule | None = getattr(trainer, "datamodule", None)
                if dm is not None:
                    try:
                        train_loader = dm.train_dataloader()
                        dataset_size = len(train_loader.dataset)
                        batch_size = getattr(dm, "batch_size", train_loader.batch_size)
                        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
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
        T_warmup: int
        T_max: int
        if "warmup_fraction" in self.scheduler_config:
            warmup_fraction = self.scheduler_config.get("warmup_fraction", 0.02)
            decay_fraction = self.scheduler_config.get("decay_fraction", 0.8)
            T_warmup = int(warmup_fraction * estimated_total_steps)
            T_max = int(decay_fraction * estimated_total_steps)
        else:
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
                return step / max(1, T_warmup)
            else:
                progress = (step - T_warmup) / max(1, T_max - T_warmup)
                progress = min(1.0, progress)
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
