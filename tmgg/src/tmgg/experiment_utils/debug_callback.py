"""Debug callback for diagnosing training issues.

This callback logs detailed statistics about model outputs, gradients, and
weights to help identify why models might get stuck at sigmoid(0) = 0.5.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch


class DebugCallback(pl.Callback):
    """Callback for logging detailed training diagnostics.

    Logs statistics about logits, gradients, prediction errors, and weights
    to help diagnose training issues like models stuck at 0.5 output.

    Parameters
    ----------
    log_interval : int
        Log statistics every N training steps. Default 50.
    log_gradients : bool
        Whether to log gradient norms. Default True.
    log_weights : bool
        Whether to log weight statistics. Default True.
    gradient_names : list of str, optional
        Specific parameter names to log gradients for. If None, logs all.

    Examples
    --------
    >>> callback = DebugCallback(log_interval=100)
    >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        log_interval: int = 50,
        log_gradients: bool = True,
        log_weights: bool = True,
        gradient_names: Optional[list] = None,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.gradient_names = gradient_names

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log statistics after each training batch."""
        if trainer.global_step % self.log_interval != 0:
            return

        # Log logit statistics if available
        if isinstance(outputs, dict) and "logits" in outputs:
            self._log_logit_stats(pl_module, outputs["logits"])

        # Log prediction error stats
        if isinstance(outputs, dict) and "logits" in outputs:
            target = batch if isinstance(batch, torch.Tensor) else batch[0]
            self._log_error_stats(pl_module, outputs["logits"], target)

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log gradient statistics after backward pass."""
        if not self.log_gradients:
            return

        if trainer.global_step % self.log_interval != 0:
            return

        self._log_gradient_stats(pl_module)

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log weight statistics at epoch start."""
        if not self.log_weights:
            return

        self._log_weight_stats(pl_module)

    def _log_logit_stats(
        self,
        pl_module: pl.LightningModule,
        logits: torch.Tensor,
    ) -> None:
        """Log statistics about raw logit values."""
        with torch.no_grad():
            pl_module.log_dict(
                {
                    "debug/logit_mean": logits.mean(),
                    "debug/logit_std": logits.std(),
                    "debug/logit_min": logits.min(),
                    "debug/logit_max": logits.max(),
                    "debug/logit_abs_mean": logits.abs().mean(),
                },
                on_step=True,
                on_epoch=False,
            )

    def _log_error_stats(
        self,
        pl_module: pl.LightningModule,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Log prediction error statistics (L∞, Frobenius, MAE)."""
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            diff = probs - target

            # Frobenius norm normalized by number of elements
            frob_normalized = diff.norm(p="fro") / (diff.numel() ** 0.5)

            pl_module.log_dict(
                {
                    "debug/linf_error": diff.abs().max(),
                    "debug/frob_error": frob_normalized,
                    "debug/mae": diff.abs().mean(),
                    "debug/pred_mean": probs.mean(),
                    "debug/pred_std": probs.std(),
                },
                on_step=True,
                on_epoch=False,
            )

    def _log_gradient_stats(self, pl_module: pl.LightningModule) -> None:
        """Log gradient norm statistics for model parameters."""
        total_norm = 0.0
        num_params = 0

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue

            # Filter by name if specified
            if self.gradient_names is not None:
                if not any(gn in name for gn in self.gradient_names):
                    continue

            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2
            num_params += 1

            # Log individual gradient norms for important parameters
            if any(key in name for key in ["W", "weight", "H", "b", "bias"]):
                safe_name = name.replace(".", "_")
                pl_module.log(
                    f"grad/{safe_name}",
                    grad_norm,
                    on_step=True,
                    on_epoch=False,
                )

        # Log total gradient norm
        if num_params > 0:
            total_norm = total_norm ** 0.5
            pl_module.log(
                "debug/total_grad_norm",
                total_norm,
                on_step=True,
                on_epoch=False,
            )

    def _log_weight_stats(self, pl_module: pl.LightningModule) -> None:
        """Log weight statistics for model parameters."""
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue

            # Log stats for key parameters
            if any(key in name for key in ["W", "weight", "H", "b", "bias"]):
                safe_name = name.replace(".", "_")
                with torch.no_grad():
                    pl_module.log_dict(
                        {
                            f"weights/{safe_name}_mean": param.mean(),
                            f"weights/{safe_name}_std": param.std(),
                            f"weights/{safe_name}_frob": param.norm(p="fro"),
                            f"weights/{safe_name}_max": param.abs().max(),
                        },
                        on_step=False,
                        on_epoch=True,
                    )
