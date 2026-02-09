"""Logging and visualization mixin for Lightning modules.

Extracted from DenoisingLightningModule to allow reuse in both the denoising
and future diffusion Lightning modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytorch_lightning.loggers import Logger


class LoggingMixin:
    """Mixin providing parameter logging, visualization gating, and config access.

    Intended for use with ``pl.LightningModule`` subclasses. The host class
    must provide:

    Attributes
    ----------
    visualization_interval : int
        Steps between logged visualizations.
    model : object
        The wrapped model, expected to expose ``parameter_count()`` and
        ``get_config()`` when available.
    logger : Logger | None
        Lightning logger (provided by LightningModule).
    global_step : int
        Current training step (provided by LightningModule).
    """

    # Provided by the host LightningModule; declared here for type checking.
    visualization_interval: int  # pyright: ignore[reportUninitializedInstanceVariable]
    model: Any  # pyright: ignore[reportExplicitAny,reportUninitializedInstanceVariable]
    logger: Logger | None  # pyright: ignore[reportUninitializedInstanceVariable]
    global_step: int  # pyright: ignore[reportUninitializedInstanceVariable]

    # ------------------------------------------------------------------
    # Model name (override in subclasses)
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        """Return the model name for display in logs and plots.

        Subclasses should override this to return a meaningful name.
        """
        return "Base"

    # ------------------------------------------------------------------
    # Parameter logging
    # ------------------------------------------------------------------

    def log_parameter_count(self) -> None:
        """Log the parameter count of the model in a formatted way.

        If the model exposes a ``parameter_count`` method the output is a
        hierarchical breakdown; otherwise a simple total is printed. Counts
        are also forwarded to the attached logger (if any).
        """
        model = self.model
        if model is None:
            return

        if not hasattr(model, "parameter_count"):
            total_params: int = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad  # pyright: ignore[reportAny]
            )
            _ = print(f"\n{'=' * 50}")
            _ = print(f"Model: {self.get_model_name()}")
            _ = print(f"Total Trainable Parameters: {total_params:,}")
            _ = print(f"{'=' * 50}\n")

            if self.logger:
                _ = self.logger.log_hyperparams({"total_parameters": total_params})
            return

        param_counts: dict[str, Any] = model.parameter_count()  # pyright: ignore[reportAny]

        def format_counts(counts: dict[str, Any], indent: int = 0) -> list[str]:
            """Recursively format parameter counts."""
            lines: list[str] = []
            prefix: str = "  " * indent

            for key, value in counts.items():
                if key == "total" and indent == 0:
                    continue
                elif key == "self":
                    if value > 0:
                        _ = lines.append(f"{prefix}├─ {key}: {value:,}")
                elif isinstance(value, dict):
                    if "total" in value:
                        _ = lines.append(f"{prefix}├─ {key}: {value['total']:,}")
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

        _ = print(f"\n{'=' * 60}")
        _ = print(f"Model Parameter Count: {self.get_model_name()}")
        _ = print(f"{'=' * 60}")
        _ = print(f"Total Trainable Parameters: {param_counts['total']:,}")
        _ = print("-" * 60)

        formatted_lines: list[str] = format_counts(param_counts)
        for line in formatted_lines:
            _ = print(line)

        _ = print(f"{'=' * 60}\n")

        if self.logger:
            _ = self.logger.log_hyperparams(
                {
                    "total_parameters": param_counts["total"],
                    "parameter_breakdown": param_counts,
                }
            )

    # ------------------------------------------------------------------
    # Visualization gating
    # ------------------------------------------------------------------

    def _should_visualize(self) -> bool:
        """Return True if the current step warrants a visualization log.

        Uses ``global_step`` (from LightningModule) and
        ``visualization_interval`` to decide.
        """
        return self.global_step % self.visualization_interval == 0

    # ------------------------------------------------------------------
    # Model config access
    # ------------------------------------------------------------------

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for logging.

        Returns
        -------
        dict[str, Any]
            Configuration dict as reported by the underlying model.
        """
        return self.model.get_config()  # pyright: ignore[reportAny]
