"""Standalone parameter-logging utilities for nn.Module models.

This module provides ``log_parameter_count``, which replaces the former
``LoggingMixin.log_parameter_count`` method with a plain function that accepts
the model, its display name, and an optional Lightning logger as explicit
arguments rather than reading them from ``self``.
"""

from __future__ import annotations

from typing import Any

from lightning_fabric.loggers import Logger
from torch import nn


def log_parameter_count(
    model: nn.Module,
    model_name: str,
    logger: Logger | None,
) -> None:
    """Log the parameter count of *model* in a formatted way.

    If the model exposes a ``parameter_count`` method the output is a
    hierarchical breakdown; otherwise a simple total is printed.  Counts
    are also forwarded to *logger* when one is provided.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are counted.
    model_name : str
        Human-readable name used in the printed header and logger tags.
    logger : Logger | None
        Optional Lightning logger.  When present, parameter counts are
        forwarded via ``log_hyperparams``.
    """
    if not hasattr(model, "parameter_count"):
        total_params: int = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad  # pyright: ignore[reportAny]
        )
        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"{'=' * 50}\n")

        if logger:
            logger.log_hyperparams({"total_parameters": total_params})
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

    print(f"\n{'=' * 60}")
    print(f"Model Parameter Count: {model_name}")
    print(f"{'=' * 60}")
    print(f"Total Trainable Parameters: {param_counts['total']:,}")
    print("-" * 60)

    formatted_lines: list[str] = format_counts(param_counts)
    for line in formatted_lines:
        print(line)

    print(f"{'=' * 60}\n")

    if logger:
        logger.log_hyperparams(
            {
                "total_parameters": param_counts["total"],
                "parameter_breakdown": param_counts,
            }
        )
