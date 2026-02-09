"""Standalone utility functions for Lightning-based training infrastructure.

These live outside any class so they can be called from datamodules, callbacks,
or other components that don't inherit from a Lightning module.
"""

from __future__ import annotations

from typing import Any, cast


def compute_datamodule_noise_levels(datamodule: Any) -> list[float]:  # pyright: ignore[reportExplicitAny]
    """Extract noise levels from a datamodule.

    Parameters
    ----------
    datamodule
        A Lightning ``DataModule`` instance expected to carry a
        ``noise_levels`` attribute.

    Returns
    -------
    list[float]
        The noise levels defined by the datamodule.

    Raises
    ------
    RuntimeError
        If the datamodule does not expose a ``noise_levels`` attribute.
    """
    levels = getattr(datamodule, "noise_levels", None)
    if levels is None:
        raise RuntimeError(
            f"DataModule {type(datamodule).__name__} does not have noise_levels. "
            "Ensure your DataModule provides a 'noise_levels' attribute."
        )
    return cast(list[float], levels)


def validate_datamodule_attributes(datamodule: Any, required: list[str]) -> None:  # pyright: ignore[reportExplicitAny]
    """Validate that a datamodule has all required attributes.

    Parameters
    ----------
    datamodule
        A Lightning ``DataModule`` instance to check.
    required
        Attribute names that must be present on *datamodule*.

    Raises
    ------
    ValueError
        If one or more required attributes are missing.
    """
    missing = [attr for attr in required if not hasattr(datamodule, attr)]
    if missing:
        raise ValueError(
            f"DataModule {type(datamodule).__name__} missing required attributes: {missing}"
        )
