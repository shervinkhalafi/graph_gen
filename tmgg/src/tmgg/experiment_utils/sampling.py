"""Shared sampling and noise schedule utilities for diffusion experiments.

Provides noise schedule functions used by both the Gaussian and DiGress
denoising modules. Extracted from the Gaussian generative lightning module
to eliminate duplication.
"""

from __future__ import annotations

import numpy as np


def get_noise_schedule(
    schedule: str,
    num_steps: int,
) -> np.ndarray:
    """Compute noise levels for each diffusion timestep.

    Parameters
    ----------
    schedule
        Schedule type controlling how noise increases over time.
        ``"linear"`` ramps uniformly, ``"cosine"`` follows a
        half-cosine curve (slower start/end), and ``"quadratic"``
        applies a squared ramp (slow start, fast end).
    num_steps
        Number of diffusion steps. The returned array has this length.

    Returns
    -------
    np.ndarray
        Noise levels in ``[0, 1]`` for each timestep, monotonically
        increasing from ``0`` to ``1``.

    Raises
    ------
    ValueError
        If *schedule* is not one of the supported types.
    """
    t = np.linspace(0, 1, num_steps)

    if schedule == "linear":
        return t
    elif schedule == "cosine":
        return 1 - np.cos(t * np.pi / 2)
    elif schedule == "quadratic":
        return t**2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
