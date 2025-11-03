"""Utility layers for digress style graph neural networks."""

from typing import Optional

import torch


def masked_softmax(
    x: torch.Tensor, mask: Optional[torch.Tensor], **kwargs
) -> torch.Tensor:
    """Compute softmax with optional masking.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    mask : torch.Tensor, optional
        Binary mask (1 for valid positions, 0 for masked)
    **kwargs
        Additional arguments for torch.softmax (e.g., dim)

    Returns
    -------
    torch.Tensor
        Softmax output with masked positions set to 0

    Notes
    -----
    When all positions are masked, returns zeros instead of NaN.
    """
    if mask is None:
        return torch.softmax(x, **kwargs)

    if mask.sum() == 0:
        return torch.zeros_like(x)

    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
