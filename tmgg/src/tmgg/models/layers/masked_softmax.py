"""Utility layers for digress style graph neural networks."""

import torch


def masked_softmax(
    x: torch.Tensor, mask: torch.Tensor | None, **kwargs
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
        Softmax output with masked positions set to 0.

    Notes
    -----
    When all positions are masked, returns the raw input ``x`` for
    upstream parity (cvignac/DiGress models/layers.py:41-46). The
    downstream attention output is zeroed by the value-side mask
    either way, so this branch is behaviourally equivalent to
    returning zeros, but matches upstream form for cross-codebase
    auditability. Per-row NaN cleanup below still applies for the
    partial-mask case.
    """
    if mask is None:
        return torch.softmax(x, **kwargs)

    if mask.sum() == 0:
        return x

    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    result = torch.softmax(x_masked, **kwargs)

    # Handle row-wise all-masked case: softmax([-inf, -inf, ...]) = NaN
    # Replace NaN rows with zeros
    nan_mask = torch.isnan(result)
    if nan_mask.any():
        result = torch.where(nan_mask, torch.zeros_like(result), result)

    return result
