"""Typed containers for the discrete diffusion pipeline.

``TransitionMatrices`` and ``LimitDistribution`` are used by
``tmgg.diffusion.protocols`` and ``tmgg.diffusion.transitions``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = [
    "TransitionMatrices",
    "LimitDistribution",
]


@dataclass(frozen=True, slots=True)
class TransitionMatrices:
    """Per-step or cumulative transition matrices for discrete diffusion.

    Parameters
    ----------
    X : Tensor
        Node class transition matrices, shape ``(bs, dx, dx)``.
    E : Tensor
        Edge class transition matrices, shape ``(bs, de, de)``.
    y : Tensor
        Global class transition matrices, shape ``(bs, dy, dy)``.
    """

    X: Tensor
    E: Tensor
    y: Tensor


@dataclass(frozen=True, slots=True)
class LimitDistribution:
    """Stationary distribution of the diffusion process.

    Parameters
    ----------
    X : Tensor
        Node class probabilities, shape ``(dx,)``.
    E : Tensor
        Edge class probabilities, shape ``(de,)``.
    y : Tensor
        Global class probabilities, shape ``(dy,)``.
    """

    X: Tensor
    E: Tensor
    y: Tensor
