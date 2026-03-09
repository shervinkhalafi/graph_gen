"""Structural protocols for the diffusion pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from .graph_types import LimitDistribution, TransitionMatrices


@runtime_checkable
class TransitionModel(Protocol):
    """Structural interface for discrete transition models.

    Both ``DiscreteUniformTransition`` and ``MarginalUniformTransition``
    satisfy this protocol without modification.  Consumers should type-hint
    against ``TransitionModel`` rather than importing the concrete classes.

    Implementations are ``nn.Module``s with registered buffers, so device
    management is handled by PyTorch's ``.to()`` — no ``device`` parameter
    on individual methods.
    """

    def get_Qt(self, beta_t: Tensor) -> TransitionMatrices:
        """Single-step transition matrices ``Q_t`` from noise level ``beta_t``.

        Each returned matrix must be row-stochastic (rows sum to 1).
        At ``beta_t = 0`` the matrix is the identity (no transition);
        at ``beta_t = 1`` it collapses to the stationary distribution.
        """
        ...

    def get_Qt_bar(self, alpha_bar_t: Tensor) -> TransitionMatrices:
        """Cumulative transition matrices ``Q̄_t`` from signal level ``alpha_bar_t``.

        Satisfies ``Q̄_t = Q_1 · Q_2 · ... · Q_t``. Row-stochastic.
        At ``alpha_bar_t = 1`` this is the identity; at ``alpha_bar_t = 0``
        every row equals the stationary (limit) distribution.
        """
        ...

    def get_limit_dist(self) -> LimitDistribution:
        """Stationary distribution reached as ``t → T`` (``alpha_bar → 0``).

        Returns node and edge distributions that each sum to 1.
        """
        ...
