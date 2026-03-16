"""Discrete transition matrices for categorical diffusion.

Provides ``DiscreteUniformTransition`` (uniform stationary distribution)
and ``MarginalUniformTransition`` (marginal stationary distribution),
both as ``nn.Module`` subclasses with registered buffers for automatic
device propagation.
"""

# pyright: reportUninitializedInstanceVariable=false
# PyTorch register_buffer() initialises these attributes at runtime;
# pyright cannot track this pattern.

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .graph_types import LimitDistribution, TransitionMatrices


class DiscreteUniformTransition(nn.Module):
    """Transition matrices for discrete diffusion with uniform stationary distribution.

    Each category's limiting distribution is ``1/K`` for ``K`` classes.
    Stationary matrices are stored as registered buffers so they follow the
    module to the correct device via ``.to()``.

    Parameters
    ----------
    x_classes : int
        Number of node feature classes.
    e_classes : int
        Number of edge feature classes.
    y_classes : int
        Number of global feature classes (0 to disable).
    """

    u_x: Tensor
    u_e: Tensor
    u_y: Tensor

    def __init__(self, x_classes: int, e_classes: int, y_classes: int) -> None:
        super().__init__()
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

        self.register_buffer(
            "u_x",
            torch.ones(1, x_classes, x_classes) / max(x_classes, 1),
        )
        self.register_buffer(
            "u_e",
            torch.ones(1, e_classes, e_classes) / max(e_classes, 1),
        )
        self.register_buffer(
            "u_y",
            torch.ones(1, y_classes, y_classes) / max(y_classes, 1),
        )

    def get_Qt(self, beta_t: Tensor) -> TransitionMatrices:
        """One-step transition matrices from step t-1 to step t.

        ``Qt = (1 - beta_t) * I + beta_t * U`` where ``U`` is the uniform
        matrix.

        Parameters
        ----------
        beta_t : Tensor
            Noise level, shape ``(bs,)``.

        Returns
        -------
        TransitionMatrices
            Transition matrices ``(q_x, q_e, q_y)`` each of shape
            ``(bs, d, d)``.
        """
        # Reshape to (bs, 1, 1) for correct broadcasting with (1, K, K) matrices.
        device = self.u_x.device
        beta_t = beta_t.reshape(-1, 1, 1).to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return TransitionMatrices(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: Tensor) -> TransitionMatrices:
        """Cumulative transition matrices from step 0 to step t.

        ``Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) * U``.

        Parameters
        ----------
        alpha_bar_t : Tensor
            Cumulative product of ``(1 - beta)``, shape ``(bs,)``.

        Returns
        -------
        TransitionMatrices
            Cumulative transition matrices ``(q_x, q_e, q_y)``.
        """
        device = self.u_x.device
        alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1).to(device)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_y
        )

        return TransitionMatrices(X=q_x, E=q_e, y=q_y)

    def get_limit_dist(self) -> LimitDistribution:
        """Return the limiting distribution as t -> infinity.

        For uniform transition this is ``1/K`` for each class.

        Returns
        -------
        LimitDistribution
            1-D distributions for X, E, and y.
        """
        x_limit = (
            torch.ones(self.X_classes) / self.X_classes
            if self.X_classes > 0
            else torch.zeros(0)
        )
        e_limit = (
            torch.ones(self.E_classes) / self.E_classes
            if self.E_classes > 0
            else torch.zeros(0)
        )
        y_limit = (
            torch.ones(self.y_classes) / self.y_classes
            if self.y_classes > 0
            else torch.zeros(0)
        )
        return LimitDistribution(X=x_limit, E=e_limit, y=y_limit)


class MarginalUniformTransition(nn.Module):
    """Transition matrices for discrete diffusion with marginal stationary distribution.

    Rows of the stationary matrix are the empirical marginal distribution
    rather than uniform ``1/K``.  For ``y`` features, falls back to uniform.
    Stationary matrices are stored as registered buffers.

    Parameters
    ----------
    x_marginals : Tensor
        Marginal distribution over node classes, shape ``(X_classes,)``.
    e_marginals : Tensor
        Marginal distribution over edge classes, shape ``(E_classes,)``.
    y_classes : int
        Number of global feature classes (0 to disable).
    """

    u_x: Tensor
    u_e: Tensor
    u_y: Tensor
    x_marginals: Tensor
    e_marginals: Tensor

    def __init__(
        self,
        x_marginals: Tensor,
        e_marginals: Tensor,
        y_classes: int,
    ) -> None:
        super().__init__()
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes

        self.register_buffer("x_marginals", x_marginals.clone())
        self.register_buffer("e_marginals", e_marginals.clone())

        # Each row of u_x is the marginal distribution — shape (1, K, K)
        self.register_buffer(
            "u_x",
            x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0).clone(),
        )
        self.register_buffer(
            "u_e",
            e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0).clone(),
        )

        u_y = torch.ones(1, y_classes, y_classes)
        if y_classes > 0:
            u_y = u_y / y_classes
        self.register_buffer("u_y", u_y)

    def get_Qt(self, beta_t: Tensor) -> TransitionMatrices:
        """One-step transition matrices from step t-1 to step t.

        ``Qt = (1 - beta_t) * I + beta_t * M`` where ``M`` has marginal
        distributions as rows.

        Parameters
        ----------
        beta_t : Tensor
            Noise level, shape ``(bs,)``.

        Returns
        -------
        TransitionMatrices
            Transition matrices ``(q_x, q_e, q_y)``.
        """
        device = self.u_x.device
        beta_t = beta_t.reshape(-1, 1, 1).to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return TransitionMatrices(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: Tensor) -> TransitionMatrices:
        """Cumulative transition matrices from step 0 to step t.

        ``Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) * M``.

        Parameters
        ----------
        alpha_bar_t : Tensor
            Cumulative product of ``(1 - beta)``, shape ``(bs,)``.

        Returns
        -------
        TransitionMatrices
            Cumulative transition matrices ``(q_x, q_e, q_y)``.
        """
        device = self.u_x.device
        alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1).to(device)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_x
        )
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_e
        )
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * self.u_y
        )

        return TransitionMatrices(X=q_x, E=q_e, y=q_y)

    def get_limit_dist(self) -> LimitDistribution:
        """Return the limiting distribution as t -> infinity.

        For marginal transition, the limit is the marginal distribution itself.

        Returns
        -------
        LimitDistribution
            1-D distributions for X, E, and y.
        """
        x_limit = self.x_marginals.clone()
        e_limit = self.e_marginals.clone()
        y_limit = (
            torch.ones(self.y_classes) / self.y_classes
            if self.y_classes > 0
            else torch.zeros(0)
        )
        return LimitDistribution(X=x_limit, E=e_limit, y=y_limit)
