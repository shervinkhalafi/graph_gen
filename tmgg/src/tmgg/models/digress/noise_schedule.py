from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

from tmgg.models.digress.diffusion_utils import (
    PlaceHolder,
    cosine_beta_schedule_discrete,
    custom_beta_schedule_discrete,
)


class PredefinedNoiseScheduleDiscrete(nn.Module):
    """Lookup table for predefined (non-learned) discrete noise schedules.

    Stores precomputed betas, alphas, and cumulative alpha-bar values.
    Supports "cosine" and "custom" schedules.

    Parameters
    ----------
    noise_schedule : str
        Name of the schedule. One of ``"cosine"`` or ``"custom"``.
    timesteps : int
        Number of diffusion steps.
    """

    betas: torch.Tensor  # pyright: ignore[reportUninitializedInstanceVariable] — set by register_buffer
    timesteps: int

    def __init__(self, noise_schedule: str, timesteps: int) -> None:
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas_np = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "custom":
            betas_np = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(f"Unknown noise schedule: {noise_schedule!r}")

        self.register_buffer("betas", torch.from_numpy(betas_np).float())

        self.alphas: torch.Tensor = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar: torch.Tensor = torch.exp(log_alpha_bar)

    @override
    def forward(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return beta values at given timesteps.

        Exactly one of *t_normalized* or *t_int* must be provided.

        Parameters
        ----------
        t_normalized : Tensor, optional
            Normalised time in ``[0, 1]``, converted to integer indices via
            rounding.
        t_int : Tensor, optional
            Integer timestep indices directly.

        Returns
        -------
        Tensor
            Beta values at the requested timesteps.
        """
        if (t_normalized is None) == (t_int is None):
            raise ValueError("Exactly one of t_normalized or t_int must be provided.")
        if t_int is None:
            assert t_normalized is not None  # for type narrowing
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(
        self,
        t_normalized: torch.Tensor | None = None,
        t_int: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return cumulative product of alphas (alpha-bar) at given timesteps.

        Parameters
        ----------
        t_normalized : Tensor, optional
            Normalised time in ``[0, 1]``.
        t_int : Tensor, optional
            Integer timestep indices.

        Returns
        -------
        Tensor
            Alpha-bar values at the requested timesteps.
        """
        if (t_normalized is None) == (t_int is None):
            raise ValueError("Exactly one of t_normalized or t_int must be provided.")
        if t_int is None:
            assert t_normalized is not None
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class DiscreteUniformTransition:
    """Transition matrices for discrete diffusion with uniform stationary distribution.

    Each category's limiting distribution is ``1/K`` for ``K`` classes.

    Parameters
    ----------
    x_classes : int
        Number of node feature classes.
    e_classes : int
        Number of edge feature classes.
    y_classes : int
        Number of global feature classes (0 to disable).
    """

    def __init__(self, x_classes: int, e_classes: int, y_classes: int) -> None:
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device | str) -> PlaceHolder:
        """One-step transition matrices from step t-1 to step t.

        ``Qt = (1 - beta_t) * I + beta_t * U`` where ``U`` is the uniform
        matrix.

        Parameters
        ----------
        beta_t : Tensor
            Noise level, shape ``(bs,)``.
        device : device or str
            Target device.

        Returns
        -------
        PlaceHolder
            Transition matrices ``(q_x, q_e, q_y)`` each of shape
            ``(bs, d, d)``.
        """
        # Reshape to (bs, 1, 1) for correct broadcasting with (1, K, K) matrices.
        # The baseline used unsqueeze(1) → (bs, 1) which only broadcasts
        # correctly when bs == K; this fixes it for arbitrary batch sizes.
        beta_t = beta_t.reshape(-1, 1, 1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(
        self, alpha_bar_t: torch.Tensor, device: torch.device | str
    ) -> PlaceHolder:
        """Cumulative transition matrices from step 0 to step t.

        ``Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) * U``.

        Parameters
        ----------
        alpha_bar_t : Tensor
            Cumulative product of ``(1 - beta)``, shape ``(bs,)``.
        device : device or str
            Target device.

        Returns
        -------
        PlaceHolder
            Cumulative transition matrices ``(q_x, q_e, q_y)``.
        """
        alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

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

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_limit_dist(self) -> PlaceHolder:
        """Return the limiting distribution as t -> infinity.

        For uniform transition this is ``1/K`` for each class.

        Returns
        -------
        PlaceHolder
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
        return PlaceHolder(X=x_limit, E=e_limit, y=y_limit)


class MarginalUniformTransition:
    """Transition matrices for discrete diffusion with marginal stationary distribution.

    Rows of the stationary matrix are the empirical marginal distribution
    rather than uniform ``1/K``.  For ``y`` features, falls back to uniform.

    Parameters
    ----------
    x_marginals : Tensor
        Marginal distribution over node classes, shape ``(X_classes,)``.
    e_marginals : Tensor
        Marginal distribution over edge classes, shape ``(E_classes,)``.
    y_classes : int
        Number of global feature classes (0 to disable).
    """

    def __init__(
        self,
        x_marginals: torch.Tensor,
        e_marginals: torch.Tensor,
        y_classes: int,
    ) -> None:
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        # Each row of u_x is the marginal distribution — shape (1, K, K)
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device | str) -> PlaceHolder:
        """One-step transition matrices from step t-1 to step t.

        ``Qt = (1 - beta_t) * I + beta_t * M`` where ``M`` has marginal
        distributions as rows.

        Parameters
        ----------
        beta_t : Tensor
            Noise level, shape ``(bs,)``.
        device : device or str
            Target device.

        Returns
        -------
        PlaceHolder
            Transition matrices ``(q_x, q_e, q_y)``.
        """
        beta_t = beta_t.reshape(-1, 1, 1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(
        self, alpha_bar_t: torch.Tensor, device: torch.device | str
    ) -> PlaceHolder:
        """Cumulative transition matrices from step 0 to step t.

        ``Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) * M``.

        Parameters
        ----------
        alpha_bar_t : Tensor
            Cumulative product of ``(1 - beta)``, shape ``(bs,)``.
        device : device or str
            Target device.

        Returns
        -------
        PlaceHolder
            Cumulative transition matrices ``(q_x, q_e, q_y)``.
        """
        alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

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

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_limit_dist(self) -> PlaceHolder:
        """Return the limiting distribution as t -> infinity.

        For marginal transition, the limit is the marginal distribution itself.

        Returns
        -------
        PlaceHolder
            1-D distributions for X, E, and y.
        """
        x_limit = self.x_marginals.clone()
        e_limit = self.e_marginals.clone()
        y_limit = (
            torch.ones(self.y_classes) / self.y_classes
            if self.y_classes > 0
            else torch.zeros(0)
        )
        return PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
