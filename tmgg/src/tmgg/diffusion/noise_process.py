"""Unified noise process hierarchy for graph diffusion.

Provides ``NoiseProcess`` as the abstract base (``ABC + nn.Module``), with two
concrete implementations:

- ``ContinuousNoiseProcess`` wraps existing ``NoiseGenerator`` subclasses and
  operates on adjacency-based graph representations.
- ``CategoricalNoiseProcess`` wraps existing transition models
  (``DiscreteUniformTransition``, ``MarginalUniformTransition``) and operates
  on one-hot categorical features via transition matrices.

Both expose the same ``apply``/``get_posterior`` interface, allowing the
training loop to be agnostic to the noise type. All subclasses are
``nn.Module``s so device management propagates automatically via ``.to()``.
"""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.noise import NoiseGenerator
from tmgg.diffusion.protocols import TransitionModel
from tmgg.diffusion.schedule import NoiseSchedule

from .diffusion_math import sum_except_batch
from .diffusion_sampling import (
    compute_posterior_distribution,
    mask_distributions,
    posterior_distributions,
    sample_discrete_features,
)


class NoiseProcess(ABC, nn.Module):
    """Abstract base for forward diffusion noise processes.

    Subclasses implement the forward noising step (``apply``) and the
    reverse posterior computation (``get_posterior``) for a specific noise
    parameterisation (continuous or categorical).

    Extends both ``ABC`` and ``nn.Module`` so that submodules (noise
    schedules, transition models) propagate device and state_dict
    automatically.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, data: GraphData, t: Tensor) -> GraphData:
        """Apply forward diffusion noise at timestep ``t``.

        Parameters
        ----------
        data
            Clean (or partially noised) graph data.
        t
            Integer timestep indices, shape ``(bs,)``. Both continuous and
            categorical subclasses convert internally to whatever their
            generator needs.

        Returns
        -------
        GraphData
            Noised graph data with the same tensor shapes as *data*.
        """
        ...

    @abstractmethod
    def get_posterior(
        self, z_t: GraphData, z_0: GraphData, t: Tensor, s: Tensor
    ) -> Any:
        """Compute the reverse posterior distribution for sampling.

        Parameters
        ----------
        z_t
            Noisy graph at timestep *t*.
        z_0
            Clean graph (or predicted clean graph).
        t
            Current timestep, shape ``(bs,)``.
        s
            Previous timestep, shape ``(bs,)``.

        Returns
        -------
        Any
            Posterior parameters. For continuous noise, a dict with ``mean``
            and ``std`` keys. For categorical noise, a ``GraphData`` holding
            posterior probability distributions.
        """
        ...


class ContinuousNoiseProcess(NoiseProcess):
    """Wraps a ``NoiseGenerator`` to produce the ``NoiseProcess`` interface.

    The forward process extracts the adjacency from ``GraphData``, delegates
    to the wrapped generator's ``add_noise``, and converts back. The timestep
    ``t`` is an integer index into the noise schedule; the noise process
    converts it to a noise level via ``get_noise_level()`` before passing
    to the generator.

    Parameters
    ----------
    generator
        An existing ``NoiseGenerator`` instance (Gaussian, EdgeFlip,
        DiGress, Logit, etc.).  Not an ``nn.Module`` — stored as a plain
        attribute.
    noise_schedule
        A ``NoiseSchedule`` used by ``get_posterior`` to look up
        ``alpha_bar`` at integer timesteps.  Registered as an ``nn.Module``
        submodule.
    """

    def __init__(
        self, generator: NoiseGenerator, noise_schedule: NoiseSchedule
    ) -> None:
        super().__init__()
        self.generator = generator
        self.noise_schedule = noise_schedule  # nn.Module — auto-registered

    def apply(self, data: GraphData, t: Tensor) -> GraphData:
        """Apply continuous noise via the wrapped generator.

        Parameters
        ----------
        data
            Batched ``GraphData`` with categorical edge features.
        t
            Integer timestep indices, shape ``(bs,)``. Converted to noise
            levels via the schedule before passing to the generator.

        Returns
        -------
        GraphData
            Noised graph data with the same shapes as *data*.
        """
        adj = data.to_adjacency()  # (bs, n, n)
        noise_levels = self.noise_schedule.get_noise_level(t)  # (bs,)
        noisy_adj = self.generator.add_noise(adj, noise_levels)
        return GraphData.from_adjacency(noisy_adj)

    def get_posterior(
        self, z_t: GraphData, z_0: GraphData, t: Tensor, s: Tensor
    ) -> dict[str, Tensor]:
        """Compute the Gaussian posterior ``q(x_{t-1} | x_t, x_0)`` for the
        reverse diffusion step.

        Implements the closed-form posterior from Ho et al., "Denoising
        Diffusion Probabilistic Models", NeurIPS 2020, Eq. 6-7
        (arXiv:2006.11239).  With ``ᾱ_t = alpha_bar(t)``, ``ᾱ_s = alpha_bar(s)``,
        and ``β_t = 1 - ᾱ_t / ᾱ_s``:

        .. math::

            \\tilde{\\mu}_t = \\frac{\\sqrt{\\bar\\alpha_s}\\,\\beta_t}
                             {1 - \\bar\\alpha_t}\\, x_0
                           + \\frac{\\sqrt{1 - \\beta_t}\\,(1 - \\bar\\alpha_s)}
                             {1 - \\bar\\alpha_t}\\, x_t

            \\tilde{\\beta}_t = \\frac{\\beta_t\\,(1 - \\bar\\alpha_s)}
                               {1 - \\bar\\alpha_t}

        Parameters
        ----------
        z_t
            Noisy graph at step *t*.
        z_0
            Clean or predicted clean graph.
        t
            Current integer timestep, shape ``(bs,)``.
        s
            Previous integer timestep (< t), shape ``(bs,)``.

        Returns
        -------
        dict[str, Tensor]
            ``"mean"`` and ``"std"`` tensors of shape ``(bs, n, n)``.
        """
        adj_t = z_t.to_adjacency()
        adj_0 = z_0.to_adjacency()

        # Cumulative signal-retention factors from the noise schedule.
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t).view(-1, 1, 1)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s).view(-1, 1, 1)

        # Single-step noise between s and t: β_t = 1 − ᾱ_t / ᾱ_s
        beta_t = 1.0 - alpha_bar_t / alpha_bar_s.clamp(min=1e-8)

        # Denominators, clamped once for numerical safety.
        one_minus_alpha_bar_t = (1.0 - alpha_bar_t).clamp(min=1e-8)

        # Posterior mean (Ho et al. 2020, Eq. 7).
        coeff_x0 = torch.sqrt(alpha_bar_s) * beta_t / one_minus_alpha_bar_t
        coeff_xt = (
            torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_s) / one_minus_alpha_bar_t
        )
        mean = coeff_x0 * adj_0 + coeff_xt * adj_t

        # Posterior variance (Ho et al. 2020, Eq. 7).
        raw_var = beta_t * (1.0 - alpha_bar_s) / one_minus_alpha_bar_t
        if (raw_var < -1e-6).any():
            raise RuntimeError(
                f"Negative posterior variance detected "
                f"(min={raw_var.min().item():.6f}). "
                f"This indicates alpha_bar_s > alpha_bar_t, i.e. the noise "
                f"schedule is non-monotonic between the requested timesteps."
            )
        var = raw_var.clamp(min=1e-8)
        std = torch.sqrt(var).expand_as(mean)

        return {"mean": mean, "std": std}


class CategoricalNoiseProcess(NoiseProcess):
    """Wraps discrete transition models for the ``NoiseProcess`` interface.

    The transition model is injected directly rather than constructed
    internally. Pass a fully-configured ``DiscreteUniformTransition`` or
    ``MarginalUniformTransition`` at construction time, or defer by
    passing ``None`` and calling ``set_transition_model()`` before first
    use.

    **Why ``None`` is allowed:** ``MarginalUniformTransition`` requires
    empirical class marginals from the dataset, but in Lightning's
    lifecycle the noise process is created in ``__init__`` — before the
    datamodule is attached.  ``DiffusionModule.setup()`` bridges this
    gap: it reads marginals from the datamodule and injects the
    transition model via ``set_transition_model()``.  Calling any method
    that needs the transition model before injection raises
    ``RuntimeError`` immediately.

    Parameters
    ----------
    noise_schedule
        A ``NoiseSchedule`` instance that maps timesteps to beta / alpha_bar
        values.  Registered as an ``nn.Module`` submodule.
    x_classes
        Number of node feature classes.
    e_classes
        Number of edge feature classes.
    y_classes
        Number of global feature classes. Default 0 (unused).
    transition_model
        An already-constructed transition model, or ``None`` to defer.
        If ``None``, ``set_transition_model()`` **must** be called before
        ``apply``, ``get_posterior``, or any other method that touches the
        transition model — otherwise those methods raise ``RuntimeError``.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        x_classes: int,
        e_classes: int,
        y_classes: int = 0,
        transition_model: TransitionModel | None = None,
    ) -> None:
        super().__init__()

        self.noise_schedule = noise_schedule  # nn.Module — auto-registered
        self.x_classes = x_classes
        self.e_classes = e_classes
        self.y_classes = y_classes

        # Store as nn.Module attribute so PyTorch tracks it as a submodule
        # when it's not None (nn.Module.__setattr__ detects Module instances).
        self._transition_model: TransitionModel | None = None
        if transition_model is not None:
            self.set_transition_model(transition_model)

    def set_transition_model(self, model: TransitionModel) -> None:
        """Set the transition model after construction.

        Typical use: construct ``MarginalUniformTransition`` in
        ``DiffusionModule.setup()`` once marginals are available from
        the datamodule, then inject it here.

        Parameters
        ----------
        model
            A fully-configured transition model.
        """
        if not isinstance(model, TransitionModel):
            raise TypeError(
                f"Expected a TransitionModel (protocol requires get_Qt, "
                f"get_Qt_bar, get_limit_dist), got {type(model).__name__}"
            )
        self._transition_model = model

    @property
    def transition_model(self) -> TransitionModel:
        """The active transition model. Raises RuntimeError if not yet set."""
        if self._transition_model is None:
            raise RuntimeError(
                "Transition model not set. For uniform transitions, pass "
                "transition_model=DiscreteUniformTransition(...) at construction. "
                "For marginal transitions, DiffusionModule.setup() should call "
                "set_transition_model() with a MarginalUniformTransition built "
                "from the datamodule's marginals — if you see this error during "
                "training, the setup() hook did not run or the datamodule has no "
                "marginals."
            )
        assert self._transition_model is not None  # narrowed by guard above
        return self._transition_model

    def apply(self, data: GraphData, t: Tensor) -> GraphData:
        """Apply categorical forward diffusion at integer timesteps.

        Multiplies one-hot features by the cumulative transition matrix
        ``Qt_bar(t)`` to get class probabilities, then samples discrete
        features and converts back to one-hot encoding.

        Parameters
        ----------
        data
            One-hot encoded ``GraphData``, shapes ``(bs, n, dx)`` for X
            and ``(bs, n, n, de)`` for E.
        t
            Integer timestep indices, shape ``(bs,)``.

        Returns
        -------
        GraphData
            Noised one-hot ``GraphData`` with the same shapes as *data*.
        """
        transition = self.transition_model

        # Get alpha_bar at timestep t
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)  # (bs,)

        # Cumulative transition matrices
        Qtb = transition.get_Qt_bar(alpha_bar_t)

        # Compute probabilities: one-hot @ transition matrix
        # X: (bs, n, dx) @ (bs, dx, dx) -> (bs, n, dx)
        prob_X = data.X.float() @ Qtb.X
        # E: (bs, n, n, de) @ (bs, de, de) -> (bs, n, n, de)
        # Need to broadcast: reshape E for matmul
        bs, n, _, de = data.E.shape
        E_flat = data.E.float().reshape(bs, n * n, de)  # (bs, n*n, de)
        prob_E_flat = E_flat @ Qtb.E  # (bs, n*n, de)
        prob_E = prob_E_flat.reshape(bs, n, n, de)  # (bs, n, n, de)

        # Sample discrete features from the computed probabilities
        sampled = sample_discrete_features(prob_X, prob_E, data.node_mask)

        # Convert sampled indices to one-hot
        X_t = F.one_hot(sampled.X.long(), num_classes=self.x_classes).float()
        E_t = F.one_hot(sampled.E.long(), num_classes=self.e_classes).float()

        return GraphData(
            X=X_t,
            E=E_t,
            y=data.y,
            node_mask=data.node_mask,
        ).mask()

    def get_posterior(
        self, z_t: GraphData, z_0: GraphData, t: Tensor, s: Tensor
    ) -> GraphData:
        """Compute categorical posterior distributions for the reverse step.

        Uses the Bayes rule formulation from DiGress:
        ``p(z_s | z_t, z_0) = z_t @ Qt(t).T * z_0 @ Qsb(s) / (z_0 @ Qtb(t) @ z_t.T)``

        Parameters
        ----------
        z_t
            Noisy one-hot graph at timestep *t*.
        z_0
            Clean (or predicted clean) one-hot graph.
        t
            Current integer timestep, shape ``(bs,)``.
        s
            Previous integer timestep (= t - 1), shape ``(bs,)``.

        Returns
        -------
        GraphData
            Posterior probability distributions. X has shape ``(bs, n, dx)``,
            E has shape ``(bs, n*n, de)`` (flattened spatial dims from the
            posterior computation).
        """
        transition = self.transition_model

        beta_t = self.noise_schedule(t_int=t)  # (bs,)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s)
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)

        Qt = transition.get_Qt(beta_t)
        Qsb = transition.get_Qt_bar(alpha_bar_s)
        Qtb = transition.get_Qt_bar(alpha_bar_t)

        prob_X = compute_posterior_distribution(
            M=z_0.X, M_t=z_t.X, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
        )
        prob_E = compute_posterior_distribution(
            M=z_0.E, M_t=z_t.E, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
        )

        return GraphData(X=prob_X, E=prob_E, y=z_t.y, node_mask=z_t.node_mask)

    def kl_prior(
        self, X: Tensor, E: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """KL divergence between the final noised distribution and the limit.

        Computes ``KL(q(z_T | x_0) || p(z_T))`` for nodes, edges, and
        global features separately, where ``p(z_T)`` is the stationary
        distribution of the transition model.

        Parameters
        ----------
        X
            Clean one-hot node features, shape ``(bs, n, dx)``.
        E
            Clean one-hot edge features, shape ``(bs, n, n, de)``.
        node_mask
            Valid-node mask, shape ``(bs, n)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Scalar KL divergences ``(kl_X, kl_E, kl_y)``, each summed
            over the batch.
        """
        transition = self.transition_model
        limit_dist = transition.get_limit_dist()

        bs, n, dx = X.shape
        de = E.shape[-1]

        # Get alpha_bar at the final timestep T
        T = self.noise_schedule.timesteps
        t_T = torch.full((bs,), T, device=X.device, dtype=torch.long)
        alpha_bar_T = self.noise_schedule.get_alpha_bar(t_int=t_T)  # (bs,)

        Qtb_T = transition.get_Qt_bar(alpha_bar_T)

        # q(z_T | x_0) = x_0 @ Qtb_T
        prob_X = X.float() @ Qtb_T.X  # (bs, n, dx)
        prob_E_flat = E.float().reshape(bs, n * n, de) @ Qtb_T.E  # (bs, n*n, de)

        # Limit distribution
        limit_X = limit_dist.X.to(X.device)  # (dx,)
        limit_E = limit_dist.E.to(X.device)  # (de,)

        # KL(q || p) = sum_k q_k * log(q_k / p_k)
        # For numerical stability, clamp probabilities
        eps = 1e-10
        kl_X = prob_X * torch.log((prob_X + eps) / (limit_X + eps))
        kl_X = kl_X.sum(dim=-1)  # (bs, n)
        # Mask invalid nodes
        kl_X = (kl_X * node_mask.float()).sum()

        kl_E = prob_E_flat * torch.log((prob_E_flat + eps) / (limit_E + eps))
        kl_E = kl_E.sum(dim=-1)  # (bs, n*n)
        # Mask invalid edges
        edge_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).reshape(bs, n * n)
        kl_E = (kl_E * edge_mask.float()).sum()

        kl_y = torch.tensor(0.0, device=X.device)

        return kl_X, kl_E, kl_y

    def compute_Lt(
        self,
        clean: GraphData,
        pred: GraphData,
        noisy: GraphData,
        t_int: Tensor,
    ) -> Tensor:
        """Diffusion KL loss term L_t, scaled by T.

        Computes KL between the true posterior ``q(z_{t-1} | z_t, x_0)``
        and the predicted posterior ``q(z_{t-1} | z_t, hat{x}_0)``, then
        multiplies by the number of timesteps ``T`` so the term is
        properly weighted in the variational lower bound.

        Parameters
        ----------
        clean
            Clean one-hot ``GraphData``.
        pred
            Model's predicted probabilities (softmaxed).
        noisy
            Noised ``GraphData`` at timestep *t*.
        t_int
            Integer timesteps, shape ``(bs,)``.

        Returns
        -------
        Tensor
            Per-sample KL scaled by ``T``, shape ``(bs,)``.
        """
        transition = self.transition_model

        if (t_int < 1).any():
            raise ValueError(
                f"compute_Lt requires t_int >= 1 (needs s = t - 1 >= 0); "
                f"got min t_int={t_int.min().item()}"
            )
        beta_t = self.noise_schedule(t_int=t_int)
        s_int = t_int - 1
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s_int)
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t_int)

        Qt = transition.get_Qt(beta_t)
        Qsb = transition.get_Qt_bar(alpha_bar_s)
        Qtb = transition.get_Qt_bar(alpha_bar_t)

        bs, n, _ = clean.X.shape
        node_mask = noisy.node_mask

        # True posterior: q(z_{t-1} | z_t, x_0)
        prob_true = posterior_distributions(
            X=clean.X,
            E=clean.E,
            y=clean.y,
            X_t=noisy.X,
            E_t=noisy.E,
            y_t=noisy.y,
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
            node_mask=node_mask,
        )
        prob_true_E = prob_true.E.reshape((bs, n, n, -1))

        # Predicted posterior: q(z_{t-1} | z_t, hat{x}_0)
        prob_pred = posterior_distributions(
            X=pred.X,
            E=pred.E,
            y=pred.y,
            X_t=noisy.X,
            E_t=noisy.E,
            y_t=noisy.y,
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
            node_mask=node_mask,
        )
        prob_pred_E = prob_pred.E.reshape((bs, n, n, -1))

        # Mask invalid positions to uniform so they don't affect the KL
        true_X, true_E, pred_X, pred_E = mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true_E,
            pred_X=prob_pred.X,
            pred_E=prob_pred_E,
            node_mask=node_mask,
        )

        kl_x = F.kl_div(
            input=pred_X.clamp(min=1e-10).log(), target=true_X, reduction="none"
        )
        kl_e = F.kl_div(
            input=pred_E.clamp(min=1e-10).log(), target=true_E, reduction="none"
        )

        T = self.noise_schedule.timesteps
        return T * (sum_except_batch(kl_x) + sum_except_batch(kl_e))

    def reconstruction_logp(self, clean: GraphData, pred_probs: GraphData) -> Tensor:
        """Reconstruction log-probability ``log p(x | z_0)``.

        Computes the cross-entropy between clean one-hot data and the
        predicted class probabilities at ``t=0``.

        Parameters
        ----------
        clean
            Clean one-hot ``GraphData`` with shapes ``(bs, n, dx)`` for X
            and ``(bs, n, n, de)`` for E.
        pred_probs
            Predicted probability distributions at ``t=0``.

        Returns
        -------
        Tensor
            Per-sample log-probability, shape ``(bs,)``.
        """
        loss_X = sum_except_batch(clean.X * pred_probs.X.clamp(min=1e-10).log())
        loss_E = sum_except_batch(clean.E * pred_probs.E.clamp(min=1e-10).log())
        return loss_X + loss_E
