"""Unified noise process hierarchy for graph diffusion.

``NoiseProcess`` is the timestep-indexed corruption interface shared by
continuous and categorical graph diffusion. It exposes three core operations:

- ``sample_prior(node_mask)``
- ``forward_sample(x_0, t)``
- ``posterior_sample(z_t, x0_param, t, s)``

Concrete processes still carry some legacy helpers while later refactor chunks
remove the remaining transition-model and exact-density compatibility surface.
All process classes are ``nn.Module`` instances so schedules and any other
stateful subcomponents follow device placement automatically.
"""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tmgg.data.datasets.graph_types import GraphData, collapse_to_indices
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import NoiseDefinition

from .diffusion_math import sum_except_batch
from .diffusion_sampling import (
    compute_posterior_distribution,
    sample_discrete_feature_noise,
    sample_discrete_features,
)


def _continuous_edge_state(data: GraphData) -> Tensor:
    """Return a dense edge-valued tensor for continuous-process math.

    Continuous states and clean binary-topology graphs both lift through
    ``to_edge_state()`` so the forward and reverse process stay in dense
    edge-state space without binary-topology projections.
    """
    return data.to_edge_state()


def _masked_graph_log_prob(sample: GraphData, probs: GraphData) -> Tensor:
    """Sum masked categorical log-probabilities per sample.

    ``sample`` is typically a one-hot graph state and ``probs`` carries the
    corresponding categorical probabilities over the same support.
    """
    node_mask = sample.node_mask
    inv = ~node_mask
    inv_edge = inv.unsqueeze(1) | inv.unsqueeze(2)

    sample_x = sample.X.clone()
    sample_e = sample.E.clone()
    prob_x = probs.X.clone()
    prob_e = probs.E.clone()

    sample_x[inv] = 0.0
    sample_e[inv_edge] = 0.0
    prob_x[inv] = 1.0
    prob_e[inv_edge] = 1.0

    log_prob_x = sum_except_batch(sample_x * prob_x.clamp(min=1e-10).log())
    log_prob_e = sum_except_batch(sample_e * prob_e.clamp(min=1e-10).log())
    return log_prob_x + log_prob_e


def _normalise_counts_or_uniform(counts: Tensor) -> Tensor:
    """Return a PMF from counts, falling back to uniform on empty support."""
    if counts.numel() == 0:
        return counts
    total = counts.sum()
    if total <= 0:
        return torch.full_like(counts, 1.0 / counts.numel())
    return counts / total


def _mix_with_limit(features: Tensor, alpha: Tensor | float, limit: Tensor) -> Tensor:
    """Interpolate one-hot features with a stationary categorical PMF."""
    if isinstance(alpha, int | float):
        alpha = torch.tensor([alpha], device=features.device, dtype=torch.float32)
    alpha = alpha.to(device=features.device, dtype=torch.float32)
    alpha = alpha.view(-1, *([1] * (features.dim() - 1)))
    limit = limit.to(device=features.device, dtype=torch.float32)
    limit = limit.view(*([1] * (features.dim() - 1)), limit.numel())
    return alpha * features.float() + (1.0 - alpha) * limit


def _categorical_kernel(identity_weight: Tensor, limit: Tensor) -> Tensor:
    """Build batched categorical kernels from identity weight and a PMF."""
    if limit.numel() == 0:
        bs = int(identity_weight.reshape(-1).shape[0])
        return limit.new_zeros((bs, 0, 0))
    identity_weight = identity_weight.to(device=limit.device, dtype=torch.float32)
    identity_weight = identity_weight.reshape(-1, 1, 1)
    classes = limit.numel()
    eye = torch.eye(classes, device=limit.device, dtype=torch.float32).unsqueeze(0)
    stationary_rows = (
        limit.to(dtype=torch.float32)
        .view(1, 1, classes)
        .expand(identity_weight.shape[0], classes, classes)
    )
    return identity_weight * eye + (1.0 - identity_weight) * stationary_rows


class NoiseProcess(ABC, nn.Module):
    """Abstract base for timestep-indexed graph corruption processes."""

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def timesteps(self) -> int:
        """Number of discrete diffusion steps owned by the process."""
        ...

    def initialize_from_data(self, train_loader: DataLoader[GraphData]) -> None:
        """Initialise any data-dependent state from the training loader."""
        _ = train_loader

    def needs_data_initialization(self) -> bool:
        """Whether the process requires a dataloader-backed setup phase."""
        return False

    def is_initialized(self) -> bool:
        """Whether any required data-dependent state is already available."""
        return True

    @abstractmethod
    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Return the model-conditioning vector for process state ``t``."""
        ...

    def model_output_to_posterior_parameter(self, model_output: GraphData) -> GraphData:
        """Convert raw model output into the reverse-process parameterisation."""
        return model_output

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        """Decode the final latent state into the public sampling output."""
        return z_0

    @abstractmethod
    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Sample the process prior at timestep ``T``."""
        ...

    @abstractmethod
    def forward_sample(self, x_0: GraphData, t: Tensor) -> GraphData:
        """Sample ``q(z_t | x_0)`` at integer timesteps ``t``."""
        ...

    @abstractmethod
    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample ``q(z_s | z_t, x_0)`` or its model-parameterised analogue."""
        ...


class ExactDensityNoiseProcess(NoiseProcess, ABC):
    """Noise process with tractable forward, posterior, and prior densities."""

    @abstractmethod
    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor:
        """Log-density of the forward process at timestep ``t``."""
        ...

    @abstractmethod
    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Log-density of the reverse posterior at timestep pair ``(t, s)``."""
        ...

    @abstractmethod
    def prior_log_prob(self, x: GraphData) -> Tensor:
        """Log-density of the prior distribution."""
        ...


class ContinuousNoiseProcess(NoiseProcess):
    """Wraps a ``NoiseDefinition`` to produce the ``NoiseProcess`` interface.

    Wraps a NoiseDefinition (from tmgg.utils.noising) with a NoiseSchedule to
    bridge stateless noise functions into the diffusion training loop. The
    definition only sees the final noise level; this class handles timestep
    conversion and posterior computation.

    The forward process operates on dense edge-valued states. Clean discrete
    graphs are lifted through explicit binary-topology accessors, while
    in-flight continuous states remain lossless through ``to_edge_state()``.
    The timestep ``t`` is an integer index into the noise schedule; the noise
    process converts it to a noise level via ``get_noise_level()`` before
    passing it to the definition.

    Parameters
    ----------
    definition
        An existing ``NoiseDefinition`` instance (GaussianNoise, EdgeFlipNoise,
        DigressNoise, LogitNoise, etc.). Not an ``nn.Module``; stored as a
        plain attribute.
    schedule
        A ``NoiseSchedule`` used by ``get_posterior`` to look up
        ``alpha_bar`` at integer timesteps.  Registered as an ``nn.Module``
        submodule.
    """

    def __init__(self, definition: NoiseDefinition, schedule: NoiseSchedule) -> None:
        super().__init__()
        self.definition = definition
        self.noise_schedule = schedule  # nn.Module — auto-registered

    @property
    def timesteps(self) -> int:
        """Number of discrete diffusion steps owned by the schedule."""
        return self.noise_schedule.timesteps

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Return the default model-conditioning vector for timestep ``t``."""
        return t.float() / self.timesteps

    def model_output_to_posterior_parameter(self, model_output: GraphData) -> GraphData:
        """Continuous reverse diffusion uses the model output directly."""
        return model_output

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        """Decode the final dense edge state into a binary graph."""
        edge_state = z_0.to_edge_state()
        adj_final = (edge_state > 0.5).float()
        adj_final = (adj_final + adj_final.transpose(1, 2)).clamp(max=1.0)

        n_max = adj_final.shape[1]
        diag_idx = torch.arange(n_max, device=adj_final.device)
        adj_final[:, diag_idx, diag_idx] = 0.0

        mask_2d = z_0.node_mask.unsqueeze(-1) * z_0.node_mask.unsqueeze(-2)
        adj_final = adj_final * mask_2d.float()

        decoded = GraphData.from_binary_adjacency(adj_final)
        return GraphData(
            X=decoded.X,
            E=decoded.E,
            y=decoded.y,
            node_mask=z_0.node_mask,
        ).mask()

    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Sample a symmetric Gaussian prior in dense edge-state space."""
        if node_mask.dim() == 1:
            node_mask = node_mask.unsqueeze(0)

        bs, n = node_mask.shape
        edge_state = torch.randn(bs, n, n, device=node_mask.device)
        edge_state = torch.triu(edge_state, diagonal=1)
        edge_state = edge_state + edge_state.transpose(1, 2)
        mask_2d = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        edge_state = edge_state * mask_2d.float()
        return GraphData.from_edge_state(edge_state, node_mask=node_mask).mask()

    def forward_sample(self, x_0: GraphData, t: Tensor) -> GraphData:
        """Sample the forward process at timestep ``t``."""
        noise_level = self._schedule_to_level(t)
        return self._apply_noise(x_0, noise_level)

    def sample_at_level(self, x_0: GraphData, level: Tensor | float) -> GraphData:
        """Sample the forward process at an explicit continuous noise level."""
        return self._apply_noise(x_0, level)

    def _schedule_to_level(self, t_int: Tensor) -> Tensor:
        """Convert timestep to noise level via ``1 - alpha_bar(t)``."""
        return self.noise_schedule.get_noise_level(t_int)

    def _apply_noise(self, data: GraphData, noise_level: Tensor | float) -> GraphData:
        """Apply continuous noise via the wrapped definition."""
        edge_state = _continuous_edge_state(data)
        noisy_edge_state = self.definition.add_noise(edge_state, noise_level)
        return GraphData.from_edge_state(noisy_edge_state, node_mask=data.node_mask)

    def _posterior_parameters(
        self, z_t: GraphData, z_0: GraphData, t: Tensor, s: Tensor
    ) -> dict[str, Tensor]:
        """Compute the Gaussian posterior ``q(x_{t-1} | x_t, x_0)`` for the
        reverse diffusion step.

        Implements the closed-form posterior from Ho et al., "Denoising
        Diffusion Probabilistic Models", NeurIPS 2020, Eq. 6-7
        (arXiv:2006.11239).  With ``alpha_bar_t = alpha_bar(t)``, ``alpha_bar_s = alpha_bar(s)``,
        and ``beta_t = 1 - alpha_bar_t / alpha_bar_s``:

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
        adj_t = _continuous_edge_state(z_t)
        adj_0 = _continuous_edge_state(z_0)

        # Cumulative signal-retention factors from the noise schedule.
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t).view(-1, 1, 1)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s).view(-1, 1, 1)

        # Single-step noise between s and t: beta_t = 1 - alpha_bar_t / alpha_bar_s
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

    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample the Gaussian reverse posterior into edge-state space."""
        posterior = self._posterior_parameters(z_t, x0_param, t, s)
        mean = posterior["mean"]
        std = posterior["std"]
        noise = torch.randn_like(mean)
        sample = mean + std * noise
        if s.dim() == 0:
            s = s.unsqueeze(0)
        no_noise_mask = (s <= 0).view(-1, 1, 1)
        sample = torch.where(no_noise_mask, mean, sample)
        sample = (sample + sample.transpose(1, 2)) / 2.0
        return GraphData.from_edge_state(sample, node_mask=z_t.node_mask).mask()


class CategoricalNoiseProcess(ExactDensityNoiseProcess):
    """Scheduled categorical diffusion over one-hot graph states.

    The process owns only two public configuration choices:

    - the timestep schedule
    - the stationary categorical distribution mode

    ``limit_distribution="uniform"`` uses uniform class PMFs immediately.
    ``limit_distribution="empirical_marginal"`` defers PMF construction until
    ``initialize_from_data()`` can count class frequencies from the train
    loader. No public transition-object hierarchy remains.
    """

    _limit_x: Tensor | None
    _limit_e: Tensor | None
    _limit_y: Tensor | None

    def __init__(
        self,
        schedule: NoiseSchedule,
        x_classes: int,
        e_classes: int,
        y_classes: int = 0,
        limit_distribution: Literal["uniform", "empirical_marginal"] = "uniform",
    ) -> None:
        super().__init__()

        self.noise_schedule = schedule  # nn.Module — auto-registered
        self.x_classes = x_classes
        self.e_classes = e_classes
        self.y_classes = y_classes
        self.limit_distribution = limit_distribution
        self.register_buffer("_limit_x", None)
        self.register_buffer("_limit_e", None)
        self.register_buffer("_limit_y", None)

        if limit_distribution == "uniform":
            self._set_stationary_distribution(
                x_probs=(
                    torch.full((x_classes,), 1.0 / x_classes)
                    if x_classes > 0
                    else torch.zeros(0)
                ),
                e_probs=(
                    torch.full((e_classes,), 1.0 / e_classes)
                    if e_classes > 0
                    else torch.zeros(0)
                ),
                y_probs=(
                    torch.full((y_classes,), 1.0 / y_classes)
                    if y_classes > 0
                    else torch.zeros(0)
                ),
            )
        elif limit_distribution != "empirical_marginal":
            raise ValueError(
                "limit_distribution must be 'uniform' or 'empirical_marginal', "
                f"got {limit_distribution!r}"
            )

    @property
    def timesteps(self) -> int:
        """Number of discrete diffusion steps owned by the schedule."""
        return self.noise_schedule.timesteps

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Return the default model-conditioning vector for timestep ``t``."""
        return t.float() / self.timesteps

    def model_output_to_posterior_parameter(self, model_output: GraphData) -> GraphData:
        """Convert model logits into categorical probabilities."""
        return GraphData(
            X=F.softmax(model_output.X, dim=-1),
            E=F.softmax(model_output.E, dim=-1),
            y=model_output.y,
            node_mask=model_output.node_mask,
        )

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        """Decode the final one-hot state into categorical class indices."""
        return collapse_to_indices(z_0.mask())

    def initialize_from_data(self, train_loader: DataLoader[GraphData]) -> None:
        """Initialise empirical stationary marginals from the training loader."""
        if self.limit_distribution == "uniform" or self._limit_x is not None:
            return

        x_counts = torch.zeros(self.x_classes, dtype=torch.float64)
        e_counts = torch.zeros(self.e_classes, dtype=torch.float64)

        for batch in train_loader:
            if batch.node_mask.dim() == 1:
                node_mask = batch.node_mask.unsqueeze(0)
                X = batch.X.unsqueeze(0)
                E = batch.E.unsqueeze(0)
            else:
                node_mask = batch.node_mask
                X = batch.X
                E = batch.E

            x_counts += X[node_mask].sum(dim=0).to(torch.float64).cpu()

            _, n = node_mask.shape
            upper_triangle = torch.triu(
                torch.ones(n, n, dtype=torch.bool, device=node_mask.device),
                diagonal=1,
            ).unsqueeze(0)
            valid_edges = (
                node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & upper_triangle
            )
            if valid_edges.any():
                e_counts += E[valid_edges].sum(dim=0).to(torch.float64).cpu()

        x_marginals = _normalise_counts_or_uniform(x_counts).to(torch.float32)
        e_marginals = _normalise_counts_or_uniform(e_counts).to(torch.float32)
        self._set_stationary_distribution(
            x_probs=x_marginals,
            e_probs=e_marginals,
            y_probs=(
                torch.full((self.y_classes,), 1.0 / self.y_classes)
                if self.y_classes > 0
                else torch.zeros(0)
            ),
        )

    def needs_data_initialization(self) -> bool:
        """Empirical-marginal mode requires training-data statistics."""
        return self.limit_distribution == "empirical_marginal"

    def is_initialized(self) -> bool:
        """Return whether the stationary marginals are already available."""
        if not self.needs_data_initialization():
            return True
        return self._limit_x is not None

    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Sample from the categorical limit distribution at timestep ``T``."""
        x_limit, e_limit, y_limit = self._stationary_distribution()
        return sample_discrete_feature_noise(x_limit, e_limit, node_mask, y_limit)

    def forward_sample(self, x_0: GraphData, t: Tensor) -> GraphData:
        """Sample the forward categorical process at timestep ``t``."""
        alpha_bar = self._schedule_to_level(t)
        return self._apply_noise(x_0, alpha_bar)

    def _set_stationary_distribution(
        self,
        *,
        x_probs: Tensor,
        e_probs: Tensor,
        y_probs: Tensor,
    ) -> None:
        """Store stationary class PMFs as registered buffers."""
        target_device = self.noise_schedule.betas.device
        x_probs = x_probs.to(device=target_device, dtype=torch.float32)
        e_probs = e_probs.to(device=target_device, dtype=torch.float32)
        y_probs = y_probs.to(device=target_device, dtype=torch.float32)

        self._validate_stationary_distribution("x_probs", x_probs, self.x_classes)
        self._validate_stationary_distribution("e_probs", e_probs, self.e_classes)
        self._validate_stationary_distribution("y_probs", y_probs, self.y_classes)

        self._limit_x = x_probs
        self._limit_e = e_probs
        self._limit_y = y_probs

    @staticmethod
    def _validate_stationary_distribution(
        name: str, probs: Tensor, expected_classes: int
    ) -> None:
        """Validate a 1-D categorical PMF exactly once at write time."""
        if probs.dim() != 1 or probs.numel() != expected_classes:
            raise ValueError(
                f"{name} must have shape ({expected_classes},), "
                f"got {tuple(probs.shape)}"
            )
        if not torch.isfinite(probs).all():
            raise ValueError(f"{name} must be finite")
        if (probs < 0).any():
            raise ValueError(f"{name} must be non-negative")
        if probs.numel() > 0 and not torch.isclose(
            probs.sum(), torch.tensor(1.0, device=probs.device), atol=1e-5
        ):
            raise ValueError(f"{name} must sum to 1, got {probs.sum().item():.6f}")

    def _stationary_distribution(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return stationary PMFs or fail loudly if setup has not run."""
        if self._limit_x is None or self._limit_e is None or self._limit_y is None:
            raise RuntimeError(
                "Stationary categorical distribution not initialised. "
                "Call initialize_from_data() first."
            )
        return self._limit_x, self._limit_e, self._limit_y

    # -- NoiseProcess interface ----------------------------------------------

    def _schedule_to_level(self, t_int: Tensor) -> Tensor:
        """Convert timestep to alpha_bar via the schedule.

        For categorical noise the "noise level" is ``alpha_bar`` itself
        (not ``1 - alpha_bar``), since the transition matrices are
        parameterised by signal retention.
        """
        return self.noise_schedule.get_alpha_bar(t_int=t_int)

    def _apply_noise(self, data: GraphData, noise_level: Tensor | float) -> GraphData:
        """Apply categorical forward noise at the given alpha_bar level."""
        x_limit, e_limit, _ = self._stationary_distribution()
        prob_X = _mix_with_limit(data.X, noise_level, x_limit)
        prob_E = _mix_with_limit(data.E, noise_level, e_limit)
        sampled = sample_discrete_features(prob_X, prob_E, data.node_mask)
        return GraphData(
            X=F.one_hot(sampled.X.long(), num_classes=self.x_classes).float(),
            E=F.one_hot(sampled.E.long(), num_classes=self.e_classes).float(),
            y=data.y,
            node_mask=data.node_mask,
        ).mask()

    def _posterior_probabilities(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Compute categorical posterior distributions for the reverse step.

        Uses the Bayes rule formulation from DiGress:
        ``p(z_s | z_t, z_0) = z_t @ Qt(t).T * z_0 @ Qsb(s) / (z_0 @ Qtb(t) @ z_t.T)``

        Parameters
        ----------
        z_t
            Noisy one-hot graph at timestep *t*.
        x0_param
            Clean graph or model-predicted clean probabilities.
        t
            Current integer timestep, shape ``(bs,)``.
        s
            Previous integer timestep (= t - 1), shape ``(bs,)``.

        Returns
        -------
        GraphData
            Posterior probability distributions with the same dense graph
            extents as ``z_t``.
        """
        x_limit, e_limit, _ = self._stationary_distribution()

        beta_t = self.noise_schedule(t_int=t)  # (bs,)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s)
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)

        q_x = _categorical_kernel(1.0 - beta_t, x_limit)
        q_e = _categorical_kernel(1.0 - beta_t, e_limit)
        qsb_x = _categorical_kernel(alpha_bar_s, x_limit)
        qsb_e = _categorical_kernel(alpha_bar_s, e_limit)
        qtb_x = _categorical_kernel(alpha_bar_t, x_limit)
        qtb_e = _categorical_kernel(alpha_bar_t, e_limit)

        prob_X = compute_posterior_distribution(
            M=x0_param.X, M_t=z_t.X, Qt_M=q_x, Qsb_M=qsb_x, Qtb_M=qtb_x
        )
        prob_E = compute_posterior_distribution(
            M=x0_param.E, M_t=z_t.E, Qt_M=q_e, Qsb_M=qsb_e, Qtb_M=qtb_e
        )
        bs, n, de = z_t.E.shape[0], z_t.E.shape[1], z_t.E.shape[-1]

        return GraphData(
            X=prob_X,
            E=prob_E.reshape(bs, n, n, de),
            y=z_t.y,
            node_mask=z_t.node_mask,
        )

    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample the categorical reverse posterior into one-hot graph states."""
        posterior = self._posterior_probabilities(z_t, x0_param, t, s)
        sampled = sample_discrete_features(
            posterior.X, posterior.E, node_mask=z_t.node_mask
        )
        return GraphData(
            X=F.one_hot(sampled.X.long(), num_classes=self.x_classes).float(),
            E=F.one_hot(sampled.E.long(), num_classes=self.e_classes).float(),
            y=z_t.y,
            node_mask=z_t.node_mask,
        ).mask()

    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor:
        """Return ``log q(z_t | x_0)`` for sampled categorical graph states."""
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)
        x_limit, e_limit, _ = self._stationary_distribution()
        prob_x = _mix_with_limit(x_0.X, alpha_bar_t, x_limit)
        prob_e = _mix_with_limit(x_0.E, alpha_bar_t, e_limit)

        probs = GraphData(X=prob_x, E=prob_e, y=x_t.y, node_mask=x_t.node_mask)
        return _masked_graph_log_prob(x_t, probs)

    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Return ``log q(z_s | z_t, x0_param)`` for sampled graph states."""
        posterior = self._posterior_probabilities(z_t, x0_param, t, s)
        return _masked_graph_log_prob(x_s, posterior)

    def prior_log_prob(self, x: GraphData) -> Tensor:
        """Return ``log p(z_T)`` under the stationary categorical prior."""
        x_limit, e_limit, _ = self._stationary_distribution()

        prob_x = x_limit.to(x.X.device).view(1, 1, -1).expand_as(x.X)
        prob_e = e_limit.to(x.E.device).view(1, 1, 1, -1).expand_as(x.E)

        probs = GraphData(X=prob_x, E=prob_e, y=x.y, node_mask=x.node_mask)
        return _masked_graph_log_prob(x, probs)
