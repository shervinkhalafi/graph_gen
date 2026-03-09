"""Composable reverse-diffusion samplers for graph generation.

Provides ``Sampler`` as the abstract base, with two concrete implementations:

- ``CategoricalSampler`` performs ancestral categorical sampling, relocating
  the logic from ``DiffusionModule.sample_batch()`` and
  ``_sample_p_zs_given_zt()`` into a standalone, model-agnostic component.
- ``ContinuousSampler`` performs Gaussian reverse diffusion over adjacency
  matrices, using the posterior mean/std from ``ContinuousNoiseProcess``.

Both accept a ``GraphModel``, a typed ``NoiseProcess``, and a
``NoiseSchedule``, and expose a single ``sample()`` method that returns a
list of individual ``GraphData`` graphs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData, collapse_to_indices
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel

from .diffusion_sampling import (
    compute_batched_over0_posterior_distribution,
    sample_discrete_feature_noise,
    sample_discrete_features,
)


class Sampler(ABC):
    """Abstract base for reverse-diffusion graph samplers.

    Subclasses implement the full reverse loop: starting from pure noise
    (limit distribution or Gaussian), iteratively denoising through
    ``T`` steps to produce clean graphs.
    """

    @abstractmethod
    def sample(
        self,
        model: GraphModel,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
    ) -> list[GraphData]:
        """Generate graphs via reverse diffusion.

        Parameters
        ----------
        model
            Trained ``GraphModel`` that predicts clean data from noisy input.
        num_graphs
            Number of graphs to generate.
        num_nodes
            Either a fixed node count (int) applied to all graphs, or a
            per-graph tensor of shape ``(num_graphs,)``.
        device
            Device on which to run sampling.

        Returns
        -------
        list[GraphData]
            One ``GraphData`` per generated graph, trimmed to its real
            node count.
        """
        ...


class CategoricalSampler(Sampler):
    """Ancestral categorical reverse-diffusion sampler.

    Relocates the sampling logic from
    ``DiffusionModule.sample_batch()`` into a composable
    component. At each reverse step, it computes the posterior
    ``p(z_s | z_t) = sum_{x_0} p(x_0|z_t) * q(z_s|z_t,x_0)`` using the
    model's prediction of the clean graph and the analytic transition
    posterior from the categorical noise process.

    Parameters
    ----------
    noise_process
        A ``CategoricalNoiseProcess`` instance (with transition model
        already initialised via ``setup()`` if using marginal transitions).
    noise_schedule
        Unified ``NoiseSchedule`` providing beta, alpha_bar lookups.

    Raises
    ------
    TypeError
        If *noise_process* is not a ``CategoricalNoiseProcess``.
    """

    def __init__(
        self,
        noise_process: CategoricalNoiseProcess,
        noise_schedule: NoiseSchedule,
    ) -> None:
        if not isinstance(noise_process, CategoricalNoiseProcess):
            raise TypeError(
                f"CategoricalSampler requires a CategoricalNoiseProcess, "
                f"got {type(noise_process).__name__}"
            )
        self.noise_process = noise_process
        self.noise_schedule = noise_schedule

    @torch.no_grad()
    def sample(
        self,
        model: GraphModel,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
    ) -> list[GraphData]:
        """Generate graphs via ancestral categorical reverse diffusion.

        Parameters
        ----------
        model
            Trained model returning predicted clean-graph probabilities
            (softmaxed X and E).
        num_graphs
            Batch size for generation.
        num_nodes
            Fixed node count or per-graph tensor.
        device
            Target device.

        Returns
        -------
        list[GraphData]
            Individual graphs with integer class indices, trimmed to
            their real node counts.
        """
        bs = num_graphs
        T = self.noise_schedule.timesteps

        # Build node mask
        if isinstance(num_nodes, int):
            n_nodes = torch.full((bs,), num_nodes, device=device, dtype=torch.long)
        else:
            n_nodes = num_nodes.to(device)

        n_max = int(n_nodes.max().item())
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample z_T from the limit distribution
        transition = self.noise_process.transition_model
        limit_dist = transition.get_limit_dist()
        z_T = sample_discrete_feature_noise(limit_dist, node_mask)
        X, E, y = z_T.X.to(device), z_T.E.to(device), z_T.y.to(device)

        dx = self.noise_process.x_classes
        de = self.noise_process.e_classes

        # Reverse diffusion loop: ancestral sampling from t=T down to t=1.
        # Implements Eq. 3-4 of Vignac et al., "DiGress: Discrete Denoising
        # Diffusion for Graph Generation" (ICLR 2023).
        for s_int in reversed(range(0, T)):
            t_int = s_int + 1
            s_tensor = torch.full((bs, 1), s_int, device=device, dtype=torch.long)
            t_tensor = torch.full((bs, 1), t_int, device=device, dtype=torch.long)

            # Schedule parameters via the unified NoiseSchedule
            beta_t = self.noise_schedule.get_beta(t_int=t_tensor)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s_tensor)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=t_tensor)

            # Transition matrices
            Qtb = transition.get_Qt_bar(alpha_t_bar)
            Qsb = transition.get_Qt_bar(alpha_s_bar)
            Qt = transition.get_Qt(beta_t)

            # Model prediction: p(x_0 | z_t)
            data = GraphData(X=X, E=E, y=y, node_mask=node_mask)
            # Pass normalised timestep to the model; squeeze to (bs,) since
            # GraphModel.forward expects 1-D timestep.
            t_norm = (t_tensor / T).squeeze(-1)
            pred = model.forward(data, t_norm)
            # Model returns logits; convert to probabilities for the posterior.
            pred_X = F.softmax(pred.X, dim=-1)  # (bs, n, dx)
            pred_E = F.softmax(pred.E, dim=-1)  # (bs, n, n, de)

            # Posterior: p(z_s | z_t, x_0) ∝ q(z_t | z_s) · q(z_s | x_0)
            # computed for every possible x_0 class (DiGress Eq. 4).
            p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
                X_t=X, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
                X_t=E, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )

            # Marginalise over x_0: p(z_s | z_t) = Σ_{x_0} p(x_0 | z_t) · p(z_s | z_t, x_0)
            # (DiGress Eq. 5), then normalise to a valid distribution.
            #
            # Degenerate posteriors (zero-sum unnormalised probability) are
            # caught and raised as errors rather than silently masked.  The
            # original DiGress codebase (github.com/cvignac/DiGress,
            # sample_p_zs_given_zt in diffusion/extra_features.py) replaces
            # zero rows with uniform 1e-5 and continues; we crash so the
            # problem surfaces immediately.
            #
            # Exception: diagonal edges (self-loops) and masked positions
            # are structurally zero by design (their one-hot features are
            # all-zero).  These are expected and handled by filling with a
            # small uniform value before normalisation.
            weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
            unnorm_prob_X = weighted_X.sum(dim=2)  # (bs, n, d_{t-1})

            zero_rows_X = torch.sum(unnorm_prob_X, dim=-1) == 0  # (bs, n)
            # For nodes, only valid (non-masked) positions matter.
            valid_zero_X = zero_rows_X & node_mask
            if valid_zero_X.any():
                n_bad = int(valid_zero_X.sum().item())
                raise RuntimeError(
                    f"Degenerate posterior: {n_bad} node(s) have zero-sum "
                    f"unnormalised probability at reverse step t={t_int}. "
                    f"This indicates the model predicted a distribution that "
                    f"is incompatible with the transition posterior. "
                    f"See DiGress reference: github.com/cvignac/DiGress"
                )
            unnorm_prob_X[zero_rows_X] = 1e-5
            prob_X = unnorm_prob_X / torch.sum(unnorm_prob_X, dim=-1, keepdim=True)

            pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
            unnorm_prob_E = weighted_E.sum(dim=-2)

            zero_rows_E = torch.sum(unnorm_prob_E, dim=-1) == 0  # (bs, n*n)
            # Build mask for valid edge positions: off-diagonal AND both
            # endpoints present.  Diagonal edges are structurally all-zero
            # (no self-loops), so their zero posterior is expected.
            diag_mask = torch.eye(n_max, device=device, dtype=torch.bool)
            diag_flat = diag_mask.reshape(-1).unsqueeze(0).expand(bs, -1)  # (bs, n*n)
            edge_valid = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).reshape(
                bs, -1
            )
            valid_zero_E = zero_rows_E & edge_valid & ~diag_flat
            if valid_zero_E.any():
                n_bad = int(valid_zero_E.sum().item())
                raise RuntimeError(
                    f"Degenerate posterior: {n_bad} edge(s) have zero-sum "
                    f"unnormalised probability at reverse step t={t_int}. "
                    f"This indicates the model predicted a distribution that "
                    f"is incompatible with the transition posterior. "
                    f"See DiGress reference: github.com/cvignac/DiGress"
                )
            unnorm_prob_E[zero_rows_E] = 1e-5
            prob_E = unnorm_prob_E / torch.sum(unnorm_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n_max, n_max, pred_E.shape[-1])

            # Sample from posterior
            sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

            X = F.one_hot(sampled_s.X.long(), num_classes=dx).float()  # pyright: ignore[reportAttributeAccessIssue]
            E = F.one_hot(sampled_s.E.long(), num_classes=de).float()  # pyright: ignore[reportAttributeAccessIssue]
            y = torch.zeros(bs, 0, device=device).type_as(X)

        # Collapse one-hot to class indices and trim per graph
        final = collapse_to_indices(
            GraphData(X=X, E=E, y=y, node_mask=node_mask).mask()
        )

        results: list[GraphData] = []
        for i in range(bs):
            n = int(n_nodes[i].item())
            results.append(
                GraphData(
                    X=final.X[i, :n].cpu(),
                    E=final.E[i, :n, :n].cpu(),
                    y=final.y[i].cpu(),
                    node_mask=final.node_mask[i, :n].cpu(),
                )
            )
        return results


class ContinuousSampler(Sampler):
    """Gaussian reverse-diffusion sampler for adjacency-based models.

    At each step, the model predicts the clean graph from the noisy
    input, and the ``ContinuousNoiseProcess`` computes the Gaussian
    posterior (mean and std). We sample from that posterior at every step
    except the final one (s=0), where we use the mean directly.

    Parameters
    ----------
    noise_process
        A ``ContinuousNoiseProcess`` instance.
    noise_schedule
        Unified ``NoiseSchedule`` providing noise-level lookups.

    Raises
    ------
    TypeError
        If *noise_process* is not a ``ContinuousNoiseProcess``.
    """

    def __init__(
        self,
        noise_process: ContinuousNoiseProcess,
        noise_schedule: NoiseSchedule,
    ) -> None:
        if not isinstance(noise_process, ContinuousNoiseProcess):
            raise TypeError(
                f"ContinuousSampler requires a ContinuousNoiseProcess, "
                f"got {type(noise_process).__name__}"
            )
        self.noise_process = noise_process
        self.noise_schedule = noise_schedule

    @torch.no_grad()
    def sample(
        self,
        model: GraphModel,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
    ) -> list[GraphData]:
        """Generate graphs via Gaussian reverse diffusion.

        Parameters
        ----------
        model
            Trained model predicting clean adjacency from noisy input.
        num_graphs
            Batch size for generation.
        num_nodes
            Fixed node count or per-graph tensor.
        device
            Target device.

        Returns
        -------
        list[GraphData]
            Individual graphs with binary adjacency, trimmed to their
            real node counts.
        """
        bs = num_graphs
        T = self.noise_schedule.timesteps

        # Build node mask
        if isinstance(num_nodes, int):
            n_nodes = torch.full((bs,), num_nodes, device=device, dtype=torch.long)
        else:
            n_nodes = num_nodes.to(device)

        n_max = int(n_nodes.max().item())

        # Start from pure symmetric Gaussian noise
        z_adj = torch.randn(bs, n_max, n_max, device=device)
        z_adj = (z_adj + z_adj.transpose(1, 2)) / 2.0

        # Reverse diffusion loop
        for s_int in reversed(range(0, T)):
            t_int_val = s_int + 1
            t_tensor = torch.full((bs,), t_int_val, device=device, dtype=torch.long)
            s_tensor = torch.full((bs,), s_int, device=device, dtype=torch.long)

            t_level = self.noise_schedule.get_noise_level(t_tensor)  # (bs,)

            # Build GraphData from current adjacency for model input
            z_data = GraphData.from_adjacency(z_adj.clamp(0, 1).round())
            # If batched, from_adjacency already handles it
            if z_data.X.device != device:
                z_data = z_data.to(device)

            # Model predicts clean graph
            x_0_pred = model.forward(z_data, t_level)

            # Compute posterior via noise process (integer timesteps —
            # the noise process queries its schedule for alpha_bar).
            posterior = self.noise_process.get_posterior(
                z_data, x_0_pred, t_tensor, s_tensor
            )
            mean = posterior["mean"]  # (bs, n, n)
            std = posterior["std"]  # (bs, n, n)

            if s_int > 0:
                # Sample from N(mean, std)
                noise = torch.randn_like(mean)
                z_adj = mean + std * noise
            else:
                # Final step: use mean directly (no noise)
                z_adj = mean

            # Keep symmetric
            z_adj = (z_adj + z_adj.transpose(1, 2)) / 2.0

        # Threshold to binary, symmetrise, zero diagonal
        adj_final = (z_adj > 0.5).float()
        adj_final = (adj_final + adj_final.transpose(1, 2)).clamp(max=1.0)
        # Zero diagonal
        diag_idx = torch.arange(n_max, device=device)
        adj_final[:, diag_idx, diag_idx] = 0.0

        # Build node mask for trimming
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Mask adjacency for invalid nodes
        mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        adj_final = adj_final * mask_2d.float()

        # Convert to GraphData per graph
        results: list[GraphData] = []
        for i in range(bs):
            n = int(n_nodes[i].item())
            g = GraphData.from_adjacency(adj_final[i, :n, :n].cpu())
            results.append(g)

        return results
