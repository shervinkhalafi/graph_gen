"""Composable reverse-diffusion sampling for graph generation.

``Sampler`` owns one reverse-chain loop that delegates process-specific
parameterisation and final decoding back to ``NoiseProcess`` hooks.
``CategoricalSampler`` and ``ContinuousSampler`` remain as semantic aliases
so configuration and call sites can still name the intended sampling mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.collectors import StepMetricCollector
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    NoiseProcess,
)
from tmgg.models.base import GraphModel


@dataclass(frozen=True, slots=True)
class DiffusionState:
    """A batched latent graph state together with its diffusion timestep."""

    graph: GraphData
    t: int
    max_t: int

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphData):
            raise TypeError(
                "DiffusionState.graph must be a GraphData instance, "
                f"got {type(self.graph).__name__}"
            )
        if isinstance(self.t, bool) or not isinstance(self.t, int):
            raise TypeError(
                f"DiffusionState.t must be an int, got {type(self.t).__name__}"
            )
        if isinstance(self.max_t, bool) or not isinstance(self.max_t, int):
            raise TypeError(
                f"DiffusionState.max_t must be an int, got {type(self.max_t).__name__}"
            )
        if self.max_t < 1:
            raise ValueError(f"DiffusionState.max_t must be >= 1, got {self.max_t}")
        if not 0 <= self.t <= self.max_t:
            raise ValueError(
                f"DiffusionState.t must satisfy 0 <= t <= max_t, got t={self.t}, "
                f"max_t={self.max_t}"
            )

        graph = self.graph
        if graph.X.dim() != 3 or graph.E.dim() != 4 or graph.y.dim() != 2:
            raise ValueError(
                "DiffusionState.graph must be batched GraphData with X (bs, n, dx), "
                "E (bs, n, n, de), and y (bs, dy)"
            )
        if graph.node_mask.dim() != 2:
            raise ValueError(
                "DiffusionState.graph.node_mask must have shape (bs, n), "
                f"got {tuple(graph.node_mask.shape)}"
            )

        bs, n = graph.node_mask.shape
        if graph.X.shape[:2] != (bs, n):
            raise ValueError(
                "DiffusionState.graph.X must align with node_mask shape (bs, n), "
                f"got X shape {tuple(graph.X.shape)} and node_mask shape {tuple(graph.node_mask.shape)}"
            )
        if graph.E.shape[:3] != (bs, n, n):
            raise ValueError(
                "DiffusionState.graph.E must align with node_mask shape (bs, n, n), "
                f"got E shape {tuple(graph.E.shape)} and node_mask shape {tuple(graph.node_mask.shape)}"
            )
        if graph.y.shape[0] != bs:
            raise ValueError(
                "DiffusionState.graph.y must align with batch size, "
                f"got y shape {tuple(graph.y.shape)} and batch size {bs}"
            )


class Sampler:
    """Unified reverse-diffusion sampler loop.

    The sampler owns the reverse-chain control flow only:
    prior or warm start -> condition vector -> model -> posterior parameter
    -> posterior sample -> finalize. Process-specific math stays on the
    ``NoiseProcess`` side of the interface.
    """

    @staticmethod
    def _build_num_nodes(
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
    ) -> Tensor:
        """Return per-graph node counts on ``device``."""
        if isinstance(num_nodes, int):
            return torch.full((num_graphs,), num_nodes, device=device, dtype=torch.long)
        return num_nodes.to(device=device, dtype=torch.long)

    @staticmethod
    def _trim_batched_graphs(final: GraphData, n_nodes: Tensor) -> list[GraphData]:
        """Split a batched final state into per-graph ``GraphData`` objects."""
        results: list[GraphData] = []
        for i, n_tensor in enumerate(n_nodes, start=0):
            n = int(n_tensor.item())
            x = final.X[i, :n].cpu()
            e = final.E[i, :n, :n].cpu()
            y = final.y[i].cpu()
            node_mask = final.node_mask[i, :n].cpu()
            results.append(GraphData(X=x, E=e, y=y, node_mask=node_mask))
        return results

    @staticmethod
    def _record_step_metrics(
        collector: StepMetricCollector,
        noise_process: NoiseProcess,
        z_t: GraphData,
        posterior_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> None:
        """Record per-step diagnostics for likelihood collectors."""
        t_int = int(t[0].item())
        s_int = int(s[0].item())

        if isinstance(noise_process, CategoricalNoiseProcess):
            posterior_probs = noise_process._posterior_probabilities(
                z_t, posterior_param, t, s
            )
            node_mask = z_t.node_mask
            ent_x = -torch.sum(
                posterior_probs.X * torch.log(posterior_probs.X.clamp(min=1e-30)),
                dim=-1,
            )
            ent_x = (ent_x * node_mask.float()).sum() / node_mask.sum().clamp(min=1)

            ent_e = -torch.sum(
                posterior_probs.E * torch.log(posterior_probs.E.clamp(min=1e-30)),
                dim=-1,
            )
            edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).float()
            ent_e = (ent_e * edge_mask).sum() / edge_mask.sum().clamp(min=1)

            collector.record(
                t_int,
                s_int,
                {
                    "kl": (ent_x + ent_e).item(),
                    "kl_X": ent_x.item(),
                    "kl_E": ent_e.item(),
                },
            )
            return

        if isinstance(noise_process, ContinuousNoiseProcess):
            posterior = noise_process._posterior_parameters(z_t, posterior_param, t, s)
            mean = posterior["mean"]
            std = posterior["std"]
            kl = 0.5 * (std.pow(2) + mean.pow(2) - 1 - 2 * std.clamp(min=1e-30).log())
            collector.record(t_int, s_int, {"kl": kl.mean().item()})
            return

        raise TypeError(
            "Collector metrics are only implemented for categorical and "
            f"continuous diffusion, got {type(noise_process).__name__}"
        )

    @torch.no_grad()
    def sample(
        self,
        model: GraphModel,
        noise_process: NoiseProcess,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
        *,
        start_from: DiffusionState | None = None,
        collector: StepMetricCollector | None = None,
    ) -> list[GraphData]:
        """Generate graphs via reverse diffusion.

        Parameters
        ----------
        model
            Trained ``GraphModel`` that predicts clean data from noisy input.
        noise_process
            Module-owned diffusion process used for the reverse chain.
        num_graphs
            Number of graphs to generate.
        num_nodes
            Either a fixed node count (int) applied to all graphs, or a
            per-graph tensor of shape ``(num_graphs,)``.
        device
            Device on which to run sampling.
        start_from
            Starting latent graph state and timestep for partial reverse
            chains. When ``None``, sampling starts from the process prior
            at ``t=T``.
        collector
            Per-step metric collector. When provided,
            ``record(t, s, metrics)`` is called at each reverse step
            with available metrics (e.g., ``{"kl": ...}``).

        Returns
        -------
        list[GraphData]
            One ``GraphData`` per generated graph, trimmed to its real
            node count.
        """
        if start_from is not None:
            if start_from.max_t != noise_process.timesteps:
                raise ValueError(
                    "Warm-start max_t must match the active noise process "
                    f"timesteps, got start_from.max_t={start_from.max_t} "
                    f"and noise_process.timesteps={noise_process.timesteps}"
                )
            z_t = start_from.graph.to(device)
            bs = z_t.node_mask.shape[0]
            if bs != num_graphs:
                raise ValueError(
                    "Warm-start batch size must match num_graphs, "
                    f"got batch size {bs} and num_graphs={num_graphs}"
                )
            n_nodes = z_t.node_mask.sum(dim=-1).long()
            t_start = start_from.t
        else:
            n_nodes = self._build_num_nodes(num_graphs, num_nodes, device)
            n_max = int(n_nodes.max().item())
            arange = (
                torch.arange(n_max, device=device).unsqueeze(0).expand(num_graphs, -1)
            )
            node_mask = arange < n_nodes.unsqueeze(1)
            z_t = noise_process.sample_prior(node_mask).to(device)
            bs = num_graphs
            t_start = noise_process.timesteps

        for s_int in reversed(range(0, t_start)):
            t_int = s_int + 1
            t_tensor = torch.full((bs,), t_int, device=device, dtype=torch.long)
            s_tensor = torch.full((bs,), s_int, device=device, dtype=torch.long)

            condition = noise_process.process_state_condition_vector(t_tensor)
            model_output = model(z_t, t=condition)
            posterior_param = noise_process.model_output_to_posterior_parameter(
                model_output
            )

            if collector is not None:
                self._record_step_metrics(
                    collector,
                    noise_process,
                    z_t,
                    posterior_param,
                    t_tensor,
                    s_tensor,
                )

            z_t = noise_process.posterior_sample(
                z_t, posterior_param, t_tensor, s_tensor
            )

        final = noise_process.finalize_sample(z_t)
        return self._trim_batched_graphs(final, n_nodes)


class CategoricalSampler(Sampler):
    """Semantic alias for the unified sampler loop."""


class ContinuousSampler(Sampler):
    """Semantic alias for the unified sampler loop."""
