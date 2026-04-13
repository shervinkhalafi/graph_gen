"""Tests for the Sampler hierarchy (CategoricalSampler, ContinuousSampler).

Test rationale
--------------
The ``Sampler`` ABC defines a ``sample()`` method that performs reverse
diffusion to generate graphs. ``CategoricalSampler`` implements ancestral
sampling for categorical discrete diffusion. ``ContinuousSampler`` implements
Gaussian reverse diffusion for adjacency-based models.

Each test validates:
- ABC contract (subclass relationship)
- Shared sampler loop across the semantic wrapper classes
- Correct output types and shapes from ``sample()``
- Graph invariants: symmetry and valid one-hot encoding

Dummy models return trivial outputs (uniform or identity) so the tests
exercise only the sampling logic, not model quality.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
)
from tmgg.diffusion.sampler import (
    CategoricalSampler,
    ContinuousSampler,
    DiffusionState,
    Sampler,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel
from tmgg.utils.noising.noise import GaussianNoise

# ---------------------------------------------------------------------------
# Dummy models for testing
# ---------------------------------------------------------------------------


class UniformCategoricalModel(GraphModel):
    """Returns uniform distribution over classes for both X and E.

    Used to test that the categorical sampler produces valid outputs
    regardless of model quality; uniform predictions still lead to valid
    probability distributions through the posterior computation.
    """

    def __init__(self, dx: int, de: int) -> None:
        super().__init__()
        self.dx = dx
        self.de = de

    def get_config(self) -> dict[str, Any]:
        return {"dx": self.dx, "de": self.de}

    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData:
        bs, n, _ = data.X.shape
        X = torch.ones(bs, n, self.dx, device=data.X.device) / self.dx
        E = torch.ones(bs, n, n, self.de, device=data.E.device) / self.de
        return GraphData(X=X, E=E, y=data.y, node_mask=data.node_mask)


class IdentityContinuousModel(GraphModel):
    """Returns the input graph unchanged.

    Used to test that the continuous sampler pipeline works end-to-end.
    With an identity model, the sampler still exercises schedule lookups,
    posterior computation, and final thresholding.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        return {}

    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData:
        return data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T_STEPS = 5
N_NODES = 4
BS = 2
DX = 3
DE = 2


# cosine_schedule_short fixture (T=5) provided by conftest.py


@pytest.fixture()
def unified_schedule() -> NoiseSchedule:
    """NoiseSchedule (our new unified wrapper) for both sampler types."""
    return NoiseSchedule("cosine_iddpm", timesteps=T_STEPS)


@pytest.fixture()
def categorical_noise_process(
    cosine_schedule_short: NoiseSchedule,
) -> CategoricalNoiseProcess:
    return CategoricalNoiseProcess(
        schedule=cosine_schedule_short,
        x_classes=DX,
        e_classes=DE,
        limit_distribution="uniform",
    )


@pytest.fixture()
def continuous_noise_process() -> ContinuousNoiseProcess:
    return ContinuousNoiseProcess(
        definition=GaussianNoise(),
        schedule=NoiseSchedule("cosine_iddpm", timesteps=T_STEPS),
    )


@pytest.fixture()
def uniform_model() -> UniformCategoricalModel:
    return UniformCategoricalModel(dx=DX, de=DE)


@pytest.fixture()
def identity_model() -> IdentityContinuousModel:
    return IdentityContinuousModel()


# ---------------------------------------------------------------------------
# DiffusionState validation tests
# ---------------------------------------------------------------------------


class TestDiffusionState:
    """DiffusionState should validate timestep metadata and batched graph shape."""

    def test_constructs_for_valid_batched_graph(self) -> None:
        graph = GraphData.from_binary_adjacency(torch.zeros(2, N_NODES, N_NODES))
        state = DiffusionState(graph=graph, t=3, max_t=T_STEPS)
        assert state.t == 3
        assert state.max_t == T_STEPS
        assert state.graph is graph

    def test_rejects_negative_timestep(self) -> None:
        graph = GraphData.from_binary_adjacency(torch.zeros(2, N_NODES, N_NODES))
        with pytest.raises(ValueError, match="0 <= t <= max_t"):
            DiffusionState(graph=graph, t=-1, max_t=T_STEPS)

    def test_rejects_timestep_above_max(self) -> None:
        graph = GraphData.from_binary_adjacency(torch.zeros(2, N_NODES, N_NODES))
        with pytest.raises(ValueError, match="0 <= t <= max_t"):
            DiffusionState(graph=graph, t=T_STEPS + 1, max_t=T_STEPS)

    def test_rejects_unbatched_graph(self) -> None:
        graph = GraphData.from_binary_adjacency(torch.zeros(N_NODES, N_NODES))
        with pytest.raises(ValueError, match="must be batched GraphData"):
            DiffusionState(graph=graph, t=1, max_t=T_STEPS)

    def test_rejects_mismatched_mask_shape(self) -> None:
        graph = GraphData(
            X=torch.zeros(BS, N_NODES, DX),
            E=torch.zeros(BS, N_NODES, N_NODES, DE),
            y=torch.zeros(BS, 0),
            node_mask=torch.ones(BS, N_NODES + 1, dtype=torch.bool),
        )
        with pytest.raises(ValueError, match="node_mask shape"):
            DiffusionState(graph=graph, t=1, max_t=T_STEPS)


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestSamplerABC:
    """Both sampler implementations must be instances of Sampler."""

    def test_categorical_sampler_is_sampler(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        _ = categorical_noise_process
        _ = unified_schedule
        sampler = CategoricalSampler()
        assert isinstance(sampler, Sampler)

    def test_continuous_sampler_is_sampler(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        _ = continuous_noise_process
        _ = unified_schedule
        sampler = ContinuousSampler()
        assert isinstance(sampler, Sampler)


# ---------------------------------------------------------------------------
# Unified-loop tests
# ---------------------------------------------------------------------------


class TestUnifiedSamplerAliases:
    """Categorical and continuous wrappers should share one sampler loop."""

    def test_categorical_alias_uses_base_sample_implementation(self) -> None:
        assert CategoricalSampler.sample is Sampler.sample

    def test_continuous_alias_uses_base_sample_implementation(self) -> None:
        assert ContinuousSampler.sample is Sampler.sample


# ---------------------------------------------------------------------------
# CategoricalSampler.sample() tests
# ---------------------------------------------------------------------------


class TestCategoricalSamplerSample:
    """Verify CategoricalSampler.sample() output type, shapes, and invariants."""

    def test_uses_noise_process_condition_vector_for_model_input(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Categorical sampling should delegate conditioning semantics.

        Test rationale: Step 1 removes sampler-owned timestep normalization.
        The sampler should ask the noise process for the model condition
        vector at each reverse step and forward that exact tensor.
        """
        sampler = CategoricalSampler()
        observed_condition: list[torch.Tensor | None] = []

        def condition_vector(t: Tensor) -> Tensor:
            return t.float() + 100.0

        original_forward = uniform_model.forward

        def spy_forward(data: GraphData, t: Tensor | None = None) -> GraphData:
            observed_condition.append(None if t is None else t.detach().clone())
            return original_forward(data, t=t)

        categorical_noise_process.process_state_condition_vector = condition_vector  # type: ignore[method-assign]
        uniform_model.forward = spy_forward  # type: ignore[assignment]

        sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )

        expected = [
            torch.full((BS,), float(step) + 100.0)
            for step in range(unified_schedule.timesteps, 0, -1)
        ]
        assert len(observed_condition) == len(expected)
        for actual, want in zip(observed_condition, expected, strict=True):
            assert actual is not None
            torch.testing.assert_close(actual, want)

    def test_returns_list_of_graph_data(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """sample() returns a list of GraphData with the correct length."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert isinstance(results, list)
        assert len(results) == BS
        for g in results:
            assert isinstance(g, GraphData)

    def test_output_shape_correctness(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Generated graphs have the expected X and E dimensions."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            # After collapse_to_indices, X is (n,) and E is (n, n)
            assert (
                g.X.dim() == 1
            ), f"Expected X to be 1-D (collapsed), got shape {g.X.shape}"
            assert g.X.shape[0] == N_NODES
            assert (
                g.E.dim() == 2
            ), f"Expected E to be 2-D (collapsed), got shape {g.E.shape}"
            assert g.E.shape == (N_NODES, N_NODES)

    def test_edge_symmetry(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Generated edge features are symmetric (E[i,j] == E[j,i])."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert torch.equal(g.E, g.E.T), "Edge indices must be symmetric"

    def test_valid_class_indices(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Node and edge class indices are within valid ranges."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            # Collapsed X values should be in [0, dx)
            assert (g.X >= 0).all() and (
                g.X < DX
            ).all(), f"X class indices out of range: min={g.X.min()}, max={g.X.max()}"
            # Collapsed E values should be in [0, de) for valid positions
            valid_mask = g.node_mask.unsqueeze(0) * g.node_mask.unsqueeze(1)
            valid_E = g.E[valid_mask.bool()]
            assert (
                (valid_E >= 0).all() and (valid_E < DE).all()
            ), f"E class indices out of range: min={valid_E.min()}, max={valid_E.max()}"

    def test_variable_node_counts(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Sampling with a Tensor of per-graph node counts works correctly."""
        sampler = CategoricalSampler()
        num_nodes = torch.tensor([3, 5])
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=2,
            num_nodes=num_nodes,
            device=torch.device("cpu"),
        )
        assert len(results) == 2
        assert results[0].X.shape[0] == 3
        assert results[1].X.shape[0] == 5

    def test_sample_no_gradients_tracked(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """sample() must not track gradients even when called outside no_grad().

        Test rationale: the reverse diffusion loop allocates many
        intermediates; tracking gradients wastes memory with no benefit
        since sampling is always inference-only.
        """
        sampler = CategoricalSampler()
        # Ensure we are NOT inside a no_grad context
        assert torch.is_grad_enabled()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert not g.X.requires_grad
            assert not g.E.requires_grad

    def test_warm_start_still_works_with_explicit_noise_process(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Warm-start sampling should still work after ownership moves.

        Test rationale: Step 2 removes sampler-owned process state. Passing the
        process explicitly must not break partial reverse chains that resume
        from an existing latent graph.
        """
        sampler = CategoricalSampler()
        node_mask = torch.ones(BS, N_NODES, dtype=torch.bool)
        start_graph = categorical_noise_process.sample_prior(node_mask)
        start_from = DiffusionState(
            graph=start_graph,
            t=unified_schedule.timesteps - 1,
            max_t=unified_schedule.timesteps,
        )

        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
            start_from=start_from,
        )

        assert len(results) == BS


# ---------------------------------------------------------------------------
# CategoricalSampler with marginal transitions (T-5)
# ---------------------------------------------------------------------------


class TestCategoricalSamplerMarginal:
    """CategoricalSampler with empirical stationary PMFs instead of uniform.

    Test rationale: all existing categorical sampler tests use the uniform
    stationary mode. These cases verify that loader-initialised empirical
    marginals also produce valid reverse sampling behavior.
    """

    @pytest.fixture()
    def marginal_noise_process(
        self,
        cosine_schedule_short: NoiseSchedule,
    ) -> CategoricalNoiseProcess:
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule_short,
            x_classes=DX,
            e_classes=DE,
            limit_distribution="empirical_marginal",
        )
        X = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        E = torch.zeros(1, N_NODES, N_NODES, DE)
        E[..., 0] = 1.0
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            E[0, i, j] = torch.tensor([0.0, 1.0])
            E[0, j, i] = torch.tensor([0.0, 1.0])
        batch = GraphData(
            X=X,
            E=E,
            y=torch.zeros(1, 0),
            node_mask=torch.ones(1, N_NODES, dtype=torch.bool),
        )
        proc.initialize_from_data([batch])  # type: ignore[arg-type]
        return proc

    def test_returns_valid_graph_data(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Marginal-transition sampling produces valid GraphData output."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert len(results) == BS
        for g in results:
            assert isinstance(g, GraphData)
            assert g.X.shape[0] == N_NODES

    def test_valid_class_indices(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Class indices remain in valid ranges with empirical marginals."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert (g.X >= 0).all() and (g.X < DX).all()

    def test_edge_symmetry(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Symmetry invariant holds with empirical marginals."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert torch.equal(g.E, g.E.T)


# ---------------------------------------------------------------------------
# ContinuousSampler.sample() tests
# ---------------------------------------------------------------------------


class TestContinuousSamplerSample:
    """Verify ContinuousSampler.sample() output type, shapes, and invariants."""

    def test_uses_noise_process_condition_vector_for_model_input(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Continuous sampling should delegate conditioning semantics.

        Test rationale: the continuous sampler previously derived schedule
        noise levels directly, which diverged from training-time conditioning.
        This test locks the sampler to the noise-process-owned condition
        vector instead.
        """
        sampler = ContinuousSampler()
        observed_condition: list[torch.Tensor | None] = []

        def condition_vector(t: Tensor) -> Tensor:
            return t.float() + 100.0

        original_forward = identity_model.forward

        def spy_forward(data: GraphData, t: Tensor | None = None) -> GraphData:
            observed_condition.append(None if t is None else t.detach().clone())
            return original_forward(data, t=t)

        continuous_noise_process.process_state_condition_vector = condition_vector  # type: ignore[method-assign]
        identity_model.forward = spy_forward  # type: ignore[assignment]

        sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )

        expected = [
            torch.full((BS,), float(step) + 100.0)
            for step in range(unified_schedule.timesteps, 0, -1)
        ]
        assert len(observed_condition) == len(expected)
        for actual, want in zip(observed_condition, expected, strict=True):
            assert actual is not None
            torch.testing.assert_close(actual, want)

    def test_returns_list_of_graph_data(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """sample() returns a list of GraphData with the correct length."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert isinstance(results, list)
        assert len(results) == BS
        for g in results:
            assert isinstance(g, GraphData)

    def test_output_shape_correctness(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Generated graphs have expected adjacency-based shapes."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            # After thresholding, the output is stored as one-hot de=2
            # but trimmed per graph. Check that E is square.
            n = g.X.shape[0] if g.X.dim() == 1 else g.X.shape[-2]
            assert g.E.shape[-2] == g.E.shape[-1] or g.E.shape[0] == n

    def test_binary_adjacency(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Final adjacency values should be binary (0 or 1)."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_binary_adjacency()
            unique_vals = torch.unique(adj)
            for v in unique_vals:
                assert v.item() in (
                    0.0,
                    1.0,
                ), f"Expected binary adjacency, found value {v.item()}"

    def test_edge_symmetry(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Generated adjacency should be symmetric."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_binary_adjacency()
            if adj.dim() == 2:
                assert torch.equal(adj, adj.T), "Adjacency must be symmetric"
            elif adj.dim() == 3:
                assert torch.equal(
                    adj, adj.transpose(-1, -2)
                ), "Adjacency must be symmetric"

    def test_zero_diagonal(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """No self-loops: diagonal of adjacency should be zero."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_binary_adjacency()
            if adj.dim() == 2:
                assert torch.diag(adj).sum() == 0, "Self-loops must be zero"
            elif adj.dim() == 3:
                for b in range(adj.shape[0]):
                    assert torch.diag(adj[b]).sum() == 0, "Self-loops must be zero"

    def test_sample_no_gradients_tracked(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """sample() must not track gradients even when called outside no_grad().

        Test rationale: the reverse diffusion loop allocates many
        intermediates; tracking gradients wastes memory with no benefit
        since sampling is always inference-only.
        """
        sampler = ContinuousSampler()
        assert torch.is_grad_enabled()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_binary_adjacency()
            assert not adj.requires_grad

    def test_warm_start_still_works_with_explicit_noise_process(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Warm-start sampling should still work after ownership moves."""
        sampler = ContinuousSampler()
        node_mask = torch.ones(BS, N_NODES, dtype=torch.bool)
        start_graph = continuous_noise_process.sample_prior(node_mask)
        start_from = DiffusionState(
            graph=start_graph,
            t=unified_schedule.timesteps - 1,
            max_t=unified_schedule.timesteps,
        )

        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
            start_from=start_from,
        )

        assert len(results) == BS
