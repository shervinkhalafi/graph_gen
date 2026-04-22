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

from tests._helpers.graph_builders import binary_graphdata
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
        assert data.X_class is not None
        assert data.E_class is not None
        bs, n, _ = data.X_class.shape
        X = torch.ones(bs, n, self.dx, device=data.X_class.device) / self.dx
        E = torch.ones(bs, n, n, self.de, device=data.E_class.device) / self.de
        return GraphData(
            y=data.y,
            node_mask=data.node_mask,
            X_class=X,
            E_class=E,
        )


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
        graph = binary_graphdata(torch.zeros(2, N_NODES, N_NODES))
        state = DiffusionState(graph=graph, t=3, max_t=T_STEPS)
        assert state.t == 3
        assert state.max_t == T_STEPS
        assert state.graph is graph

    def test_rejects_negative_timestep(self) -> None:
        graph = binary_graphdata(torch.zeros(2, N_NODES, N_NODES))
        with pytest.raises(ValueError, match="0 <= t <= max_t"):
            DiffusionState(graph=graph, t=-1, max_t=T_STEPS)

    def test_rejects_timestep_above_max(self) -> None:
        graph = binary_graphdata(torch.zeros(2, N_NODES, N_NODES))
        with pytest.raises(ValueError, match="0 <= t <= max_t"):
            DiffusionState(graph=graph, t=T_STEPS + 1, max_t=T_STEPS)

    def test_rejects_unbatched_graph(self) -> None:
        graph = binary_graphdata(torch.zeros(N_NODES, N_NODES))
        with pytest.raises(ValueError, match="node_mask must have shape"):
            DiffusionState(graph=graph, t=1, max_t=T_STEPS)

    def test_rejects_mismatched_mask_shape(self) -> None:
        # The GraphData constructor itself now rejects mismatched leading
        # dims; the DiffusionState contract still implicitly benefits
        # from that validation. We verify the failure surface at the
        # dataclass level.
        with pytest.raises(ValueError, match="leading dims"):
            GraphData(
                X_class=torch.zeros(BS, N_NODES, DX),
                E_class=torch.zeros(BS, N_NODES, N_NODES, DE),
                y=torch.zeros(BS, 0),
                node_mask=torch.ones(BS, N_NODES + 1, dtype=torch.bool),
            )


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
            # finalize_sample returns one-hot per-graph tensors; per-node
            # shape is (n, dx) and per-edge shape is (n, n, de).
            assert g.X_class is not None and g.E_class is not None
            assert g.X_class.dim() == 2
            assert g.X_class.shape == (N_NODES, DX)
            assert g.E_class.dim() == 3
            assert g.E_class.shape == (N_NODES, N_NODES, DE)

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
            assert g.E_class is not None
            assert torch.equal(
                g.E_class, g.E_class.transpose(0, 1)
            ), "Edge one-hots must be symmetric"

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
            # finalize_sample returns one-hot tensors. Each row of X_class
            # and each (i, j) position of E_class must sum to 1 at valid
            # positions (not counting the diagonal for edges).
            assert g.X_class is not None and g.E_class is not None
            x_sums = g.X_class.sum(dim=-1)
            assert torch.allclose(
                x_sums[g.node_mask], torch.ones_like(x_sums[g.node_mask])
            )
            valid_mask = g.node_mask.unsqueeze(0) * g.node_mask.unsqueeze(1)
            e_sums = g.E_class.sum(dim=-1)
            ones_like = torch.ones_like(e_sums[valid_mask.bool()])
            assert torch.allclose(e_sums[valid_mask.bool()], ones_like)

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
        assert results[0].node_mask.shape[0] == 3
        assert results[1].node_mask.shape[0] == 5

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
            assert g.X_class is not None
            assert g.E_class is not None
            assert not g.X_class.requires_grad
            assert not g.E_class.requires_grad

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
        from torch_geometric.data import Batch, Data

        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule_short,
            x_classes=DX,
            e_classes=DE,
            limit_distribution="empirical_marginal",
        )
        # 4-node graph with edges (0,1), (0,2), (1,2). PyG enumerates both
        # directions per undirected edge → 6 directed edges total.
        edges = [(0, 1), (0, 2), (1, 2)]
        directed = [(i, j) for i, j in edges] + [(j, i) for i, j in edges]
        edge_index = torch.tensor(
            [[u for u, _ in directed], [v for _, v in directed]],
            dtype=torch.long,
        )
        pyg_batch = Batch.from_data_list(
            [Data(edge_index=edge_index, num_nodes=N_NODES)]
        )
        proc.initialize_from_data([pyg_batch])
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
            assert g.X_class is not None
            assert g.X_class.shape[0] == N_NODES

    def test_valid_class_distributions(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """X_class rows are valid one-hot PMFs with empirical marginals."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert g.X_class is not None
            x_sums = g.X_class.sum(dim=-1)
            assert torch.allclose(
                x_sums[g.node_mask], torch.ones_like(x_sums[g.node_mask])
            )

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
            assert g.E_class is not None
            assert torch.equal(g.E_class, g.E_class.transpose(0, 1))


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
            n = (
                g.node_mask.shape[0]
                if g.node_mask.dim() == 1
                else g.node_mask.shape[-1]
            )
            e_class = g.E_class
            assert e_class is not None
            assert e_class.shape[-2] == e_class.shape[-1] or e_class.shape[0] == n

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
            adj = g.binarised_adjacency()
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
            adj = g.binarised_adjacency()
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
            adj = g.binarised_adjacency()
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
            adj = g.binarised_adjacency()
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


class TestAssertSymmetricEToggle:
    """``Sampler.assert_symmetric_e`` controls the per-step E-symmetry guard.

    Test rationale (parity #28 / #29 / D-5): the toggle mirrors upstream
    DiGress's per-reverse-step ``assert (E_s == E_s.transpose(1, 2)).all()``
    at ``diffusion_model_discrete.py:649``. Default ``True`` matches
    upstream semantics; ``False`` lets production hot loops skip the
    check when fp32 batch sizes make it expensive. We pin both the
    default and the failure surface so a future bypass of the canonical
    symmetrisation primitive cannot silently land asymmetric edges
    downstream.
    """

    def test_default_passes_on_normal_categorical_sampling(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Standard categorical sampling produces symmetric E_class with default toggle."""
        sampler = CategoricalSampler()  # default assert_symmetric_e=True
        assert sampler._assert_symmetric_e is True  # noqa: SLF001
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        # Reaches the end without firing the per-step assertion.
        assert len(results) == BS

    def test_disabling_toggle_lets_loop_complete_without_check(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """``assert_symmetric_e=False`` bypasses the per-step symmetry guard."""
        sampler = CategoricalSampler(assert_symmetric_e=False)
        assert sampler._assert_symmetric_e is False  # noqa: SLF001
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert len(results) == BS

    def test_assertion_fires_on_forced_asymmetric_e_class(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """A noise process that returns asymmetric E_class trips the guard."""
        sampler = CategoricalSampler()  # default assert_symmetric_e=True

        # Wrap posterior_sample_from_model_output to force asymmetry.
        original_hook = categorical_noise_process.posterior_sample_from_model_output

        def asymmetric_hook(
            z_t: GraphData,
            x0_param: GraphData,
            t: Tensor,
            s: Tensor,
        ) -> GraphData:
            out = original_hook(z_t, x0_param, t, s)
            assert out.E_class is not None
            corrupted = out.E_class.clone()
            # Flip a single off-diagonal entry to break symmetry.
            corrupted[..., 0, 1, 0] = 1.0 - corrupted[..., 0, 1, 0]
            return out.replace(E_class=corrupted)

        categorical_noise_process.posterior_sample_from_model_output = (  # type: ignore[method-assign]
            asymmetric_hook
        )

        with pytest.raises(AssertionError, match="asymmetric E_class"):
            sampler.sample(
                model=uniform_model,
                noise_process=categorical_noise_process,
                num_graphs=BS,
                num_nodes=N_NODES,
                device=torch.device("cpu"),
            )
