"""Tests for the Sampler hierarchy (CategoricalSampler, ContinuousSampler).

Test rationale
--------------
The ``Sampler`` ABC defines a ``sample()`` method that performs reverse
diffusion to generate graphs. ``CategoricalSampler`` implements ancestral
sampling for categorical discrete diffusion. ``ContinuousSampler`` implements
Gaussian reverse diffusion for adjacency-based models.

Each test validates:
- ABC contract (subclass relationship)
- Construction-time type checking (wrong noise process type raises TypeError)
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
from tmgg.data.noising.noise import GaussianNoiseGenerator
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
)
from tmgg.diffusion.sampler import (
    CategoricalSampler,
    ContinuousSampler,
    Sampler,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.diffusion.transitions import DiscreteUniformTransition
from tmgg.models.base import GraphModel

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
        noise_schedule=cosine_schedule_short,
        x_classes=DX,
        e_classes=DE,
        transition_model=DiscreteUniformTransition(DX, DE, 0),
    )


@pytest.fixture()
def continuous_noise_process() -> ContinuousNoiseProcess:
    return ContinuousNoiseProcess(
        generator=GaussianNoiseGenerator(),
        noise_schedule=NoiseSchedule("cosine_iddpm", timesteps=T_STEPS),
    )


@pytest.fixture()
def uniform_model() -> UniformCategoricalModel:
    return UniformCategoricalModel(dx=DX, de=DE)


@pytest.fixture()
def identity_model() -> IdentityContinuousModel:
    return IdentityContinuousModel()


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
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        assert isinstance(sampler, Sampler)

    def test_continuous_sampler_is_sampler(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        assert isinstance(sampler, Sampler)


# ---------------------------------------------------------------------------
# Type-checking tests
# ---------------------------------------------------------------------------


class TestSamplerTypeChecking:
    """Construction with the wrong noise process type must raise TypeError."""

    def test_categorical_sampler_rejects_continuous_process(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        with pytest.raises(TypeError, match="CategoricalNoiseProcess"):
            CategoricalSampler(continuous_noise_process, unified_schedule)  # type: ignore[arg-type]

    def test_continuous_sampler_rejects_categorical_process(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        with pytest.raises(TypeError, match="ContinuousNoiseProcess"):
            ContinuousSampler(categorical_noise_process, unified_schedule)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CategoricalSampler.sample() tests
# ---------------------------------------------------------------------------


class TestCategoricalSamplerSample:
    """Verify CategoricalSampler.sample() output type, shapes, and invariants."""

    def test_returns_list_of_graph_data(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """sample() returns a list of GraphData with the correct length."""
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        num_nodes = torch.tensor([3, 5])
        results = sampler.sample(
            model=uniform_model,
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
        sampler = CategoricalSampler(categorical_noise_process, unified_schedule)
        # Ensure we are NOT inside a no_grad context
        assert torch.is_grad_enabled()
        results = sampler.sample(
            model=uniform_model,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert not g.X.requires_grad
            assert not g.E.requires_grad


# ---------------------------------------------------------------------------
# CategoricalSampler with marginal transitions (T-5)
# ---------------------------------------------------------------------------


class TestCategoricalSamplerMarginal:
    """CategoricalSampler with MarginalUniformTransition instead of uniform.

    Test rationale: all existing CategoricalSampler tests use
    DiscreteUniformTransition. MarginalUniformTransition produces
    different transition matrices (rows are the marginal distribution,
    not 1/K uniform), which affects both the limit distribution used
    for initial sampling and the posterior computation. These tests
    verify the sampler handles marginal transitions correctly.
    """

    @pytest.fixture()
    def marginal_noise_process(
        self,
        cosine_schedule_short: NoiseSchedule,
    ) -> CategoricalNoiseProcess:
        from tmgg.diffusion.transitions import MarginalUniformTransition

        x_marginals = torch.tensor([0.6, 0.2, 0.2])
        e_marginals = torch.tensor([0.8, 0.2])
        tm = MarginalUniformTransition(x_marginals, e_marginals, 0)
        return CategoricalNoiseProcess(
            noise_schedule=cosine_schedule_short,
            x_classes=DX,
            e_classes=DE,
            transition_model=tm,
        )

    def test_returns_valid_graph_data(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Marginal-transition sampling produces valid GraphData output."""
        sampler = CategoricalSampler(marginal_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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
        """Class indices remain in valid ranges with marginal transitions."""
        sampler = CategoricalSampler(marginal_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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
        """Symmetry invariant holds with marginal transitions."""
        sampler = CategoricalSampler(marginal_noise_process, unified_schedule)
        results = sampler.sample(
            model=uniform_model,
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

    def test_returns_list_of_graph_data(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """sample() returns a list of GraphData with the correct length."""
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        results = sampler.sample(
            model=identity_model,
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
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        results = sampler.sample(
            model=identity_model,
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
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        results = sampler.sample(
            model=identity_model,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_adjacency()
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
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        results = sampler.sample(
            model=identity_model,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_adjacency()
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
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        results = sampler.sample(
            model=identity_model,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_adjacency()
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
        sampler = ContinuousSampler(continuous_noise_process, unified_schedule)
        assert torch.is_grad_enabled()
        results = sampler.sample(
            model=identity_model,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.to_adjacency()
            assert not adj.requires_grad
