"""Tests for the NoiseProcess hierarchy.

Test rationale
--------------
The ``NoiseProcess`` hierarchy unifies two separate noise systems under a
single abstract interface. ``ContinuousNoiseProcess`` wraps the existing
``NoiseDefinition`` subclasses and operates on dense edge-valued states.
``CategoricalNoiseProcess`` owns its stationary categorical PMFs directly and
operates on one-hot categorical representations.

Each test validates that the wrapper correctly delegates to the underlying
noise model, preserves tensor shapes and graph invariants (symmetry,
masking), and satisfies the ``NoiseProcess`` ABC contract.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    ExactDensityNoiseProcess,
    NoiseProcess,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import (
    DigressNoise,
    EdgeFlipNoise,
    GaussianNoise,
    LogitNoise,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_adjacency_batch() -> Tensor:
    """A small batch of binary adjacency matrices (bs=2, n=4)."""
    torch.manual_seed(42)
    adj = torch.zeros(2, 4, 4)
    # Graph 0: path 0-1-2-3
    adj[0, 0, 1] = adj[0, 1, 0] = 1
    adj[0, 1, 2] = adj[0, 2, 1] = 1
    adj[0, 2, 3] = adj[0, 3, 2] = 1
    # Graph 1: complete K4
    for i in range(4):
        for j in range(4):
            if i != j:
                adj[1, i, j] = 1
    return adj


@pytest.fixture()
def graph_data_from_adj(simple_adjacency_batch: Tensor) -> GraphData:
    """GraphData created from the simple adjacency batch."""
    return GraphData.from_binary_adjacency(simple_adjacency_batch)


@pytest.fixture()
def categorical_graph_data() -> GraphData:
    """A small batch of one-hot categorical GraphData (bs=2, n=5, dx=3, de=2)."""
    torch.manual_seed(42)
    bs, n, dx, de = 2, 5, 3, 2
    # Random one-hot X
    x_idx = torch.randint(0, dx, (bs, n))
    X = torch.zeros(bs, n, dx)
    X.scatter_(2, x_idx.unsqueeze(-1), 1.0)
    # Random one-hot E (symmetric, zero diagonal)
    e_idx = torch.randint(0, de, (bs, n, n))
    # Make symmetric
    e_idx = torch.triu(e_idx, diagonal=1)
    e_idx = e_idx + e_idx.transpose(1, 2)
    E = torch.zeros(bs, n, n, de)
    E.scatter_(3, e_idx.unsqueeze(-1), 1.0)
    # Zero diagonal -> class 0
    diag = torch.arange(n)
    E[:, diag, diag, :] = 0
    E[:, diag, diag, 0] = 1.0

    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    return GraphData(X=X, E=E, y=y, node_mask=node_mask)


# cosine_schedule fixture (T=50) provided by conftest.py


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestNoiseProcessABC:
    """Verify that both subclasses satisfy the NoiseProcess interface."""

    def test_continuous_is_noise_process(self, cosine_schedule: NoiseSchedule) -> None:
        gen = GaussianNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        assert isinstance(proc, NoiseProcess)
        assert proc.timesteps == cosine_schedule.timesteps

    def test_categorical_is_noise_process(self, cosine_schedule: NoiseSchedule) -> None:
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        assert isinstance(proc, NoiseProcess)
        assert proc.timesteps == cosine_schedule.timesteps


# ---------------------------------------------------------------------------
# ContinuousNoiseProcess tests
# ---------------------------------------------------------------------------


class TestContinuousNoiseProcess:
    """Tests for ContinuousNoiseProcess wrapping NoiseDefinition subclasses."""

    def test_process_state_condition_vector_uses_normalized_timestep(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Default conditioning should be the normalized timestep ``t / T``.

        Test rationale: Step 1 moves model-conditioning ownership into the
        noise process without changing experiment semantics yet. The process
        therefore needs to expose the same normalized timestep representation
        that training used before the refactor.
        """
        proc = ContinuousNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
        )
        t = torch.tensor([1, 25, 50], dtype=torch.long)

        expected = t.float() / proc.timesteps
        actual = proc.process_state_condition_vector(t)

        torch.testing.assert_close(actual, expected)

    def test_initialization_hooks_are_noops(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Continuous processes should report no dataloader-backed setup needs."""
        proc = ContinuousNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
        )

        assert proc.needs_data_initialization() is False
        assert proc.is_initialized() is True

    def test_sample_at_level_preserves_graph_shape(
        self,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Direct-level noising should preserve the GraphData batch structure."""
        proc = ContinuousNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
        )

        result = proc.sample_at_level(graph_data_from_adj, level=0.25)

        assert isinstance(result, GraphData)
        assert result.X.shape == graph_data_from_adj.X.shape
        assert (
            result.to_edge_state().shape
            == graph_data_from_adj.to_binary_adjacency().shape
        )
        assert result.node_mask.shape == graph_data_from_adj.node_mask.shape

    @pytest.mark.parametrize(
        "generator_cls",
        [
            GaussianNoise,
            EdgeFlipNoise,
            DigressNoise,
            LogitNoise,
        ],
        ids=["gaussian", "edge_flip", "digress", "logit"],
    )
    def test_forward_sample_returns_graph_data_with_same_shape(
        self,
        generator_cls: type,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """forward_sample() preserves the graph extents of the input batch."""
        gen = generator_cls()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        t = torch.tensor([25, 25], dtype=torch.long)  # integer timesteps
        result = proc.forward_sample(graph_data_from_adj, t)

        assert isinstance(result, GraphData)
        assert result.X.shape == graph_data_from_adj.X.shape
        assert (
            result.to_edge_state().shape
            == graph_data_from_adj.to_binary_adjacency().shape
        )
        assert result.y.shape == graph_data_from_adj.y.shape
        assert result.node_mask.shape == graph_data_from_adj.node_mask.shape

    def test_forward_sample_no_noise_at_t_zero(
        self,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """At t=0, DiGress noise should produce no change (alpha_bar=1)."""
        gen = DigressNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        t = torch.tensor([0, 0], dtype=torch.long)
        result = proc.forward_sample(graph_data_from_adj, t)
        # With eps=0 -> alpha_bar=1 -> flip_prob=0, output == input
        adj_in = graph_data_from_adj.to_binary_adjacency()
        adj_out = result.to_edge_state()
        assert torch.allclose(adj_in, adj_out)

    def test_posterior_sample_returns_graph_data_with_same_shape(
        self,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """posterior_sample() returns a sampled GraphData state at timestep s."""
        gen = GaussianNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        # Pass integer timesteps — the posterior uses the schedule to look up alpha_bar.
        t = torch.tensor([40, 40], dtype=torch.long)
        s = torch.tensor([30, 30], dtype=torch.long)
        z_t = proc.forward_sample(graph_data_from_adj, t)
        posterior = proc.posterior_sample(z_t, graph_data_from_adj, t, s)
        assert isinstance(posterior, GraphData)
        assert (
            posterior.to_edge_state().shape
            == graph_data_from_adj.to_binary_adjacency().shape
        )
        assert posterior.node_mask.shape == graph_data_from_adj.node_mask.shape

    def test_posterior_parameters_use_schedule_alpha_bar(
        self,
        graph_data_from_adj: GraphData,
    ) -> None:
        """Posterior alpha values should come from the schedule, not a hardcoded
        linear mapping.

        Test rationale: before this fix, ContinuousNoiseProcess posterior math
        computed alpha_t = 1 - t assuming t was a noise level and the schedule
        was linear.  With a cosine schedule, the relationship between integer
        timestep t and alpha_bar(t) is non-linear; a hardcoded ``1 - t``
        would produce wrong posterior statistics.  We verify that the posterior
        mean helper uses the schedule's alpha_bar values by comparing cosine
        and linear schedules.
        """
        cosine = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
        linear = NoiseSchedule(schedule_type="linear_ddpm", timesteps=50)

        gen = GaussianNoise()
        proc_cosine = ContinuousNoiseProcess(definition=gen, schedule=cosine)
        proc_linear = ContinuousNoiseProcess(definition=gen, schedule=linear)

        t = torch.tensor([25, 25], dtype=torch.long)
        s = torch.tensor([20, 20], dtype=torch.long)

        z_t = proc_cosine.forward_sample(
            graph_data_from_adj, torch.tensor([25, 25], dtype=torch.long)
        )

        post_cosine = proc_cosine._posterior_parameters(z_t, graph_data_from_adj, t, s)
        post_linear = proc_linear._posterior_parameters(z_t, graph_data_from_adj, t, s)

        # Cosine and linear schedules have different alpha_bar at t=25.
        # The posterior means must differ because the interpolation weights
        # (alpha_bar_s) differ between schedules.
        alpha_s_cosine = cosine.get_alpha_bar(t_int=s)
        alpha_s_linear = linear.get_alpha_bar(t_int=s)
        assert not torch.allclose(
            alpha_s_cosine, alpha_s_linear, atol=1e-3
        ), "Precondition: cosine and linear alpha_bar should differ at t=25"
        assert not torch.allclose(
            post_cosine["mean"], post_linear["mean"], atol=1e-3
        ), "Posterior means should differ when using different schedules"

    def test_apply_edge_symmetry(
        self,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Edge features remain symmetric after forward sampling."""
        gen = GaussianNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        t = torch.tensor([25, 25], dtype=torch.long)
        result = proc.forward_sample(graph_data_from_adj, t)
        adj = result.to_edge_state()
        assert torch.allclose(adj, adj.transpose(1, 2), atol=1e-5)


# ---------------------------------------------------------------------------
# CategoricalNoiseProcess tests
# ---------------------------------------------------------------------------


class TestCategoricalNoiseProcess:
    """Tests for CategoricalNoiseProcess stationary-PMF behavior."""

    def test_process_state_condition_vector_uses_normalized_timestep(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Categorical conditioning also defaults to the normalized timestep.

        Test rationale: the refactor changes ownership, not semantics. Both
        concrete process types should initially expose the existing ``t / T``
        representation so training and sampling stay aligned during Step 1.
        """
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([1, 25, 50], dtype=torch.long)

        expected = t.float() / proc.timesteps
        actual = proc.process_state_condition_vector(t)

        torch.testing.assert_close(actual, expected)

    def test_uniform_construction(self, cosine_schedule: NoiseSchedule) -> None:
        """Uniform mode constructs immediate stationary PMFs."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        assert proc._limit_x is not None
        assert proc._limit_e is not None
        assert isinstance(proc, ExactDensityNoiseProcess)
        assert proc.needs_data_initialization() is False
        assert proc.is_initialized() is True

    def test_empirical_marginal_starts_uninitialised(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Empirical-marginal mode defers PMF construction to loader setup."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )
        assert proc._limit_x is None
        assert proc._limit_e is None
        assert proc.needs_data_initialization() is True
        assert proc.is_initialized() is False

    def test_initialize_from_data_uses_strict_upper_triangle(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Empirical edge marginals ignore mirrored edges and diagonal mass."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=2,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )

        X = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        E = torch.zeros(1, 2, 2, 2)
        E[..., 0] = 1.0
        E[0, 0, 1] = torch.tensor([0.0, 1.0])
        E[0, 1, 0] = torch.tensor([0.0, 1.0])
        node_mask = torch.tensor([[True, True]])
        batch = GraphData(X=X, E=E, y=torch.zeros(1, 0), node_mask=node_mask)

        proc.initialize_from_data([batch])  # type: ignore[arg-type]

        torch.testing.assert_close(
            proc._limit_e,
            torch.tensor([0.0, 1.0]),
        )

    def test_initialize_from_data_falls_back_to_uniform_on_zero_edge_counts(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Empirical edge marginals fall back to uniform when no real edges exist."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=2,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )

        X = torch.tensor([[[1.0, 0.0]]])
        E = torch.tensor([[[[1.0, 0.0]]]])
        node_mask = torch.tensor([[True]])
        batch = GraphData(X=X, E=E, y=torch.zeros(1, 0), node_mask=node_mask)

        proc.initialize_from_data([batch])  # type: ignore[arg-type]

        torch.testing.assert_close(
            proc._limit_e,
            torch.tensor([0.5, 0.5]),
        )

    def test_rejects_invalid_limit_distribution(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Unknown limit_distribution values raise ValueError."""
        with pytest.raises(ValueError, match="limit_distribution"):
            CategoricalNoiseProcess(
                schedule=cosine_schedule,
                x_classes=3,
                e_classes=2,
                limit_distribution="bogus",  # type: ignore[arg-type]
            )

    def test_stationary_distribution_raises_before_forward_sample(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """Empirical-marginal mode raises until loader setup populates PMFs."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )
        t = torch.tensor([10, 20])
        with pytest.raises(RuntimeError, match="not initialised"):
            proc.forward_sample(categorical_graph_data, t)

    def test_forward_sample_returns_valid_one_hot(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """forward_sample() returns one-hot encoded GraphData."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        # Use integer timestep indices
        t = torch.tensor([10, 20])
        result = proc.forward_sample(categorical_graph_data, t)

        assert isinstance(result, GraphData)
        # X should be one-hot: each row sums to 1 for valid nodes
        x_sums = result.X[categorical_graph_data.node_mask].sum(dim=-1)
        assert torch.allclose(x_sums, torch.ones_like(x_sums))

        # E should be one-hot for valid edges (off-diagonal, valid nodes)
        bs, n = categorical_graph_data.node_mask.shape
        mask_2d = categorical_graph_data.node_mask.unsqueeze(
            1
        ) * categorical_graph_data.node_mask.unsqueeze(2)
        diag = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
        valid_edges = mask_2d & ~diag
        e_sums = result.E[valid_edges].sum(dim=-1)
        assert torch.allclose(e_sums, torch.ones_like(e_sums))

    def test_forward_sample_preserves_shape(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """forward_sample() preserves the shape of all GraphData fields."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([10, 20])
        result = proc.forward_sample(categorical_graph_data, t)
        assert result.X.shape == categorical_graph_data.X.shape
        assert result.E.shape == categorical_graph_data.E.shape
        assert result.node_mask.shape == categorical_graph_data.node_mask.shape

    def test_prior_log_prob_returns_per_sample_shape(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """prior_log_prob() returns one finite value per batch element."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        result = proc.prior_log_prob(categorical_graph_data)
        assert result.shape == (categorical_graph_data.X.shape[0],)
        assert torch.isfinite(result).all()

    def test_forward_log_prob_uses_requested_final_timestep(
        self,
    ) -> None:
        """forward_log_prob() should distinguish timestep T from T-1."""
        T = 10
        schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=T)
        proc = CategoricalNoiseProcess(
            schedule=schedule,
            x_classes=2,
            e_classes=2,
            limit_distribution="uniform",
        )

        bs, n = 2, 5
        X = torch.zeros(bs, n, 2)
        X[:, :, 0] = 1.0  # all nodes class 0
        E = torch.zeros(bs, n, n, 2)
        E[:, :, :, 0] = 1.0  # all edges class 0
        node_mask = torch.ones(bs, n, dtype=torch.bool)
        clean = GraphData(X=X, E=E, y=torch.zeros(bs, 0), node_mask=node_mask)

        log_prob_T = proc.forward_log_prob(
            clean, clean, torch.full((bs,), T, dtype=torch.long)
        )
        log_prob_T_minus_1 = proc.forward_log_prob(
            clean, clean, torch.full((bs,), T - 1, dtype=torch.long)
        )
        assert torch.isfinite(log_prob_T).all()
        assert torch.isfinite(log_prob_T_minus_1).all()
        assert torch.all(log_prob_T < log_prob_T_minus_1)

    def test_posterior_sample_returns_one_hot_graph_data(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """posterior_sample() returns a sampled one-hot GraphData state."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([20, 30])
        s = torch.tensor([19, 29])
        z_t = proc.forward_sample(categorical_graph_data, t)
        posterior = proc.posterior_sample(z_t, categorical_graph_data, t, s)
        assert isinstance(posterior, GraphData)
        assert posterior.X.shape == categorical_graph_data.X.shape
        assert posterior.E.shape == categorical_graph_data.E.shape
        x_sums = posterior.X[categorical_graph_data.node_mask].sum(dim=-1)
        assert torch.allclose(x_sums, torch.ones_like(x_sums))

        bs, n = categorical_graph_data.node_mask.shape
        mask_2d = categorical_graph_data.node_mask.unsqueeze(
            1
        ) * categorical_graph_data.node_mask.unsqueeze(2)
        diag = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
        valid_edges = mask_2d & ~diag
        e_sums = posterior.E[valid_edges].sum(dim=-1)
        assert torch.allclose(e_sums, torch.ones_like(e_sums))

    def test_apply_edge_symmetry(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """Edge features remain symmetric after categorical forward sampling."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([15, 25])
        result = proc.forward_sample(categorical_graph_data, t)
        # E should be symmetric in the one-hot encoding
        assert torch.allclose(
            result.E, result.E.transpose(1, 2)
        ), "Edge features must remain symmetric after categorical noise"
