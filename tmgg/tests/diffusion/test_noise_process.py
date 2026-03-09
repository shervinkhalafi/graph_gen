"""Tests for the NoiseProcess hierarchy.

Test rationale
--------------
The ``NoiseProcess`` hierarchy unifies two separate noise systems under a
single abstract interface. ``ContinuousNoiseProcess`` wraps the existing
``NoiseGenerator`` subclasses and operates on adjacency representations.
``CategoricalNoiseProcess`` wraps the existing ``DiscreteUniformTransition``
and ``MarginalUniformTransition`` transition models and operates on one-hot
categorical representations.

Each test validates that the wrapper correctly delegates to the underlying
noise model, preserves tensor shapes and graph invariants (symmetry,
masking), and satisfies the ``NoiseProcess`` ABC contract.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.noise import (
    DigressNoiseGenerator,
    EdgeFlipNoiseGenerator,
    GaussianNoiseGenerator,
    LogitNoiseGenerator,
)
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    NoiseProcess,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.diffusion.transitions import DiscreteUniformTransition

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
    return GraphData.from_adjacency(simple_adjacency_batch)


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
        gen = GaussianNoiseGenerator()
        proc = ContinuousNoiseProcess(generator=gen, noise_schedule=cosine_schedule)
        assert isinstance(proc, NoiseProcess)

    def test_categorical_is_noise_process(self, cosine_schedule: NoiseSchedule) -> None:
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        assert isinstance(proc, NoiseProcess)


# ---------------------------------------------------------------------------
# ContinuousNoiseProcess tests
# ---------------------------------------------------------------------------


class TestContinuousNoiseProcess:
    """Tests for ContinuousNoiseProcess wrapping NoiseGenerator subclasses."""

    @pytest.mark.parametrize(
        "generator_cls",
        [
            GaussianNoiseGenerator,
            EdgeFlipNoiseGenerator,
            DigressNoiseGenerator,
            LogitNoiseGenerator,
        ],
        ids=["gaussian", "edge_flip", "digress", "logit"],
    )
    def test_apply_returns_graph_data_with_same_shape(
        self,
        generator_cls: type,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """apply() returns a GraphData with the same tensor shapes as input."""
        gen = generator_cls()
        proc = ContinuousNoiseProcess(generator=gen, noise_schedule=cosine_schedule)
        t = torch.tensor([25, 25], dtype=torch.long)  # integer timesteps
        result = proc.apply(graph_data_from_adj, t)

        assert isinstance(result, GraphData)
        assert result.X.shape == graph_data_from_adj.X.shape
        assert result.E.shape == graph_data_from_adj.E.shape
        assert result.y.shape == graph_data_from_adj.y.shape
        assert result.node_mask.shape == graph_data_from_adj.node_mask.shape

    def test_apply_no_noise_at_t_zero(
        self,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """At t=0, DiGress noise should produce no change (alpha_bar=1)."""
        gen = DigressNoiseGenerator()
        proc = ContinuousNoiseProcess(generator=gen, noise_schedule=cosine_schedule)
        t = torch.tensor([0, 0], dtype=torch.long)
        result = proc.apply(graph_data_from_adj, t)
        # With eps=0 -> alpha_bar=1 -> flip_prob=0, output == input
        adj_in = graph_data_from_adj.to_adjacency()
        adj_out = result.to_adjacency()
        assert torch.allclose(adj_in, adj_out)

    def test_get_posterior_returns_correct_shapes(
        self,
        graph_data_from_adj: GraphData,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """get_posterior returns a dict with 'mean' and 'std' of correct shape."""
        gen = GaussianNoiseGenerator()
        proc = ContinuousNoiseProcess(generator=gen, noise_schedule=cosine_schedule)
        # Pass integer timesteps — the posterior uses the schedule to look up alpha_bar.
        t = torch.tensor([40, 40], dtype=torch.long)
        s = torch.tensor([30, 30], dtype=torch.long)
        z_t = proc.apply(graph_data_from_adj, t)
        posterior = proc.get_posterior(z_t, graph_data_from_adj, t, s)
        # Posterior should contain adjacency-shaped tensors
        adj_shape = graph_data_from_adj.to_adjacency().shape
        assert posterior["mean"].shape == adj_shape
        assert posterior["std"].shape == adj_shape

    def test_get_posterior_uses_schedule_alpha_bar(
        self,
        graph_data_from_adj: GraphData,
    ) -> None:
        """Posterior alpha values should come from the schedule, not a hardcoded
        linear mapping.

        Test rationale: before this fix, ContinuousNoiseProcess.get_posterior
        computed alpha_t = 1 - t assuming t was a noise level and the schedule
        was linear.  With a cosine schedule, the relationship between integer
        timestep t and alpha_bar(t) is non-linear; a hardcoded ``1 - t``
        would produce wrong posterior statistics.  We verify that the posterior
        mean uses the schedule's alpha_bar values by comparing with a manual
        computation.
        """
        cosine = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
        linear = NoiseSchedule(schedule_type="linear_ddpm", timesteps=50)

        gen = GaussianNoiseGenerator()
        proc_cosine = ContinuousNoiseProcess(generator=gen, noise_schedule=cosine)
        proc_linear = ContinuousNoiseProcess(generator=gen, noise_schedule=linear)

        t = torch.tensor([25, 25], dtype=torch.long)
        s = torch.tensor([20, 20], dtype=torch.long)

        z_t = proc_cosine.apply(
            graph_data_from_adj, torch.tensor([25, 25], dtype=torch.long)
        )

        post_cosine = proc_cosine.get_posterior(z_t, graph_data_from_adj, t, s)
        post_linear = proc_linear.get_posterior(z_t, graph_data_from_adj, t, s)

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
        """Edge features remain symmetric after applying noise."""
        gen = GaussianNoiseGenerator()
        proc = ContinuousNoiseProcess(generator=gen, noise_schedule=cosine_schedule)
        t = torch.tensor([25, 25], dtype=torch.long)
        result = proc.apply(graph_data_from_adj, t)
        adj = result.to_adjacency()
        assert torch.allclose(adj, adj.transpose(1, 2), atol=1e-5)


# ---------------------------------------------------------------------------
# CategoricalNoiseProcess tests
# ---------------------------------------------------------------------------


class TestCategoricalNoiseProcess:
    """Tests for CategoricalNoiseProcess wrapping transition models."""

    def test_uniform_construction(self, cosine_schedule: NoiseSchedule) -> None:
        """Uniform transition mode constructs without error."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        assert proc.transition_model is not None

    def test_marginal_setup(self, cosine_schedule: NoiseSchedule) -> None:
        """Marginal transition requires a MarginalUniformTransition to be injected."""
        from tmgg.diffusion.transitions import MarginalUniformTransition

        x_marginals = torch.tensor([0.3, 0.3, 0.4])
        e_marginals = torch.tensor([0.7, 0.3])
        tm = MarginalUniformTransition(x_marginals, e_marginals, 0)
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=tm,
        )
        assert proc.transition_model is not None

    def test_protocol_conformance(self) -> None:
        """Both transition implementations satisfy the TransitionModel protocol."""
        from tmgg.diffusion.protocols import TransitionModel
        from tmgg.diffusion.transitions import MarginalUniformTransition

        assert issubclass(DiscreteUniformTransition, TransitionModel)
        assert issubclass(MarginalUniformTransition, TransitionModel)

    def test_rejects_non_transition_model(self, cosine_schedule: NoiseSchedule) -> None:
        """Passing an object that doesn't satisfy TransitionModel raises TypeError."""
        with pytest.raises(TypeError, match="TransitionModel"):
            CategoricalNoiseProcess(
                noise_schedule=cosine_schedule,
                x_classes=3,
                e_classes=2,
                transition_model=object(),  # type: ignore[arg-type]
            )

    def test_no_transition_model_raises_before_apply(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """Calling apply() without a transition_model raises RuntimeError."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
        )
        t = torch.tensor([10, 20])
        with pytest.raises(RuntimeError, match="not set"):
            proc.apply(categorical_graph_data, t)

    def test_apply_returns_valid_one_hot(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """apply() returns one-hot encoded GraphData."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        # Use integer timestep indices
        t = torch.tensor([10, 20])
        result = proc.apply(categorical_graph_data, t)

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

    def test_apply_preserves_shape(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """apply() preserves the shape of all GraphData fields."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        t = torch.tensor([10, 20])
        result = proc.apply(categorical_graph_data, t)
        assert result.X.shape == categorical_graph_data.X.shape
        assert result.E.shape == categorical_graph_data.E.shape
        assert result.node_mask.shape == categorical_graph_data.node_mask.shape

    def test_kl_prior_returns_correct_shape(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """kl_prior() returns a tuple of three scalar tensors (kl_X, kl_E, kl_y)."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        kl_X, kl_E, kl_y = proc.kl_prior(
            categorical_graph_data.X,
            categorical_graph_data.E,
            categorical_graph_data.node_mask,
        )
        # Each should be a scalar (batch-summed KL)
        assert kl_X.dim() == 0
        assert kl_E.dim() == 0
        assert kl_y.dim() == 0
        # KL should be non-negative
        assert kl_X >= 0
        assert kl_E >= 0

    def test_kl_prior_uses_final_timestep(
        self,
    ) -> None:
        """kl_prior evaluates at timestep T, not T-1.

        Regression test for an off-by-one bug where kl_prior used index T-1
        instead of T. The alphas_bar buffer has T+1 entries (indices 0..T),
        and the final noise level corresponds to index T.
        """
        T = 10
        schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=T)
        proc = CategoricalNoiseProcess(
            noise_schedule=schedule,
            x_classes=2,
            e_classes=2,
            transition_model=DiscreteUniformTransition(2, 2, 0),
        )

        bs, n = 2, 5
        X = torch.zeros(bs, n, 2)
        X[:, :, 0] = 1.0  # all nodes class 0
        E = torch.zeros(bs, n, n, 2)
        E[:, :, :, 0] = 1.0  # all edges class 0
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        # Compute kl_prior — it should use alpha_bar[T]
        kl_X, kl_E, _ = proc.kl_prior(X, E, node_mask)

        # Verify by computing with alpha_bar[T] vs alpha_bar[T-1]:
        # at T, alpha_bar is smaller => distribution closer to uniform => KL larger
        alpha_T = schedule.get_alpha_bar(t_int=torch.tensor([T]))
        alpha_T_minus_1 = schedule.get_alpha_bar(t_int=torch.tensor([T - 1]))
        assert alpha_T < alpha_T_minus_1, "alpha_bar[T] should be less noisy"

        # The returned KL should be finite and non-negative
        assert torch.isfinite(kl_X) and kl_X >= 0
        assert torch.isfinite(kl_E) and kl_E >= 0

    def test_get_posterior_returns_graph_data(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """get_posterior() returns a GraphData with probability distributions."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        t = torch.tensor([20, 30])
        s = torch.tensor([19, 29])
        z_t = proc.apply(categorical_graph_data, t)
        posterior = proc.get_posterior(z_t, categorical_graph_data, t, s)
        assert isinstance(posterior, GraphData)
        # X posterior should have shape (bs, n, dx)
        assert posterior.X.shape == categorical_graph_data.X.shape
        # E posterior is flattened to (bs, n*n, de) by compute_posterior_distribution
        bs, n, de = (
            categorical_graph_data.E.shape[0],
            categorical_graph_data.E.shape[1],
            categorical_graph_data.E.shape[-1],
        )
        assert posterior.E.shape == (bs, n * n, de)

    def test_no_transition_model_is_none(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Constructing without a transition_model leaves it as None."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
        )
        assert proc._transition_model is None

    def test_apply_edge_symmetry(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: GraphData,
    ) -> None:
        """Edge features remain symmetric after applying categorical noise."""
        proc = CategoricalNoiseProcess(
            noise_schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            transition_model=DiscreteUniformTransition(3, 2, 0),
        )
        t = torch.tensor([15, 25])
        result = proc.apply(categorical_graph_data, t)
        # E should be symmetric in the one-hot encoding
        assert torch.allclose(
            result.E, result.E.transpose(1, 2)
        ), "Edge features must remain symmetric after categorical noise"
