"""Tests for the composed MMDEvaluator helper.

Test Rationale
--------------
MMDEvaluator replaces GenerativeEvalMixin, owning accumulation state and
MMD computation without coupling to LightningModule logging. These tests
verify the accumulation cap, early-return on insufficient references,
successful metric computation, and correct state clearing.

Key invariants:
1. ``accumulate`` stops accepting graphs once ``eval_num_samples`` is reached.
2. ``evaluate`` returns ``None`` when fewer than 2 reference graphs exist.
3. ``evaluate`` returns ``MMDResults`` when enough references are present.
4. ``clear`` resets both the reference list and the recorded node count.
5. ``evaluate`` always clears state after completion (success or skip).
"""

import networkx as nx
import pytest

from tmgg.experiment_utils.mmd_evaluator import MMDEvaluator
from tmgg.experiment_utils.mmd_metrics import MMDResults


@pytest.fixture()
def evaluator() -> MMDEvaluator:
    """Default evaluator with a small sample cap for testing."""
    return MMDEvaluator(eval_num_samples=5, kernel="gaussian_tv", sigma=1.0)


def _make_er_graphs(n: int, count: int, seed_offset: int = 0) -> list[nx.Graph]:
    """Generate Erdos-Renyi graphs with distinct seeds."""
    return [nx.erdos_renyi_graph(n, 0.3, seed=i + seed_offset) for i in range(count)]


class TestAccumulate:
    """Tests for reference graph accumulation."""

    def test_accumulate_respects_limit(self, evaluator: MMDEvaluator) -> None:
        """Accumulation stops once eval_num_samples graphs have been added.

        Starting state: empty evaluator with eval_num_samples=5.
        Invariant: after feeding 8 graphs, only the first 5 are kept.
        """
        graphs = _make_er_graphs(10, 8)
        for g in graphs:
            evaluator.accumulate(g)
        assert evaluator.num_ref_graphs == 5


class TestEvaluateInsufficientRefs:
    """Tests for early-return when references are insufficient."""

    def test_evaluate_returns_none_with_too_few_refs(
        self, evaluator: MMDEvaluator
    ) -> None:
        """evaluate() returns None when fewer than 2 reference graphs exist.

        Starting state: evaluator with only 1 accumulated reference graph.
        Invariant: at least 2 reference graphs are needed for the unbiased
        MMD estimator, so a single-graph reference set must yield None.
        """
        evaluator.accumulate(nx.erdos_renyi_graph(10, 0.3, seed=0))
        evaluator.set_num_nodes(10)
        generated = _make_er_graphs(10, 3, seed_offset=100)
        result = evaluator.evaluate(generated)
        assert result is None

    def test_evaluate_returns_none_without_num_nodes(
        self, evaluator: MMDEvaluator
    ) -> None:
        """evaluate() returns None when num_nodes has not been set.

        Starting state: evaluator with 3 reference graphs but no set_num_nodes call.
        Invariant: num_nodes must be known for the caller to generate graphs,
        so a missing value signals the epoch was incomplete.
        """
        for g in _make_er_graphs(10, 3):
            evaluator.accumulate(g)
        generated = _make_er_graphs(10, 3, seed_offset=100)
        result = evaluator.evaluate(generated)
        assert result is None


class TestEvaluateSuccess:
    """Tests for successful MMD computation."""

    def test_evaluate_returns_results(self, evaluator: MMDEvaluator) -> None:
        """evaluate() returns MMDResults when enough references are present.

        Starting state: evaluator with 5 accumulated ER(10, 0.3) graphs and
        num_nodes set.
        Invariant: with >= 2 references, the MMD computation should succeed
        and return an MMDResults dataclass with finite, non-negative values.
        """
        refs = _make_er_graphs(10, 5)
        for g in refs:
            evaluator.accumulate(g)
        evaluator.set_num_nodes(10)

        generated = _make_er_graphs(10, 5, seed_offset=100)
        result = evaluator.evaluate(generated)

        assert isinstance(result, MMDResults)
        assert result.degree_mmd >= 0.0
        assert result.clustering_mmd >= 0.0
        assert result.spectral_mmd >= 0.0


class TestClear:
    """Tests for state reset."""

    def test_clear_resets_state(self, evaluator: MMDEvaluator) -> None:
        """clear() removes all accumulated references and resets num_nodes.

        Starting state: evaluator with 3 accumulated graphs and num_nodes=10.
        Invariant: after clear(), num_ref_graphs == 0 and num_nodes is None.
        """
        for g in _make_er_graphs(10, 3):
            evaluator.accumulate(g)
        evaluator.set_num_nodes(10)

        evaluator.clear()

        assert evaluator.num_ref_graphs == 0
        assert evaluator.num_nodes is None

    def test_evaluate_clears_state_after_completion(
        self, evaluator: MMDEvaluator
    ) -> None:
        """State is empty after a successful evaluate() call.

        Starting state: evaluator with 5 refs and num_nodes set.
        Invariant: evaluate() calls clear() internally, so the evaluator
        should be ready for the next epoch with no leftover state.
        """
        for g in _make_er_graphs(10, 5):
            evaluator.accumulate(g)
        evaluator.set_num_nodes(10)

        generated = _make_er_graphs(10, 5, seed_offset=100)
        evaluator.evaluate(generated)

        assert evaluator.num_ref_graphs == 0
        assert evaluator.num_nodes is None

    def test_evaluate_clears_state_on_skip(self, evaluator: MMDEvaluator) -> None:
        """State is cleared even when evaluate() returns None (too few refs).

        Starting state: evaluator with 1 ref graph.
        Invariant: the early-return path also calls clear().
        """
        evaluator.accumulate(nx.erdos_renyi_graph(10, 0.3, seed=0))
        evaluator.set_num_nodes(10)

        evaluator.evaluate(_make_er_graphs(10, 3, seed_offset=100))

        assert evaluator.num_ref_graphs == 0
        assert evaluator.num_nodes is None
