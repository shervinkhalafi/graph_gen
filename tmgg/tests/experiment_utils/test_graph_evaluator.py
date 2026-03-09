"""Tests for GraphEvaluator -- unified evaluator merging MMD + structural metrics.

Test Rationale
--------------
GraphEvaluator combines MMDEvaluator's accumulate lifecycle with the full set of
graph generation metrics (degree/clustering/spectral MMD, orbit MMD, SBM accuracy,
planarity accuracy, uniqueness, novelty). These tests verify the accumulation
lifecycle, that evaluate() returns all expected fields, and that edge cases (too
few references, missing setup) are handled correctly.

The compute_orbit_mmd and compute_sbm_accuracy functions depend on external tools
(orca binary, graph-tool) that may not be available in all environments. Tests
for those metrics are skipped when the dependencies are missing.

All other metrics (degree/clustering/spectral MMD, planarity, uniqueness, novelty)
use only networkx and are always testable.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import networkx as nx
import pytest

from tmgg.experiments._shared_utils.evaluation_metrics.graph_evaluator import (
    EvaluationResults,
    GraphEvaluator,
)

# ---------------------------------------------------------------------------
# Dependency availability checks
# ---------------------------------------------------------------------------
_graph_tool_available = importlib.util.find_spec("graph_tool") is not None

_orca_available: bool
try:
    from tmgg.experiments._shared_utils.evaluation_metrics.orca import (
        is_available as _orca_is_available,
    )

    _orca_available = _orca_is_available()
except Exception:
    _orca_available = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def er_graphs() -> list[nx.Graph[Any]]:
    """Ten Erdos-Renyi graphs with 12 nodes and ~30% edge density."""
    return [nx.erdos_renyi_graph(12, 0.3, seed=i) for i in range(10)]


@pytest.fixture
def different_er_graphs() -> list[nx.Graph[Any]]:
    """Ten Erdos-Renyi graphs with different parameters (higher density)."""
    return [nx.erdos_renyi_graph(12, 0.7, seed=i + 100) for i in range(10)]


@pytest.fixture
def evaluator() -> GraphEvaluator:
    """Default GraphEvaluator with eval_num_samples=10."""
    return GraphEvaluator(eval_num_samples=10)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Verify GraphEvaluator accepts constructor parameters."""

    def test_default_construction(self) -> None:
        ev = GraphEvaluator(eval_num_samples=20)
        assert ev.eval_num_samples == 20

    def test_custom_kernel_and_sigma(self) -> None:
        ev = GraphEvaluator(eval_num_samples=5, kernel="gaussian", sigma=2.0)
        assert ev.kernel == "gaussian"
        assert ev.sigma == 2.0

    def test_default_kernel_is_gaussian_tv(self) -> None:
        ev = GraphEvaluator(eval_num_samples=5)
        assert ev.kernel == "gaussian_tv"

    def test_sbm_params_stored(self) -> None:
        ev = GraphEvaluator(eval_num_samples=5, p_intra=0.4, p_inter=0.01)
        assert ev.p_intra == 0.4
        assert ev.p_inter == 0.01


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
class TestSetup:
    """Verify setup() stores train graphs for novelty computation."""

    def test_setup_stores_train_graphs(
        self, evaluator: GraphEvaluator, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        evaluator.setup(train_graphs=er_graphs)
        assert evaluator.train_graphs is not None
        assert len(evaluator.train_graphs) == len(er_graphs)

    def test_setup_overwrites_previous(
        self, evaluator: GraphEvaluator, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        evaluator.setup(train_graphs=er_graphs)
        assert len(evaluator.train_graphs) == len(er_graphs)


# ---------------------------------------------------------------------------
# Accumulation lifecycle
# ---------------------------------------------------------------------------
class TestAccumulation:
    """Verify accumulate/clear lifecycle."""

    def test_accumulate_stores_graphs(
        self, evaluator: GraphEvaluator, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        for g in er_graphs:
            evaluator.accumulate(g)
        assert len(evaluator.ref_graphs) == len(er_graphs)

    def test_accumulate_caps_at_eval_num_samples(self) -> None:
        ev = GraphEvaluator(eval_num_samples=3)
        graphs = [nx.erdos_renyi_graph(8, 0.3, seed=i) for i in range(10)]
        for g in graphs:
            ev.accumulate(g)
        assert len(ev.ref_graphs) == 3

    def test_clear_resets_ref_graphs(
        self, evaluator: GraphEvaluator, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        for g in er_graphs:
            evaluator.accumulate(g)
        evaluator.clear()
        assert len(evaluator.ref_graphs) == 0

    def test_clear_preserves_train_graphs(
        self, evaluator: GraphEvaluator, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        evaluator.setup(train_graphs=er_graphs)
        for g in er_graphs:
            evaluator.accumulate(g)
        evaluator.clear()
        # train_graphs should survive clear()
        assert len(evaluator.train_graphs) == len(er_graphs)


# ---------------------------------------------------------------------------
# Evaluate -- core MMD metrics (always available)
# ---------------------------------------------------------------------------
class TestEvaluateCore:
    """Verify evaluate() returns EvaluationResults with all fields."""

    def test_evaluate_returns_evaluation_results(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert isinstance(result, EvaluationResults)

    def test_evaluate_has_mmd_fields(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert isinstance(result.degree_mmd, float)
        assert isinstance(result.clustering_mmd, float)
        assert isinstance(result.spectral_mmd, float)

    def test_evaluate_has_planarity(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert isinstance(result.planarity_accuracy, float)
        assert 0.0 <= result.planarity_accuracy <= 1.0

    def test_evaluate_has_uniqueness(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert isinstance(result.uniqueness, float)
        assert 0.0 <= result.uniqueness <= 1.0

    def test_evaluate_has_novelty(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert isinstance(result.novelty, float)
        assert 0.0 <= result.novelty <= 1.0

    def test_evaluate_returns_none_when_too_few_refs(
        self,
        evaluator: GraphEvaluator,
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        """evaluate() requires at least 2 reference graphs for MMD."""
        evaluator.accumulate(nx.erdos_renyi_graph(10, 0.3, seed=0))
        result = evaluator.evaluate(different_er_graphs)
        assert result is None

    def test_evaluate_returns_none_when_no_refs(
        self,
        evaluator: GraphEvaluator,
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        result = evaluator.evaluate(different_er_graphs)
        assert result is None

    def test_novelty_is_none_without_train_graphs(
        self, er_graphs: list[nx.Graph[Any]], different_er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """If setup() was never called, novelty should be None."""
        ev = GraphEvaluator(eval_num_samples=10)
        for g in er_graphs:
            ev.accumulate(g)
        result = ev.evaluate(different_er_graphs)
        assert result is not None
        assert result.novelty is None

    def test_mmd_near_zero_for_identical_distributions(
        self, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """MMD between a distribution and itself should be near zero."""
        ev = GraphEvaluator(eval_num_samples=10)
        for g in er_graphs:
            ev.accumulate(g)
        result = ev.evaluate(er_graphs)
        assert result is not None
        assert result.degree_mmd == pytest.approx(0.0, abs=0.01)

    def test_mmd_positive_for_different_distributions(
        self, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """MMD between different graph families should be clearly positive."""
        ev = GraphEvaluator(eval_num_samples=10)
        for g in er_graphs:
            ev.accumulate(g)
        # Regular graphs (very different degree distribution from ER)
        regular = [nx.random_regular_graph(3, 12, seed=i) for i in range(10)]
        result = ev.evaluate(regular)
        assert result is not None
        assert result.degree_mmd > 0.01


# ---------------------------------------------------------------------------
# Evaluate -- orbit and SBM (require external tools)
# ---------------------------------------------------------------------------
class TestEvaluateOrbitSBM:
    """Tests for metrics that need graph-tool or orca."""

    @pytest.mark.skipif(
        not _orca_available, reason="orca binary not compiled/available"
    )
    def test_orbit_mmd_is_float_when_orca_available(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert isinstance(result.orbit_mmd, float)

    @pytest.mark.skipif(not _graph_tool_available, reason="graph-tool not installed")
    def test_sbm_accuracy_is_float_when_graph_tool_available(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert isinstance(result.sbm_accuracy, float)

    def test_orbit_mmd_none_when_orca_unavailable(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        """When orca is not available, orbit_mmd should be None."""
        if _orca_available:
            pytest.skip("orca is available; this test checks the unavailable case")
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert result.orbit_mmd is None

    def test_sbm_accuracy_none_when_graph_tool_unavailable(
        self,
        evaluator: GraphEvaluator,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        """When graph-tool is not available, sbm_accuracy should be None."""
        if _graph_tool_available:
            pytest.skip(
                "graph-tool is available; this test checks the unavailable case"
            )
        evaluator.setup(train_graphs=er_graphs[:3])
        for g in er_graphs:
            evaluator.accumulate(g)
        result = evaluator.evaluate(different_er_graphs)
        assert result is not None
        assert result.sbm_accuracy is None


# ---------------------------------------------------------------------------
# EvaluationResults.to_dict
# ---------------------------------------------------------------------------
class TestEvaluationResultsToDict:
    """Verify to_dict() returns a flat dict suitable for logging."""

    def test_to_dict_keys(self) -> None:
        res = EvaluationResults(
            degree_mmd=0.1,
            clustering_mmd=0.2,
            spectral_mmd=0.3,
            orbit_mmd=0.4,
            sbm_accuracy=0.5,
            planarity_accuracy=0.6,
            uniqueness=0.7,
            novelty=0.8,
        )
        d = res.to_dict()
        expected_keys = {
            "degree_mmd",
            "clustering_mmd",
            "spectral_mmd",
            "orbit_mmd",
            "sbm_accuracy",
            "planarity_accuracy",
            "uniqueness",
            "novelty",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self) -> None:
        res = EvaluationResults(
            degree_mmd=0.1,
            clustering_mmd=0.2,
            spectral_mmd=0.3,
            orbit_mmd=None,
            sbm_accuracy=None,
            planarity_accuracy=0.6,
            uniqueness=0.7,
            novelty=None,
        )
        d = res.to_dict()
        assert d["degree_mmd"] == 0.1
        assert d["orbit_mmd"] is None
        assert d["novelty"] is None

    def test_to_dict_all_values_are_float_or_none(self) -> None:
        res = EvaluationResults(
            degree_mmd=0.1,
            clustering_mmd=0.2,
            spectral_mmd=0.3,
            orbit_mmd=None,
            sbm_accuracy=None,
            planarity_accuracy=0.6,
            uniqueness=0.7,
            novelty=0.8,
        )
        d = res.to_dict()
        for v in d.values():
            assert v is None or isinstance(v, float)


# ---------------------------------------------------------------------------
# Full accumulation lifecycle
# ---------------------------------------------------------------------------
class TestFullLifecycle:
    """End-to-end: accumulate, evaluate, clear, accumulate again."""

    def test_accumulate_evaluate_clear_cycle(
        self, er_graphs: list[nx.Graph[Any]], different_er_graphs: list[nx.Graph[Any]]
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10)
        ev.setup(train_graphs=er_graphs[:3])

        # First cycle
        for g in er_graphs:
            ev.accumulate(g)
        result1 = ev.evaluate(different_er_graphs)
        assert result1 is not None

        ev.clear()
        assert len(ev.ref_graphs) == 0

        # Second cycle with different refs
        for g in different_er_graphs:
            ev.accumulate(g)
        result2 = ev.evaluate(er_graphs)
        assert result2 is not None

        # The two results should differ (different ref/gen pairings)
        assert (
            result1.degree_mmd != result2.degree_mmd
            or result1.clustering_mmd != result2.clustering_mmd
        )
