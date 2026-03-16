"""Tests for BaseGraphDataModule.get_reference_graphs().

Test Rationale
--------------
get_reference_graphs() is the public API for extracting NetworkX graphs
from dataset splits. It replaces the per-step accumulation pattern in
Lightning modules. Tests verify correctness across val/test stages,
truncation to max_graphs, and error handling for invalid stages.

The fixture uses MultiGraphDataModule with SBM and fixed-size graphs
(num_nodes=8, num_graphs=20) so node counts are deterministic and the
test dataset is small enough to be fast.
"""

from __future__ import annotations

import networkx as nx
import pytest

from tmgg.data.data_modules.multigraph_data_module import MultiGraphDataModule


@pytest.fixture
def data_module() -> MultiGraphDataModule:
    """Small SBM data module for testing."""
    dm = MultiGraphDataModule(
        graph_type="sbm",
        num_graphs=20,
        num_nodes=8,
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")
    return dm


class TestGetReferenceGraphs:
    """Verify get_reference_graphs() extracts NetworkX graphs correctly."""

    def test_val_returns_networkx_graphs(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=5)
        assert len(graphs) > 0
        assert all(isinstance(g, nx.Graph) for g in graphs)

    def test_test_returns_networkx_graphs(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("test", max_graphs=5)
        assert len(graphs) > 0
        assert all(isinstance(g, nx.Graph) for g in graphs)

    def test_max_graphs_caps_output(self, data_module: MultiGraphDataModule) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=3)
        assert len(graphs) <= 3

    def test_max_graphs_larger_than_dataset(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=10000)
        # Should return all available, not crash
        assert len(graphs) > 0

    def test_invalid_stage_raises(self, data_module: MultiGraphDataModule) -> None:
        with pytest.raises(ValueError, match="stage must be"):
            data_module.get_reference_graphs("train", max_graphs=5)

    def test_graphs_have_correct_node_count(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=5)
        for g in graphs:
            # SBM graphs with num_nodes=8 should have 8 nodes
            assert g.number_of_nodes() == 8

    def test_graphs_are_undirected(self, data_module: MultiGraphDataModule) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=5)
        for g in graphs:
            assert not g.is_directed()
