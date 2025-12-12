"""Tests for synthetic graph generators."""

import pytest
import numpy as np
import networkx as nx

from tmgg.experiment_utils.data.synthetic_graphs import (
    generate_watts_strogatz_graphs,
    generate_random_geometric_graphs,
    generate_configuration_model_graphs,
    generate_regular_graphs,
    generate_erdos_renyi_graphs,
    generate_tree_graphs,
    SyntheticGraphDataset,
)


class TestWattsStrogatzGraphs:
    """Tests for Watts-Strogatz small-world graph generator."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n, num_graphs = 20, 5
        graphs = generate_watts_strogatz_graphs(n, num_graphs)
        assert graphs.shape == (num_graphs, n, n)

    def test_symmetry(self):
        """Test that graphs are symmetric (undirected)."""
        graphs = generate_watts_strogatz_graphs(15, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_binary_edges(self):
        """Test that adjacency matrices are binary."""
        graphs = generate_watts_strogatz_graphs(15, 3)
        assert np.all((graphs == 0) | (graphs == 1))

    def test_k_parameter(self):
        """Test that k parameter affects connectivity."""
        n = 20
        graphs_k2 = generate_watts_strogatz_graphs(n, 5, k=2, p=0.0)
        graphs_k4 = generate_watts_strogatz_graphs(n, 5, k=4, p=0.0)

        # Higher k means more edges (at p=0, each node has exactly k neighbors)
        edges_k2 = graphs_k2.sum() / 2 / 5  # avg edges per graph
        edges_k4 = graphs_k4.sum() / 2 / 5
        assert edges_k4 > edges_k2

    def test_p_parameter_extremes(self):
        """Test rewiring probability at extremes."""
        n = 20
        # p=0: ring lattice (highly structured)
        graphs_p0 = generate_watts_strogatz_graphs(n, 3, k=4, p=0.0, seed=42)
        # p=1: random-like (less structured)
        graphs_p1 = generate_watts_strogatz_graphs(n, 3, k=4, p=1.0, seed=42)

        # Both should have same number of edges but different structure
        assert graphs_p0.shape == graphs_p1.shape

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        g1 = generate_watts_strogatz_graphs(15, 3, seed=123)
        g2 = generate_watts_strogatz_graphs(15, 3, seed=123)
        assert np.allclose(g1, g2)

    def test_invalid_k_raises(self):
        """Test that invalid k values raise errors."""
        with pytest.raises(ValueError, match="k=.*must be less than"):
            generate_watts_strogatz_graphs(10, 3, k=10)

        with pytest.raises(ValueError, match="k must be even"):
            generate_watts_strogatz_graphs(10, 3, k=3)


class TestRandomGeometricGraphs:
    """Tests for random geometric graph generator."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n, num_graphs = 20, 5
        graphs = generate_random_geometric_graphs(n, num_graphs)
        assert graphs.shape == (num_graphs, n, n)

    def test_symmetry(self):
        """Test that graphs are symmetric (undirected)."""
        graphs = generate_random_geometric_graphs(15, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_binary_edges(self):
        """Test that adjacency matrices are binary."""
        graphs = generate_random_geometric_graphs(15, 3)
        assert np.all((graphs == 0) | (graphs == 1))

    def test_radius_parameter(self):
        """Test that radius affects edge density."""
        n = 30
        graphs_small_r = generate_random_geometric_graphs(n, 5, radius=0.2, seed=42)
        graphs_large_r = generate_random_geometric_graphs(n, 5, radius=0.5, seed=42)

        # Larger radius means more edges
        edges_small = graphs_small_r.sum()
        edges_large = graphs_large_r.sum()
        assert edges_large > edges_small

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        g1 = generate_random_geometric_graphs(15, 3, seed=123)
        g2 = generate_random_geometric_graphs(15, 3, seed=123)
        assert np.allclose(g1, g2)


class TestConfigurationModelGraphs:
    """Tests for configuration model graph generator."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n, num_graphs = 20, 5
        graphs = generate_configuration_model_graphs(n, num_graphs)
        assert graphs.shape == (num_graphs, n, n)

    def test_symmetry(self):
        """Test that graphs are symmetric (undirected)."""
        graphs = generate_configuration_model_graphs(15, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_no_self_loops(self):
        """Test that graphs have no self-loops."""
        graphs = generate_configuration_model_graphs(15, 3)
        for A in graphs:
            assert np.allclose(np.diag(A), 0)

    def test_custom_degree_sequence(self):
        """Test with custom degree sequence."""
        n = 10
        # Regular degree sequence (all nodes degree 3)
        deg_seq = [3] * n
        graphs = generate_configuration_model_graphs(n, 3, degree_sequence=deg_seq)

        assert graphs.shape == (3, n, n)
        # Degrees should be close to 3 (may be slightly lower due to simplification)
        for A in graphs:
            degrees = A.sum(axis=1)
            assert np.all(degrees <= 3)

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        g1 = generate_configuration_model_graphs(15, 3, seed=123)
        g2 = generate_configuration_model_graphs(15, 3, seed=123)
        assert np.allclose(g1, g2)


class TestSyntheticGraphDataset:
    """Tests for SyntheticGraphDataset class."""

    def test_watts_strogatz_type(self):
        """Test Watts-Strogatz graph generation via dataset."""
        dataset = SyntheticGraphDataset("watts_strogatz", n=15, num_graphs=5)
        assert len(dataset) == 5
        assert dataset[0].shape == (15, 15)
        assert dataset.graph_type == "watts_strogatz"

    def test_ws_alias(self):
        """Test ws alias for Watts-Strogatz."""
        dataset = SyntheticGraphDataset("ws", n=15, num_graphs=5, k=4, p=0.2)
        assert dataset.graph_type == "watts_strogatz"
        assert len(dataset) == 5

    def test_random_geometric_type(self):
        """Test random geometric graph generation via dataset."""
        dataset = SyntheticGraphDataset("random_geometric", n=20, num_graphs=5)
        assert len(dataset) == 5
        assert dataset[0].shape == (20, 20)
        assert dataset.graph_type == "random_geometric"

    def test_rg_alias(self):
        """Test rg alias for random geometric."""
        dataset = SyntheticGraphDataset("rg", n=15, num_graphs=5, radius=0.4)
        assert dataset.graph_type == "random_geometric"

    def test_configuration_model_type(self):
        """Test configuration model graph generation via dataset."""
        dataset = SyntheticGraphDataset("configuration_model", n=15, num_graphs=5)
        assert len(dataset) == 5
        assert dataset[0].shape == (15, 15)
        assert dataset.graph_type == "configuration_model"

    def test_cm_alias(self):
        """Test cm alias for configuration model."""
        dataset = SyntheticGraphDataset("cm", n=15, num_graphs=5)
        assert dataset.graph_type == "configuration_model"

    def test_er_alias(self):
        """Test er alias for Erdős-Rényi."""
        dataset = SyntheticGraphDataset("er", n=15, num_graphs=5, p=0.3)
        assert dataset.graph_type == "erdos_renyi"

    def test_invalid_type_raises(self):
        """Test that invalid graph type raises error."""
        with pytest.raises(ValueError, match="graph_type must be one of"):
            SyntheticGraphDataset("invalid_type", n=10, num_graphs=5)

    def test_to_torch(self):
        """Test conversion to PyTorch tensor."""
        import torch

        dataset = SyntheticGraphDataset("ws", n=10, num_graphs=3)
        tensor = dataset.to_torch()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 10, 10)
        assert tensor.dtype == torch.float32

    def test_train_val_test_split(self):
        """Test dataset splitting."""
        dataset = SyntheticGraphDataset("rg", n=10, num_graphs=100)
        train, val, test = dataset.train_val_test_split(
            train_ratio=0.7, val_ratio=0.15, seed=42
        )

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert train.shape[1:] == (10, 10)


class TestExistingGenerators:
    """Regression tests for existing generators to ensure they still work."""

    def test_regular_graphs(self):
        """Test regular graph generator."""
        graphs = generate_regular_graphs(n=20, d=3, num_graphs=5, seed=42)
        assert graphs.shape == (5, 20, 20)
        # Check all nodes have degree 3
        for A in graphs:
            degrees = A.sum(axis=1)
            assert np.allclose(degrees, 3)

    def test_erdos_renyi_graphs(self):
        """Test Erdős-Rényi graph generator."""
        graphs = generate_erdos_renyi_graphs(n=20, p=0.3, num_graphs=5, seed=42)
        assert graphs.shape == (5, 20, 20)

    def test_tree_graphs(self):
        """Test tree graph generator."""
        graphs = generate_tree_graphs(n=15, num_graphs=5, seed=42)
        assert graphs.shape == (5, 15, 15)
        # Check each graph is a tree (n-1 edges)
        for A in graphs:
            edges = A.sum() / 2
            assert edges == 14  # n-1 edges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
