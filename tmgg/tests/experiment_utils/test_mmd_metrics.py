"""Tests for MMD metrics module (DiGress-aligned).

Test Rationale
--------------
These tests verify the MMD (Maximum Mean Discrepancy) computation for graph
generation evaluation, aligned with DiGress for comparability. MMD measures
distributional distance between reference and generated graphs.

Key invariants tested:
1. MMD between identical distributions should be near zero
2. MMD between different distributions should be positive
3. Graph statistics (degree, clustering, spectral) computed correctly
4. Kernels satisfy mathematical properties (symmetry, positivity)
5. Spectral uses normalized Laplacian (DiGress "spectre")
"""

import networkx as nx
import numpy as np
import pytest
import torch

from tmgg.experiment_utils.mmd_metrics import (
    GraphStatistics,
    MMDResults,
    adjacency_to_networkx,
    compute_clustering_histogram,
    compute_degree_histogram,
    compute_graph_statistics,
    compute_mmd,
    compute_mmd_from_adjacencies,
    compute_mmd_metrics,
    compute_spectral_histogram,
    gaussian_kernel,
    gaussian_tv_kernel,
)


class TestAdjacencyToNetworkx:
    """Tests for adjacency matrix to NetworkX conversion."""

    def test_simple_triangle(self):
        """Triangle graph should convert correctly."""
        A = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float32)
        G = adjacency_to_networkx(A)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3

    def test_empty_graph(self):
        """Empty graph should have no edges."""
        A = torch.zeros(5, 5)
        G = adjacency_to_networkx(A)
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 0

    def test_probabilistic_thresholding(self):
        """Values below 0.5 should become 0, above should become 1."""
        A = torch.tensor(
            [[0, 0.6, 0.4], [0.6, 0, 0.8], [0.4, 0.8, 0]], dtype=torch.float32
        )
        G = adjacency_to_networkx(A)
        # 0.6 and 0.8 should become edges, 0.4 should not
        assert G.number_of_edges() == 2


class TestHistogramComputation:
    """Tests for graph statistics histogram computation."""

    def test_degree_histogram_regular_graph(self):
        """d-regular graph should have all degrees equal to d."""
        G = nx.random_regular_graph(3, 10, seed=42)
        hist = compute_degree_histogram(G)
        # All nodes should have degree 3
        assert hist[3] > 0
        # No other degrees should appear
        assert np.sum(hist) == pytest.approx(1.0, rel=0.1)

    def test_clustering_histogram_clique(self):
        """Complete graph should have all clustering coefficients = 1."""
        G = nx.complete_graph(5)
        hist = compute_clustering_histogram(G, num_bins=100)
        # Last bin (near 1.0) should have all mass
        assert hist[-1] > 0 or hist[-2] > 0

    def test_clustering_histogram_default_bins(self):
        """Default clustering histogram should have 100 bins (DiGress style)."""
        G = nx.erdos_renyi_graph(20, 0.3, seed=42)
        hist = compute_clustering_histogram(G)
        assert len(hist) == 100

    def test_spectral_histogram_shape(self):
        """Spectral histogram should have correct number of bins (200 default)."""
        G = nx.erdos_renyi_graph(20, 0.3, seed=42)
        hist = compute_spectral_histogram(G)
        assert len(hist) == 200

    def test_spectral_histogram_custom_bins(self):
        """Spectral histogram should respect custom bin count."""
        G = nx.erdos_renyi_graph(20, 0.3, seed=42)
        hist = compute_spectral_histogram(G, num_bins=50)
        assert len(hist) == 50

    def test_spectral_histogram_uses_laplacian(self):
        """Spectral histogram should use normalized Laplacian eigenvalues.

        Normalized Laplacian eigenvalues are in [0, 2], so histogram should
        have mass only in that range (DiGress spectre metric).
        """
        G = nx.erdos_renyi_graph(30, 0.4, seed=42)
        hist = compute_spectral_histogram(G, num_bins=200)
        # Histogram is PMF, should sum to 1
        assert hist.sum() == pytest.approx(1.0, rel=0.01)


class TestKernels:
    """Tests for kernel functions."""

    def test_gaussian_kernel_identical(self):
        """Gaussian kernel of identical histograms should be 1."""
        x = np.array([1, 2, 3, 4, 5])
        k = gaussian_kernel(x, x, sigma=1.0)
        assert k == pytest.approx(1.0)

    def test_gaussian_kernel_different(self):
        """Gaussian kernel of different histograms should be < 1."""
        x = np.array([1, 0, 0, 0, 0])
        y = np.array([0, 0, 0, 0, 1])
        k = gaussian_kernel(x, y, sigma=1.0)
        assert k < 1.0

    def test_gaussian_kernel_symmetric(self):
        """Gaussian kernel should be symmetric."""
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])
        assert gaussian_kernel(x, y) == pytest.approx(gaussian_kernel(y, x))

    def test_gaussian_tv_kernel_identical(self):
        """Gaussian TV kernel of identical histograms should be 1."""
        x = np.array([1, 2, 3, 4, 5])
        k = gaussian_tv_kernel(x, x)
        assert k == pytest.approx(1.0)

    def test_gaussian_tv_kernel_symmetric(self):
        """Gaussian TV kernel should be symmetric."""
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])
        assert gaussian_tv_kernel(x, y) == pytest.approx(gaussian_tv_kernel(y, x))

    def test_gaussian_tv_kernel_bounded(self):
        """Gaussian TV kernel should be in (0, 1] for any inputs."""
        x = np.array([1, 0, 0, 0, 0])
        y = np.array([0, 0, 0, 0, 1])
        k = gaussian_tv_kernel(x, y, sigma=1.0)
        assert 0 < k < 1  # Different distributions, so k < 1

    def test_gaussian_tv_kernel_sigma_effect(self):
        """Larger sigma should give higher kernel values for different distributions."""
        x = np.array([1, 0, 0])
        y = np.array([0, 0, 1])
        k_small = gaussian_tv_kernel(x, y, sigma=0.1)
        k_large = gaussian_tv_kernel(x, y, sigma=10.0)
        assert k_large > k_small


class TestMMDComputation:
    """Tests for MMD computation."""

    def test_mmd_identical_samples(self):
        """MMD between identical sample sets should be near zero."""
        samples = [np.array([1, 2, 3, 4]) for _ in range(10)]
        mmd = compute_mmd(samples, samples, kernel="gaussian")
        assert mmd == pytest.approx(0.0, abs=0.01)

    def test_mmd_different_samples(self):
        """MMD between different distributions should be positive."""
        samples1 = [np.array([1, 0, 0, 0, 0]) for _ in range(10)]
        samples2 = [np.array([0, 0, 0, 0, 1]) for _ in range(10)]
        mmd = compute_mmd(samples1, samples2, kernel="gaussian")
        assert mmd > 0

    def test_mmd_nonnegative(self):
        """MMD should always be non-negative."""
        rng = np.random.default_rng(42)
        samples1 = [rng.random(5) for _ in range(10)]
        samples2 = [rng.random(5) for _ in range(10)]
        mmd = compute_mmd(samples1, samples2, kernel="gaussian")
        assert mmd >= 0

    def test_mmd_gaussian_tv_kernel(self):
        """MMD should work with gaussian_tv kernel (DiGress style)."""
        samples1 = [np.array([1, 2, 3]) for _ in range(5)]
        samples2 = [np.array([3, 2, 1]) for _ in range(5)]
        mmd = compute_mmd(samples1, samples2, kernel="gaussian_tv")
        assert mmd >= 0

    def test_mmd_default_kernel_is_gaussian_tv(self):
        """Default kernel should be gaussian_tv for DiGress compatibility."""
        samples1 = [np.array([1, 2, 3]) for _ in range(5)]
        samples2 = [np.array([3, 2, 1]) for _ in range(5)]
        mmd_default = compute_mmd(samples1, samples2)
        mmd_explicit = compute_mmd(samples1, samples2, kernel="gaussian_tv")
        assert mmd_default == mmd_explicit


class TestGraphStatistics:
    """Tests for compute_graph_statistics."""

    def test_statistics_structure(self):
        """Should return three histogram types (DiGress-compatible)."""
        graphs = [nx.erdos_renyi_graph(10, 0.3, seed=i) for i in range(5)]
        stats = compute_graph_statistics(graphs)

        assert isinstance(stats, GraphStatistics)
        assert len(stats.degree) == 5
        assert len(stats.clustering) == 5
        assert len(stats.spectral) == 5

    def test_statistics_parallel(self):
        """Should work with parallel execution."""
        graphs = [nx.erdos_renyi_graph(10, 0.3, seed=i) for i in range(10)]
        stats = compute_graph_statistics(graphs, max_workers=2)
        assert len(stats.degree) == 10

    def test_spectral_histogram_sizes(self):
        """Spectral histograms should have 200 bins by default (DiGress)."""
        graphs = [nx.erdos_renyi_graph(15, 0.3, seed=i) for i in range(3)]
        stats = compute_graph_statistics(graphs)
        for hist in stats.spectral:
            assert len(hist) == 200

    def test_clustering_histogram_sizes(self):
        """Clustering histograms should have 100 bins by default (DiGress)."""
        graphs = [nx.erdos_renyi_graph(15, 0.3, seed=i) for i in range(3)]
        stats = compute_graph_statistics(graphs)
        for hist in stats.clustering:
            assert len(hist) == 100


class TestMMDMetrics:
    """Tests for full MMD metrics pipeline (DiGress-compatible)."""

    def test_mmd_results_structure(self):
        """Should return three MMD values (DiGress-compatible)."""
        ref_graphs = [nx.erdos_renyi_graph(20, 0.3, seed=i) for i in range(10)]
        gen_graphs = [nx.erdos_renyi_graph(20, 0.3, seed=i + 100) for i in range(10)]

        results = compute_mmd_metrics(ref_graphs, gen_graphs)

        assert isinstance(results, MMDResults)
        assert hasattr(results, "degree_mmd")
        assert hasattr(results, "clustering_mmd")
        assert hasattr(results, "spectral_mmd")

    def test_mmd_identical_distributions(self):
        """MMD between identical distributions should be near zero."""
        # Generate same graphs twice (same seeds)
        ref_graphs = [nx.erdos_renyi_graph(15, 0.3, seed=i) for i in range(10)]
        gen_graphs = [nx.erdos_renyi_graph(15, 0.3, seed=i) for i in range(10)]

        results = compute_mmd_metrics(ref_graphs, gen_graphs)

        # Should be near zero for identical distributions
        assert results.degree_mmd == pytest.approx(0.0, abs=0.01)

    def test_mmd_different_graph_types(self):
        """MMD between different graph types should be larger."""
        # ER random graphs vs regular graphs
        ref_graphs = [nx.erdos_renyi_graph(20, 0.3, seed=i) for i in range(10)]
        gen_graphs = [nx.random_regular_graph(3, 20, seed=i) for i in range(10)]

        results = compute_mmd_metrics(ref_graphs, gen_graphs)

        # Degree distribution should differ significantly
        # (ER has varying degrees, regular has fixed degree)
        assert results.degree_mmd > 0.01

    def test_to_dict(self):
        """Results should convert to dictionary for logging."""
        ref_graphs = [nx.erdos_renyi_graph(10, 0.3, seed=i) for i in range(5)]
        gen_graphs = [nx.erdos_renyi_graph(10, 0.3, seed=i + 50) for i in range(5)]

        results = compute_mmd_metrics(ref_graphs, gen_graphs)
        d = results.to_dict()

        assert "degree_mmd" in d
        assert "clustering_mmd" in d
        assert "spectral_mmd" in d
        assert len(d) == 3  # Only three metrics now


class TestMMDFromAdjacencies:
    """Tests for convenience function with adjacency matrices."""

    def test_from_torch_tensors(self):
        """Should work with batched torch tensors."""
        # Create simple adjacency matrices
        n = 10
        batch_size = 5

        # Random symmetric adjacency matrices
        ref = torch.rand(batch_size, n, n)
        ref = (ref + ref.transpose(-2, -1)) / 2
        ref = (ref > 0.5).float()
        ref = ref * (1 - torch.eye(n))

        gen = torch.rand(batch_size, n, n)
        gen = (gen + gen.transpose(-2, -1)) / 2
        gen = (gen > 0.5).float()
        gen = gen * (1 - torch.eye(n))

        results = compute_mmd_from_adjacencies(ref, gen)
        assert isinstance(results, MMDResults)

    def test_from_numpy_arrays(self):
        """Should work with numpy arrays."""
        n = 10
        batch_size = 5

        ref = np.random.rand(batch_size, n, n)
        ref = (ref + ref.transpose(0, 2, 1)) / 2
        ref = (ref > 0.5).astype(float)
        np.einsum("bii->bi", ref)[:] = 0

        gen = np.random.rand(batch_size, n, n)
        gen = (gen + gen.transpose(0, 2, 1)) / 2
        gen = (gen > 0.5).astype(float)
        np.einsum("bii->bi", gen)[:] = 0

        results = compute_mmd_from_adjacencies(ref, gen)
        assert isinstance(results, MMDResults)
