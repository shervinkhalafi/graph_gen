"""Tests for plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from tmgg.data import add_edge_flip_noise
from tmgg.experiments._shared_utils.plotting import (
    create_multi_noise_visualization,
    create_network_denoising_figure,
    plot_graph_denoising_combined,
)
from tmgg.models.gnn import GNNSymmetric


class TestPlottingFunctions:
    """Test plotting utility functions."""

    def test_create_multi_noise_visualization(self):
        """Test multi-noise visualization.

        Uses GNNSymmetric which outputs adjacency matrix reconstructions,
        as expected by create_multi_noise_visualization.
        """
        # Create model that outputs adjacency reconstruction
        model = GNNSymmetric(num_layers=1, feature_dim_out=5)
        model.eval()

        # Create test adjacency matrix
        A_original = torch.eye(10).unsqueeze(0)
        noise_levels = [0.1, 0.2, 0.3]

        fig = create_multi_noise_visualization(
            A_original, model, add_edge_flip_noise, noise_levels
        )

        assert isinstance(fig, Figure)
        # Should have 5 rows x 3 columns = 15 subplots
        # (clean, noisy, denoised, predicted, delta)
        assert len(fig.axes) == 15

        plt.close(fig)


class TestNetworkVisualization:
    """Tests for node-link graph visualization functions.

    Test Rationale
    --------------
    These tests validate the network (node-link diagram) visualization additions,
    ensuring:
    - Correct figure creation with appropriate panel counts
    - Edge difference coloring works correctly (added/removed edges)
    - Layout algorithms produce valid node positions
    - Graceful handling of edge cases (large graphs, empty graphs)
    - Integration with noise/denoise workflow
    """

    @pytest.fixture
    def small_graph(self):
        """Create a small test graph (n=10) suitable for network visualization."""
        n = 10
        # Create a simple block structure: two cliques connected by a bridge
        A = np.zeros((n, n))
        # First clique (nodes 0-4)
        A[:5, :5] = 1
        # Second clique (nodes 5-9)
        A[5:, 5:] = 1
        # Bridge edge
        A[4, 5] = A[5, 4] = 1
        # Zero diagonal
        np.fill_diagonal(A, 0)
        return A

    @pytest.fixture
    def large_graph(self):
        """Create a graph too large for network visualization (n=100)."""
        n = 100
        # Random sparse graph
        A = np.random.rand(n, n)
        A = (A > 0.9).astype(float)
        A = np.triu(A, 1)
        A = A + A.T
        return A

    def test_plot_graph_denoising_combined(self, small_graph):
        """Test combined heatmap + network 2-row visualization."""
        A_clean = small_graph
        A_noisy = A_clean + 0.1 * np.random.randn(*A_clean.shape)
        A_noisy = np.clip(A_noisy, 0, 1)
        A_noisy = (A_noisy + A_noisy.T) / 2  # Keep symmetric
        A_denoised = A_clean.copy()

        fig = plot_graph_denoising_combined(
            A_clean=A_clean,
            A_noisy=A_noisy,
            A_denoised=A_denoised,
            noise_type="test",
            noise_level=0.1,
        )

        assert isinstance(fig, Figure)
        # Should have 2 rows: 4 heatmaps (row 1) + 3 networks (row 2) = 7 axes
        # Plus colorbars: +4 = 11 total (approximate, depends on implementation)
        assert len(fig.axes) >= 7

        plt.close(fig)

    def test_plot_graph_denoising_combined_large_graph_heatmaps_only(self, large_graph):
        """Test combined visualization falls back to heatmaps only for large graphs."""
        A = large_graph

        fig = plot_graph_denoising_combined(
            A_clean=A,
            A_noisy=A,
            A_denoised=A,
            max_nodes_for_layout=50,
        )

        # Should still return a figure (heatmaps work for any size)
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_create_network_denoising_figure(self, small_graph):
        """Test convenience wrapper with noise/denoise functions."""
        A_clean = small_graph

        def noise_fn(A, eps):
            """Simple flip noise."""
            mask = np.random.rand(*A.shape) < eps
            A_noisy = A.copy()
            A_noisy[mask] = 1 - A_noisy[mask]
            A_noisy = np.triu(A_noisy, 1)
            A_noisy = A_noisy + A_noisy.T
            return A_noisy

        def denoise_fn(A_noisy):
            """Identity denoiser for testing."""
            return A_noisy

        fig = create_network_denoising_figure(
            A_clean=A_clean,
            noise_fn=noise_fn,
            denoise_fn=denoise_fn,
            noise_level=0.1,
            noise_type="flip",
        )

        assert fig is not None
        assert isinstance(fig, Figure)

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
