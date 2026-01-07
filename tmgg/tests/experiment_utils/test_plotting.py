"""Tests for plotting utilities."""

import os
import tempfile
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from tmgg.experiment_utils.data import add_digress_noise
from tmgg.experiment_utils.plotting import (
    create_multi_noise_visualization,
    create_network_denoising_figure,
    plot_denoising_results,
    plot_eigenvalue_comparison,
    plot_graph_denoising_combined,
    plot_graph_network_comparison,
    plot_noise_level_comparison,
    plot_training_curves,
    plot_training_metrics_grid,
)
from tmgg.models.gnn import GNNSymmetric


class TestPlottingFunctions:
    """Test plotting utility functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for plotting tests."""
        # Sample losses
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.55]

        # Sample adjacency matrices
        A_original = np.eye(10)
        A_noisy = A_original + 0.1 * np.random.randn(10, 10)
        A_denoised = A_original + 0.05 * np.random.randn(10, 10)

        # Sample eigenvalues
        eigenvals_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        eigenvals_noisy = eigenvals_true + 0.1 * np.random.randn(5)
        eigenvals_denoised = eigenvals_true + 0.05 * np.random.randn(5)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "A_original": A_original,
            "A_noisy": A_noisy,
            "A_denoised": A_denoised,
            "eigenvals_true": eigenvals_true,
            "eigenvals_noisy": eigenvals_noisy,
            "eigenvals_denoised": eigenvals_denoised,
        }

    def test_plot_training_curves(self, sample_data):
        """Test training curves plotting."""
        fig = plot_training_curves(
            sample_data["train_losses"],
            sample_data["val_losses"],
            title="Test Training Curves",
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        # Check that both lines are plotted
        ax = fig.axes[0]
        assert len(ax.lines) == 2

        # Check labels
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Loss"

        plt.close(fig)

    def test_plot_training_curves_with_save(self, sample_data):
        """Test saving training curves plot."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig = plot_training_curves(
                sample_data["train_losses"],
                sample_data["val_losses"],
                save_path=tmp_path,
            )

            # Check file was created
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            plt.close(fig)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_plot_denoising_results(self, sample_data):
        """Test denoising results plotting."""
        fig = plot_denoising_results(
            sample_data["A_original"],
            sample_data["A_noisy"],
            sample_data["A_denoised"],
            noise_type="Gaussian",
            eps=0.1,
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 8  # 4 main plots + 4 colorbars (includes Delta)

        # Check titles - get non-colorbar axes
        main_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        titles = [ax.get_title() for ax in main_axes]
        assert "Original" in titles[0]
        assert "Noisy" in titles[1]
        assert "Denoised" in titles[2]
        assert "Delta" in titles[3]

        plt.close(fig)

    def test_plot_noise_level_comparison(self):
        """Test noise level comparison plotting."""
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        metrics_dict = {
            "Train": [0.1, 0.15, 0.2, 0.25, 0.3],
            "Test": [0.12, 0.18, 0.24, 0.30, 0.36],
        }

        fig = plot_noise_level_comparison(noise_levels, metrics_dict, metric_name="MSE")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert len(ax.lines) == 2  # Two datasets
        assert ax.get_xlabel() == "Noise Level (ε)"
        assert ax.get_ylabel() == "MSE"

        # Check log scale
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

        plt.close(fig)

    def test_plot_eigenvalue_comparison(self, sample_data):
        """Test eigenvalue comparison plotting."""
        fig = plot_eigenvalue_comparison(
            sample_data["eigenvals_true"],
            sample_data["eigenvals_noisy"],
            sample_data["eigenvals_denoised"],
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        # Check that we have 3 sets of bars
        assert len(ax.patches) == 15  # 5 eigenvalues * 3 sets

        plt.close(fig)

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
            A_original, model, add_digress_noise, noise_levels
        )

        assert isinstance(fig, Figure)
        # Should have 5 rows x 3 columns = 15 subplots
        # (clean, noisy, denoised, predicted, delta)
        assert len(fig.axes) == 15

        plt.close(fig)

    def test_plot_training_metrics_grid(self):
        """Test plotting grid of training metrics."""
        metrics_history = {
            "loss": [1.0, 0.8, 0.6, 0.4],
            "accuracy": [0.6, 0.7, 0.8, 0.85],
            "learning_rate": [0.01, 0.008, 0.006, 0.004],
        }

        fig = plot_training_metrics_grid(metrics_history)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3

        # Check titles
        titles = [ax.get_title() for ax in fig.axes]
        assert any("loss" in title.lower() for title in titles)
        assert any("accuracy" in title.lower() for title in titles)
        assert any("learning_rate" in title.lower() for title in titles)

        # Check that loss plot uses log scale
        for ax, (name, _) in zip(fig.axes, metrics_history.items(), strict=False):
            if "loss" in name.lower():
                assert ax.get_yscale() == "log"

        plt.close(fig)

    def test_plot_training_metrics_grid_single_metric(self):
        """Test plotting grid with single metric."""
        metrics_history = {"loss": [1.0, 0.8, 0.6, 0.4]}

        fig = plot_training_metrics_grid(metrics_history)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_empty_data(self):
        """Test that functions handle empty data gracefully."""
        # Empty training curves
        fig = plot_training_curves([], [])
        assert isinstance(fig, Figure)
        plt.close(fig)

        # Empty metrics
        fig = plot_noise_level_comparison([], {})
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mismatched_dimensions(self, sample_data):
        """Test handling of mismatched dimensions."""
        # Different sized arrays for eigenvalues
        eigenvals_short = sample_data["eigenvals_true"][:3]

        # This should still work, just plot available data
        fig = plot_eigenvalue_comparison(
            eigenvals_short,
            sample_data["eigenvals_noisy"][:3],
            sample_data["eigenvals_denoised"][:3],
        )

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # Should have 3 * 3 = 9 bars
        assert len(ax.patches) == 9

        plt.close(fig)

    def test_figure_properties(self, sample_data):
        """Test that figures have correct properties."""
        fig = plot_training_curves(
            sample_data["train_losses"], sample_data["val_losses"]
        )

        # Check figure size
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6

        # Check that grid is enabled
        ax = fig.axes[0]
        assert ax.xaxis.get_gridlines()[0].get_visible()

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

    def test_plot_graph_network_comparison_basic(self, small_graph):
        """Test basic 3-panel network comparison visualization."""
        A_clean = small_graph
        # Simulated noisy version (flip some edges)
        A_noisy = A_clean.copy()
        A_noisy[0, 1] = A_noisy[1, 0] = 0  # Remove edge
        A_noisy[0, 8] = A_noisy[8, 0] = 1  # Add edge
        # Simulated denoised version
        A_denoised = A_clean.copy()
        A_denoised[0, 1] = A_denoised[1, 0] = 0.8  # Soft prediction

        fig = plot_graph_network_comparison(
            A_clean=A_clean,
            A_noisy=A_noisy,
            A_denoised=A_denoised,
            layout="spring",
            seed=42,
        )

        assert fig is not None
        assert isinstance(fig, Figure)
        # Should have 3 axes (clean, noisy, denoised)
        assert len(fig.axes) == 3

        plt.close(fig)

    def test_plot_graph_network_comparison_layouts(self, small_graph):
        """Test different layout algorithms for network visualization."""
        A = small_graph

        layouts: list[Literal["spring", "spectral", "kamada_kawai"]] = [
            "spring",
            "spectral",
            "kamada_kawai",
        ]
        for layout_name in layouts:
            fig = plot_graph_network_comparison(
                A_clean=A,
                A_noisy=A,
                A_denoised=A,
                layout=layout_name,
                seed=42,
            )

            assert fig is not None
            assert isinstance(fig, Figure)
            plt.close(fig)

    def test_plot_graph_network_comparison_skips_large_graphs(self, large_graph):
        """Test that network visualization returns None for large graphs."""
        A = large_graph

        fig = plot_graph_network_comparison(
            A_clean=A,
            A_noisy=A,
            A_denoised=A,
            max_nodes_for_layout=50,
        )

        # Should return None for graphs exceeding max_nodes
        assert fig is None

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

    def test_plot_graph_network_comparison_with_tensor(self, small_graph):
        """Test that torch tensors are handled correctly."""
        A_clean = torch.from_numpy(small_graph).float()
        A_noisy = A_clean.clone()
        A_denoised = A_clean.clone()

        fig = plot_graph_network_comparison(
            A_clean=A_clean,
            A_noisy=A_noisy,
            A_denoised=A_denoised,
        )

        assert fig is not None
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_plot_graph_network_empty_graph(self):
        """Test handling of graph with no edges."""
        n = 10
        A_empty = np.zeros((n, n))

        fig = plot_graph_network_comparison(
            A_clean=A_empty,
            A_noisy=A_empty,
            A_denoised=A_empty,
        )

        assert fig is not None
        assert isinstance(fig, Figure)

        plt.close(fig)

    def test_plot_graph_network_full_graph(self):
        """Test handling of complete graph."""
        n = 10
        A_full = np.ones((n, n))
        np.fill_diagonal(A_full, 0)

        fig = plot_graph_network_comparison(
            A_clean=A_full,
            A_noisy=A_full,
            A_denoised=A_full,
        )

        assert fig is not None
        assert isinstance(fig, Figure)

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
