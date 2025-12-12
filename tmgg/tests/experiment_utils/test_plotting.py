"""Tests for plotting utilities."""

import pytest
import matplotlib.pyplot as plt
import numpy as np
import torch
import tempfile
import os

from tmgg.experiment_utils.plotting import (
    plot_training_curves,
    plot_denoising_results,
    plot_noise_level_comparison,
    plot_eigenvalue_comparison,
    create_multi_noise_visualization,
    plot_training_metrics_grid
)
from tmgg.models.gnn import GNNSymmetric
from tmgg.experiment_utils.data import add_digress_noise


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
            'train_losses': train_losses,
            'val_losses': val_losses,
            'A_original': A_original,
            'A_noisy': A_noisy,
            'A_denoised': A_denoised,
            'eigenvals_true': eigenvals_true,
            'eigenvals_noisy': eigenvals_noisy,
            'eigenvals_denoised': eigenvals_denoised
        }
    
    def test_plot_training_curves(self, sample_data):
        """Test training curves plotting."""
        fig = plot_training_curves(
            sample_data['train_losses'],
            sample_data['val_losses'],
            title="Test Training Curves"
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check that both lines are plotted
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        
        # Check labels
        assert ax.get_xlabel() == 'Epoch'
        assert ax.get_ylabel() == 'Loss'
        
        plt.close(fig)
    
    def test_plot_training_curves_with_save(self, sample_data):
        """Test saving training curves plot."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig = plot_training_curves(
                sample_data['train_losses'],
                sample_data['val_losses'],
                save_path=tmp_path
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
            sample_data['A_original'],
            sample_data['A_noisy'],
            sample_data['A_denoised'],
            noise_type="Gaussian",
            eps=0.1
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 8  # 4 main plots + 4 colorbars (includes Delta)

        # Check titles - get non-colorbar axes
        main_axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
        titles = [ax.get_title() for ax in main_axes]
        assert 'Original' in titles[0]
        assert 'Noisy' in titles[1]
        assert 'Denoised' in titles[2]
        assert 'Delta' in titles[3]

        plt.close(fig)
    
    def test_plot_noise_level_comparison(self):
        """Test noise level comparison plotting."""
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        metrics_dict = {
            'Train': [0.1, 0.15, 0.2, 0.25, 0.3],
            'Test': [0.12, 0.18, 0.24, 0.30, 0.36]
        }
        
        fig = plot_noise_level_comparison(
            noise_levels,
            metrics_dict,
            metric_name="MSE"
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert len(ax.lines) == 2  # Two datasets
        assert ax.get_xlabel() == 'Noise Level (ε)'
        assert ax.get_ylabel() == 'MSE'
        
        # Check log scale
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_plot_eigenvalue_comparison(self, sample_data):
        """Test eigenvalue comparison plotting."""
        fig = plot_eigenvalue_comparison(
            sample_data['eigenvals_true'],
            sample_data['eigenvals_noisy'],
            sample_data['eigenvals_denoised']
        )
        
        assert isinstance(fig, plt.Figure)
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
            A_original,
            model,
            add_digress_noise,
            noise_levels
        )

        assert isinstance(fig, plt.Figure)
        # Should have 4 rows x 3 columns = 12 subplots (clean, noisy, denoised, delta)
        assert len(fig.axes) == 12

        plt.close(fig)
    
    def test_plot_training_metrics_grid(self):
        """Test plotting grid of training metrics."""
        metrics_history = {
            'loss': [1.0, 0.8, 0.6, 0.4],
            'accuracy': [0.6, 0.7, 0.8, 0.85],
            'learning_rate': [0.01, 0.008, 0.006, 0.004]
        }
        
        fig = plot_training_metrics_grid(metrics_history)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        
        # Check titles
        titles = [ax.get_title() for ax in fig.axes]
        assert any('loss' in title.lower() for title in titles)
        assert any('accuracy' in title.lower() for title in titles)
        assert any('learning_rate' in title.lower() for title in titles)
        
        # Check that loss plot uses log scale
        for ax, (name, _) in zip(fig.axes, metrics_history.items()):
            if 'loss' in name.lower():
                assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_plot_training_metrics_grid_single_metric(self):
        """Test plotting grid with single metric."""
        metrics_history = {
            'loss': [1.0, 0.8, 0.6, 0.4]
        }
        
        fig = plot_training_metrics_grid(metrics_history)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    def test_plot_empty_data(self):
        """Test that functions handle empty data gracefully."""
        # Empty training curves
        fig = plot_training_curves([], [])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Empty metrics
        fig = plot_noise_level_comparison([], {})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_mismatched_dimensions(self, sample_data):
        """Test handling of mismatched dimensions."""
        # Different sized arrays for eigenvalues
        eigenvals_short = sample_data['eigenvals_true'][:3]
        
        # This should still work, just plot available data
        fig = plot_eigenvalue_comparison(
            eigenvals_short,
            sample_data['eigenvals_noisy'][:3],
            sample_data['eigenvals_denoised'][:3]
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should have 3 * 3 = 9 bars
        assert len(ax.patches) == 9
        
        plt.close(fig)
    
    def test_figure_properties(self, sample_data):
        """Test that figures have correct properties."""
        fig = plot_training_curves(
            sample_data['train_losses'],
            sample_data['val_losses']
        )
        
        # Check figure size
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        
        # Check that grid is enabled
        ax = fig.axes[0]
        assert ax.xaxis.get_gridlines()[0].get_visible()
        
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])