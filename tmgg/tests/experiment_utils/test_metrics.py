"""Tests for metrics utilities."""

import numpy as np
import pytest
import torch

from tmgg.experiment_utils.metrics import (
    compute_batch_metrics,
    compute_eigenvalue_error,
    compute_reconstruction_metrics,
    compute_subspace_distance,
)


class TestMetrics:
    """Test metrics computation functions."""

    def test_compute_eigenvalue_error(self):
        """Test eigenvalue error computation."""
        A_true = np.eye(10)
        A_pred = np.eye(10) + 0.1 * np.random.randn(10, 10)
        A_pred = (A_pred + A_pred.T) / 2  # Make symmetric

        error = compute_eigenvalue_error(A_true, A_pred, k=4)

        assert isinstance(error, float)
        assert error >= 0

    def test_compute_subspace_distance(self):
        """Test subspace distance computation."""
        A_true = np.eye(10)
        A_pred = np.eye(10) + 0.1 * np.random.randn(10, 10)
        A_pred = (A_pred + A_pred.T) / 2  # Make symmetric

        distance = compute_subspace_distance(A_true, A_pred, k=4)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_compute_reconstruction_metrics(self):
        """Test reconstruction metrics (pruned set: mse, frobenius, spectral)."""
        A_true = torch.eye(5)
        A_pred = torch.eye(5) + 0.1 * torch.randn(5, 5)

        metrics = compute_reconstruction_metrics(A_true, A_pred)

        expected_keys = [
            "mse",
            "frobenius_error",
            "eigenvalue_error",
            "subspace_distance",
        ]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert metrics[key] >= 0

    def test_compute_batch_metrics(self):
        """Test batch metrics computation."""
        batch_size = 3
        matrix_size = 5

        A_true_batch = torch.eye(matrix_size).unsqueeze(0).repeat(batch_size, 1, 1)
        A_pred_batch = A_true_batch + 0.1 * torch.randn(
            batch_size, matrix_size, matrix_size
        )

        metrics = compute_batch_metrics(A_true_batch, A_pred_batch)

        expected_keys = [
            "mse",
            "frobenius_error",
            "eigenvalue_error",
            "subspace_distance",
        ]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert metrics[key] >= 0

    def test_perfect_reconstruction(self):
        """Test metrics with perfect reconstruction."""
        A_true = torch.eye(5)
        A_pred = A_true.clone()

        metrics = compute_reconstruction_metrics(A_true, A_pred)

        assert metrics["mse"] < 1e-6
        assert metrics["frobenius_error"] < 1e-6

    def test_tensor_input_handling(self):
        """Test that function handles different tensor types."""
        A_true_np = np.eye(5)
        A_pred_np = np.eye(5) + 0.1 * np.random.randn(5, 5)

        metrics_np = compute_reconstruction_metrics(A_true_np, A_pred_np)
        assert isinstance(metrics_np["mse"], float)

        A_true_torch = torch.from_numpy(A_true_np).float()
        A_pred_torch = torch.from_numpy(A_pred_np).float()

        metrics_torch = compute_reconstruction_metrics(A_true_torch, A_pred_torch)
        assert isinstance(metrics_torch["mse"], float)

        assert abs(metrics_np["mse"] - metrics_torch["mse"]) < 1e-5

    def test_batch_dimension_handling(self):
        """Test handling of batch dimensions."""
        A_true_batch = torch.eye(5).unsqueeze(0)
        A_pred_batch = A_true_batch + 0.1 * torch.randn(1, 5, 5)

        metrics_batch = compute_reconstruction_metrics(A_true_batch, A_pred_batch)

        A_true_single = A_true_batch.squeeze(0)
        A_pred_single = A_pred_batch.squeeze(0)

        metrics_single = compute_reconstruction_metrics(A_true_single, A_pred_single)

        assert abs(metrics_batch["mse"] - metrics_single["mse"]) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__])
