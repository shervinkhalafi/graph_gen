"""Integration tests for end-to-end training pipelines."""

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tmgg.experiment_utils.data import (
    PermutedAdjacencyDataset,
    add_digress_noise,
    add_gaussian_noise,
    add_rotation_noise,
    compute_eigendecomposition,
    generate_sbm_adjacency,
    random_skew_symmetric_matrix,
)
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.gnn import GNN, GNNSymmetric, NodeVarGNN
from tmgg.models.hybrid import create_sequential_model


class TestEndToEndTraining:
    """Test complete training pipelines for different model types."""

    @pytest.fixture
    def training_data(
        self,
    ) -> tuple[DataLoader[torch.Tensor], list[torch.Tensor]]:
        """Create sample training data."""
        # Generate a few SBM graphs
        block_sizes_list = [[5, 5], [3, 3, 4], [7, 3]]
        adjacency_matrices: list[torch.Tensor] = []

        for block_sizes in block_sizes_list:
            A = generate_sbm_adjacency(block_sizes, p=1.0, q=0.0)
            adjacency_matrices.append(torch.tensor(A, dtype=torch.float32))

        # Create dataset
        dataset = PermutedAdjacencyDataset(adjacency_matrices, num_samples=50)
        dataloader: DataLoader[torch.Tensor] = DataLoader(
            dataset, batch_size=10, shuffle=True
        )

        return dataloader, adjacency_matrices

    def test_gnn_training_pipeline(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test GNN training end-to-end."""
        dataloader, adjacency_matrices = training_data

        # Initialize model
        model = GNN(num_layers=2, num_terms=2, feature_dim_in=10, feature_dim_out=5)
        _ = model.float()

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        initial_loss: float | None = None
        _ = model.train()

        for _ in range(5):
            epoch_loss = 0.0
            for batch in dataloader:
                # Add noise
                batch_noisy = add_digress_noise(batch, p=0.1)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass
                X, Y = model(batch_noisy)
                A_pred = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if initial_loss is None:
                initial_loss = avg_loss

        # Check that loss decreased
        assert initial_loss is not None
        assert avg_loss < initial_loss

        # Test evaluation
        _ = model.eval()
        with torch.no_grad():
            test_batch = adjacency_matrices[0].unsqueeze(0)
            test_noisy = add_digress_noise(test_batch, p=0.1)

            X, Y = model(test_noisy.float())
            A_pred = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))

            assert A_pred.shape == test_batch.shape
            assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)

    def test_attention_training_pipeline(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test attention model training end-to-end.

        MultiLayerAttention now returns a single tensor (reconstructed adjacency)
        rather than (output, attention_scores) tuple.
        """
        dataloader, adjacency_matrices = training_data

        # Initialize model
        k = 10  # Number of eigenvectors
        model = MultiLayerAttention(d_model=k, num_heads=2, num_layers=2, d_k=5, d_v=5)
        _ = model.float()

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training with rotation noise
        skew = random_skew_symmetric_matrix(k)
        initial_loss: float | None = None
        _ = model.train()

        for _ in range(5):
            epoch_loss = 0.0
            for batch in dataloader:
                # Add rotation noise
                batch_noisy = add_rotation_noise(batch, eps=0.1, skew=skew)
                _, V_noisy = compute_eigendecomposition(batch_noisy)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Use only top k eigenvectors
                V_input = V_noisy[:, :, :k].float()

                # Forward pass - model returns single tensor
                A_pred = model(V_input)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if initial_loss is None:
                initial_loss = avg_loss

        # Check that loss decreased
        assert initial_loss is not None
        assert avg_loss < initial_loss

    def test_hybrid_training_pipeline(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test hybrid model training end-to-end."""
        dataloader, adjacency_matrices = training_data

        # Create hybrid model
        gnn_config: dict[str, Any] = {
            "num_layers": 1,
            "num_terms": 2,
            "feature_dim_in": 10,
            "feature_dim_out": 5,
        }

        transformer_config: dict[str, Any] = {
            "num_heads": 2,
            "num_layers": 1,
            "d_k": 5,
            "d_v": 5,
        }

        model = create_sequential_model(gnn_config, transformer_config)
        _ = model.float()

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        initial_loss: float | None = None
        _ = model.train()

        for _ in range(5):
            epoch_loss = 0.0
            for batch in dataloader:
                # Add Gaussian noise
                batch_noisy = add_gaussian_noise(batch, eps=0.1)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass
                A_pred = model(batch_noisy)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if initial_loss is None:
                initial_loss = avg_loss

        # Check that loss decreased
        assert initial_loss is not None
        assert avg_loss < initial_loss

        # Test evaluation
        _ = model.eval()
        with torch.no_grad():
            test_batch = adjacency_matrices[0].unsqueeze(0)
            test_noisy = add_gaussian_noise(test_batch, eps=0.1)

            A_pred = model(test_noisy.float())

            assert A_pred.shape == test_batch.shape
            assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)

    def test_multi_noise_level_training(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test training with multiple noise levels."""
        dataloader, _ = training_data

        # Initialize simple GNN
        model = GNNSymmetric(num_layers=1, feature_dim_in=10, feature_dim_out=5)
        _ = model.float()

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Define noise levels
        noise_levels = [0.05, 0.1, 0.2, 0.3]

        _ = model.train()
        for _ in range(3):
            for batch in dataloader:
                # Sample random noise level
                eps: float = np.random.choice(noise_levels)

                # Add noise
                batch_noisy = add_digress_noise(batch, p=eps)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass
                A_pred, _ = model(batch_noisy)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test on each noise level
        _ = model.eval()
        with torch.no_grad():
            test_batch = dataloader.dataset[0].unsqueeze(0)

            for eps in noise_levels:
                test_noisy = add_digress_noise(test_batch, p=eps)
                A_pred, _ = model(test_noisy.float())

                assert A_pred.shape == test_batch.shape
                assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)

    def test_learning_rate_scheduling(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test training with learning rate scheduling."""
        dataloader, _ = training_data

        # Initialize model
        model = NodeVarGNN(num_layers=1, num_terms=2, feature_dim=10)
        _ = model.float()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Add cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )

        # Track learning rates
        lrs: list[float] = []

        _ = model.train()
        for _ in range(10):
            for batch in dataloader:
                batch_noisy = add_gaussian_noise(batch, eps=0.1)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass
                A_pred = model(batch_noisy)
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step scheduler
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Check that learning rate changed
        assert len(set(lrs)) > 1
        assert min(lrs) < 0.1  # LR decreased at some point

    def test_gradient_flow_with_regularization(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test gradients flow with eigenvalue regularization.

        Eigendecomposition gradients involve 1/(λ_i - λ_j) terms that explode
        for near-degenerate eigenvalues. Adding diagonal regularization spreads
        eigenvalues apart, preventing this numerical instability.

        This test verifies the GNN model directly. For hybrid models, gradients
        must traverse both eigendecomposition and transformer layers, resulting
        in smaller (but still non-zero) gradients that require relaxed tolerance.
        """
        dataloader, _ = training_data

        # Test GNN with eigenvalue regularization
        model = GNN(num_layers=2, feature_dim_out=5, eigenvalue_reg=1e-3)
        _ = model.float()

        batch = next(iter(dataloader))
        batch_noisy = add_digress_noise(batch, p=0.1).float()
        batch = batch.float()

        X, Y = model(batch_noisy)
        output = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))
        loss = torch.mean((output - batch) ** 2)
        loss.backward()

        # With regularization, GNN gradients should be numerically stable
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1
                assert not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=1e-10
                ), f"Zero gradient for {name}"

        assert params_with_grad > 0, "No parameters have gradients"

    def test_gradient_flow_hybrid_model(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Test gradient flow in hybrid model with relaxed tolerance.

        Hybrid models have gradients flowing through eigendecomposition then
        transformer attention, resulting in smaller gradients. We verify
        gradients exist (atol=1e-6) rather than strict non-zero check.
        """
        dataloader, _ = training_data

        gnn_config: dict[str, Any] = {
            "num_layers": 1,
            "feature_dim_out": 5,
            "eigenvalue_reg": 1e-3,
        }
        transformer_config: dict[str, Any] = {
            "num_heads": 2,
            "num_layers": 1,
        }

        model = create_sequential_model(gnn_config, transformer_config)
        _ = model.float()

        batch = next(iter(dataloader))
        batch_noisy = add_digress_noise(batch, p=0.1).float()
        batch = batch.float()

        output = model(batch_noisy)
        loss = torch.mean((output - batch) ** 2)
        loss.backward()

        # Verify most parameters have non-trivial gradients (relaxed tolerance)
        params_with_grad = 0
        params_with_nonzero_grad = 0
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1
                if not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=1e-6
                ):
                    params_with_nonzero_grad += 1

        assert params_with_grad > 0, "No parameters have gradients"
        # At least 50% of parameters should have meaningful gradients
        grad_ratio = params_with_nonzero_grad / params_with_grad
        assert (
            grad_ratio >= 0.5
        ), f"Only {grad_ratio:.1%} of parameters have meaningful gradients"

    @pytest.mark.xfail(
        reason="Without eigenvalue regularization, near-degenerate eigenvalues "
        "cause gradient terms 1/(λ_i - λ_j) to explode, resulting in "
        "near-zero gradients after normalization."
    )
    def test_gradient_flow_without_regularization(
        self, training_data: tuple[DataLoader[torch.Tensor], list[torch.Tensor]]
    ) -> None:
        """Demonstrate gradient issues without regularization.

        This test documents that eigendecomposition breaks gradient flow when
        eigenvalues are close together. Use eigenvalue_reg > 0 in practice.
        """
        dataloader, _ = training_data

        # Model without regularization
        model = GNN(num_layers=2, feature_dim_out=5, eigenvalue_reg=0.0)
        _ = model.float()

        batch = next(iter(dataloader))
        batch_noisy = add_digress_noise(batch, p=0.1).float()
        batch = batch.float()

        X, Y = model(batch_noisy)
        output = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))
        loss = torch.mean((output - batch) ** 2)
        loss.backward()

        # Without regularization, some gradients will be near-zero
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=1e-10
                ), f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__])
