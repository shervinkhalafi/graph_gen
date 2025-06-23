"""Integration tests for end-to-end training pipelines."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tmgg.models.gnn import GNN, GNNSymmetric, NodeVarGNN
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.hybrid import SequentialDenoisingModel, create_sequential_model
from tmgg.experiment_utils.data import (
    generate_sbm_adjacency, 
    PermutedAdjacencyDataset,
    add_digress_noise,
    add_gaussian_noise,
    add_rotation_noise,
    random_skew_symmetric_matrix
)


class TestEndToEndTraining:
    """Test complete training pipelines for different model types."""
    
    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        # Generate a few SBM graphs
        block_sizes_list = [[5, 5], [3, 3, 4], [7, 3]]
        adjacency_matrices = []
        
        for block_sizes in block_sizes_list:
            A = generate_sbm_adjacency(block_sizes, p=1.0, q=0.0)
            adjacency_matrices.append(torch.tensor(A, dtype=torch.float32))
        
        # Create dataset
        dataset = PermutedAdjacencyDataset(adjacency_matrices, num_samples=50)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        return dataloader, adjacency_matrices
    
    def test_gnn_training_pipeline(self, training_data):
        """Test GNN training end-to-end."""
        dataloader, adjacency_matrices = training_data
        
        # Initialize model
        model = GNN(num_layers=2, num_terms=2, feature_dim_in=10, feature_dim_out=5)
        model = model.float()
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        initial_loss = None
        model.train()
        
        for epoch in range(5):
            epoch_loss = 0.0
            for batch in dataloader:
                # Add noise
                batch_noisy, _, _ = add_digress_noise(batch, p=0.1)
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
        assert avg_loss < initial_loss
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_batch = adjacency_matrices[0].unsqueeze(0)
            test_noisy, _, _ = add_digress_noise(test_batch, p=0.1)
            
            X, Y = model(test_noisy.float())
            A_pred = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))
            
            assert A_pred.shape == test_batch.shape
            assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)
    
    def test_attention_training_pipeline(self, training_data):
        """Test attention model training end-to-end."""
        dataloader, adjacency_matrices = training_data
        
        # Initialize model
        k = 10  # Number of eigenvectors
        model = MultiLayerAttention(
            d_model=k, 
            num_heads=2, 
            num_layers=2,
            d_k=5,
            d_v=5
        )
        model = model.float()
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training with rotation noise
        skew = random_skew_symmetric_matrix(k)
        initial_loss = None
        model.train()
        
        for epoch in range(5):
            epoch_loss = 0.0
            for batch in dataloader:
                # Add rotation noise
                batch_noisy, V_noisy, _ = add_rotation_noise(batch, eps=0.1, skew=skew)
                batch_noisy = batch_noisy.float()
                batch = batch.float()
                
                # Use only top k eigenvectors
                V_input = V_noisy[:, :, :k].float()
                
                # Forward pass
                _, attention_scores = model(V_input)
                A_pred = attention_scores[-1]  # Last layer's attention
                
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
        assert avg_loss < initial_loss
    
    def test_hybrid_training_pipeline(self, training_data):
        """Test hybrid model training end-to-end."""
        dataloader, adjacency_matrices = training_data
        
        # Create hybrid model
        gnn_config = {
            "num_layers": 1,
            "num_terms": 2,
            "feature_dim_in": 10,
            "feature_dim_out": 5
        }
        
        transformer_config = {
            "num_heads": 2,
            "num_layers": 1,
            "d_k": 5,
            "d_v": 5
        }
        
        model = create_sequential_model(gnn_config, transformer_config)
        model = model.float()
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        initial_loss = None
        model.train()
        
        for epoch in range(5):
            epoch_loss = 0.0
            for batch in dataloader:
                # Add Gaussian noise
                batch_noisy, _, _ = add_gaussian_noise(batch, eps=0.1)
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
        assert avg_loss < initial_loss
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_batch = adjacency_matrices[0].unsqueeze(0)
            test_noisy, _, _ = add_gaussian_noise(test_batch, eps=0.1)
            
            A_pred = model(test_noisy.float())
            
            assert A_pred.shape == test_batch.shape
            assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)
    
    def test_multi_noise_level_training(self, training_data):
        """Test training with multiple noise levels."""
        dataloader, _ = training_data
        
        # Initialize simple GNN
        model = GNNSymmetric(num_layers=1, feature_dim_in=10, feature_dim_out=5)
        model = model.float()
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Define noise levels
        noise_levels = [0.05, 0.1, 0.2, 0.3]
        
        model.train()
        for epoch in range(3):
            for batch in dataloader:
                # Sample random noise level
                eps = np.random.choice(noise_levels)
                
                # Add noise
                batch_noisy, _, _ = add_digress_noise(batch, p=eps)
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
        model.eval()
        with torch.no_grad():
            test_batch = dataloader.dataset[0].unsqueeze(0)
            
            for eps in noise_levels:
                test_noisy, _, _ = add_digress_noise(test_batch, p=eps)
                A_pred, _ = model(test_noisy.float())
                
                assert A_pred.shape == test_batch.shape
                assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)
    
    def test_learning_rate_scheduling(self, training_data):
        """Test training with learning rate scheduling."""
        dataloader, _ = training_data
        
        # Initialize model
        model = NodeVarGNN(num_layers=1, num_terms=2, feature_dim=10)
        model = model.float()
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # Add cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )
        
        # Track learning rates
        lrs = []
        
        model.train()
        for epoch in range(10):
            for batch in dataloader:
                batch_noisy, _, _ = add_gaussian_noise(batch, eps=0.1)
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
            lrs.append(optimizer.param_groups[0]['lr'])
        
        # Check that learning rate changed
        assert len(set(lrs)) > 1
        assert min(lrs) < 0.1  # LR decreased at some point
    
    def test_gradient_flow(self, training_data):
        """Test that gradients flow properly through all model types."""
        dataloader, _ = training_data
        
        models = [
            GNN(num_layers=2, feature_dim_out=5),
            create_sequential_model(
                {"num_layers": 1, "feature_dim_out": 5},
                {"num_heads": 2, "num_layers": 1}
            )
        ]
        
        for model in models:
            model = model.float()
            
            # Get one batch
            batch = next(iter(dataloader))
            batch_noisy, _, _ = add_digress_noise(batch, p=0.1)
            batch_noisy = batch_noisy.float()
            batch = batch.float()
            
            # Forward pass
            if isinstance(model, GNN):
                X, Y = model(batch_noisy)
                output = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))
            else:
                output = model(batch_noisy)
            
            # Compute loss
            loss = torch.mean((output - batch) ** 2)
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist and are non-zero
            # Note: Some parameters like score_combination.weight might not have gradients
            # if they're not used in the loss computation (e.g., attention scores that are discarded)
            params_with_grad = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    params_with_grad += 1
                    assert not torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-10), \
                        f"Zero gradient for {name}"
            
            # Ensure at least some parameters have gradients
            assert params_with_grad > 0, "No parameters have gradients"


if __name__ == "__main__":
    pytest.main([__file__])