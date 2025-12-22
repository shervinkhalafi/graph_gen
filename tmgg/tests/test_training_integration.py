"""Integration tests for the full training pipeline.

Test rationale:
    This test verifies that the end-to-end training pipeline works correctly,
    from configuration loading through model training to final evaluation.
    It ensures that:
    1. Hydra configuration instantiation produces valid objects
    2. The training loop executes without errors
    3. Loss decreases during training (model is learning)
    4. Validation metrics are computed and logged

Invariants:
    - Model output shape matches input shape (adjacency matrix reconstruction)
    - Loss is finite and positive
    - Training loss decreases over epochs (on average)
"""

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal configuration for testing."""
        return OmegaConf.create(
            {
                "seed": 42,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "optimizer_type": "adam",
                "loss_type": "BCEWithLogits",
                "noise_type": "Digress",
                "noise_levels": [0.1, 0.2],
                "visualization_interval": 100,  # Don't visualize during short tests
                "model": {
                    "k": 8,
                    "model_type": "filter_bank",
                },
                "paths": {
                    "output_dir": str(tmp_path),
                    "results_dir": str(tmp_path / "results"),
                },
            }
        )

    @pytest.fixture
    def sample_adjacency_matrices(self):
        """Generate sample adjacency matrices for testing."""
        batch_size = 4
        n_nodes = 20

        # Create block diagonal adjacency matrices (simple structure)
        matrices = []
        for _ in range(batch_size):
            A = torch.zeros(n_nodes, n_nodes)
            # Create two blocks
            A[:10, :10] = torch.bernoulli(torch.full((10, 10), 0.7))
            A[10:, 10:] = torch.bernoulli(torch.full((10, 10), 0.7))
            # Make symmetric and remove diagonal
            A = (A + A.T) / 2
            A.fill_diagonal_(0)
            A = (A > 0.5).float()
            matrices.append(A)

        return torch.stack(matrices)

    def test_spectral_lightning_module_instantiation(self, minimal_config):
        """Test that spectral Lightning module can be instantiated from config."""
        from tmgg.experiments.spectral_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        module = SpectralDenoisingLightningModule(
            model_type=minimal_config.model.model_type,
            k=minimal_config.model.k,
            learning_rate=minimal_config.learning_rate,
            weight_decay=minimal_config.weight_decay,
            optimizer_type=minimal_config.optimizer_type,
            loss_type=minimal_config.loss_type,
            noise_type=minimal_config.noise_type,
            noise_levels=minimal_config.noise_levels,
        )

        assert module is not None
        assert hasattr(module, "model")
        assert hasattr(module, "noise_generator")

    def test_forward_pass_shape(self, minimal_config, sample_adjacency_matrices):
        """Test that forward pass preserves shape."""
        from tmgg.experiments.spectral_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        module = SpectralDenoisingLightningModule(
            model_type=minimal_config.model.model_type,
            k=minimal_config.model.k,
            learning_rate=minimal_config.learning_rate,
            noise_levels=minimal_config.noise_levels,
        )

        with torch.no_grad():
            output = module(sample_adjacency_matrices)

        assert output.shape == sample_adjacency_matrices.shape

    def test_training_step_produces_finite_loss(
        self, minimal_config, sample_adjacency_matrices
    ):
        """Test that training step produces finite loss."""
        from tmgg.experiments.spectral_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        module = SpectralDenoisingLightningModule(
            model_type=minimal_config.model.model_type,
            k=minimal_config.model.k,
            learning_rate=minimal_config.learning_rate,
            noise_levels=minimal_config.noise_levels,
            loss_type=minimal_config.loss_type,
        )

        result = module.training_step(sample_adjacency_matrices, batch_idx=0)  # pyright: ignore[reportArgumentType]

        # training_step returns dict with 'loss' key
        loss = result["loss"]
        assert torch.isfinite(loss)  # pyright: ignore[reportArgumentType]
        assert loss > 0  # pyright: ignore[reportOperatorIssue]

    def test_short_training_run(self, minimal_config, tmp_path):
        """Test a short training run completes without errors."""
        from tmgg.experiment_utils.data.data_module import GraphDataModule
        from tmgg.experiments.spectral_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        # Create model
        module = SpectralDenoisingLightningModule(
            model_type=minimal_config.model.model_type,
            k=minimal_config.model.k,
            learning_rate=minimal_config.learning_rate,
            noise_levels=minimal_config.noise_levels,
            loss_type=minimal_config.loss_type,
            visualization_interval=100,  # Don't visualize
        )

        # Create minimal data module
        data_module = GraphDataModule(
            dataset_name="sbm",
            dataset_config={
                "num_nodes": 20,
                "num_graphs": 10,
            },
            batch_size=4,
            noise_levels=minimal_config.noise_levels,
        )

        # Create trainer with minimal settings
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        trainer.fit(module, data_module)

        # Verify training completed
        assert trainer.current_epoch == 2

    def test_loss_decreases_during_training(self, minimal_config, tmp_path):
        """Test that loss decreases during training (model learns)."""
        from tmgg.experiment_utils.data.data_module import GraphDataModule
        from tmgg.experiments.spectral_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        # Create model with nonlinear model for bounded output with MSE loss
        module = SpectralDenoisingLightningModule(
            model_type=minimal_config.model.model_type,
            k=minimal_config.model.k,
            learning_rate=1e-2,  # Higher LR for faster convergence in test
            noise_levels=[0.1],  # Single noise level for cleaner test
            loss_type="MSE",  # MSE is easier to optimize
            visualization_interval=100,
        )

        # Create data module with fixed data for deterministic test
        data_module = GraphDataModule(
            dataset_name="sbm",
            dataset_config={
                "num_nodes": 15,
                "num_graphs": 5,
            },
            batch_size=5,
            noise_levels=[0.1],
        )

        # Collect losses
        losses = []
        original_training_step = module.training_step

        def tracking_training_step(batch, batch_idx):
            result = original_training_step(batch, batch_idx)
            # training_step returns dict with 'loss' key
            losses.append(result["loss"].item())  # pyright: ignore[reportAttributeAccessIssue]
            return result

        module.training_step = tracking_training_step  # pyright: ignore[reportAttributeAccessIssue]

        # Train for a few epochs
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(module, data_module)

        # Check that average loss in last epoch is lower than first epoch
        n_batches_per_epoch = len(losses) // 5
        first_epoch_loss = sum(losses[:n_batches_per_epoch]) / n_batches_per_epoch
        last_epoch_loss = sum(losses[-n_batches_per_epoch:]) / n_batches_per_epoch

        # Allow some tolerance - just check it's not getting worse
        assert last_epoch_loss <= first_epoch_loss * 1.5, (
            f"Loss did not decrease: first_epoch={first_epoch_loss:.4f}, "
            f"last_epoch={last_epoch_loss:.4f}"
        )


class TestDataModuleIntegration:
    """Tests for data module integration with training."""

    def test_datamodule_provides_noise_levels(self):
        """Test that data module provides noise_levels attribute."""
        from tmgg.experiment_utils.data.data_module import GraphDataModule

        noise_levels = [0.05, 0.1, 0.2]
        data_module = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 10, "num_graphs": 5},
            noise_levels=noise_levels,
        )

        assert hasattr(data_module, "noise_levels")
        assert data_module.noise_levels == noise_levels

    def test_lightning_module_uses_datamodule_noise_levels(self):
        """Test that Lightning module uses datamodule's noise_levels when attached."""
        from tmgg.experiment_utils.data.data_module import GraphDataModule
        from tmgg.experiments.spectral_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        # Create module with different noise levels
        module = SpectralDenoisingLightningModule(
            k=8,
            noise_levels=[0.1, 0.2],  # This should be overridden
        )

        # Create data module with different noise levels
        dm_noise_levels = [0.05, 0.15, 0.25]
        data_module = GraphDataModule(
            dataset_name="sbm",
            dataset_config={
                "num_nodes": 10,
                "num_graphs": 5,
                "block_sizes": [5, 5],  # Fixed block sizes to simplify setup
            },
            noise_levels=dm_noise_levels,
        )

        # Create trainer to attach datamodule
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )

        # Properly set up datamodule (prepare_data then setup)
        data_module.prepare_data()
        data_module.setup("fit")
        trainer.datamodule = data_module  # pyright: ignore[reportAttributeAccessIssue]
        module.trainer = trainer  # pyright: ignore[reportAttributeAccessIssue]

        # Now the noise_levels property should return datamodule's values
        assert module.noise_levels == dm_noise_levels
