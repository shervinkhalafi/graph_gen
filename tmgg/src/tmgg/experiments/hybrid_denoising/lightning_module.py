"""PyTorch Lightning module for hybrid GNN+Transformer denoising."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import wandb
import numpy as np

from tmgg.models.hybrid import SequentialDenoisingModel, create_sequential_model
from tmgg.experiment_utils import (
    compute_reconstruction_metrics,
    compute_batch_metrics,
    create_wandb_visualization,
    add_gaussian_noise,
    add_rotation_noise,
    add_digress_noise,
    create_noise_generator,
)


class HybridDenoisingLightningModule(pl.LightningModule):
    """PyTorch Lightning module for hybrid GNN+Transformer graph denoising."""
    
    def __init__(self, 
                 # GNN configuration
                 gnn_num_layers: int = 2,
                 gnn_num_terms: int = 2,
                 gnn_feature_dim_in: int = 20,
                 gnn_feature_dim_out: int = 5,
                 # Transformer configuration
                 use_transformer: bool = True,
                 transformer_num_layers: int = 4,
                 transformer_num_heads: int = 4,
                 transformer_d_k: Optional[int] = None,
                 transformer_d_v: Optional[int] = None,
                 transformer_dropout: float = 0.0,
                 transformer_bias: bool = True,
                 # Training configuration
                 learning_rate: float = 0.005,
                 loss_type: str = "BCE",
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 noise_levels: Optional[List[float]] = None,
                 noise_type: str = "Digress",
                 rotation_k: int = 20,
                 seed: Optional[int] = None):
        """
        Initialize the Lightning module.
        
        Args:
            gnn_num_layers: Number of GNN layers
            gnn_num_terms: Number of polynomial terms in GNN
            gnn_feature_dim_in: GNN input feature dimension
            gnn_feature_dim_out: GNN output feature dimension
            use_transformer: Whether to use transformer denoising
            transformer_num_layers: Number of transformer layers
            transformer_num_heads: Number of attention heads
            transformer_d_k: Key dimension for transformer
            transformer_d_v: Value dimension for transformer
            transformer_dropout: Transformer dropout rate
            transformer_bias: Whether to use bias in transformer
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration
            noise_levels: List of noise levels for evaluation
            noise_type: Type of noise to apply ("gaussian", "digress", "rotation")
            rotation_k: Dimension for rotation noise skew matrix
            seed: Random seed for reproducible noise generation
        """
        super().__init__()
        self.save_hyperparameters()
        
        # GNN configuration
        gnn_config = {
            "num_layers": gnn_num_layers,
            "num_terms": gnn_num_terms,
            "feature_dim_in": gnn_feature_dim_in,
            "feature_dim_out": gnn_feature_dim_out,
        }
        
        # Transformer configuration
        transformer_config = None
        if use_transformer:
            transformer_config = {
                "num_layers": transformer_num_layers,
                "num_heads": transformer_num_heads,
                "d_k": transformer_d_k,
                "d_v": transformer_d_v,
                "dropout": transformer_dropout,
                "bias": transformer_bias,
            }
        
        # Create hybrid model
        self.model = create_sequential_model(gnn_config, transformer_config)
        
        # Loss function
        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "BCE":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Store configuration
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config
        self.use_transformer = use_transformer
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2, 0.3]
        self.noise_type = noise_type
        
        # Create noise generator
        self.noise_generator = create_noise_generator(
            noise_type=noise_type,
            rotation_k=rotation_k,
            seed=seed
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input noisy adjacency matrix
            
        Returns:
            Reconstructed adjacency matrix
        """
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        # Sample random noise level for this batch
        eps = np.random.choice(self.noise_levels)
        
        # Add noise using noise generator
        batch_noisy, _, _ = self.noise_generator.add_noise(batch, eps)
        
        # Ensure double precision if needed
        batch_noisy = batch_noisy.double()
        batch = batch.double()
        
        # Forward pass using noisy adjacency matrix
        output = self.forward(batch_noisy)
        
        # Compute loss
        loss = self.criterion(output, batch)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_noise_level', eps, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index
            
        Returns:
            Dictionary of validation metrics
        """
        # Use fixed noise level for validation
        eps = 0.2
        
        # Add noise using noise generator
        batch_noisy, _, _ = self.noise_generator.add_noise(batch, eps)
        
        # Ensure double precision if needed
        batch_noisy = batch_noisy.double()
        batch = batch.double()
        
        # Forward pass
        output = self.forward(batch_noisy)
        
        # Compute loss
        val_loss = self.criterion(output, batch)
        
        # Compute reconstruction metrics
        batch_metrics = compute_batch_metrics(batch, output)
        
        # Log metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, value in batch_metrics.items():
            self.log(f'val_{metric_name}', value, on_step=False, on_epoch=True)
        
        return {'val_loss': val_loss, **batch_metrics}
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Batch of adjacency matrices
            batch_idx: Batch index
            
        Returns:
            Dictionary of test metrics
        """
        metrics_by_noise = {}
        
        # Test across multiple noise levels
        for eps in self.noise_levels:
            batch_noisy, _, _ = self.noise_generator.add_noise(batch, eps)
            
            # Ensure double precision if needed
            batch_noisy = batch_noisy.double()
            batch = batch.double()
            
            output = self.forward(batch_noisy)
            
            # Compute metrics for this noise level
            batch_metrics = compute_batch_metrics(batch, output)
            
            for metric_name, value in batch_metrics.items():
                key = f'test_{metric_name}_eps_{eps}'
                metrics_by_noise[key] = value
                self.log(key, value, on_step=False, on_epoch=True)
        
        return metrics_by_noise
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch for visualization."""
        if self.current_epoch % 10 == 0:  # Log visualizations every 10 epochs
            self._log_visualizations("val")
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch for visualization."""
        self._log_visualizations("test")
    
    def _log_visualizations(self, stage: str) -> None:
        """
        Log visualizations to wandb.
        
        Args:
            stage: Stage name ("val" or "test")
        """
        if not isinstance(self.logger, pl.loggers.WandbLogger) or self.trainer.sanity_checking:
            return

        # Get a sample from the appropriate dataloader
        if stage == "val":
            dataloader = self.trainer.datamodule.val_dataloader()
        elif stage == "test":
            dataloader = self.trainer.datamodule.test_dataloader()
        else:
            return  # Should not happen

        try:
            # Get a single batch and take the first sample for visualization
            batch = next(iter(dataloader))
            A_sample = batch[0]
        except StopIteration:
            # Dataloader might be empty
            return
        
        try:
            wandb_images = create_wandb_visualization(
                A_sample, self, self.noise_generator.add_noise, 
                self.noise_levels, stage, self.device
            )
            self.logger.experiment.log(wandb_images)
        except Exception as e:
            self.logger.experiment.log({f"{stage}_viz_error_msg": str(e)})
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.scheduler_config is None:
            # Default scheduler configuration
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        scheduler_type = self.scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_config.get('T_0', 20),
                T_mult=self.scheduler_config.get('T_mult', 2)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 50),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for logging."""
        config = self.model.get_config()
        config["use_transformer"] = self.use_transformer
        return config
