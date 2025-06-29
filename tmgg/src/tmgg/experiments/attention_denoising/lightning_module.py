"""PyTorch Lightning module for attention-based denoising."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import wandb
import numpy as np

from tmgg.models.attention import MultiLayerAttention
from tmgg.experiment_utils import (
    compute_reconstruction_metrics,
    compute_batch_metrics,
    create_wandb_visualization,
    create_graph_denoising_wandb_image,
    add_gaussian_noise,
    add_rotation_noise,
    add_digress_noise,
)


class AttentionDenoisingLightningModule(pl.LightningModule):
    """PyTorch Lightning module for attention-based graph denoising."""
    
    def __init__(self, 
                 d_model: int = 20,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 dropout: float = 0.0,
                 bias: bool = True,
                 learning_rate: float = 0.001,
                 loss_type: str = "MSE",
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 noise_levels: Optional[List[float]] = None,
                 noise_type: str = "Digress"):
        """
        Initialize the Lightning module.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            d_k: Key dimension (default: d_model // num_heads)
            d_v: Value dimension (default: d_model // num_heads)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            learning_rate: Learning rate for optimizer
            loss_type: Loss function type ("MSE" or "BCE")
            scheduler_config: Optional scheduler configuration
            noise_levels: List of noise levels to sample from during training
            noise_type: Type of noise to use ("Gaussian", "Rotation", "Digress")
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = MultiLayerAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            bias=bias
        )
        
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
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2, 0.3]
        self.noise_type = noise_type
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention model.
        
        Args:
            x: Input tensor (eigenvectors)
            
        Returns:
            Attention scores from the last layer
        """
        _, attention_scores = self.model(x)
        return attention_scores[-1]  # Return attention scores from last layer
    
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
        
        # Get the appropriate noise function
        noise_fn_map = {
            'Gaussian': add_gaussian_noise,
            'Rotation': add_rotation_noise,
            'Digress': add_digress_noise,
        }
        noise_fn = noise_fn_map.get(self.noise_type, add_digress_noise)
        
        # Add noise
        batch_noisy, V_noisy, _ = noise_fn(batch, eps)
        
        # Forward pass using noisy eigenvectors
        output = self.forward(V_noisy)
        
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
        
        # Get the appropriate noise function
        noise_fn_map = {
            'Gaussian': add_gaussian_noise,
            'Rotation': add_rotation_noise,
            'Digress': add_digress_noise,
        }
        noise_fn = noise_fn_map.get(self.noise_type, add_digress_noise)
        
        # Add noise
        batch_noisy, V_noisy, _ = noise_fn(batch, eps)
        
        # Forward pass
        output = self.forward(V_noisy)
        
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
        
        # Get the appropriate noise function
        noise_fn_map = {
            'Gaussian': add_gaussian_noise,
            'Rotation': add_rotation_noise,
            'Digress': add_digress_noise,
        }
        noise_fn = noise_fn_map.get(self.noise_type, add_digress_noise)
        
        # Test across multiple noise levels
        for eps in self.noise_levels:
            batch_noisy, V_noisy, _ = noise_fn(batch, eps)
            output = self.forward(V_noisy)
            
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
            # Use the new flexible visualization
            noise_levels = self.trainer.datamodule.noise_levels
            noise_type = getattr(self.trainer.datamodule, 'noise_type', 'Digress')
            
            # Get the appropriate noise function
            noise_fn_map = {
                'Gaussian': add_gaussian_noise,
                'Rotation': add_rotation_noise,
                'Digress': add_digress_noise,
            }
            noise_fn = noise_fn_map.get(noise_type, add_digress_noise)
            
            # Create denoise function that properly handles the model output
            def denoise_fn(A_noisy):
                with torch.no_grad():
                    self.eval()
                    A_input = A_noisy.to(self.device)
                    if A_input.ndim == 2:
                        A_input = A_input.unsqueeze(0)
                    output = self.forward(A_input)
                    return output.squeeze(0)
            
            # Log visualization for each noise level
            wandb_images = {}
            for eps in noise_levels[:3]:  # Limit to first 3 noise levels for space
                wandb_image = create_graph_denoising_wandb_image(
                    A_clean=A_sample,
                    noise_fn=noise_fn,
                    denoise_fn=denoise_fn,
                    noise_level=eps,
                    noise_type=noise_type,
                    title_prefix="Attention - "
                )
                wandb_images[f"{stage}_denoising_eps_{eps}"] = wandb_image
            
            self.logger.experiment.log(wandb_images)
            
            # Also use the legacy multi-noise visualization if desired
            legacy_images = create_wandb_visualization(
                A_sample, self, noise_fn, 
                noise_levels, stage, self.device
            )
            self.logger.experiment.log(legacy_images)
        except Exception as e:
            # Use a different key pattern to avoid wandb type confusion
            # Also provide more context about the error
            import traceback
            error_msg = f"Visualization failed: {str(e)}\nTraceback: {traceback.format_exc()}"
            self.logger.experiment.log({f"{stage}_viz_error_msg": error_msg})
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.scheduler_config is None:
            return optimizer
        
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
        return self.model.get_config()
