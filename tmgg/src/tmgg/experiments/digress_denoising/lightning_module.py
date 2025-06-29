"""PyTorch Lightning module for Digress GraphTransformer-based denoising."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List

from tmgg.models.digress.transformer_model import GraphTransformer
from tmgg.experiment_utils import (
    compute_batch_metrics,
    create_wandb_visualization,
    add_digress_noise,
)


class DigressDenoisingLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Digress GraphTransformer-based graph denoising."""

    def __init__(
        self,
        use_eigenvectors: bool = False,
        node_feature_dim: int = 20,
        n_layers: int = 4,
        hidden_mlp_dims: Optional[Dict[str, int]] = None,
        hidden_dims: Optional[Dict[str, int]] = None,
        output_dims: Optional[Dict[str, int]] = None,
        learning_rate: float = 0.001,
        loss_type: str = "MSE",
        scheduler_config: Optional[Dict[str, Any]] = None,
        noise_levels: Optional[List[float]] = None,
        noise_type: str = "Digress",
    ):
        super().__init__()
        self.save_hyperparameters()

        input_dims = {
            "X": self.hparams.node_feature_dim if self.hparams.use_eigenvectors else 1,
            "E": 1,
            "y": 0,
        }

        self.model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
        )

        if loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "BCE":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2, 0.3]
        self.noise_type = noise_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        eps = self.noise_levels[torch.randint(len(self.noise_levels), (1,)).item()]
        batch_noisy, V_noisy, _ = add_digress_noise(batch, eps)

        model_input = V_noisy if self.hparams.use_eigenvectors else batch_noisy
        output = self.forward(model_input)

        loss = self.criterion(output, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_noise_level", eps, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        eps = 0.2
        batch_noisy, V_noisy, _ = add_digress_noise(batch, eps)

        model_input = V_noisy if self.hparams.use_eigenvectors else batch_noisy
        output = self.forward(model_input)

        val_loss = self.criterion(output, batch)
        batch_metrics = compute_batch_metrics(batch, output)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, value in batch_metrics.items():
            self.log(f"val_{metric_name}", value, on_step=False, on_epoch=True)

        return {"val_loss": val_loss, **batch_metrics}

    def test_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        metrics_by_noise = {}
        for eps in self.noise_levels:
            batch_noisy, V_noisy, _ = add_digress_noise(batch, eps)
            model_input = V_noisy if self.hparams.use_eigenvectors else batch_noisy
            output = self.forward(model_input)

            batch_metrics = compute_batch_metrics(batch, output)
            for metric_name, value in batch_metrics.items():
                key = f"test_{metric_name}_eps_{eps}"
                metrics_by_noise[key] = value
                self.log(key, value, on_step=False, on_epoch=True)

        return metrics_by_noise

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch > 0 and self.current_epoch % 10 == 0:
            self._log_visualizations("val")

    def on_test_epoch_end(self) -> None:
        self._log_visualizations("test")

    def _log_visualizations(self, stage: str) -> None:
        if not isinstance(self.logger, pl.loggers.WandbLogger):
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
                A_sample,
                self,
                add_digress_noise,
                self.noise_levels,
                stage,
                self.device,
            )
            self.logger.experiment.log(wandb_images)
        except Exception as e:
            self.logger.experiment.log({f"{stage}_viz_error_msg": str(e)})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.scheduler_config is None:
            return optimizer

        scheduler_type = self.hparams.scheduler_config.get("type", "cosine")
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.scheduler_config.get("T_0", 20),
                T_mult=self.hparams.scheduler_config.get("T_mult", 2),
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.scheduler_config.get("step_size", 50),
                gamma=self.hparams.scheduler_config.get("gamma", 0.1),
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_model_config(self) -> Dict[str, Any]:
        return self.model.get_config()
