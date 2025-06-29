"""Main runner for attention-based denoising experiments."""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import wandb

from .lightning_module import AttentionDenoisingLightningModule
from tmgg.experiment_utils.data.data_module import GraphDataModule


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def create_callbacks(config: DictConfig) -> list:
    """Create PyTorch Lightning callbacks."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.paths.output_dir) / "checkpoints",
        filename="attention-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        verbose=True,
        min_delta=1e-4,
    )
    callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


def create_logger(config: DictConfig) -> Optional[pl.loggers.Logger]:
    """Create logger for experiment tracking."""
    if "wandb" in config:
        logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            save_dir=config.paths.output_dir,
        )
        return logger
    return None


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "config")


@hydra.main(
    version_base="1.3", config_path=CONFIG_PATH, config_name="experiment/default"
)
def main(config: DictConfig) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        config: Hydra configuration

    Returns:
        Dictionary with training results
    """
    # Set random seed
    set_seed(config.seed)

    # Create output directories
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(Path(config.paths.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Initialize data module
    data_module = hydra.utils.instantiate(config.data)

    # Initialize model
    model = hydra.utils.instantiate(config.model)

    # Create callbacks and logger
    callbacks = create_callbacks(config)
    logger = create_logger(config)

    # Initialize trainer
    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Log hyperparameters
    if logger:
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)

    # Get best model path
    best_model_path = trainer.checkpoint_callback.best_model_path

    # Load best model for final evaluation
    if best_model_path:
        best_model = AttentionDenoisingLightningModule.load_from_checkpoint(
            best_model_path
        )

        # Perform final evaluation across noise levels
        noise_levels = data_module.noise_levels
        final_results = evaluate_across_noise_levels(
            best_model, data_module, noise_levels
        )

        # Log final results
        if logger:
            for metric_name, values in final_results.items():
                for i, eps in enumerate(noise_levels):
                    logger.experiment.log({f"final_{metric_name}_eps_{eps}": values[i]})

    # Cleanup
    if logger and isinstance(logger, WandbLogger):
        wandb.finish()

    return {
        "best_model_path": best_model_path,
        "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
    }


def evaluate_across_noise_levels(
    model: AttentionDenoisingLightningModule,
    data_module: GraphDataModule,
    noise_levels: list,
) -> Dict[str, list]:
    """
    Evaluate model across different noise levels.

    Args:
        model: Trained model
        data_module: Data module
        noise_levels: List of noise levels to evaluate

    Returns:
        Dictionary with evaluation results
    """
    from tmgg.experiment_utils import add_digress_noise, compute_reconstruction_metrics

    model.eval()
    results = {"mse": [], "eigenvalue_error": [], "subspace_distance": []}

    # Get a sample for evaluation
    sample_A = data_module.get_sample_adjacency_matrix("test")

    with torch.no_grad():
        for eps in noise_levels:
            # Add noise
            A_noisy, V_noisy, _ = add_digress_noise(sample_A, eps)

            # Predict
            if A_noisy.ndim == 2:
                A_noisy_input = A_noisy.unsqueeze(0)
                V_noisy_input = V_noisy.unsqueeze(0)
            else:
                A_noisy_input = A_noisy
                V_noisy_input = V_noisy

            # Move to model's device
            V_noisy_input = V_noisy_input.to(model.device)
            A_pred = model(V_noisy_input)

            # Compute metrics (move prediction back to CPU)
            metrics = compute_reconstruction_metrics(sample_A, A_pred.squeeze(0).cpu())

            results["mse"].append(metrics["mse"])
            results["eigenvalue_error"].append(metrics["eigenvalue_error"])
            results["subspace_distance"].append(metrics["subspace_distance"])

    return results


if __name__ == "__main__":
    main()
