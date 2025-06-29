"""Main runner for Digress Transformer-based denoising experiments."""

import random
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import os
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

from .lightning_module import DigressDenoisingLightningModule
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.paths.output_dir) / "checkpoints",
        filename="digress-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=20, verbose=True, min_delta=1e-4
    )
    callbacks.append(early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


def create_logger(config: DictConfig) -> Optional[pl.loggers.Logger]:
    """Create logger for experiment tracking."""
    if "wandb" in config:
        return WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            save_dir=config.paths.output_dir,
        )
    return None


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(THIS_DIR, "config")


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(config: DictConfig) -> Dict[str, Any]:
    """Main training function."""
    set_seed(config.seed)

    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(config.paths.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    data_module = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)

    callbacks = create_callbacks(config)
    logger = create_logger(config)

    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    if logger:
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    trainer.fit(model, data_module)
    trainer.test(model, data_module, ckpt_path="best")

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        best_model = DigressDenoisingLightningModule.load_from_checkpoint(
            best_model_path
        )
        noise_levels = data_module.noise_levels
        final_results = evaluate_across_noise_levels(
            best_model, data_module, noise_levels
        )
        if logger:
            for metric_name, values in final_results.items():
                for i, eps in enumerate(noise_levels):
                    logger.experiment.log({f"final_{metric_name}_eps_{eps}": values[i]})

    if logger and isinstance(logger, WandbLogger):
        wandb.finish()

    return {
        "best_model_path": best_model_path,
        "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
    }


def evaluate_across_noise_levels(
    model: DigressDenoisingLightningModule,
    data_module: GraphDataModule,
    noise_levels: list,
) -> Dict[str, list]:
    """Evaluate model across different noise levels."""
    from tmgg.experiment_utils import add_digress_noise, compute_reconstruction_metrics

    model.eval()
    results = {"mse": [], "eigenvalue_error": [], "subspace_distance": []}
    sample_A = data_module.get_sample_adjacency_matrix("test")

    with torch.no_grad():
        for eps in noise_levels:
            A_noisy, V_noisy, _ = add_digress_noise(sample_A, eps)
            model_input = V_noisy if model.hparams.use_eigenvectors else A_noisy

            if model_input.ndim == 2:
                model_input = model_input.unsqueeze(0)

            A_pred = model(model_input.to(model.device))

            metrics = compute_reconstruction_metrics(sample_A, A_pred.squeeze(0).cpu())
            results["mse"].append(metrics["mse"])
            results["eigenvalue_error"].append(metrics["eigenvalue_error"])
            results["subspace_distance"].append(metrics["subspace_distance"])

    return results


if __name__ == "__main__":
    main()
