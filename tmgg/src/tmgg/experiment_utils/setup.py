from pathlib import Path
import random
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger
import torch
import pytorch_lightning as pl

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

    # Enable Tensor Core optimization for GPUs that support it
    torch.set_float32_matmul_precision("high")


def create_callbacks(config: DictConfig) -> list[pl.Callback]:
    """Create PyTorch Lightning callbacks from config.

    Reads callback parameters from config.callbacks if available,
    otherwise uses sensible defaults. All step-based, no epoch references.
    """
    callbacks = []

    # Get callback config with defaults
    cb_config = config.get("callbacks", {})
    ckpt_config = cb_config.get("checkpoint", {})
    es_config = cb_config.get("early_stopping", {})

    # Model checkpointing (step-based filename)
    # auto_insert_metric_name=False required because metric name val/loss contains a slash
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.paths.output_dir) / "checkpoints",
        filename=ckpt_config.get("filename", "model-step={step:06d}-val_loss={val/loss:.4f}"),
        monitor=ckpt_config.get("monitor", "val/loss"),
        mode=ckpt_config.get("mode", "min"),
        save_top_k=ckpt_config.get("save_top_k", 3),
        save_last=ckpt_config.get("save_last", True),
        every_n_train_steps=ckpt_config.get("every_n_train_steps", None),
        auto_insert_metric_name=ckpt_config.get("auto_insert_metric_name", False),
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping (patience = validation checks, not epochs)
    early_stopping = EarlyStopping(
        monitor=es_config.get("monitor", "val/loss"),
        mode=es_config.get("mode", "min"),
        patience=es_config.get("patience", 10),
        min_delta=es_config.get("min_delta", 1e-4),
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Learning rate monitoring (step-based)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Progress bar (step-based Rich)
    # Note: trainer.enable_progress_bar must be false to avoid conflict with Lightning's default
    from tmgg.experiment_utils.progress import StepProgressBar

    pb_config = config.get("progress_bar", {})
    callbacks.append(
        StepProgressBar(
            metrics_to_show=pb_config.get("metrics_to_show", ["train_loss", "val/loss"]),
            show_epoch=pb_config.get("show_epoch", True),
            metrics_format=pb_config.get("metrics_format", ".4f"),
        )
    )

    return callbacks
