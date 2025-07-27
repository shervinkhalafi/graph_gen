"""Logging utilities for experiment tracking across different backends."""

from typing import List, Optional, Union, Dict, Any
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    WandbLogger,
    CSVLogger,
    Logger
)
from omegaconf import DictConfig, ListConfig
import matplotlib.pyplot as plt
import matplotlib.figure


def create_loggers(config: DictConfig) -> List[Logger]:
    """
    Create PyTorch Lightning loggers based on configuration.
    
    Args:
        config: Hydra configuration containing logger settings
        
    Returns:
        List of configured PyTorch Lightning loggers
    """
    loggers = []
    
    # Check for legacy wandb configuration (backward compatibility)
    if "wandb" in config and config.wandb is not None:
        wandb_logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.get("tags", []),
            save_dir=config.paths.output_dir,
        )
        loggers.append(wandb_logger)
    
    # Process new logger configuration
    if "logger" in config:
        logger_configs = config.logger
        
        # Handle single logger config or list of configs
        if not isinstance(logger_configs, (list, ListConfig)):
            logger_configs = [logger_configs]
            
        for logger_config in logger_configs:
            if isinstance(logger_config, dict) or hasattr(logger_config, 'items'):
                # Get the logger type (first key in the dict)
                logger_type = list(logger_config.keys())[0]
                logger_params = logger_config[logger_type]
                
                if logger_type == "tensorboard":
                    # TensorBoard supports S3 paths via fsspec
                    save_dir = logger_params.get("save_dir", 
                                               os.path.join(config.paths.output_dir, "tensorboard"))
                    tb_logger = TensorBoardLogger(
                        save_dir=save_dir,
                        name=logger_params.get("name", config.get("experiment_name", "default")),
                        version=logger_params.get("version", None),
                        log_graph=logger_params.get("log_graph", False),
                        default_hp_metric=logger_params.get("default_hp_metric", False),
                    )
                    loggers.append(tb_logger)
                    
                elif logger_type == "wandb":
                    wandb_logger = WandbLogger(
                        project=logger_params.get("project"),
                        name=logger_params.get("name"),
                        tags=logger_params.get("tags", []),
                        save_dir=logger_params.get("save_dir", config.paths.output_dir),
                        entity=logger_params.get("entity", None),
                        log_model=logger_params.get("log_model", False),
                    )
                    loggers.append(wandb_logger)
                    
                elif logger_type == "csv":
                    save_dir = logger_params.get("save_dir", 
                                               os.path.join(config.paths.output_dir, "csv"))
                    csv_logger = CSVLogger(
                        save_dir=save_dir,
                        name=logger_params.get("name", config.get("experiment_name", "default")),
                        version=logger_params.get("version", None),
                        flush_logs_every_n_steps=logger_params.get("flush_logs_every_n_steps", 100),
                    )
                    loggers.append(csv_logger)
    
    # Default to TensorBoard if no loggers configured
    if not loggers:
        default_save_dir = os.path.join(config.paths.output_dir, "tensorboard")
        tb_logger = TensorBoardLogger(
            save_dir=default_save_dir,
            name=config.get("experiment_name", "default"),
        )
        loggers.append(tb_logger)
    
    return loggers


def log_figure(
    loggers: Union[Logger, List[Logger]],
    tag: str,
    figure: matplotlib.figure.Figure,
    global_step: Optional[int] = None,
    close: bool = True
) -> None:
    """
    Log a matplotlib figure to all configured loggers.
    
    Args:
        loggers: Single logger or list of loggers
        tag: Tag/name for the figure
        figure: Matplotlib figure to log
        global_step: Global step for logging (optional)
        close: Whether to close the figure after logging (default: True)
    """
    # Ensure loggers is a list
    if not isinstance(loggers, list):
        loggers = [loggers]
    
    # Handle the case where logger might be None
    loggers = [logger for logger in loggers if logger is not None]
    
    for logger in loggers:
        try:
            if isinstance(logger, TensorBoardLogger):
                # TensorBoard uses add_figure
                logger.experiment.add_figure(tag, figure, global_step=global_step)
                
            elif isinstance(logger, WandbLogger):
                # Wandb uses Image wrapper
                import wandb
                logger.experiment.log({tag: wandb.Image(figure)})
                
            elif isinstance(logger, CSVLogger):
                # CSV logger saves figures to disk
                save_dir = Path(logger.log_dir) / "figures"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename with optional step
                if global_step is not None:
                    filename = f"{tag}_step_{global_step}.png"
                else:
                    filename = f"{tag}.png"
                
                # Save figure
                save_path = save_dir / filename
                figure.savefig(save_path, dpi=300, bbox_inches='tight')
                
        except Exception as e:
            # Log errors but don't fail the entire logging process
            print(f"Failed to log figure '{tag}' to {type(logger).__name__}: {str(e)}")
    
    # Close figure to prevent memory leaks
    if close:
        plt.close(figure)


def log_figures(
    loggers: Union[Logger, List[Logger]],
    figures: Dict[str, matplotlib.figure.Figure],
    global_step: Optional[int] = None,
    close: bool = True
) -> None:
    """
    Log multiple matplotlib figures to all configured loggers.
    
    Args:
        loggers: Single logger or list of loggers
        figures: Dictionary mapping tags to matplotlib figures
        global_step: Global step for logging (optional)
        close: Whether to close figures after logging (default: True)
    """
    for tag, figure in figures.items():
        log_figure(loggers, tag, figure, global_step=global_step, close=close)