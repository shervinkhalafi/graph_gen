"""Logging utilities for experiment tracking across different backends."""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
from loguru import logger as loguru
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger

if TYPE_CHECKING:
    from rich.console import Console


def setup_rich_logging(console: "Console") -> None:
    """Configure loguru to output through Rich console for progress bar compatibility.

    When using Rich's Progress bar (a live display), any output that bypasses
    Rich's console corrupts the progress bar rendering. This function redirects
    loguru to write through the Rich console, which coordinates with live displays.

    Parameters
    ----------
    console
        The Rich Console instance (typically from `rich.get_console()`).
    """
    loguru.remove()  # Remove default stderr handler
    loguru.add(
        lambda msg: console.print(msg, end="", highlight=False, markup=False),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        colorize=True,
    )


def _is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URL."""
    return path.startswith("s3://")


def _has_s3_credentials() -> bool:
    """Check if all required S3/Tigris credentials are set."""
    return all([
        os.environ.get("TMGG_S3_BUCKET"),
        os.environ.get("AWS_S3_ENDPOINT"),
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
    ])


def _has_wandb_credentials() -> bool:
    """Check if WandB API key is set."""
    return bool(os.environ.get("WANDB_API_KEY"))


def _setup_s3_env_for_fsspec() -> None:
    """Configure fsspec/s3fs to use custom S3 endpoint from AWS_S3_ENDPOINT.

    s3fs reads credentials from AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY but
    does NOT read endpoint_url from any env var. We use fsspec's config system
    to set the endpoint globally.
    """
    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    if endpoint:
        import fsspec
        fsspec.config.conf["s3"] = {"endpoint_url": endpoint}

    try:
        import s3fs
        s3fs.S3FileSystem.clear_instance_cache()
    except (ImportError, AttributeError):
        pass


def create_loggers(config: DictConfig) -> List[Logger]:
    """
    Create PyTorch Lightning loggers based on configuration.

    Args:
        config: Hydra configuration containing logger settings

    Returns:
        List of configured PyTorch Lightning loggers
    """
    # Bridge TMGG_S3_* env vars to fsspec-compatible format
    _setup_s3_env_for_fsspec()

    loggers = []

    # Check for legacy wandb configuration (backward compatibility)
    if "wandb" in config and config.wandb is not None:
        if not _has_wandb_credentials():
            loguru.warning("Skipping legacy WandB logger: WANDB_API_KEY not set")
        else:
            loguru.info("Using wandb logger")
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
            if isinstance(logger_config, dict) or hasattr(logger_config, "items"):
                # Get the logger type (first key in the dict)
                logger_type = list(logger_config.keys())[0]
                logger_params = logger_config[logger_type]

                if logger_type == "tensorboard":
                    # TensorBoard supports S3 paths via fsspec
                    save_dir = logger_params.get(
                        "save_dir", os.path.join(config.paths.output_dir, "tensorboard")
                    )

                    # Skip S3-based TensorBoard if credentials are missing
                    if _is_s3_path(save_dir) and not _has_s3_credentials():
                        loguru.warning(
                            "Skipping TensorBoard S3 logger: S3 credentials not set"
                        )
                        continue

                    loguru.info("Using {logger_type} logger", logger_type=logger_type)
                    loguru.info("Saving to {save_dir}", save_dir=save_dir)
                    tb_logger = TensorBoardLogger(
                        save_dir=save_dir,
                        name=logger_params.get(
                            "name", config.get("experiment_name", "default")
                        ),
                        version=logger_params.get("version", None),
                        log_graph=logger_params.get("log_graph", False),
                        default_hp_metric=logger_params.get("default_hp_metric", False),
                    )
                    loggers.append(tb_logger)

                elif logger_type == "wandb":
                    # Skip WandB if credentials are missing
                    if not _has_wandb_credentials():
                        loguru.warning("Skipping WandB logger: WANDB_API_KEY not set")
                        continue

                    loguru.info("Using {logger_type} logger", logger_type=logger_type)
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
                    save_dir = logger_params.get(
                        "save_dir", os.path.join(config.paths.output_dir, "csv")
                    )
                    csv_logger = CSVLogger(
                        save_dir=save_dir,
                        name=logger_params.get(
                            "name", config.get("experiment_name", "default")
                        ),
                        version=logger_params.get("version", None),
                        flush_logs_every_n_steps=logger_params.get(
                            "flush_logs_every_n_steps", 100
                        ),
                    )
                    loggers.append(csv_logger)

    # Default to TensorBoard if no loggers configured
    if not loggers:
        loguru.info("Using {logger_type} logger", logger_type="tensorboard")
        default_save_dir = os.path.join(config.paths.output_dir, "tensorboard")
        loguru.info("Saving to to {save_dir}", save_dir=default_save_dir)
        tb_logger = TensorBoardLogger(
            save_dir=default_save_dir,
            name=config.get("experiment_name", "default"),
        )
        loggers.append(tb_logger)

    # Auto-inject TensorBoard S3 logger if credentials available and not already configured
    if _has_s3_credentials():
        has_s3_tensorboard = any(
            isinstance(lg, TensorBoardLogger) and _is_s3_path(lg.save_dir)
            for lg in loggers
        )
        if not has_s3_tensorboard:
            bucket = os.environ["TMGG_S3_BUCKET"]
            prefix = os.environ.get("TMGG_S3_PREFIX", "")
            experiment_name = config.get("experiment_name", "default")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Write TensorBoard logs locally, then sync to S3 at end of training
            # Direct S3 writes are too slow (synchronous network I/O on every log)
            local_tb_dir = Path(config.paths.output_dir) / "tensorboard_s3"
            local_tb_dir.mkdir(parents=True, exist_ok=True)
            s3_path = f"s3://{bucket}/{prefix}tensorboard/{experiment_name}/{timestamp}"

            # Store S3 path for later sync (accessible via logger.save_dir metadata)
            loguru.info(
                "TensorBoard S3 logger: writing to {local_dir}, will sync to {s3_path} on close",
                local_dir=local_tb_dir,
                s3_path=s3_path,
            )

            auto_tb_logger = TensorBoardLogger(
                save_dir=str(local_tb_dir),
                name=experiment_name,
                version=None,
                log_graph=False,
                default_hp_metric=False,
                flush_secs=120,
            )
            # Store S3 destination for sync_tensorboard_to_s3()
            auto_tb_logger._s3_sync_path = s3_path
            loggers.append(auto_tb_logger)

    return loggers


def sync_tensorboard_to_s3(loggers: List[Logger]) -> None:
    """Sync any TensorBoard loggers with _s3_sync_path to S3.

    Called at end of training to upload local TensorBoard logs to S3.
    This avoids slow synchronous S3 writes during training.
    """
    import fsspec

    for logger in loggers:
        if isinstance(logger, TensorBoardLogger) and hasattr(logger, "_s3_sync_path"):
            s3_path = logger._s3_sync_path
            local_path = Path(logger.log_dir)

            if not local_path.exists():
                loguru.warning(f"TensorBoard log dir not found: {local_path}")
                continue

            loguru.info(f"Syncing TensorBoard logs to S3: {local_path} -> {s3_path}")
            try:
                fs = fsspec.filesystem("s3")
                # Upload files individually (fs.put recursive has bugs with some backends)
                uploaded = 0
                for local_file in local_path.rglob("*"):
                    if local_file.is_file():
                        rel_path = local_file.relative_to(local_path)
                        s3_file = f"{s3_path}/{rel_path}"
                        fs.put_file(str(local_file), s3_file)
                        uploaded += 1
                loguru.info(f"TensorBoard S3 sync complete: {uploaded} files -> {s3_path}")
            except Exception as e:
                loguru.error(f"Failed to sync TensorBoard to S3: {e}")


def log_figure(
    loggers: Union[Logger, list[Logger]],
    tag: str,
    figure: matplotlib.figure.Figure,
    global_step: Optional[int] = None,
    close: bool = True,
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
        if isinstance(logger, TensorBoardLogger):
            # TensorBoard uses add_figure
            loguru.info("Saving figure to tensor_board")
            logger.experiment.add_figure(tag, figure, global_step=global_step)

        elif isinstance(logger, WandbLogger):
            # Wandb uses Image wrapper
            import wandb

            loguru.info("Saving figure to wandb")
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
            loguru.info(f"Saving figure to {save_path=}")
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
    # Close figure to prevent memory leaks
    if close:
        plt.close(figure)


def log_figures(
    loggers: Union[Logger, List[Logger]],
    figures: Dict[str, matplotlib.figure.Figure],
    global_step: Optional[int] = None,
    close: bool = True,
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


def log_metrics(
    logger: Union[Logger, List[Logger]],
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """
    Log scalar metrics to any PyTorch Lightning logger with consistent API.

    This function provides a unified interface for logging metrics across
    different logger types (TensorBoard, W&B, CSV, etc.), handling the
    different APIs transparently.

    Args:
        logger: Single logger or list of loggers
        metrics: Dictionary mapping metric names to scalar values
        step: Optional global step (required for TensorBoard, optional for others)

    Example:
        ```python
        # Log a single metric
        log_metrics(logger, {"loss": 0.5}, step=100)

        # Log multiple metrics at once
        metrics = {
            "train_loss": 0.5,
            "val_loss": 0.6,
            "accuracy": 0.92
        }
        log_metrics(logger, metrics, step=trainer.global_step)
        ```
    """
    if logger is None:
        return

    # Ensure logger is a list
    if not isinstance(logger, list):
        loggers = [logger]
    else:
        loggers = logger

    # Filter out None loggers
    loggers = [lg for lg in loggers if lg is not None]

    for lg in loggers:
        try:
            if isinstance(lg, TensorBoardLogger):
                # TensorBoard logs scalars one at a time
                for key, value in metrics.items():
                    lg.experiment.add_scalar(key, value, global_step=step or 0)

            elif isinstance(lg, WandbLogger):
                # W&B can log all metrics at once
                lg.experiment.log(metrics)

            elif isinstance(lg, CSVLogger):
                # Write metrics directly to a CSV file in the log directory
                import csv

                csv_dir = Path(lg.log_dir)
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_path = csv_dir / "manual_metrics.csv"

                # Check if file exists to determine if we need headers
                file_exists = csv_path.exists()

                # Prepare row with optional step column
                row_data = {"step": step} if step is not None else {}
                row_data.update(metrics)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row_data)

            # Add other logger types as needed
            # elif isinstance(lg, MLFlowLogger):
            #     for key, value in metrics.items():
            #         lg.experiment.log_metric(key, value, step=step)

        except Exception as e:
            # Log errors but don't fail the entire logging process
            print(f"Failed to log metrics to {type(lg).__name__}: {str(e)}")
