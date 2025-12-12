import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from loguru import logger as loguru
from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.final_eval import final_eval
from tmgg.experiment_utils.logging import create_loggers, sync_tensorboard_to_s3
from tmgg.experiment_utils.sanity_check import maybe_run_sanity_check
from tmgg.experiment_utils.setup import create_callbacks, set_seed


def _is_training_complete(checkpoint_path: Path, max_steps: int) -> bool:
    """Check if checkpoint has reached max_steps."""
    if not checkpoint_path.exists():
        return False
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    return ckpt.get("global_step", 0) >= max_steps


def run_experiment(config: DictConfig) -> dict[str, Any]:
    # Set random seed
    set_seed(config.seed)

    # Create output directories
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)

    # Check if experiment is already complete (skip W&B run creation if so)
    checkpoint_dir = Path(config.paths.output_dir) / "checkpoints"
    last_ckpt = checkpoint_dir / "last.ckpt"
    test_results_path = Path(config.paths.output_dir) / "test_results.json"

    training_complete = _is_training_complete(last_ckpt, config.trainer.max_steps)
    testing_complete = test_results_path.exists()

    if training_complete and testing_complete and not config.get("force_fresh", False):
        loguru.info(f"Skipping completed experiment: {config.paths.output_dir}")
        return {"skipped": True, "reason": "already_complete"}

    # Save configuration
    with open(Path(config.paths.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Initialize data module
    data_module = hydra.utils.instantiate(config.data)

    # Initialize model
    model = hydra.utils.instantiate(config.model)

    # Create callbacks and logger
    callbacks = create_callbacks(config)
    logger = create_loggers(config)

    # Initialize trainer
    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Run sanity check if enabled
    maybe_run_sanity_check(config=config, data_module=data_module, model=model)

    # Log hyperparameters
    # TODO: I think this might be redundant?
    if logger:
        # Handle multiple loggers
        if isinstance(logger, list):
            for lg in logger:
                lg.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        else:
            logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # Checkpoint resumption: look for last.ckpt unless force_fresh is set
    ckpt_path = None
    if config.get("force_fresh", False):
        loguru.info(f"force_fresh=True, starting fresh in: {config.paths.output_dir}")
    elif last_ckpt.exists():
        ckpt_path = str(last_ckpt)
        loguru.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        loguru.info(f"Starting fresh training in: {config.paths.output_dir}")

    # Train model
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Test model (skip if already tested)
    if testing_complete and not config.get("force_retest", False):
        loguru.info(f"Test results already exist, skipping: {test_results_path}")
    else:
        trainer.test(model, data_module)
        # Save test marker so we can skip on resume
        test_results_path.write_text(
            json.dumps({"tested_at": datetime.now().isoformat()})
        )

    # Get best model path
    best_model_path = trainer.checkpoint_callback.best_model_path

    # Load best model for final evaluation
    if best_model_path:
        # Ensure test data is set up (may have been skipped if test results existed)
        data_module.setup("test")
        eval_noise_levels = config.get("evaluation", {}).get("noise_levels", None)
        final_eval(model, data_module, logger, trainer, best_model_path, eval_noise_levels)

    # Sync TensorBoard logs to S3 (if configured)
    if isinstance(logger, list):
        sync_tensorboard_to_s3(logger)

    # Close W&B run to ensure clean separation in sweeps
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return {
        "best_model_path": best_model_path,
        "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
    }
