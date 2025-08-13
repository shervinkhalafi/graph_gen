from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.final_eval import final_eval
from tmgg.experiment_utils.logging import create_loggers
from tmgg.experiment_utils.sanity_check import maybe_run_sanity_check
from tmgg.experiment_utils.setup import create_callbacks, set_seed


def run_experiment(config: DictConfig) -> dict[str, Any]:
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

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)

    # Get best model path
    best_model_path = trainer.checkpoint_callback.best_model_path

    # Load best model for final evaluation
    if best_model_path:
        final_eval(model, data_module, logger, trainer)

    return {
        "best_model_path": best_model_path,
        "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
    }
