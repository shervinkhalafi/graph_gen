import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — set before any matplotlib imports
import torch
from loguru import logger as loguru
from omegaconf import DictConfig, OmegaConf

from tmgg.training.logging import create_loggers
from tmgg.training.orchestration.experiment_setup import (
    configure_matmul_precision,
    create_callbacks,
    set_seed,
)
from tmgg.training.orchestration.sanity_check import (
    maybe_run_sanity_check,
)


def _format_hp(value: float) -> str:
    """Format a hyperparameter value: scientific notation for < 0.01, plain otherwise."""
    if isinstance(value, float) and value < 0.01:
        return f"{value:.0e}".replace("e-0", "e-")
    return str(value)


def generate_run_id(config: DictConfig) -> str:
    """Generate a human-readable run ID from resolved config values.

    If ``run_id`` already exists in the config, returns it unchanged.
    Otherwise, builds a compact identifier from experiment name, model class,
    key hyperparameters, and seed. Small float values use scientific notation
    (e.g. ``lr1e-4``), matching the convention of the former ``run_id_template``.

    Parameters
    ----------
    config : DictConfig
        Resolved Hydra configuration.

    Returns
    -------
    str
        Run identifier like ``stage1_SpectralArch_lr1e-4_wd1e-2_k8_s1``.
    """
    existing = config.get("run_id")
    if existing is not None:
        return str(existing)

    parts: list[str] = []

    exp_name = config.get("experiment_name")
    if exp_name:
        parts.append(str(exp_name))

    model_cfg = config.get("model", {})
    if not hasattr(model_cfg, "get"):
        model_cfg = {}
    target = model_cfg.get("_target_", "") if model_cfg else ""
    if target:
        parts.append(str(target).split(".")[-1])

    lr = config.get("learning_rate")
    if lr is not None:
        parts.append(f"lr{_format_hp(lr)}")

    wd = config.get("weight_decay")
    if wd is not None:
        parts.append(f"wd{_format_hp(wd)}")

    k = model_cfg.get("k") if model_cfg else None
    if k is not None:
        parts.append(f"k{k}")

    diff_steps = model_cfg.get("diffusion_steps") if model_cfg else None
    if diff_steps is not None:
        parts.append(f"T{diff_steps}")

    seed = config.get("seed")
    if seed is not None:
        parts.append(f"s{seed}")

    return "_".join(parts)


def check_wandb_run_exists(entity: str, project: str, run_name: str) -> bool:
    """Check whether a W&B run with this display name already exists.

    Uses the ``displayName`` filter for an efficient single-run lookup.
    Returns False on ``CommError`` (network issues) so that transient
    connectivity problems don't block experiment execution. All other
    exceptions propagate -- authentication failures, programming errors,
    etc. must surface immediately rather than silently returning False
    (which would cause expensive experiments to re-run).

    Parameters
    ----------
    entity : str
        W&B entity (team or username).
    project : str
        W&B project name.
    run_name : str
        Display name to search for.

    Returns
    -------
    bool
        True if a run with that display name exists.
    """
    import wandb
    from wandb.errors import CommError

    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"displayName": run_name})
        return len(list(runs)) > 0
    except CommError:
        loguru.warning(
            "W&B API unreachable, skipping dedup check for {run_name}",
            run_name=run_name,
        )
        return False


# Matches the `save_last: true` setting in base/callbacks/default.yaml
_LAST_CHECKPOINT = "last.ckpt"


def _find_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent 'last' checkpoint in the directory.

    Matches ``last.ckpt`` (standard) or ``last-v*.ckpt`` (Lightning version
    counter, which creates ``last-v1.ckpt``, ``last-v2.ckpt``, etc. when
    a checkpoint with that name already exists).

    Returns None if no checkpoint is found.
    """
    exact = checkpoint_dir / _LAST_CHECKPOINT
    if exact.exists():
        return exact
    # Fallback: Lightning version counter produces last-v1.ckpt, last-v2.ckpt, etc.
    candidates = sorted(checkpoint_dir.glob("last-v*.ckpt"))
    return candidates[-1] if candidates else None


def _is_training_complete(checkpoint_path: Path | None, max_steps: int) -> bool:
    """Check if checkpoint has reached max_steps."""
    if checkpoint_path is None or not checkpoint_path.exists():
        return False
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    return ckpt.get("global_step", 0) >= max_steps


def run_experiment(config: DictConfig) -> dict[str, Any]:
    # Set random seed and GPU matmul precision
    set_seed(config.seed)
    configure_matmul_precision()

    # Auto-generate run_id if not set
    if config.get("run_id") is None:
        from omegaconf import open_dict

        with open_dict(config):
            config.run_id = generate_run_id(config)

    # W&B preflight: skip if a run with this name already exists
    if config.get("skip_if_wandb_exists", False):
        entity = config.get("wandb_entity")
        project = config.get("wandb_project")
        run_name = config.get("run_id")
        if (
            entity
            and project
            and run_name
            and check_wandb_run_exists(str(entity), str(project), str(run_name))
        ):
            loguru.info(
                f"Skipping: W&B run '{run_name}' already exists in {entity}/{project}"
            )
            return {"skipped": True, "reason": "wandb_run_exists"}

    # Create output directories
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)

    # Check if experiment is already complete (skip W&B run creation if so)
    checkpoint_dir = Path(config.paths.output_dir) / "checkpoints"
    last_ckpt = _find_last_checkpoint(checkpoint_dir)
    test_results_path = Path(config.paths.output_dir) / "test_results.json"

    max_steps = OmegaConf.select(config, "trainer.max_steps", default=None)
    training_complete = (
        _is_training_complete(last_ckpt, max_steps) if max_steps is not None else False
    )
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

    # Checkpoint resumption: look for last checkpoint unless force_fresh is set
    ckpt_path = None
    if config.get("force_fresh", False):
        loguru.info(f"force_fresh=True, starting fresh in: {config.paths.output_dir}")
    elif last_ckpt is not None:
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

    # Close W&B run to ensure clean separation in sweeps
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    assert trainer.checkpoint_callback is not None, (
        "ModelCheckpoint callback is required but missing from trainer. "
        "Check callbacks config (base/callbacks/default.yaml)."
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    return {
        "best_model_path": best_model_path,
        "best_val_loss": best_score.item() if best_score is not None else None,
    }
