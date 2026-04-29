import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — set before any matplotlib imports
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger as loguru
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from tmgg.training.logging import create_loggers
from tmgg.training.orchestration.sanity_check import (
    maybe_run_sanity_check,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all RNG backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def configure_matmul_precision(precision: str = "high") -> None:
    """Set float32 matmul precision for Tensor Core GPUs.

    Parameters
    ----------
    precision : str
        One of ``"highest"``, ``"high"``, ``"medium"``. Default ``"high"``
        enables TF32 on Ampere+ GPUs for ~2x speedup with minimal accuracy loss.
    """
    torch.set_float32_matmul_precision(precision)


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
        filename=ckpt_config.get(
            "filename", "model-step={step:06d}-val_loss={val/loss:.4f}"
        ),
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
    from tmgg.training.progress import StepProgressBar

    pb_config = config.get("progress_bar", {})
    callbacks.append(
        StepProgressBar(
            metrics_to_show=pb_config.get(
                "metrics_to_show", ["train_loss", "val/loss"]
            ),
            show_epoch=pb_config.get("show_epoch", True),
            metrics_format=pb_config.get("metrics_format", ".4f"),
        )
    )

    # EMA shadow weights (parity #45 / D-15). Mirrors upstream DiGress's
    # cfg.train.ema_decay > 0 gate (main.py:181-183): non-zero ema_decay
    # registers the callback, zero (default) leaves it out.
    ema_decay = float(cb_config.get("ema_decay", 0.0))
    if ema_decay > 0.0:
        from tmgg.training.callbacks import EMACallback

        callbacks.append(EMACallback(decay=ema_decay))

    # Final-sample dump at fit-end (parity #46 / D-16b). Upstream
    # gates on final_model_samples_to_generate > 0; we mirror that
    # exactly. Block reads exclusively from training.final_sample_dump.*;
    # the legacy flat-key shim was removed per CLAUDE.md "no fallbacks,
    # no transitions". Wave configs supply the block via Hydra defaults
    # (training/final_sample_dump@training.final_sample_dump: default).
    training_cfg = config.get("training", {})
    fsd_cfg = training_cfg.get("final_sample_dump", {})
    num_final_samples = int(fsd_cfg.get("num_samples", 0))
    if num_final_samples > 0:
        from tmgg.training.callbacks import FinalSampleDumpCallback

        callbacks.append(
            FinalSampleDumpCallback(
                num_samples=num_final_samples,
                sample_batch_size=int(fsd_cfg.get("sample_batch_size", 64)),
                save_path=fsd_cfg.get("save_path", None),
                run_name=str(config.get("run_id", "run")),
            )
        )

    # Reverse-chain snapshot recording (parity #46 / D-16a). Mirrors
    # upstream DiGress's ``number_chain_steps > 0`` gate: leaving
    # num_chains_to_save at 0 disables the callback entirely. Wave
    # configs compose the block via Hydra defaults
    # (training/chain_saving@training.chain_saving: default) so the
    # block lives in one place.
    cs_cfg = training_cfg.get("chain_saving", {})
    num_chains_to_save = int(cs_cfg.get("num_chains_to_save", 0))
    if num_chains_to_save > 0:
        from tmgg.training.callbacks import ChainSavingCallback

        callbacks.append(
            ChainSavingCallback(
                num_chains_to_save=num_chains_to_save,
                snapshot_step_interval=int(cs_cfg.get("snapshot_step_interval", 50)),
                chain_save_every_n_val=int(cs_cfg.get("chain_save_every_n_val", 20)),
                chain_save_at_fit_end=bool(cs_cfg.get("chain_save_at_fit_end", True)),
                chain_save_path=cs_cfg.get("chain_save_path", None),
                run_name=str(config.get("run_id", "run")),
            )
        )

    # Async-eval spawn callback (smallest-config sweep). Mirrors the
    # gate pattern used by ema/final_sample_dump/chain_saving: read
    # the config block; if enabled, instantiate via hydra.utils.instantiate
    # so the YAML's _target_ + all init kwargs flow through cleanly. The
    # ``enabled`` key is the gate only and is not an ``__init__`` kwarg
    # of ``AsyncEvalSpawnCallback``, so we strip it before instantiation
    # (instantiate would otherwise pass it as a kwarg and crash).
    aes_cfg = cb_config.get("async_eval_spawn", None)
    if aes_cfg is not None and bool(aes_cfg.get("enabled", False)):
        from omegaconf import DictConfig as _DictConfig
        from omegaconf import open_dict

        aes_container = OmegaConf.to_container(aes_cfg, resolve=True)
        if not isinstance(aes_container, dict):
            raise TypeError(
                "callbacks.async_eval_spawn must resolve to a dict; "
                f"got {type(aes_container).__name__}"
            )
        aes_cfg_for_init = OmegaConf.create(aes_container)
        assert isinstance(aes_cfg_for_init, _DictConfig)
        with open_dict(aes_cfg_for_init):
            if "enabled" in aes_cfg_for_init:
                del aes_cfg_for_init["enabled"]
        callbacks.append(hydra.utils.instantiate(aes_cfg_for_init))

    return callbacks


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

    When ``force_fresh=true`` is set without an explicit ``run_id``
    override, the auto-generated id gets a ``_fresh_<UTC-timestamp>``
    suffix so the output directory (computed downstream as
    ``outputs/<experiment_name>/<run_id>/``) is guaranteed unique. Without
    this suffix, ``force_fresh`` would only bypass the Lightning
    checkpoint resume but would still write into the same directory as a
    prior run with identical hyperparams, risking checkpoint clobbering
    in ``ModelCheckpoint``'s ``last.ckpt`` slot.

    Parameters
    ----------
    config : DictConfig
        Resolved Hydra configuration.

    Returns
    -------
    str
        Run identifier like ``stage1_SpectralArch_lr1e-4_wd1e-2_k8_s1`` or,
        under ``force_fresh=true``, the same id with a
        ``_fresh_<YYYYmmddTHHMMSS>`` suffix.
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

    base = "_".join(parts)

    if config.get("force_fresh", False):
        from datetime import UTC, datetime

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        return f"{base}_fresh_{stamp}"

    return base


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
    # Defaults to "high" (TF32 on Ampere+) but configurable via base_infra.
    # Set to "highest" to disable TF32 for byte-exact fp32 reproducibility.
    matmul_prec = str(config.get("matmul_precision", "high"))
    configure_matmul_precision(matmul_prec)

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
