"""Modal function for MMD evaluation of trained DiGress checkpoints.

Provides a Modal function that evaluates trained checkpoints using MMD metrics
against train/val/test splits from the same data distribution used during training.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import modal

from tmgg.modal.app import DEFAULT_SCALEDOWN_WINDOW, GPU_CONFIGS, app
from tmgg.modal.image import create_tmgg_image
from tmgg.modal.volumes import OUTPUTS_MOUNT, get_volume_mounts

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationInput:
    """Input for MMD evaluation task.

    Parameters
    ----------
    run_id
        Run ID whose checkpoint to evaluate.
    checkpoint_path
        Optional explicit checkpoint path. If None, defaults to
        ``/data/outputs/{run_id}/checkpoints/last.ckpt``.
    splits
        Which splits to evaluate (subset of ["train", "val", "test"]).
    num_samples
        Number of graphs to generate per split for evaluation.
    num_steps
        Number of denoising steps for sampling.
    mmd_kernel
        Kernel type for MMD computation.
    mmd_sigma
        Bandwidth for kernel.
    seed
        Random seed for reproducibility.
    """

    run_id: str
    checkpoint_path: str | None = None
    splits: list[str] = field(default_factory=lambda: ["train", "val", "test"])
    num_samples: int = 500
    num_steps: int = 100
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv"
    mmd_sigma: float = 1.0
    seed: int = 42


@dataclass
class EvaluationOutput:
    """Output from MMD evaluation.

    Parameters
    ----------
    run_id
        Run ID that was evaluated.
    checkpoint_name
        Name of the checkpoint file evaluated (e.g., "last", "best", "epoch_50").
    status
        Completion status: 'completed' or 'failed'.
    results
        Nested dict mapping split -> metric -> value.
        E.g., {"train": {"degree_mmd": 0.01, "clustering_mmd": 0.02, ...}}.
    error_message
        Error details if status is 'failed'.
    evaluation_params
        Parameters used for evaluation (num_samples, num_steps, etc.).
    timestamp
        ISO timestamp of when evaluation completed.
    """

    run_id: str
    checkpoint_name: str = "last"
    status: Literal["completed", "failed"] = "completed"
    results: dict[str, dict[str, float]] = field(default_factory=dict)
    error_message: str | None = None
    evaluation_params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


# Create experiment image with fallback for testing
try:
    from tmgg.modal.paths import discover_tmgg_path

    _tmgg_path = discover_tmgg_path()
    experiment_image = create_tmgg_image(_tmgg_path)
except Exception:
    # During testing with mocked modal, image creation may fail
    experiment_image = None

# Modal secrets for Tigris storage and W&B
tigris_secret = modal.Secret.from_name(
    "tigris-credentials",
    required_keys=[
        "TMGG_TIGRIS_BUCKET",
        "TMGG_TIGRIS_ACCESS_KEY",
        "TMGG_TIGRIS_SECRET_KEY",
    ],
)

wandb_secret = modal.Secret.from_name(
    "wandb-credentials",
    required_keys=["WANDB_API_KEY"],
)


@app.function(
    name="modal_evaluate_mmd",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],  # A10G sufficient for evaluation
    timeout=3600,  # 1 hour max
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_evaluate_mmd(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on Modal.

    This function runs inside a Modal container with GPU access and mounted volumes.
    It loads a trained checkpoint, reconstructs the data module to get reference
    graphs, generates samples from the model, and computes MMD metrics.

    Parameters
    ----------
    task_dict
        Serialized EvaluationInput as a dictionary.

    Returns
    -------
    dict
        EvaluationOutput as a dictionary.
    """
    import torch
    from omegaconf import OmegaConf

    from tmgg.experiment_utils.mmd_metrics import (
        MMDResults,
        adjacency_to_networkx,
        compute_mmd_metrics,
    )
    from tmgg.experiments.generative.evaluate_checkpoint import (
        load_model_from_checkpoint,
    )

    # Configure logging for Modal container
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Override any existing config in Modal
    )
    logger = logging.getLogger("mmd_eval")

    # Reconstruct input
    task = EvaluationInput(**task_dict)
    run_id = task.run_id

    # Start timing
    start_total = time.time()

    # Log startup with all params
    logger.info("=== MMD Evaluation Starting ===")
    logger.info(f"run_id: {run_id}")
    logger.info(f"checkpoint: {task.checkpoint_path or 'last.ckpt'}")
    logger.info(f"splits: {task.splits}")
    logger.info(f"num_samples: {task.num_samples}")
    logger.info(f"num_steps: {task.num_steps}")
    logger.info(f"mmd_kernel: {task.mmd_kernel}")

    # Resolve paths
    output_dir = Path(OUTPUTS_MOUNT) / run_id
    config_path = output_dir / "config.yaml"

    if task.checkpoint_path:
        checkpoint_path = Path(task.checkpoint_path)
    else:
        checkpoint_path = output_dir / "checkpoints" / "last.ckpt"

    # Extract checkpoint name (without .ckpt extension) for output filename
    checkpoint_name = checkpoint_path.stem  # e.g., "last", "best", "epoch_50"

    # Validate paths exist
    if not config_path.exists():
        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="failed",
                error_message=f"Config not found: {config_path}",
                timestamp=datetime.now().isoformat(),
            )
        )

    if not checkpoint_path.exists():
        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="failed",
                error_message=f"Checkpoint not found: {checkpoint_path}",
                timestamp=datetime.now().isoformat(),
            )
        )

    try:
        # Load config
        logger.info(f"Loading config from: {config_path}")
        config = OmegaConf.load(config_path)

        # Load model
        logger.info(f"Loading model from: {checkpoint_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model_from_checkpoint(checkpoint_path, device)
        logger.info("Model loaded successfully")

        # Reconstruct data module from config
        logger.info("Reconstructing data module...")
        data_module = _reconstruct_data_module(config)
        data_module.setup()

        # Get reference graphs for each split
        split_data = {
            "train": data_module._train_data,
            "val": data_module._val_data,
            "test": data_module._test_data,
        }

        results: dict[str, dict[str, float]] = {}

        for split in task.splits:
            if split not in split_data:
                logger.warning(f"Unknown split '{split}', skipping")
                continue

            ref_data = split_data[split]
            if ref_data is None or len(ref_data) == 0:
                logger.warning(f"No data for split '{split}', skipping")
                continue

            split_start = time.time()
            logger.info(
                f"[{split}] Starting evaluation ({len(ref_data)} reference graphs)"
            )

            # Determine number of samples (use min of requested and available)
            num_eval = min(task.num_samples, len(ref_data))
            num_nodes = ref_data.shape[1]  # Assuming shape is (N, num_nodes, num_nodes)

            # Select reference subset
            ref_subset = ref_data[:num_eval]

            # Generate samples
            logger.info(
                f"[{split}] Generating {num_eval} graphs ({task.num_steps} steps)..."
            )
            gen_start = time.time()
            with torch.no_grad():
                generated = model.sample(
                    num_graphs=num_eval,
                    num_nodes=num_nodes,
                    num_steps=task.num_steps,
                )
            logger.info(
                f"[{split}] Generation complete: {len(generated)} graphs in {time.time() - gen_start:.1f}s"
            )

            # Convert to NetworkX for MMD computation
            logger.info(f"[{split}] Computing MMD metrics...")
            ref_nx = [adjacency_to_networkx(g) for g in ref_subset]
            gen_nx = [adjacency_to_networkx(g) for g in generated]

            mmd_results: MMDResults = compute_mmd_metrics(
                ref_nx,
                gen_nx,
                kernel=task.mmd_kernel,
                sigma=task.mmd_sigma,
            )

            results[split] = mmd_results.to_dict()
            logger.info(
                f"[{split}] MMD: degree={mmd_results.degree_mmd:.4f}, "
                f"clustering={mmd_results.clustering_mmd:.4f}, "
                f"spectral={mmd_results.spectral_mmd:.4f}"
            )
            logger.info(f"[{split}] Completed in {time.time() - split_start:.1f}s")

        # Save results to volume with checkpoint-specific filename
        eval_output_path = output_dir / f"mmd_evaluation_{checkpoint_name}.json"
        eval_data = {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "params": {
                "num_samples": task.num_samples,
                "num_steps": task.num_steps,
                "mmd_kernel": task.mmd_kernel,
                "mmd_sigma": task.mmd_sigma,
                "seed": task.seed,
            },
            "results": results,
        }
        with open(eval_output_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        logger.info(f"Results saved to: {eval_output_path}")

        output = EvaluationOutput(
            run_id=run_id,
            checkpoint_name=checkpoint_name,
            status="completed",
            results=results,
            evaluation_params={
                "num_samples": task.num_samples,
                "num_steps": task.num_steps,
                "mmd_kernel": task.mmd_kernel,
                "mmd_sigma": task.mmd_sigma,
                "seed": task.seed,
                "checkpoint_path": str(checkpoint_path),
            },
            timestamp=datetime.now().isoformat(),
        )

        logger.info(f"=== Evaluation complete in {time.time() - start_total:.1f}s ===")
        return asdict(output)

    except Exception as e:
        import traceback

        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(f"Evaluation failed: {error_msg}")

        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="failed",
                error_message=error_msg,
                timestamp=datetime.now().isoformat(),
            )
        )


def _reconstruct_data_module(config):
    """Reconstruct GraphDistributionDataModule from saved config.

    Parameters
    ----------
    config
        OmegaConf configuration loaded from config.yaml.

    Returns
    -------
    GraphDistributionDataModule
        Configured data module ready for setup().
    """
    from tmgg.experiments.generative.datamodule import GraphDistributionDataModule

    data_cfg = config.get("data", {})

    # Extract parameters from config - handle both direct and nested configs
    dataset_type = data_cfg.get("dataset_type", "sbm")
    num_nodes = data_cfg.get("num_nodes", 20)
    num_graphs = data_cfg.get("num_graphs", 1000)
    train_ratio = data_cfg.get("train_ratio", 0.8)
    val_ratio = data_cfg.get("val_ratio", 0.1)
    batch_size = data_cfg.get("batch_size", 32)
    seed = data_cfg.get("seed", 42)
    dataset_config = dict(data_cfg.get("dataset_config", {}))

    return GraphDistributionDataModule(
        dataset_type=dataset_type,
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        seed=seed,
        dataset_config=dataset_config,
    )


@app.function(
    name="modal_list_checkpoints",
    image=experiment_image,
    timeout=60,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_list_checkpoints(run_id: str) -> dict[str, Any]:
    """List all checkpoints available for a run.

    Parameters
    ----------
    run_id
        Run ID to list checkpoints for.

    Returns
    -------
    dict
        Contains 'run_id', 'checkpoints' (list of checkpoint paths), and 'status'.
    """
    output_dir = Path(OUTPUTS_MOUNT) / run_id
    checkpoints_dir = output_dir / "checkpoints"

    if not checkpoints_dir.exists():
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": f"Checkpoints directory not found: {checkpoints_dir}",
            "checkpoints": [],
        }

    # Find all .ckpt files
    checkpoint_files = sorted(checkpoints_dir.glob("*.ckpt"))
    checkpoints = [
        {
            "name": ckpt.stem,
            "path": str(ckpt),
            "size_mb": ckpt.stat().st_size / (1024 * 1024),
        }
        for ckpt in checkpoint_files
    ]

    return {
        "run_id": run_id,
        "status": "completed",
        "checkpoints": checkpoints,
    }


# Variants for different GPU tiers
@app.function(
    name="modal_evaluate_mmd_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],  # T4 for quick tests
    timeout=1800,  # 30 min
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_evaluate_mmd_debug(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate on debug (T4) GPU tier - delegates to modal_evaluate_mmd."""
    return modal_evaluate_mmd.local(task_dict)


@app.function(
    name="modal_evaluate_mmd_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],  # A100 for larger evaluations
    timeout=7200,  # 2 hours
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_evaluate_mmd_fast(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate on fast (A100) GPU tier - delegates to modal_evaluate_mmd."""
    return modal_evaluate_mmd.local(task_dict)
