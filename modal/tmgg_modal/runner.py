"""Modal-specific CloudRunner implementation.

Provides ModalRunner that executes TMGG experiments on Modal GPUs
with Tigris storage for checkpoints and metrics.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import modal
from omegaconf import DictConfig, OmegaConf

from tmgg_modal.app import GPU_CONFIGS, DEFAULT_TIMEOUTS, app  # pyright: ignore[reportImplicitRelativeImport]
from tmgg_modal.image import create_tmgg_image  # pyright: ignore[reportImplicitRelativeImport]
from tmgg_modal.storage import TigrisStorage, get_storage_from_env  # pyright: ignore[reportImplicitRelativeImport]
from tmgg_modal.volumes import get_volume_mounts, OUTPUTS_MOUNT  # pyright: ignore[reportImplicitRelativeImport]

logger = logging.getLogger(__name__)

# Import base classes from tmgg
try:
    from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult  # pyright: ignore[reportAssignmentType]
except ImportError:
    # Fallback definitions for when tmgg is not installed
    from dataclasses import dataclass, field

    @dataclass
    class ExperimentResult:
        run_id: str
        config: dict[str, Any]
        metrics: dict[str, float] = field(default_factory=dict)
        checkpoint_path: str | None = None
        status: str = "completed"
        error_message: str | None = None
        duration_seconds: float = 0.0

    class CloudRunner:
        pass


# Create experiment image, with fallback for testing
try:
    from tmgg_modal.paths import discover_tmgg_path  # pyright: ignore[reportImplicitRelativeImport]

    _tmgg_path = discover_tmgg_path()
    experiment_image = create_tmgg_image(_tmgg_path)
except Exception:
    # During testing with mocked modal, image creation may fail
    # This is fine as the decorated functions won't actually run
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
    required_keys=[
        "WANDB_API_KEY",
    ],
)


def _get_timeout_for_gpu(gpu_type: str) -> int:
    """Get the timeout in seconds for a given GPU tier."""
    return DEFAULT_TIMEOUTS.get(gpu_type, DEFAULT_TIMEOUTS["standard"])


@app.function(
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=DEFAULT_TIMEOUTS["standard"],
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def run_single_experiment(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single experiment on Modal.

    Parameters
    ----------
    config_dict
        Serialized Hydra configuration.

    Returns
    -------
    dict
        Experiment result as dictionary including timing information.
    """
    from datetime import datetime
    from tmgg.experiment_utils.run_experiment import run_experiment

    start_time = time.time()
    started_at = datetime.now().isoformat()
    run_id = config_dict.get("run_id", str(uuid.uuid4())[:8])

    # Get timeout for this GPU tier (for tracking purposes)
    gpu_tier = "standard"  # Default tier for this function
    timeout_seconds = _get_timeout_for_gpu(gpu_tier)

    # Reconstruct OmegaConf
    config = OmegaConf.create(config_dict)

    # Override output directory to use Modal volume
    config.paths = OmegaConf.create(
        {
            "output_dir": f"{OUTPUTS_MOUNT}/{run_id}",
            "results_dir": f"{OUTPUTS_MOUNT}/{run_id}/results",
        }
    )

    try:
        result = run_experiment(config)
        duration = time.time() - start_time

        # Upload checkpoint and metrics to Tigris
        storage = get_storage_from_env()
        checkpoint_uri = None
        storage_warnings = []

        if not storage:
            storage_warnings.append(
                f"[{run_id}] Cloud storage not configured - results saved locally only"
            )
            logger.warning(storage_warnings[-1])
        else:
            # Upload checkpoint if available
            if result.get("best_model_path"):
                checkpoint_path = Path(result["best_model_path"])
                if checkpoint_path.exists():
                    try:
                        checkpoint_uri = storage.upload_checkpoint(
                            checkpoint_path, run_id
                        )
                        logger.info(
                            f"[{run_id}] Uploaded checkpoint to {checkpoint_uri}"
                        )
                    except Exception as e:
                        storage_warnings.append(
                            f"[{run_id}] Failed to upload checkpoint: {e}"
                        )
                        logger.error(storage_warnings[-1])

            # Upload metrics
            try:
                metrics_data = {
                    "best_val_loss": result.get("best_val_loss", float("inf")),
                    "config": config_dict,
                    "duration_seconds": duration,
                }
                storage.upload_metrics(metrics_data, run_id)
                logger.info(f"[{run_id}] Uploaded metrics to storage")
            except Exception as e:
                storage_warnings.append(f"[{run_id}] Failed to upload metrics: {e}")
                logger.error(storage_warnings[-1])

        metrics = {"best_val_loss": result.get("best_val_loss", float("inf"))}

        # Check if we're approaching timeout (> 90% of limit)
        timeout_warning = None
        if duration > timeout_seconds * 0.9:
            timeout_warning = (
                f"Experiment used {duration:.0f}s of {timeout_seconds}s timeout "
                f"({duration/timeout_seconds*100:.1f}%)"
            )
            logger.warning(f"[{run_id}] {timeout_warning}")

        return {
            "run_id": run_id,
            "config": config_dict,
            "metrics": metrics,
            "checkpoint_path": checkpoint_uri,
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "timeout_seconds": timeout_seconds,
            "gpu_tier": gpu_tier,
            "timeout_warning": timeout_warning,
            "storage_warnings": storage_warnings if storage_warnings else None,
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "run_id": run_id,
            "config": config_dict,
            "metrics": {},
            "checkpoint_path": None,
            "status": "failed",
            "error_message": str(e),
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "timeout_seconds": timeout_seconds,
            "gpu_tier": gpu_tier,
        }


@app.function(
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    timeout=DEFAULT_TIMEOUTS["fast"],
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def run_single_experiment_fast(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single experiment on fast (A100) GPU."""
    return run_single_experiment.local(config_dict)


class ModalRunner(CloudRunner):
    """Modal-specific experiment runner.

    Executes experiments on Modal GPUs with automatic scaling
    and parallel execution support.
    """

    def __init__(
        self,
        gpu_type: str = "standard",
        storage: TigrisStorage | None = None,
    ):
        """Initialize Modal runner.

        Parameters
        ----------
        gpu_type
            Default GPU tier for experiments.
        storage
            Tigris storage for results. If None, creates from env.
        """
        self.gpu_type = gpu_type
        self.storage = storage
        self._active_runs: dict[str, str] = {}

    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
    ) -> ExperimentResult:
        """Run a single experiment on Modal.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            GPU tier override.
        timeout_seconds
            Timeout override.

        Returns
        -------
        ExperimentResult
            Result of the experiment.
        """
        gpu = gpu_type or self.gpu_type
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Select appropriate function based on GPU tier
        if gpu in ("fast", "multi", "h100"):
            result_dict = run_single_experiment_fast.remote(config_dict)  # pyright: ignore[reportArgumentType]
        else:
            result_dict = run_single_experiment.remote(config_dict)  # pyright: ignore[reportArgumentType]

        return ExperimentResult(**result_dict)

    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str | None = None,
        parallelism: int = 4,
        timeout_seconds: int | None = None,
    ) -> list[ExperimentResult]:
        """Run multiple experiments in parallel on Modal.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            GPU tier for all experiments.
        parallelism
            Maximum concurrent experiments.
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[ExperimentResult]
            Results from all experiments.
        """
        gpu = gpu_type or self.gpu_type
        config_dicts = [OmegaConf.to_container(c, resolve=True) for c in configs]

        # Use Modal's starmap for parallel execution
        if gpu in ("fast", "multi", "h100"):
            results = list(run_single_experiment_fast.map(config_dicts))
        else:
            results = list(run_single_experiment.map(config_dicts))

        return [ExperimentResult(**r) for r in results]

    def get_status(self, run_id: str) -> str:
        """Get status of a Modal run.

        Note: Modal doesn't provide easy status checking for completed
        function calls. This returns 'completed' or 'unknown'.
        """
        return self._active_runs.get(run_id, "unknown")

    def cancel(self, run_id: str) -> bool:
        """Cancel not directly supported in Modal."""
        return False


def create_runner(gpu_type: str = "standard") -> ModalRunner:
    """Factory function to create a ModalRunner.

    Parameters
    ----------
    gpu_type
        Default GPU tier.

    Returns
    -------
    ModalRunner
        Configured runner instance.
    """
    storage = get_storage_from_env()
    return ModalRunner(gpu_type=gpu_type, storage=storage)
