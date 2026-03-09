"""MMD evaluation dispatch for trained DiGress checkpoints.

No ``import modal`` at module level. The ``@app.function`` decorated
wrappers live in ``_functions.py``; this module builds CLI arguments
and runs the evaluate_checkpoint CLI as a subprocess, mirroring the
training dispatch pattern in ``modal_run_cli``.

The subprocess boundary keeps ``modal/`` free of ``tmgg.experiments``
imports --- those happen inside the subprocess, not here.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from tmgg.modal._lib.volumes import OUTPUTS_MOUNT

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
    num_samples
        Number of graphs to generate for evaluation.
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
        Nested dict mapping evaluation label -> metric -> value.
        E.g., {"eval": {"degree_mmd": 0.01, "clustering_mmd": 0.02, ...}}.
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


def _make_failed_output(
    run_id: str,
    checkpoint_name: str,
    error_message: str,
) -> dict[str, Any]:
    """Build a serialized EvaluationOutput for a failure case."""
    return asdict(
        EvaluationOutput(
            run_id=run_id,
            checkpoint_name=checkpoint_name,
            status="failed",
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
        )
    )


def _cuda_available() -> bool:
    """Check CUDA availability without importing torch at module level."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run_mmd_evaluation(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics via CLI subprocess.

    Mirrors the training dispatch pattern in ``_functions.py:modal_run_cli``:
    builds CLI arguments from the evaluation parameters and calls the
    ``evaluate_checkpoint`` CLI as a subprocess. The subprocess handles
    model loading, data reconstruction, sampling, and MMD computation.

    Parameters
    ----------
    task_dict
        Serialized EvaluationInput as a dictionary.

    Returns
    -------
    dict
        Serialized EvaluationOutput.
    """
    from omegaconf import OmegaConf

    task = EvaluationInput(**task_dict)
    run_id = task.run_id
    output_dir = Path(OUTPUTS_MOUNT) / run_id
    config_path = output_dir / "config.yaml"

    if task.checkpoint_path:
        checkpoint_path = Path(task.checkpoint_path)
    else:
        checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    checkpoint_name = checkpoint_path.stem

    # ------------------------------------------------------------------
    # Fail fast on missing files (before spending time on subprocess)
    # ------------------------------------------------------------------
    if not config_path.exists():
        return _make_failed_output(
            run_id, checkpoint_name, f"Config not found: {config_path}"
        )
    if not checkpoint_path.exists():
        return _make_failed_output(
            run_id, checkpoint_name, f"Checkpoint not found: {checkpoint_path}"
        )

    # ------------------------------------------------------------------
    # Read dataset config for CLI args
    # ------------------------------------------------------------------
    raw_config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(
            f"Expected dict from OmegaConf.to_container, got {type(config)}"
        )
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise TypeError(f"Expected dict for config['data'], got {type(data_cfg)}")
    dataset_type = data_cfg.get("graph_type", "sbm")
    num_nodes = data_cfg.get("num_nodes", 20)

    device = "cuda" if _cuda_available() else "cpu"

    # Temp file for CLI to write JSON results into.
    # Created inside the try block so the finally cleanup always applies.
    result_path = Path(tempfile.mktemp(suffix=".json", prefix="mmd_eval_"))

    try:
        cli_args = [
            "python",
            "-m",
            "tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint",
            "--checkpoint",
            str(checkpoint_path),
            "--dataset",
            str(dataset_type),
            "--num-samples",
            str(task.num_samples),
            "--num-nodes",
            str(num_nodes),
            "--num-steps",
            str(task.num_steps),
            "--mmd-kernel",
            task.mmd_kernel,
            "--mmd-sigma",
            str(task.mmd_sigma),
            "--seed",
            str(task.seed),
            "--device",
            device,
            "--output",
            str(result_path),
        ]

        logger.info("Running evaluation CLI: %s", " ".join(cli_args))
        proc = subprocess.run(cli_args, capture_output=True, text=True)

        if proc.returncode != 0:
            error_tail = proc.stderr[-2000:] if proc.stderr else "no stderr"
            logger.error(
                "Evaluation CLI failed (exit %d): %s", proc.returncode, error_tail
            )
            return _make_failed_output(
                run_id,
                checkpoint_name,
                f"CLI exited with code {proc.returncode}: {error_tail}",
            )

        if not result_path.exists():
            return _make_failed_output(
                run_id,
                checkpoint_name,
                "CLI completed but produced no output file",
            )

        with open(result_path) as f:
            cli_results = json.load(f)

        mmd_results = cli_results.get("mmd_results", {})

        # Write combined results to the shared volume
        eval_params = {
            "num_samples": task.num_samples,
            "num_steps": task.num_steps,
            "mmd_kernel": task.mmd_kernel,
            "mmd_sigma": task.mmd_sigma,
            "seed": task.seed,
        }
        eval_output_path = output_dir / f"mmd_evaluation_{checkpoint_name}.json"
        eval_data = {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "params": eval_params,
            "results": {"eval": mmd_results},
        }
        with open(eval_output_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        logger.info("Results saved to: %s", eval_output_path)

        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="completed",
                results={"eval": mmd_results},
                evaluation_params=eval_params,
                timestamp=datetime.now().isoformat(),
            )
        )

    finally:
        if result_path.exists():
            result_path.unlink()


def list_checkpoints_for_run(run_id: str) -> dict[str, Any]:
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
