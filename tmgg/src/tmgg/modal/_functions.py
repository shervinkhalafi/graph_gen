"""Modal deployment definitions for tmgg-spectral.

This module is the **single entry point** for ``modal deploy``. All
``@app.function`` decorated functions and the ``@app.local_entrypoint``
are defined here. Runtime code (ModalRunner, CLI) never imports this
module — they use ``modal.Function.from_name()`` to get references
to the deployed functions.

Deploy::

    uv run modal deploy -m tmgg.modal._functions

Serve (hot-reload)::

    uv run modal serve -m tmgg.modal._functions
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

from tmgg.modal.app import (
    DEFAULT_SCALEDOWN_WINDOW,
    DEFAULT_TIMEOUTS,
    GPU_CONFIGS,
    MEMORY_CONFIGS,
    MODAL_APP_NAME,
)
from tmgg.modal.image import create_tmgg_image
from tmgg.modal.volumes import get_eigenstructure_volume_mounts, get_volume_mounts

# ---------------------------------------------------------------------------
# App, secrets, image — only executed during `modal deploy` / `modal serve`
# ---------------------------------------------------------------------------

app = modal.App(MODAL_APP_NAME)

try:
    from tmgg.modal.paths import discover_tmgg_path

    experiment_image = create_tmgg_image(discover_tmgg_path())
except (ImportError, RuntimeError):
    experiment_image = None  # pyright: ignore[reportAssignmentType]

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


# ======================================================================
# Experiment execution functions (logic in experiment_utils/task.py)
# ======================================================================


@app.function(
    name="modal_execute_task",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=DEFAULT_TIMEOUTS["standard"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_execute_task(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute a single experiment on Modal (A10G GPU).

    Parameters
    ----------
    task_dict
        Serialized TaskInput as a dictionary (via ``asdict()``).

    Returns
    -------
    dict
        TaskOutput as a dictionary (via ``asdict()``).
    """
    from tmgg.experiment_utils.task import TaskInput, execute_task
    from tmgg.modal.storage import get_storage_from_env

    task = TaskInput(**task_dict)
    output = execute_task(task, get_storage=get_storage_from_env)
    return asdict(output)


@app.function(
    name="modal_execute_task_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    timeout=DEFAULT_TIMEOUTS["fast"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_execute_task_fast(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute task on fast (A100) GPU — delegates to modal_execute_task."""
    return modal_execute_task.local(task_dict)


@app.function(
    name="modal_execute_task_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    timeout=DEFAULT_TIMEOUTS["debug"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_execute_task_debug(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute task on debug (T4) GPU — delegates to modal_execute_task."""
    return modal_execute_task.local(task_dict)


# ======================================================================
# Evaluation functions (logic in evaluate.py)
# ======================================================================


@app.function(
    name="modal_evaluate_mmd",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=3600,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_evaluate_mmd(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on Modal (A10G GPU)."""
    from tmgg.modal.evaluate import run_mmd_evaluation

    return run_mmd_evaluation(task_dict)


@app.function(
    name="modal_evaluate_mmd_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    timeout=1800,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_evaluate_mmd_debug(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate on debug (T4) GPU — delegates to modal_evaluate_mmd."""
    return modal_evaluate_mmd.local(task_dict)


@app.function(
    name="modal_evaluate_mmd_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    timeout=7200,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_evaluate_mmd_fast(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate on fast (A100) GPU — delegates to modal_evaluate_mmd."""
    return modal_evaluate_mmd.local(task_dict)


@app.function(
    name="modal_list_checkpoints",
    image=experiment_image,
    timeout=60,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def modal_list_checkpoints(run_id: str) -> dict[str, Any]:
    """List all checkpoints available for a run."""
    from tmgg.modal.evaluate import list_checkpoints_for_run

    return list_checkpoints_for_run(run_id)


# ======================================================================
# Eigenstructure functions (logic in eigenstructure.py)
# ======================================================================


@app.function(
    name="eigenstructure_collect",
    image=experiment_image,
    memory=MEMORY_CONFIGS["medium"],
    timeout=3600,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_collect(
    dataset_name: str,
    dataset_config: dict[str, Any],
    output_path: str,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Collect eigendecompositions for a dataset on Modal."""
    from tmgg.modal.eigenstructure import eigenstructure_collect_impl

    return eigenstructure_collect_impl(
        dataset_name, dataset_config, output_path, batch_size, seed
    )


@app.function(
    name="eigenstructure_analyze",
    image=experiment_image,
    memory=MEMORY_CONFIGS["medium"],
    timeout=1800,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_analyze(
    input_path: str,
    output_path: str,
    subspace_k: int = 10,
    compute_covariance: bool = True,
    matrix_type: str = "adjacency",
) -> dict[str, Any]:
    """Analyze collected eigenstructure data on Modal."""
    from tmgg.modal.eigenstructure import eigenstructure_analyze_impl

    return eigenstructure_analyze_impl(
        input_path, output_path, subspace_k, compute_covariance, matrix_type
    )


@app.function(
    name="eigenstructure_noised",
    image=experiment_image,
    memory=MEMORY_CONFIGS["large"],
    timeout=7200,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_noised(
    input_path: str,
    output_path: str,
    noise_type: str,
    noise_levels: list[float],
    rotation_k: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Collect eigendecompositions for noised graphs on Modal."""
    from tmgg.modal.eigenstructure import eigenstructure_noised_impl

    return eigenstructure_noised_impl(
        input_path, output_path, noise_type, noise_levels, rotation_k, seed
    )


@app.function(
    name="eigenstructure_compare",
    image=experiment_image,
    memory=MEMORY_CONFIGS["large"],
    timeout=3600,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_compare(
    original_path: str,
    noised_path: str,
    output_path: str,
    subspace_k: int = 10,
    procrustes_k_values: list[int] | None = None,
    compute_covariance_evolution: bool = True,
    matrix_type: str = "adjacency",
) -> dict[str, Any]:
    """Compare original and noised eigenstructure on Modal."""
    from tmgg.modal.eigenstructure import eigenstructure_compare_impl

    return eigenstructure_compare_impl(
        original_path,
        noised_path,
        output_path,
        subspace_k,
        procrustes_k_values,
        compute_covariance_evolution,
        matrix_type,
    )


@app.function(
    name="eigenstructure_list",
    image=experiment_image,
    memory=MEMORY_CONFIGS["small"],
    timeout=60,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_list() -> list[dict[str, Any]]:
    """List all eigenstructure studies in the volume."""
    from tmgg.modal.eigenstructure import eigenstructure_list_impl

    return eigenstructure_list_impl()


@app.function(
    name="eigenstructure_list_files",
    image=experiment_image,
    memory=MEMORY_CONFIGS["small"],
    timeout=120,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_list_files(study_path: str) -> list[dict[str, Any]]:
    """List all files in an eigenstructure study directory."""
    from tmgg.modal.eigenstructure import eigenstructure_list_files_impl

    return eigenstructure_list_files_impl(study_path)


@app.function(
    name="eigenstructure_read_file",
    image=experiment_image,
    memory=MEMORY_CONFIGS["medium"],
    timeout=300,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_read_file(file_path: str) -> bytes:
    """Read a file from the eigenstructure volume."""
    from tmgg.modal.eigenstructure import eigenstructure_read_file_impl

    return eigenstructure_read_file_impl(file_path)


# ======================================================================
# Local entrypoint (moved from run_single.py)
# ======================================================================


@app.local_entrypoint()
def main(
    config: str,
    gpu: str = "standard",
    timeout: int | None = None,
    tags: str = "",
) -> dict[str, Any]:
    """Run a single experiment from a config file.

    Parameters
    ----------
    config
        Path to experiment config JSON file.
    gpu
        GPU tier: debug (T4), standard (A10G), fast (A100).
    timeout
        Optional timeout in seconds (overrides config).
    tags
        Comma-separated additional W&B tags.
    """
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data: dict[str, Any] = json.load(f)

    run_id = config_data.get("run_id", config_path.stem)

    task_dict: dict[str, Any] = {
        "config": config_data.get("config", config_data),
        "run_id": run_id,
        "gpu_tier": gpu,
        "timeout_seconds": timeout or config_data.get("timeout_seconds", 3600),
        "additional_tags": [t.strip() for t in tags.split(",") if t.strip()],
    }

    print(f"Running experiment: {run_id}")
    print(f"GPU tier: {gpu}")
    print(f"Config: {config_path}")

    # Select function by tier
    if gpu == "debug":
        result: dict[str, Any] = modal_execute_task_debug.remote(task_dict)
    elif gpu in ("fast", "multi", "h100"):
        result = modal_execute_task_fast.remote(task_dict)
    else:
        result = modal_execute_task.remote(task_dict)

    status = result.get("status", "unknown")
    print(f"\nExperiment {run_id} finished: {status}")

    if status == "completed":
        metrics = result.get("metrics", {})
        if "best_val_loss" in metrics:
            print(f"  best_val_loss: {metrics['best_val_loss']:.6f}")
    elif status == "failed":
        print(f"  error: {result.get('error_message', 'unknown')}")

    return result
