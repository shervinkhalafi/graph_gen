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

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# Modal's type stubs are incomplete for volume mount and image parameters.

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import modal

from tmgg.modal._lib.image import create_tmgg_image
from tmgg.modal._lib.volumes import get_volume_mounts
from tmgg.modal.app import (
    DEFAULT_SCALEDOWN_WINDOW,
    DEFAULT_TIMEOUTS,
    GPU_CONFIGS,
    MODAL_APP_NAME,
)

# ---------------------------------------------------------------------------
# App, secrets, image — only executed during `modal deploy` / `modal serve`
# ---------------------------------------------------------------------------

app = modal.App(MODAL_APP_NAME)

try:
    from tmgg.modal._lib.paths import discover_tmgg_path

    experiment_image = create_tmgg_image(discover_tmgg_path())
except (ImportError, RuntimeError):
    experiment_image = None

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
# Helpers
# ======================================================================


def _extract_wandb_run_id(stdout: str) -> str | None:
    """Extract W&B run ID from subprocess stdout.

    Looks for the run-page URL that W&B prints at the end of every run,
    e.g. ``https://wandb.ai/entity/project/runs/abcd1234``.

    Parameters
    ----------
    stdout
        Combined stdout text from the subprocess.

    Returns
    -------
    str or None
        The 8-character W&B run ID, or ``None`` if not found.
    """
    match = re.search(r"https://wandb\.ai/[^/]+/[^/]+/runs/([a-z0-9]+)", stdout)
    if match:
        return match.group(1)
    return None


def _run_cli_impl(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a CLI experiment subprocess inside a Modal container.

    This is the shared implementation behind all ``modal_run_cli*`` functions.
    Each GPU-tier function delegates here — Modal requires a separate
    ``@app.function`` per GPU/timeout configuration, but the execution logic
    is identical.

    Parameters
    ----------
    cmd
        CLI entry point name (e.g. ``"tmgg-discrete-gen"``).
    config_yaml
        Fully resolved Hydra config serialized as YAML.
    run_id
        Experiment identifier for the confirmation log.
    overrides
        Optional Hydra CLI overrides.

    Returns
    -------
    dict
        ``{"status": "completed"|"failed", "run_id": ..., ...}``
    """
    import subprocess
    import tempfile

    from tmgg.modal._lib.confirmation import (
        DEFAULT_CONFIRMATION_PATH,
        append_confirmation,
    )

    append_confirmation(
        DEFAULT_CONFIRMATION_PATH, run_id=run_id, status="started", cmd=cmd
    )

    # Write resolved config to a temp directory so Hydra can find it
    config_dir = Path(tempfile.mkdtemp(prefix="tmgg_run_"))
    config_file = config_dir / "run_config.yaml"
    config_file.write_text(config_yaml)

    cli_args = [cmd, f"--config-path={config_dir}", "--config-name=run_config"]
    if overrides:
        cli_args.extend(overrides)

    result = subprocess.run(cli_args, capture_output=True, text=True)

    wandb_run_id = _extract_wandb_run_id(result.stdout)

    if result.returncode == 0:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="completed",
            exit_code=0,
            wandb_run_id=wandb_run_id,
            cmd=cmd,
        )
        return {"status": "completed", "run_id": run_id, "wandb_run_id": wandb_run_id}
    else:
        error_tail = result.stderr[-500:] if result.stderr else ""
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            exit_code=result.returncode,
            error=error_tail,
            cmd=cmd,
        )
        return {
            "status": "failed",
            "run_id": run_id,
            "exit_code": result.returncode,
            "error": error_tail,
        }


def _evaluate_mmd_impl(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Run MMD evaluation on a checkpoint.

    Shared implementation for all ``modal_evaluate_mmd*`` functions.
    Each GPU tier has its own ``@app.function`` (Modal requires separate
    decorators per GPU/timeout config), but delegates here.
    """
    from tmgg.modal._lib.evaluate import run_mmd_evaluation

    return run_mmd_evaluation(task_dict)


# ======================================================================
# Experiment execution functions (CLI subprocess transport)
#
# Modal requires a separate @app.function per GPU/timeout configuration —
# there is no way to parameterize GPU type at call time. Each function
# below is a thin wrapper that delegates to _run_cli_impl.
# ======================================================================


@app.function(
    name="modal_run_cli",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=DEFAULT_TIMEOUTS["standard"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI on standard GPU (A10G)."""
    return _run_cli_impl(cmd, config_yaml, run_id, overrides)


@app.function(
    name="modal_run_cli_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    timeout=DEFAULT_TIMEOUTS["fast"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli_fast(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI on fast GPU (A100)."""
    return _run_cli_impl(cmd, config_yaml, run_id, overrides)


@app.function(
    name="modal_run_cli_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    timeout=DEFAULT_TIMEOUTS["debug"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli_debug(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI on debug GPU (T4)."""
    return _run_cli_impl(cmd, config_yaml, run_id, overrides)


# ======================================================================
# Evaluation functions
#
# Same pattern: one @app.function per GPU tier, all delegate to
# _evaluate_mmd_impl.
# ======================================================================


@app.function(
    name="modal_evaluate_mmd",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    timeout=3600,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on standard GPU (A10G)."""
    return _evaluate_mmd_impl(task_dict)


@app.function(
    name="modal_evaluate_mmd_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    timeout=1800,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_debug(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on debug GPU (T4)."""
    return _evaluate_mmd_impl(task_dict)


@app.function(
    name="modal_evaluate_mmd_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    timeout=7200,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_fast(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on fast GPU (A100)."""
    return _evaluate_mmd_impl(task_dict)


@app.function(
    name="modal_list_checkpoints",
    image=experiment_image,
    timeout=60,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_list_checkpoints(run_id: str) -> dict[str, Any]:
    """List all checkpoints available for a run."""
    from tmgg.modal._lib.evaluate import list_checkpoints_for_run

    return list_checkpoints_for_run(run_id)
