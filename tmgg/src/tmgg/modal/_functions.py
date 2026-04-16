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
import shlex
import signal
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

app = modal.App(MODAL_APP_NAME, include_source=False)

try:
    from tmgg.modal._lib.paths import discover_source_checkout_path

    experiment_image = create_tmgg_image(discover_source_checkout_path())
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

IMPORT_PREFLIGHTS: tuple[tuple[str, str], ...] = (
    ("torch", "import torch"),
    ("torch_geometric", "import torch_geometric"),
    ("ot", "import ot"),
    ("graph_tool", "import graph_tool.all as gt"),
)
CONFIG_PREFLIGHT_SNIPPET = """
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

config_path = Path({config_path!r})
cfg = OmegaConf.load(config_path)

print("Config preflight: instantiate datamodule", flush=True)
dm = hydra.utils.instantiate(cfg.data)
dm.setup("fit")
print("Config preflight OK: datamodule setup", flush=True)

print("Config preflight: fetch validation batch", flush=True)
batch = next(iter(dm.val_dataloader()))

def _shape_summary(data):
    parts = [("node_mask", tuple(data.node_mask.shape))]
    for name in ("X_class", "X_feat", "E_class", "E_feat"):
        t = getattr(data, name)
        parts.append((name, tuple(t.shape) if t is not None else None))
    return parts

print(
    "Config preflight OK: validation batch",
    _shape_summary(batch),
    flush=True,
)

print("Config preflight: instantiate LightningModule", flush=True)
module = hydra.utils.instantiate(cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
module = module.to(device)
print("Config preflight OK: module to device", str(device), flush=True)

if hasattr(module, "noise_process") and module.noise_process.needs_data_initialization():
    print("Config preflight: initialize noise process", flush=True)
    module.noise_process.initialize_from_data(dm.train_dataloader())
    print("Config preflight OK: noise process initialized", flush=True)

print("Config preflight: transfer batch to device", flush=True)
batch = module.transfer_batch_to_device(batch, device, 0)
print(
    "Config preflight OK: batch to device",
    str(batch.node_mask.device),
    _shape_summary(batch),
    flush=True,
)

if hasattr(module, "noise_process") and hasattr(module, "_compute_loss") and hasattr(module, "T"):
    with torch.no_grad():
        bs = int(batch.node_mask.shape[0])
        t_int = torch.randint(1, int(module.T) + 1, (bs,), device=device)

        print("Config preflight: noise forward_sample", flush=True)
        z_t = module.noise_process.forward_sample(batch, t_int)
        print(
            "Config preflight OK: noise forward_sample",
            _shape_summary(z_t),
            flush=True,
        )

        print("Config preflight: condition vector", flush=True)
        condition = module.noise_process.process_state_condition_vector(t_int)
        print("Config preflight OK: condition vector", tuple(condition.shape), flush=True)

        print("Config preflight: model forward", flush=True)
        pred = module.model(z_t, t=condition)
        print(
            "Config preflight OK: model forward",
            _shape_summary(pred),
            flush=True,
        )

        print("Config preflight: compute loss", flush=True)
        loss = module._compute_loss(pred, batch)
        print("Config preflight OK: compute loss", float(loss.item()), flush=True)
"""


# ======================================================================
# Helpers
# ======================================================================


def _extract_wandb_run_url(output: str) -> str | None:
    """Extract the W&B run URL from subprocess output.

    Looks for the run-page URL that W&B prints at the end of every run,
    e.g. ``https://wandb.ai/entity/project/runs/abcd1234``.

    Parameters
    ----------
    output
        Combined subprocess output text.

    Returns
    -------
    str or None
        The full W&B run URL, or ``None`` if not found.
    """
    matches = re.findall(r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/[a-z0-9]+", output)
    if matches:
        return matches[-1]
    return None


def _extract_wandb_run_id(output: str) -> str | None:
    """Extract the W&B run ID from subprocess output."""
    wandb_url = _extract_wandb_run_url(output)
    if wandb_url is None:
        return None
    return wandb_url.rsplit("/", 1)[-1]


def _format_return_code(return_code: int) -> str:
    """Return a human-readable subprocess exit description."""
    if return_code >= 0:
        return str(return_code)

    try:
        signame = signal.Signals(-return_code).name
    except ValueError:
        return str(return_code)
    return f"{return_code} ({signame})"


def _run_import_preflight() -> None:
    """Probe native imports in isolated subprocesses before launching the CLI.

    Native-extension import failures can terminate Python with signals like
    ``SIGILL`` before any traceback is written. Running one import per
    subprocess gives us a precise module name and exit code in Modal logs.
    """
    import subprocess
    import sys

    for module_name, import_stmt in IMPORT_PREFLIGHTS:
        print(f"Preflight import check: {module_name}", flush=True)
        result = subprocess.run(
            [sys.executable, "-c", import_stmt],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Preflight import OK: {module_name}", flush=True)
            continue

        output = (result.stdout or "") + (result.stderr or "")
        output_tail = output[-1000:] if output else "<no output>"
        raise RuntimeError(
            "Modal import preflight failed.\n"
            f"Module: {module_name}\n"
            f"Statement: {shlex.quote(import_stmt)}\n"
            f"Exit code: {_format_return_code(result.returncode)}\n"
            f"Output tail:\n{output_tail}"
        )


def _run_config_preflight(config_file: Path) -> None:
    """Exercise datamodule setup and first-batch transfer in isolation."""
    import subprocess
    import sys

    print("Config preflight: start", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CONFIG_PREFLIGHT_SNIPPET.format(config_path=str(config_file)),
        ],
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode == 0:
        print("Config preflight OK: complete", flush=True)
        return

    output_tail = output[-1200:] if output else "<no output>"
    raise RuntimeError(
        "Modal config preflight failed.\n"
        f"Exit code: {_format_return_code(result.returncode)}\n"
        f"Output tail:\n{output_tail}"
    )


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

    try:
        _run_import_preflight()
    except RuntimeError as exc:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            error=str(exc)[-1000:],
            stage="preflight",
            cmd=cmd,
        )
        raise

    # Write resolved config to a temp directory so Hydra can find it
    config_dir = Path(tempfile.mkdtemp(prefix="tmgg_run_"))
    config_file = config_dir / "run_config.yaml"
    config_file.write_text(config_yaml)

    try:
        _run_config_preflight(config_file)
    except RuntimeError as exc:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            error=str(exc)[-1200:],
            stage="config_preflight",
            cmd=cmd,
        )
        raise

    cli_args = [cmd, f"--config-path={config_dir}", "--config-name=run_config"]
    if overrides:
        cli_args.extend(overrides)

    process = subprocess.Popen(
        cli_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    output_lines: list[str] = []
    wandb_url: str | None = None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
        if wandb_url is None:
            wandb_url = _extract_wandb_run_url(line)
            if wandb_url is not None:
                print(f"W&B URL: {wandb_url}", flush=True)

    return_code = process.wait()
    combined_output = "".join(output_lines)
    if wandb_url is None:
        wandb_url = _extract_wandb_run_url(combined_output)
        if wandb_url is not None:
            print(f"W&B URL: {wandb_url}", flush=True)
    wandb_run_id = _extract_wandb_run_id(combined_output)

    if return_code == 0:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="completed",
            exit_code=0,
            wandb_run_id=wandb_run_id,
            wandb_url=wandb_url,
            cmd=cmd,
        )
        return {
            "status": "completed",
            "run_id": run_id,
            "wandb_run_id": wandb_run_id,
            "wandb_url": wandb_url,
        }
    else:
        error_tail = combined_output[-500:] if combined_output else ""
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            exit_code=return_code,
            error=error_tail,
            wandb_url=wandb_url,
            cmd=cmd,
        )
        raise RuntimeError(
            "Experiment subprocess failed inside Modal container.\n"
            f"Exit code: {_format_return_code(return_code)}\n"
            f"Command: {cli_args}\n"
            f"Output tail:\n{error_tail}"
        )


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
