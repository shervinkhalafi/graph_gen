"""Modal local entrypoint for running a single experiment.

Takes a config JSON file path and runs the experiment on Modal.
Use with `--detach` for fire-and-forget execution.

Usage
-----
doppler run -- uv run modal run src/tmgg/modal/run_single.py \
    --config ./configs/stage1/2026-01-05/stage1_linear_pe_lr1e-4_wd1e-2_k8_s1.json \
    --gpu debug

With detach (fire-and-forget):
doppler run -- uv run modal run --detach src/tmgg/modal/run_single.py \
    --config ./configs/stage1/2026-01-05/stage1_linear_pe_lr1e-4_wd1e-2_k8_s1.json \
    --gpu standard
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from tmgg.modal.app import app

# Import Modal functions - these are defined in runner.py
# We use from_name() to get references to deployed functions
MODAL_APP_NAME = "tmgg-spectral"


def _select_modal_function(gpu_tier: str) -> Any:
    """Select the appropriate Modal function for the GPU tier.

    Tier mapping:
    - debug -> T4 (modal_execute_task_debug)
    - standard -> A10G (modal_execute_task)
    - fast, multi, h100 -> A100 (modal_execute_task_fast)
    """
    if gpu_tier == "debug":
        func_name = "modal_execute_task_debug"
    elif gpu_tier in ("fast", "multi", "h100"):
        func_name = "modal_execute_task_fast"
    else:
        func_name = "modal_execute_task"
    return modal.Function.from_name(MODAL_APP_NAME, func_name)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment config from JSON file.

    Parameters
    ----------
    config_path
        Path to config JSON file.

    Returns
    -------
    dict
        Config dictionary containing run_id and nested config.
    """
    with open(config_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(
    config: str,
    gpu: str = "standard",
    timeout: int | None = None,
    tags: str = "",
):
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

    config_data = load_config(config_path)
    run_id = config_data.get("run_id", config_path.stem)

    # Build task input dict
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

    # Get the deployed Modal function and run it
    modal_fn = _select_modal_function(gpu)
    result = modal_fn.remote(task_dict)

    # Print result summary
    status = result.get("status", "unknown")
    print(f"\nExperiment {run_id} finished: {status}")

    if status == "completed":
        metrics = result.get("metrics", {})
        if "best_val_loss" in metrics:
            print(f"  best_val_loss: {metrics['best_val_loss']:.6f}")
    elif status == "failed":
        print(f"  error: {result.get('error_message', 'unknown')}")

    return result
