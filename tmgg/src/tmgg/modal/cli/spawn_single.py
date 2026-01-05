"""Spawn a single experiment on the deployed Modal app.

Uses Function.from_name() to call the deployed app directly without
creating an ephemeral app context. Much faster than `modal run`.

Usage
-----
doppler run -- uv run python -m tmgg.modal.cli.spawn_single \
    --config ./configs/stage1/2026-01-05/stage1_linear_pe_lr1e-4_wd1e-2_k8_s1.json \
    --gpu debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import modal

MODAL_APP_NAME = "tmgg-spectral"


def select_modal_function(gpu_tier: str) -> modal.Function:
    """Select the appropriate deployed Modal function for the GPU tier."""
    if gpu_tier == "debug":
        func_name = "modal_execute_task_debug"
    elif gpu_tier in ("fast", "multi", "h100"):
        func_name = "modal_execute_task_fast"
    else:
        func_name = "modal_execute_task"
    return modal.Function.from_name(MODAL_APP_NAME, func_name)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment config from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spawn a single experiment on the deployed Modal app."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to experiment config JSON file",
    )
    parser.add_argument(
        "--gpu",
        default="standard",
        choices=["debug", "standard", "fast"],
        help="GPU tier (default: standard)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds (overrides config)",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated additional W&B tags",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for result instead of fire-and-forget",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config_data = load_config(args.config)
    run_id = config_data.get("run_id", args.config.stem)

    # Build task input dict
    task_dict: dict[str, Any] = {
        "config": config_data.get("config", config_data),
        "run_id": run_id,
        "gpu_tier": args.gpu,
        "timeout_seconds": args.timeout or config_data.get("timeout_seconds", 3600),
        "additional_tags": [t.strip() for t in args.tags.split(",") if t.strip()],
    }

    # Get deployed function
    modal_fn = select_modal_function(args.gpu)

    if args.wait:
        # Blocking call - wait for result
        print(f"Running {run_id} on {args.gpu} GPU (blocking)...")
        result = modal_fn.remote(task_dict)
        status = result.get("status", "unknown")
        print(f"Finished: {status}")
        if status == "completed":
            metrics = result.get("metrics", {})
            if "best_val_loss" in metrics:
                print(f"  best_val_loss: {metrics['best_val_loss']:.6f}")
        elif status == "failed":
            print(f"  error: {result.get('error_message', 'unknown')}")
    else:
        # Fire-and-forget using spawn()
        fc = modal_fn.spawn(task_dict)
        print(f"Spawned {run_id} -> {fc.object_id}")


if __name__ == "__main__":
    main()
