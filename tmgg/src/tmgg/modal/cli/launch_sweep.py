"""Launch multiple experiments from a config directory.

Uses Function.from_name().spawn() to call the deployed Modal app directly,
avoiding ephemeral app creation overhead.

Usage
-----
doppler run -- uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/stage1/2026-01-05/ \
    --gpu debug

With filtering:
doppler run -- uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/stage1/2026-01-05/ \
    --filter "linear_pe" \
    --gpu standard
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import modal
import wandb
from tqdm import tqdm

MODAL_APP_NAME = "tmgg-spectral"


def get_existing_run_names(entity: str, project: str) -> set[str]:
    """Fetch existing run names from W&B to skip already-completed experiments.

    Parameters
    ----------
    entity
        W&B entity (team/username).
    project
        W&B project name.

    Returns
    -------
    set[str]
        Set of existing run display names (used as run_id in our configs).
    """
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}")
        # Use displayName if available, otherwise name
        return {r.name for r in runs}
    except Exception as e:
        print(f"Warning: Could not fetch existing runs from W&B: {e}", file=sys.stderr)
        return set()


def get_config_files(config_dir: Path, filter_pattern: str | None = None) -> list[Path]:
    """Get all config JSON files from directory, optionally filtered.

    Parameters
    ----------
    config_dir
        Directory containing config JSON files.
    filter_pattern
        Optional substring filter for run_ids.

    Returns
    -------
    list[Path]
        Sorted list of matching config file paths.
    """
    all_configs = sorted(config_dir.glob("*.json"))

    if filter_pattern:
        all_configs = [c for c in all_configs if filter_pattern in c.stem]

    return all_configs


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


def launch_single(
    config_path: Path,
    modal_fn: modal.Function,
    gpu: str,
    timeout: int | None = None,
    tags: str = "",
    dry_run: bool = False,
    wait: bool = False,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> tuple[bool, str]:
    """Launch a single experiment on the deployed Modal app.

    Parameters
    ----------
    config_path
        Path to config JSON file.
    modal_fn
        Modal function reference from select_modal_function().
    gpu
        GPU tier: debug (T4), standard (A10G), fast (A100).
    timeout
        Optional timeout in seconds.
    tags
        Comma-separated additional W&B tags.
    dry_run
        If True, print what would be done but don't execute.
    wait
        If True, wait for result instead of fire-and-forget.
    wandb_entity
        W&B entity (team/username) to log runs to.
    wandb_project
        W&B project name to log runs to.

    Returns
    -------
    tuple[bool, str]
        (success, message) tuple.
    """
    config_data = load_config(config_path)
    run_id = config_data.get("run_id", config_path.stem)

    if dry_run:
        return True, f"[dry-run] Would spawn {run_id}"

    # Get the actual config (might be nested under "config" key or be the whole dict)
    actual_config = config_data.get("config", config_data)

    # Inject _wandb_config if entity or project specified
    if wandb_entity or wandb_project:
        wandb_cfg = actual_config.get("_wandb_config", {})
        if wandb_entity:
            wandb_cfg["entity"] = wandb_entity
        if wandb_project:
            wandb_cfg["project"] = wandb_project
        actual_config["_wandb_config"] = wandb_cfg

    # Build task input dict
    task_dict: dict[str, Any] = {
        "config": actual_config,
        "run_id": run_id,
        "gpu_tier": gpu,
        "timeout_seconds": timeout or config_data.get("timeout_seconds", 3600),
        "additional_tags": [t.strip() for t in tags.split(",") if t.strip()],
    }

    try:
        if wait:
            result = modal_fn.remote(task_dict)
            status = result.get("status", "unknown")
            return status == "completed", f"Finished: {status}"
        else:
            fc = modal_fn.spawn(task_dict)
            return True, f"Spawned -> {fc.object_id}"
    except Exception as e:
        return False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch multiple experiments from a config directory."
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        type=Path,
        help="Directory containing config JSON files",
    )
    parser.add_argument(
        "--gpu",
        default="standard",
        choices=["debug", "standard", "fast"],
        help="GPU tier (default: standard)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_pattern",
        help="Filter configs by run_id substring",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout override in seconds",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated additional W&B tags",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for each experiment to complete (blocking)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between launches in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of experiments to launch",
    )
    parser.add_argument(
        "--wandb-entity",
        help="W&B entity (team/username) to log runs to",
    )
    parser.add_argument(
        "--wandb-project",
        help="W&B project name to log runs to",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already exist in W&B (requires --wandb-entity and --wandb-project)",
    )

    args = parser.parse_args()

    if not args.config_dir.is_dir():
        print(f"Error: {args.config_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Validate skip-existing requirements
    if args.skip_existing and not (args.wandb_entity and args.wandb_project):
        print(
            "Error: --skip-existing requires --wandb-entity and --wandb-project",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get config files
    configs = get_config_files(args.config_dir, args.filter_pattern)

    if not configs:
        print("No config files found matching criteria", file=sys.stderr)
        sys.exit(1)

    # Filter out existing runs if requested
    skipped_count = 0
    if args.skip_existing:
        print(
            f"Checking W&B for existing runs in {args.wandb_entity}/{args.wandb_project}..."
        )
        existing_runs = get_existing_run_names(args.wandb_entity, args.wandb_project)
        print(f"Found {len(existing_runs)} existing runs in W&B")

        original_count = len(configs)
        configs = [c for c in configs if c.stem not in existing_runs]
        skipped_count = original_count - len(configs)

        if skipped_count > 0:
            print(f"Skipping {skipped_count} already-completed experiments")

    if args.limit:
        configs = configs[: args.limit]

    print(f"Found {len(configs)} configs to launch")
    print(f"GPU tier: {args.gpu}")
    print(f"Wait mode: {args.wait}")

    if args.dry_run:
        print("\n[DRY RUN - no commands will be executed]")

    print()

    # Get Modal function once (avoids repeated lookups)
    modal_fn = select_modal_function(args.gpu) if not args.dry_run else None

    # Launch experiments
    success_count = 0
    fail_count = 0

    pbar = tqdm(configs, desc="Launching", unit="exp", ncols=80)
    for config_path in pbar:
        run_id = config_path.stem
        pbar.set_postfix_str(run_id[:30])

        success, message = launch_single(
            config_path=config_path,
            modal_fn=modal_fn,  # type: ignore[arg-type]
            gpu=args.gpu,
            timeout=args.timeout,
            tags=args.tags,
            dry_run=args.dry_run,
            wait=args.wait,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            tqdm.write(f"FAILED {run_id}: {message}", file=sys.stderr)

        # Small delay between spawns
        if not args.dry_run and args.delay > 0:
            time.sleep(args.delay)

    print()
    print(f"Summary: {success_count} launched, {fail_count} failed")


if __name__ == "__main__":
    main()
