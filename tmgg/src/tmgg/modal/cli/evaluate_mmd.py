"""Launch MMD evaluation on Modal for trained checkpoints.

Uses Function.from_name() to call the deployed Modal app directly.

Usage
-----
# Evaluate single run on all splits (last checkpoint only)
doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \
    --run-id digress_sbm_vanilla \
    --splits train val test

# Evaluate ALL checkpoints for a run
doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \
    --run-id digress_sbm_vanilla \
    --all-checkpoints

# Evaluate specific checkpoint
doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \
    --run-id digress_sbm_vanilla \
    --checkpoint /data/outputs/digress_sbm_vanilla/checkpoints/best.ckpt

# Fire-and-forget (non-blocking)
doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \
    --run-id digress_sbm_vanilla \
    --all-checkpoints \
    --no-wait
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import modal

MODAL_APP_NAME = "tmgg-spectral"


def select_modal_function(gpu_tier: str) -> modal.Function:
    """Select the appropriate deployed Modal function for the GPU tier."""
    if gpu_tier == "debug":
        func_name = "modal_evaluate_mmd_debug"
    elif gpu_tier in ("fast", "multi", "h100"):
        func_name = "modal_evaluate_mmd_fast"
    else:
        func_name = "modal_evaluate_mmd"
    return modal.Function.from_name(MODAL_APP_NAME, func_name)


def get_list_checkpoints_function() -> modal.Function:
    """Get the modal_list_checkpoints function."""
    return modal.Function.from_name(MODAL_APP_NAME, "modal_list_checkpoints")


def evaluate_single_checkpoint(
    modal_fn: modal.Function,
    task_dict: dict[str, Any],
    wait: bool,
    run_id: str,
    checkpoint_name: str,
) -> int:
    """Evaluate a single checkpoint and return exit code."""
    if wait:
        print(f"\nEvaluating {checkpoint_name}...")
        try:
            result = modal_fn.remote(task_dict)
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)
            return 1

        status = result.get("status", "unknown")
        ckpt_name = result.get("checkpoint_name", checkpoint_name)
        print(f"  Status: {status}")

        if status == "completed":
            results = result.get("results", {})
            for split, metrics in results.items():
                print(
                    f"    {split}: degree={metrics.get('degree_mmd', 0):.6f}, "
                    f"clustering={metrics.get('clustering_mmd', 0):.6f}, "
                    f"spectral={metrics.get('spectral_mmd', 0):.6f}"
                )
            print(f"  Saved to: /data/outputs/{run_id}/mmd_evaluation_{ckpt_name}.json")
            return 0
        else:
            error = result.get("error_message", "Unknown error")
            print(f"  Error: {error}", file=sys.stderr)
            return 1
    else:
        fc = modal_fn.spawn(task_dict)
        print(f"  Spawned {checkpoint_name} -> {fc.object_id}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch MMD evaluation on Modal for trained checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate last checkpoint on all splits
  doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \\
      --run-id digress_sbm_vanilla

  # Evaluate ALL checkpoints (last, best, epoch_*, etc.)
  doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \\
      --run-id digress_sbm_vanilla \\
      --all-checkpoints

  # Evaluate specific checkpoint
  doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \\
      --run-id digress_sbm_vanilla \\
      --checkpoint /data/outputs/digress_sbm_vanilla/checkpoints/best.ckpt

  # All checkpoints, fire-and-forget (spawns parallel evaluations)
  doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \\
      --run-id digress_sbm_vanilla \\
      --all-checkpoints \\
      --no-wait

  # Use debug GPU (T4) for quick tests
  doppler run -- uv run python -m tmgg.modal.cli.evaluate_mmd \\
      --run-id digress_sbm_vanilla \\
      --gpu debug \\
      --num-samples 100
""",
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID whose checkpoint(s) to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path (defaults to last.ckpt)",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate ALL checkpoints in the run (spawns separate evaluations)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to evaluate (default: train val test)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of graphs to generate per split (default: 500)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of denoising steps (default: 100)",
    )
    parser.add_argument(
        "--mmd-kernel",
        default="gaussian_tv",
        choices=["gaussian", "gaussian_tv"],
        help="Kernel for MMD computation (default: gaussian_tv)",
    )
    parser.add_argument(
        "--mmd-sigma",
        type=float,
        default=1.0,
        help="Kernel bandwidth (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        default="standard",
        choices=["debug", "standard", "fast"],
        help="GPU tier (default: standard)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        dest="wait",
        help="Wait for result (default)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_false",
        dest="wait",
        help="Fire-and-forget mode (return immediately)",
    )

    args = parser.parse_args()

    # Validate args
    if args.checkpoint and args.all_checkpoints:
        print(
            "Error: Cannot use both --checkpoint and --all-checkpoints", file=sys.stderr
        )
        return 1

    # Get deployed functions
    try:
        modal_fn = select_modal_function(args.gpu)
    except modal.exception.NotFoundError as e:
        print(
            f"Error: Modal app '{MODAL_APP_NAME}' not deployed or function not found.",
            file=sys.stderr,
        )
        print(
            "Run 'doppler run -- uv run modal deploy -m tmgg.modal.runner' first.",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        return 1

    print(f"Evaluating run: {args.run_id}")
    print(f"  Splits: {', '.join(args.splits)}")
    print(f"  Samples: {args.num_samples}, Steps: {args.num_steps}")
    print(f"  GPU tier: {args.gpu}")

    # Handle --all-checkpoints
    if args.all_checkpoints:
        print("\nDiscovering checkpoints...")
        try:
            list_fn = get_list_checkpoints_function()
            list_result = list_fn.remote(args.run_id)
        except modal.exception.NotFoundError:
            print(
                "Error: modal_list_checkpoints function not found. Redeploy the Modal app.",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            print(f"Error listing checkpoints: {e}", file=sys.stderr)
            return 1

        if list_result.get("status") != "completed":
            print(
                f"Error: {list_result.get('error_message', 'Unknown error')}",
                file=sys.stderr,
            )
            return 1

        checkpoints = list_result.get("checkpoints", [])
        if not checkpoints:
            print("No checkpoints found for this run.", file=sys.stderr)
            return 1

        print(f"Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            print(f"  - {ckpt['name']} ({ckpt['size_mb']:.1f} MB)")

        # Spawn/run evaluations for each checkpoint
        exit_code = 0
        for ckpt in checkpoints:
            task_dict: dict[str, Any] = {
                "run_id": args.run_id,
                "checkpoint_path": ckpt["path"],
                "splits": args.splits,
                "num_samples": args.num_samples,
                "num_steps": args.num_steps,
                "mmd_kernel": args.mmd_kernel,
                "mmd_sigma": args.mmd_sigma,
                "seed": args.seed,
            }
            result = evaluate_single_checkpoint(
                modal_fn, task_dict, args.wait, args.run_id, ckpt["name"]
            )
            if result != 0:
                exit_code = 1

        if not args.wait:
            print("\nAll evaluations spawned. Check Modal dashboard for progress.")

        return exit_code

    # Single checkpoint evaluation
    task_dict = {
        "run_id": args.run_id,
        "checkpoint_path": args.checkpoint,
        "splits": args.splits,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "mmd_kernel": args.mmd_kernel,
        "mmd_sigma": args.mmd_sigma,
        "seed": args.seed,
    }

    checkpoint_name = (
        "last"
        if not args.checkpoint
        else args.checkpoint.split("/")[-1].replace(".ckpt", "")
    )

    if args.wait:
        print("\nRunning evaluation (blocking)...")
        try:
            result = modal_fn.remote(task_dict)
        except Exception as e:
            print(f"Error during evaluation: {e}", file=sys.stderr)
            return 1

        status = result.get("status", "unknown")
        ckpt_name = result.get("checkpoint_name", checkpoint_name)
        print(f"\nStatus: {status}")
        print(f"Checkpoint: {ckpt_name}")

        if status == "completed":
            results = result.get("results", {})
            print("\nMMD Results:")
            print("-" * 50)
            for split, metrics in results.items():
                print(f"\n  {split}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.6f}")
            print("-" * 50)
            print(
                f"\nResults saved to: /data/outputs/{args.run_id}/mmd_evaluation_{ckpt_name}.json"
            )
            return 0
        else:
            error = result.get("error_message", "Unknown error")
            print(f"\nError: {error}", file=sys.stderr)
            return 1
    else:
        fc = modal_fn.spawn(task_dict)
        print(f"\nSpawned evaluation -> {fc.object_id}")
        print("Use Modal dashboard or 'modal function list' to check status.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
