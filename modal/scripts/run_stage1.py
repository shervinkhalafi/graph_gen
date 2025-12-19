#!/usr/bin/env python
"""CLI script to run Stage 1 experiments on Modal.

Usage
-----
# Run with default settings
uv run python scripts/run_stage1.py

# Run with custom parallelism and GPU
uv run python scripts/run_stage1.py --parallelism 8 --gpu fast

# Dry run to see configurations
uv run python scripts/run_stage1.py --dry-run

# Using Modal directly
modal run scripts/run_stage1.py
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1: Proof of Concept experiments on Modal"
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Maximum concurrent experiments (default: 4)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="standard",
        choices=["debug", "standard", "fast", "multi", "h100"],
        help="GPU tier (default: standard)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on Modal (for testing)",
    )

    args = parser.parse_args()

    if args.local:
        # Local execution for testing
        from tmgg_modal.stages.stage1 import generate_stage1_configs  # pyright: ignore[reportImplicitRelativeImport]

        configs = generate_stage1_configs()
        print(f"Generated {len(configs)} configurations for Stage 1")
        if args.dry_run:
            for cfg in configs[:5]:
                print(f"  {cfg.get('run_id')}")
            if len(configs) > 5:
                print(f"  ... and {len(configs) - 5} more")
        return

    # Modal execution
    from tmgg_modal.stages.stage1 import run_stage1  # pyright: ignore[reportImplicitRelativeImport]

    print(
        f"Starting Stage 1 on Modal with parallelism={args.parallelism}, gpu={args.gpu}"
    )
    result = run_stage1.remote(
        parallelism=args.parallelism,
        gpu_type=args.gpu,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("Stage 1 Results")
    print("=" * 60)
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
