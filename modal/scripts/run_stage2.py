#!/usr/bin/env python
"""CLI script to run Stage 2 experiments on Modal.

Usage
-----
# Run with default settings (uses Stage 1 best config)
uv run python scripts/run_stage2.py

# Run without using Stage 1 results
uv run python scripts/run_stage2.py --no-stage1-best

# Run with custom parallelism and GPU
uv run python scripts/run_stage2.py --parallelism 8 --gpu fast

# Dry run to see configurations
uv run python scripts/run_stage2.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 2: Core Validation experiments on Modal"
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
        "--no-stage1-best",
        action="store_true",
        help="Don't use Stage 1 best config to narrow search",
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
        from tmgg_modal.stages.stage2 import generate_stage2_configs  # pyright: ignore[reportImplicitRelativeImport]

        configs = generate_stage2_configs()
        print(f"Generated {len(configs)} configurations for Stage 2")
        if args.dry_run:
            for cfg in configs[:5]:
                print(f"  {cfg.get('run_id')}")
            if len(configs) > 5:
                print(f"  ... and {len(configs) - 5} more")
        return

    # Modal execution
    from tmgg_modal.stages.stage2 import run_stage2  # pyright: ignore[reportImplicitRelativeImport]

    print(
        f"Starting Stage 2 on Modal with parallelism={args.parallelism}, gpu={args.gpu}"
    )
    result = run_stage2.remote(
        parallelism=args.parallelism,
        gpu_type=args.gpu,
        use_stage1_best=not args.no_stage1_best,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("Stage 2 Results")
    print("=" * 60)
    for key, value in result.items():
        if key == "best_by_architecture":
            print(f"  {key}:")
            for arch, data in value.items():
                print(f"    {arch}: {data}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
