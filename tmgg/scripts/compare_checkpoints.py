#!/usr/bin/env python
"""Compare multiple checkpoints on multiple datasets with MMD metrics.

Produces a comparison table (CSV) and optional visualizations showing
performance differences between checkpoints across datasets.

Usage
-----
```bash
python scripts/compare_checkpoints.py \
    --checkpoints baseline.ckpt tmgg.ckpt \
    --datasets sbm erdos_renyi watts_strogatz \
    --output comparison.csv
```
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


def run_comparison(
    checkpoint_paths: list[str],
    datasets: list[str],
    num_samples: int = 500,
    num_nodes: int = 20,
    num_steps: int = 100,
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    mmd_sigma: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run comparison across checkpoints and datasets.

    Parameters
    ----------
    checkpoint_paths
        List of checkpoint file paths.
    datasets
        List of dataset types to evaluate on.
    num_samples
        Number of graphs per evaluation.
    num_nodes
        Number of nodes per graph.
    num_steps
        Denoising steps for sampling.
    mmd_kernel
        Kernel for MMD computation.
    mmd_sigma
        Kernel bandwidth.
    device
        Computation device.
    seed
        Random seed.

    Returns
    -------
    list[dict]
        Results for each checkpoint-dataset combination.
    """
    from tmgg.experiments.generative.evaluate_checkpoint import evaluate_checkpoint

    results: list[dict[str, Any]] = []
    total_evals = len(checkpoint_paths) * len(datasets)
    current = 0

    for checkpoint_path in checkpoint_paths:
        for dataset in datasets:
            current += 1
            print(
                f"\n[{current}/{total_evals}] Evaluating {Path(checkpoint_path).name} on {dataset}"
            )

            try:
                eval_result = evaluate_checkpoint(
                    checkpoint_path=checkpoint_path,
                    dataset_type=dataset,
                    num_samples=num_samples,
                    num_nodes=num_nodes,
                    num_steps=num_steps,
                    mmd_kernel=mmd_kernel,
                    mmd_sigma=mmd_sigma,
                    device=device,
                    seed=seed,
                )
                results.append(eval_result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_name": Path(checkpoint_path).name,
                        "dataset_type": dataset,
                        "error": str(e),
                        "mmd_results": {
                            "degree_mmd": float("nan"),
                            "clustering_mmd": float("nan"),
                            "spectral_mmd": float("nan"),
                        },
                    }
                )

    return results


def results_to_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write results to CSV file.

    Parameters
    ----------
    results
        List of evaluation results.
    output_path
        Output CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint",
        "dataset",
        "degree_mmd",
        "clustering_mmd",
        "spectral_mmd",
        "num_samples",
        "num_nodes",
        "num_steps",
        "error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            mmd = result.get("mmd_results", {})
            row = {
                "checkpoint": result.get("checkpoint_name", ""),
                "dataset": result.get("dataset_type", ""),
                "degree_mmd": mmd.get("degree_mmd", ""),
                "clustering_mmd": mmd.get("clustering_mmd", ""),
                "spectral_mmd": mmd.get("spectral_mmd", ""),
                "num_samples": result.get("num_generated", ""),
                "num_nodes": result.get("num_nodes", ""),
                "num_steps": result.get("num_steps", ""),
                "error": result.get("error", ""),
            }
            writer.writerow(row)


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print results as a formatted comparison table.

    Parameters
    ----------
    results
        List of evaluation results.
    """
    # Group by dataset
    datasets = sorted(set(r.get("dataset_type", "") for r in results))
    checkpoints = sorted(set(r.get("checkpoint_name", "") for r in results))

    # Create lookup table
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        key = (r.get("checkpoint_name", ""), r.get("dataset_type", ""))
        lookup[key] = r.get("mmd_results", {})

    # Print header
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Print per-metric tables
    metrics = [
        ("degree_mmd", "Degree MMD"),
        ("clustering_mmd", "Clustering MMD"),
        ("spectral_mmd", "Spectral MMD"),
    ]

    for metric_key, metric_name in metrics:
        print(f"\n{metric_name}:")
        print("-" * 60)

        # Header row
        header = f"{'Checkpoint':<30}"
        for ds in datasets:
            header += f" {ds:>12}"
        print(header)
        print("-" * 60)

        # Data rows
        for ckpt in checkpoints:
            row = f"{ckpt[:30]:<30}"
            for ds in datasets:
                mmd = lookup.get((ckpt, ds), {})
                val = mmd.get(metric_key, float("nan"))
                if isinstance(val, float) and val == val:  # not nan
                    row += f" {val:>12.6f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    print("\n" + "=" * 80)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare DiGress checkpoints across datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two checkpoints on SBM and Erdos-Renyi
  python scripts/compare_checkpoints.py \\
      --checkpoints baseline.ckpt tmgg.ckpt \\
      --datasets sbm erdos_renyi \\
      --output comparison.csv

  # Full comparison with multiple datasets
  python scripts/compare_checkpoints.py \\
      --checkpoints outputs/run1/checkpoints/last.ckpt \\
                    outputs/run2/checkpoints/last.ckpt \\
      --datasets sbm erdos_renyi watts_strogatz regular tree \\
      --num-samples 1000 --output results/comparison.csv
""",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint files to compare",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        choices=["sbm", "erdos_renyi", "er", "watts_strogatz", "ws", "regular", "tree"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of graphs per evaluation (default: 500)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20,
        help="Number of nodes per graph (default: 20)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Denoising steps (default: 100)",
    )
    parser.add_argument(
        "--mmd-kernel",
        type=str,
        default="gaussian_tv",
        choices=["gaussian", "gaussian_tv"],
        help="MMD kernel (default: gaussian_tv)",
    )
    parser.add_argument(
        "--mmd-sigma",
        type=float,
        default=1.0,
        help="Kernel bandwidth (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Also save detailed results as JSON (optional)",
    )

    args = parser.parse_args()

    # Validate checkpoint files exist
    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"Error: Checkpoint not found: {ckpt}")
            return 1

    # Run comparison
    results = run_comparison(
        checkpoint_paths=args.checkpoints,
        datasets=args.datasets,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        num_steps=args.num_steps,
        mmd_kernel=args.mmd_kernel,
        mmd_sigma=args.mmd_sigma,
        device=args.device,
        seed=args.seed,
    )

    # Print table
    print_comparison_table(results)

    # Save CSV
    output_path = Path(args.output)
    results_to_csv(results, output_path)
    print(f"\nResults saved to: {output_path}")

    # Save JSON if requested
    if args.json:
        import json

        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "comparison_timestamp": datetime.now().isoformat(),
                    "config": {
                        "checkpoints": args.checkpoints,
                        "datasets": args.datasets,
                        "num_samples": args.num_samples,
                        "num_nodes": args.num_nodes,
                        "num_steps": args.num_steps,
                        "mmd_kernel": args.mmd_kernel,
                        "mmd_sigma": args.mmd_sigma,
                        "seed": args.seed,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Detailed JSON saved to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
