"""CLI for evaluating DiGress checkpoints with MMD metrics.

Loads a pretrained checkpoint (from any supported format), generates graphs,
and computes MMD metrics against a reference dataset.

Usage
-----
```bash
uv run python -m tmgg.experiments.generative.evaluate_checkpoint \
    --checkpoint path/to/checkpoint.ckpt \
    --dataset sbm \
    --num-samples 500 \
    --num-nodes 20 \
    --output results.json
```
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch

from tmgg.experiment_utils.digress_checkpoint_compat import (
    CheckpointFormat,
    load_digress_checkpoint,
)
from tmgg.experiment_utils.mmd_metrics import (
    MMDResults,
    adjacency_to_networkx,
    compute_mmd_metrics,
)
from tmgg.experiments.digress_denoising.lightning_module import (
    DigressDenoisingLightningModule,
)


def generate_reference_graphs(
    dataset_type: str,
    num_graphs: int,
    num_nodes: int,
    seed: int = 42,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Generate reference graphs from a synthetic dataset.

    Parameters
    ----------
    dataset_type
        Type of dataset. Supported: "sbm", "erdos_renyi", "watts_strogatz",
        "regular", "tree".
    num_graphs
        Number of graphs to generate.
    num_nodes
        Number of nodes per graph.
    seed
        Random seed.
    **kwargs
        Additional parameters for the dataset generator.

    Returns
    -------
    list[torch.Tensor]
        List of adjacency matrices.
    """
    import numpy as np

    from tmgg.experiment_utils.data.sbm import generate_sbm_adjacency
    from tmgg.experiment_utils.data.synthetic_graphs import SyntheticGraphDataset

    if dataset_type == "sbm":
        # SBM with default parameters
        p = kwargs.get("p", 0.5)
        q = kwargs.get("q", 0.1)
        num_blocks = kwargs.get("num_blocks", 2)

        rng = np.random.default_rng(seed)
        adjacencies = []
        for _ in range(num_graphs):
            # Equal-sized blocks
            block_size = num_nodes // num_blocks
            remainder = num_nodes % num_blocks
            block_sizes = [block_size] * num_blocks
            for j in range(remainder):
                block_sizes[j] += 1

            adj = generate_sbm_adjacency(block_sizes, p, q, rng)
            # Zero diagonal and make symmetric
            np.fill_diagonal(adj, 0)
            adj = (adj + adj.T) / 2
            adj = (adj > 0.5).astype(np.float32)
            adjacencies.append(torch.from_numpy(adj))
        return adjacencies

    # Use SyntheticGraphDataset for other types
    type_map = {
        "erdos_renyi": "erdos_renyi",
        "er": "erdos_renyi",
        "watts_strogatz": "watts_strogatz",
        "ws": "watts_strogatz",
        "regular": "regular",
        "tree": "tree",
    }

    if dataset_type not in type_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset = SyntheticGraphDataset(
        graph_type=type_map[dataset_type],
        n=num_nodes,
        num_graphs=num_graphs,
        seed=seed,
        **kwargs,
    )

    return [torch.from_numpy(dataset[i]).float() for i in range(len(dataset))]


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> DigressDenoisingLightningModule:
    """Load a DigressDenoisingLightningModule from checkpoint.

    Handles automatic format detection and state dict remapping.

    Parameters
    ----------
    checkpoint_path
        Path to checkpoint file.
    device
        Device to load model on.

    Returns
    -------
    DigressDenoisingLightningModule
        Loaded model ready for inference.
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint with format detection
    loaded = load_digress_checkpoint(
        checkpoint_path,
        target_format=CheckpointFormat.TMGG_LIGHTNING,
        map_location=device,
    )

    print(f"Detected checkpoint format: {loaded.original_format.name}")
    print(f"Remapped to: {loaded.target_format.name}")

    # Extract hyperparameters
    hparams = loaded.hyper_parameters

    # Create model with hyperparameters
    # Filter to only accepted parameters
    accepted_params = {
        "use_eigenvectors",
        "k",
        "n_layers",
        "hidden_mlp_dims",
        "hidden_dims",
        "output_dims",
        "learning_rate",
        "weight_decay",
        "optimizer_type",
        "amsgrad",
        "loss_type",
        "scheduler_config",
        "noise_levels",
        "noise_type",
        "rotation_k",
        "seed",
        "node_feature_dim",  # Legacy
    }

    filtered_hparams = {k: v for k, v in hparams.items() if k in accepted_params}

    # Create module
    module = DigressDenoisingLightningModule(**filtered_hparams)

    # Load state dict
    module.load_state_dict(loaded.state_dict, strict=False)

    module = module.to(device)
    module.eval()

    return module


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_type: str,
    num_samples: int = 500,
    num_nodes: int = 20,
    num_steps: int = 100,
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    mmd_sigma: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
    **dataset_kwargs: Any,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a synthetic dataset.

    Parameters
    ----------
    checkpoint_path
        Path to checkpoint file.
    dataset_type
        Type of reference dataset.
    num_samples
        Number of graphs to generate/compare.
    num_nodes
        Number of nodes per graph.
    num_steps
        Number of denoising steps for sampling.
    mmd_kernel
        Kernel for MMD computation.
    mmd_sigma
        Bandwidth for kernel.
    device
        Device for computation.
    seed
        Random seed.
    **dataset_kwargs
        Additional parameters for the dataset generator.

    Returns
    -------
    dict
        Evaluation results including MMD metrics.
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'='*60}")

    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Generate reference graphs
    print(f"\nGenerating {num_samples} reference graphs ({dataset_type})...")
    ref_graphs = generate_reference_graphs(
        dataset_type=dataset_type,
        num_graphs=num_samples,
        num_nodes=num_nodes,
        seed=seed,
        **dataset_kwargs,
    )

    # Generate graphs from model
    print(f"Sampling {num_samples} graphs from model ({num_steps} steps)...")
    with torch.no_grad():
        generated = model.sample(
            num_graphs=num_samples,
            num_nodes=num_nodes,
            num_steps=num_steps,
        )

    # Convert to NetworkX for MMD computation
    print("Computing MMD metrics...")
    ref_nx = [adjacency_to_networkx(g) for g in ref_graphs]
    gen_nx = [adjacency_to_networkx(g) for g in generated]

    mmd_results: MMDResults = compute_mmd_metrics(
        ref_nx,
        gen_nx,
        kernel=mmd_kernel,
        sigma=mmd_sigma,
    )

    # Build results
    results = {
        "checkpoint_path": str(checkpoint_path.absolute()),
        "checkpoint_name": checkpoint_path.name,
        "dataset_type": dataset_type,
        "num_generated": num_samples,
        "num_reference": num_samples,
        "num_nodes": num_nodes,
        "num_steps": num_steps,
        "mmd_kernel": mmd_kernel,
        "mmd_sigma": mmd_sigma,
        "mmd_results": mmd_results.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
    }

    # Print results
    print(f"\n{'='*60}")
    print("MMD Results")
    print(f"{'='*60}")
    print(f"  Degree MMD:     {mmd_results.degree_mmd:.6f}")
    print(f"  Clustering MMD: {mmd_results.clustering_mmd:.6f}")
    print(f"  Spectral MMD:   {mmd_results.spectral_mmd:.6f}")
    print(f"{'='*60}\n")

    return results


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate DiGress checkpoint with MMD metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on SBM dataset
  python -m tmgg.experiments.generative.evaluate_checkpoint \\
      --checkpoint outputs/run1/checkpoints/last.ckpt \\
      --dataset sbm --num-samples 500 --num-nodes 20

  # Evaluate with custom parameters and save to JSON
  python -m tmgg.experiments.generative.evaluate_checkpoint \\
      --checkpoint model.ckpt --dataset erdos_renyi \\
      --num-samples 1000 --output results.json

  # Evaluate on GPU
  python -m tmgg.experiments.generative.evaluate_checkpoint \\
      --checkpoint model.ckpt --dataset sbm --device cuda:0
""",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sbm", "erdos_renyi", "er", "watts_strogatz", "ws", "regular", "tree"],
        help="Type of reference dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of graphs to generate/compare (default: 500)",
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
        help="Number of denoising steps (default: 100)",
    )
    parser.add_argument(
        "--mmd-kernel",
        type=str,
        default="gaussian_tv",
        choices=["gaussian", "gaussian_tv"],
        help="Kernel for MMD (default: gaussian_tv)",
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
        help="Device for computation (default: cpu)",
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
        default=None,
        help="Output JSON file (optional)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project for logging (optional)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)",
    )

    # Dataset-specific parameters
    parser.add_argument(
        "--sbm-p",
        type=float,
        default=0.5,
        help="SBM intra-block edge probability (default: 0.5)",
    )
    parser.add_argument(
        "--sbm-q",
        type=float,
        default=0.1,
        help="SBM inter-block edge probability (default: 0.1)",
    )
    parser.add_argument(
        "--sbm-num-blocks",
        type=int,
        default=2,
        help="SBM number of blocks (default: 2)",
    )
    parser.add_argument(
        "--er-p",
        type=float,
        default=0.1,
        help="Erdos-Renyi edge probability (default: 0.1)",
    )
    parser.add_argument(
        "--ws-k",
        type=int,
        default=4,
        help="Watts-Strogatz k parameter (default: 4)",
    )
    parser.add_argument(
        "--ws-p",
        type=float,
        default=0.3,
        help="Watts-Strogatz rewiring probability (default: 0.3)",
    )
    parser.add_argument(
        "--regular-d",
        type=int,
        default=3,
        help="Regular graph degree (default: 3)",
    )

    args = parser.parse_args()

    # Build dataset kwargs based on type
    dataset_kwargs: dict[str, Any] = {}
    if args.dataset == "sbm":
        dataset_kwargs = {
            "p": args.sbm_p,
            "q": args.sbm_q,
            "num_blocks": args.sbm_num_blocks,
        }
    elif args.dataset in ("erdos_renyi", "er"):
        dataset_kwargs = {"p": args.er_p}
    elif args.dataset in ("watts_strogatz", "ws"):
        dataset_kwargs = {"k": args.ws_k, "p": args.ws_p}
    elif args.dataset == "regular":
        dataset_kwargs = {"d": args.regular_d}

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_type=args.dataset,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        num_steps=args.num_steps,
        mmd_kernel=args.mmd_kernel,
        mmd_sigma=args.mmd_sigma,
        device=args.device,
        seed=args.seed,
        **dataset_kwargs,
    )

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Log to W&B if requested
    if args.wandb_project:
        try:
            import wandb

            run_name = args.wandb_run_name or f"eval_{Path(args.checkpoint).stem}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "dataset": args.dataset,
                    "num_samples": args.num_samples,
                    "num_nodes": args.num_nodes,
                    "num_steps": args.num_steps,
                    **dataset_kwargs,
                },
            )
            wandb.log(results["mmd_results"])
            wandb.finish()
            print(f"Results logged to W&B project: {args.wandb_project}")
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")

    return 0


if __name__ == "__main__":
    sys.exit(main())
