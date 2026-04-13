"""CLI for discrete diffusion checkpoint evaluation.

Loads a trained checkpoint, samples graphs, and reports MMD metrics
against a reference distribution. Separated from ``evaluate`` to avoid
an import cycle with ``lightning_module``.

Only DiffusionModule checkpoints are supported. Legacy checkpoints from
earlier LightningModule implementations are incompatible.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch

from tmgg.evaluation.mmd_metrics import (
    adjacency_to_networkx,
    compute_mmd_metrics,
)
from tmgg.evaluation.reference_graphs import (
    generate_reference_graphs,
)
from tmgg.models.base import GraphModel
from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
)


def _instantiate_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: str,
) -> GraphModel:
    """Reconstruct the nested graph model from checkpoint metadata.

    ``DiffusionModule`` intentionally excludes the graph model from
    ``save_hyperparameters`` because the instantiated module object is not a
    stable constructor argument. The checkpoint still stores
    ``model_class`` and ``model_config`` in the hyperparameters, which is the
    canonical source for rebuilding that nested model at load time.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise TypeError(
            f"Expected checkpoint hyperparameters dict in {checkpoint_path}, "
            f"got {type(hparams).__name__}"
        )

    model_class_path = hparams.get("model_class")
    model_config = hparams.get("model_config")
    if not isinstance(model_class_path, str):
        raise TypeError(
            f"Checkpoint {checkpoint_path} is missing string model_class metadata"
        )
    if not isinstance(model_config, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} is missing dict model_config metadata"
        )

    module_path, class_name = model_class_path.rsplit(".", 1)
    model_module = importlib.import_module(module_path)
    model_class = getattr(model_module, class_name)
    if not isinstance(model_class, type) or not issubclass(model_class, GraphModel):
        raise TypeError(
            f"Checkpoint model class {model_class_path!r} is not a GraphModel subclass"
        )

    model = model_class(**model_config)
    return model


def _load_diffusion_module(
    checkpoint_path: Path,
    *,
    device: str,
) -> DiffusionModule:
    """Load a diffusion checkpoint, rebuilding the nested model if needed."""
    try:
        return DiffusionModule.load_from_checkpoint(
            str(checkpoint_path), map_location=device
        )
    except TypeError as exc:
        if "missing 1 required keyword-only argument: 'model'" not in str(exc):
            raise

    # Newer checkpoints persist model metadata but not the instantiated model.
    # Rebuild the nested GraphModel explicitly and retry the Lightning load.
    model = _instantiate_checkpoint_model(checkpoint_path, device=device)
    return DiffusionModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        model=model,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_type: str = "sbm",
    num_samples: int = 500,
    num_nodes: int = 20,
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    mmd_sigma: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
    **dataset_kwargs: Any,
) -> dict[str, Any]:
    """Load a discrete diffusion checkpoint, sample graphs, and compute MMD.

    Parameters
    ----------
    checkpoint_path
        Path to a Lightning checkpoint (``.ckpt``).
    dataset_type
        Reference dataset type (``"sbm"``, ``"erdos_renyi"``, etc.).
    num_samples
        Number of graphs to generate and compare.
    num_nodes
        Nodes per graph.
    mmd_kernel
        Kernel for MMD computation.
    mmd_sigma
        Kernel bandwidth.
    device
        Torch device.
    seed
        Random seed for reference graph generation.
    **dataset_kwargs
        Forwarded to ``generate_reference_graphs``.

    Returns
    -------
    dict
        Evaluation results including MMD metrics.
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'=' * 60}")

    # Load model from checkpoint (save_hyperparameters stores constructor args)
    # Only DiffusionModule checkpoints are supported.
    print("\nLoading model...")
    module = _load_diffusion_module(checkpoint_path, device=device)
    module = module.to(device)
    module.eval()

    # Generate reference graphs
    print(f"\nGenerating {num_samples} reference graphs ({dataset_type})...")
    ref_adjacencies = generate_reference_graphs(
        dataset_type=dataset_type,
        num_graphs=num_samples,
        num_nodes=num_nodes,
        seed=seed,
        **dataset_kwargs,
    )
    ref_graphs = [adjacency_to_networkx(adj) for adj in ref_adjacencies]

    # Sample from model using the sampler
    print(f"Sampling {num_samples} graphs from model...")
    with torch.no_grad():
        generated_graph_data = module.generate_graphs(num_samples)

    mmd_results = compute_mmd_metrics(
        ref_graphs, generated_graph_data, kernel=mmd_kernel, sigma=mmd_sigma
    )

    results: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.absolute()),
        "checkpoint_name": checkpoint_path.name,
        "dataset_type": dataset_type,
        "num_generated": num_samples,
        "num_reference": num_samples,
        "num_nodes": num_nodes,
        "mmd_kernel": mmd_kernel,
        "mmd_sigma": mmd_sigma,
        "mmd_results": mmd_results.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
    }

    print(f"\n{'=' * 60}")
    print("MMD Results")
    print(f"{'=' * 60}")
    print(f"  Degree MMD:     {mmd_results.degree_mmd:.6f}")
    print(f"  Clustering MMD: {mmd_results.clustering_mmd:.6f}")
    print(f"  Spectral MMD:   {mmd_results.spectral_mmd:.6f}")
    print(f"{'=' * 60}\n")

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for discrete diffusion checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a discrete diffusion checkpoint with MMD metrics",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--dataset",
        default="sbm",
        choices=["sbm", "erdos_renyi", "er", "watts_strogatz", "ws", "regular", "tree"],
        help="Reference dataset type (default: sbm)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Graphs to generate/compare"
    )
    parser.add_argument("--num-nodes", type=int, default=20, help="Nodes per graph")
    parser.add_argument(
        "--kernel",
        default="gaussian_tv",
        choices=["gaussian", "gaussian_tv"],
        help="MMD kernel (default: gaussian_tv)",
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="Kernel bandwidth")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Write results to JSON file"
    )

    args = parser.parse_args(argv)

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_type=args.dataset,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        mmd_kernel=args.kernel,
        mmd_sigma=args.sigma,
        device=args.device,
        seed=args.seed,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
