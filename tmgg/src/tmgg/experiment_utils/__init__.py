"""Shared utilities for graph denoising experiments."""

from .data import (
    generate_sbm_adjacency,
    generate_block_sizes,
    add_gaussian_noise,
    add_rotation_noise,
    add_digress_noise,
    random_skew_symmetric_matrix,
    AdjacencyMatrixDataset,
    PermutedAdjacencyDataset,
    GraphDataset,
)
from .metrics import (
    compute_eigenvalue_error,
    compute_subspace_distance,
    compute_reconstruction_metrics,
)
from .plotting import (
    plot_training_curves,
    plot_denoising_results,
    plot_noise_level_comparison,
    plot_eigenvalue_comparison,
    plot_eigenvalue_denoising,
    create_wandb_visualization,
)

__all__ = [
    # Data utilities
    "generate_sbm_adjacency",
    "generate_block_sizes", 
    "add_gaussian_noise",
    "add_rotation_noise",
    "add_digress_noise",
    "random_skew_symmetric_matrix",
    "AdjacencyMatrixDataset",
    "PermutedAdjacencyDataset",
    "GraphDataset",
    # Metrics
    "compute_eigenvalue_error",
    "compute_subspace_distance", 
    "compute_reconstruction_metrics",
    # Plotting
    "plot_training_curves",
    "plot_denoising_results",
    "plot_noise_level_comparison",
    "plot_eigenvalue_comparison",
    "plot_eigenvalue_denoising",
    "create_wandb_visualization",
]