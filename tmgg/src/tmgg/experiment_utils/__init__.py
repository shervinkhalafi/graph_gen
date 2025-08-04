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
    NoiseGenerator,
    GaussianNoiseGenerator,
    DigressNoiseGenerator,
    RotationNoiseGenerator,
    create_noise_generator,
    compute_eigendecomposition,
)
from .metrics import (
    compute_eigenvalue_error,
    compute_subspace_distance,
    compute_reconstruction_metrics,
    compute_batch_metrics,
)
from .plotting import (
    plot_graph_denoising_comparison,
    create_graph_denoising_wandb_image,
    create_graph_denoising_figure,
    plot_training_curves,
    plot_denoising_results,
    plot_noise_level_comparison,
    plot_eigenvalue_comparison,
    plot_eigenvalue_denoising,
    create_wandb_visualization,
)
from .sanity_check import (
    SanityCheckResult,
    check_noise_generator,
    check_model_forward_pass,
    check_data_loader,
    check_loss_computation,
    run_experiment_sanity_check,
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
    "compute_eigendecomposition",
    # Noise generators
    "NoiseGenerator",
    "GaussianNoiseGenerator", 
    "DigressNoiseGenerator",
    "RotationNoiseGenerator",
    "create_noise_generator",
    # Metrics
    "compute_eigenvalue_error",
    "compute_subspace_distance",
    "compute_reconstruction_metrics",
    "compute_batch_metrics",
    # Plotting
    "plot_graph_denoising_comparison",
    "create_graph_denoising_wandb_image",
    "create_graph_denoising_figure",
    "plot_training_curves",
    "plot_denoising_results",
    "plot_noise_level_comparison",
    "plot_eigenvalue_comparison",
    "plot_eigenvalue_denoising",
    "create_wandb_visualization",
    # Sanity checks
    "SanityCheckResult",
    "check_noise_generator",
    "check_model_forward_pass",
    "check_data_loader",
    "check_loss_computation",
    "run_experiment_sanity_check",
]

