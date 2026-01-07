"""Shared utilities for graph denoising experiments."""

from .checkpoint_utils import load_checkpoint_with_fallback
from .data import (
    AdjacencyMatrixDataset,
    DigressNoiseGenerator,
    GaussianNoiseGenerator,
    GraphDataset,
    NoiseGenerator,
    PermutedAdjacencyDataset,
    RotationNoiseGenerator,
    add_digress_noise,
    add_gaussian_noise,
    add_rotation_noise,
    compute_eigendecomposition,
    create_noise_generator,
    generate_block_sizes,
    generate_sbm_adjacency,
    random_skew_symmetric_matrix,
)
from .debug_callback import DebugCallback
from .exceptions import (
    CheckpointMismatchError,
    ConfigurationError,
    DataModuleStateError,
    ExperimentUtilsError,
)
from .metrics import (
    compute_accuracy,
    compute_batch_metrics,
    compute_eigenvalue_error,
    compute_reconstruction_metrics,
    compute_subspace_distance,
)
from .plotting import (
    create_graph_denoising_figure,
    create_graph_denoising_wandb_image,
    create_network_denoising_figure,
    create_wandb_visualization,
    plot_denoising_results,
    plot_eigenvalue_comparison,
    plot_eigenvalue_denoising,
    plot_graph_denoising_combined,
    plot_graph_denoising_comparison,
    plot_graph_network_comparison,
    plot_noise_level_comparison,
    plot_training_curves,
)
from .sanity_check import (
    SanityCheckResult,
    check_data_loader,
    check_loss_computation,
    check_model_forward_pass,
    check_noise_generator,
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
    "compute_accuracy",
    "compute_eigenvalue_error",
    "compute_subspace_distance",
    "compute_reconstruction_metrics",
    "compute_batch_metrics",
    # Plotting
    "plot_graph_denoising_comparison",
    "plot_graph_network_comparison",
    "plot_graph_denoising_combined",
    "create_graph_denoising_wandb_image",
    "create_graph_denoising_figure",
    "create_network_denoising_figure",
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
    # Debug callback
    "DebugCallback",
    # Checkpoint utilities
    "load_checkpoint_with_fallback",
    # Exceptions
    "ExperimentUtilsError",
    "ConfigurationError",
    "CheckpointMismatchError",
    "DataModuleStateError",
]
