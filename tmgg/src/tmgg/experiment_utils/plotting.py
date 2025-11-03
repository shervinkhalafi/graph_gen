"""Plotting utilities for graph denoising experiments."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def plot_graph_denoising_comparison(
    A_clean: Union[np.ndarray, torch.Tensor],
    noise_fn: Callable[
        [Union[np.ndarray, torch.Tensor], float],
        Tuple[Union[np.ndarray, torch.Tensor], Any, Any],
    ],
    denoise_fn: Callable[
        [Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]
    ],
    noise_level: float,
    noise_type: str = "Unknown",
    title_prefix: str = "",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (20, 5),
    save_path: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    General-purpose visualization for graph denoising experiments.

    This function creates a 4-panel visualization showing:
    1. Clean graph
    2. Noisy graph
    3. Denoised graph
    4. Delta (Denoised - Clean)

    Args:
        A_clean: Clean adjacency matrix (numpy array or torch tensor)
        noise_fn: Function that adds noise. Should accept (matrix, noise_level) and
                 return (noisy_matrix, noise_info1, noise_info2)
        denoise_fn: Function that denoises. Should accept noisy matrix and return denoised matrix
        noise_level: Noise level parameter (e.g., epsilon)
        noise_type: Name of the noise type for labeling
        title_prefix: Optional prefix for the plot title
        cmap: Colormap to use for visualization
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        vmin: Minimum value for color scale (if None, auto-determined)
        vmax: Maximum value for color scale (if None, auto-determined)

    Returns:
        Matplotlib figure object

    Example:
        ```python
        # For GNN experiments
        fig = plot_graph_denoising_comparison(
            A_clean=clean_adjacency,
            noise_fn=add_gaussian_noise,
            denoise_fn=lambda A: model(A.unsqueeze(0)).squeeze(0),
            noise_level=0.1,
            noise_type="Gaussian"
        )

        # For DiGress experiments with edge flipping
        fig = plot_graph_denoising_comparison(
            A_clean=clean_adjacency,
            noise_fn=add_digress_noise,
            denoise_fn=lambda A: digress_model.denoise(A),
            noise_level=0.3,
            noise_type="Edge Flipping (DiGress)"
        )
        ```
    """
    # Convert to numpy if needed
    if isinstance(A_clean, torch.Tensor):
        A_clean_np = A_clean.detach().cpu().numpy()
    else:
        A_clean_np = A_clean

    # Add noise
    A_noisy = noise_fn(A_clean, noise_level)
    if isinstance(A_noisy, torch.Tensor):
        A_noisy_np = A_noisy.detach().cpu().numpy()
    else:
        A_noisy_np = A_noisy

    # Denoise
    A_denoised = denoise_fn(A_noisy)
    if isinstance(A_denoised, torch.Tensor):
        A_denoised_np = A_denoised.detach().cpu().numpy()
    else:
        A_denoised_np = A_denoised

    # Handle batch dimensions if present
    while A_clean_np.ndim > 2:
        A_clean_np = A_clean_np[0]
    while A_noisy_np.ndim > 2:
        A_noisy_np = A_noisy_np[0]
    while A_denoised_np.ndim > 2:
        A_denoised_np = A_denoised_np[0]

    # Calculate the difference
    A_delta_np = A_denoised_np - A_clean_np

    # Create figure with 4 subplots arranged horizontally
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Determine color scale if not provided
    if vmin is None:
        vmin = min(A_clean_np.min(), A_noisy_np.min(), A_denoised_np.min())
    if vmax is None:
        vmax = max(A_clean_np.max(), A_noisy_np.max(), A_denoised_np.max())

    # Plot clean graph (left)
    im1 = axes[0].imshow(A_clean_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title(
        f"{title_prefix}Clean Graph" if title_prefix else "Clean Graph", fontsize=12
    )
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot noisy graph (middle)
    im2 = axes[1].imshow(A_noisy_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title(f"Noisy Graph ({noise_type}, ε={noise_level})", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot denoised graph (right)
    im3 = axes[2].imshow(A_denoised_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[2].set_title("Denoised Graph", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Plot delta (right-most)
    delta_vmax = np.abs(A_delta_np).max()
    im4 = axes[3].imshow(
        A_delta_np, cmap="bwr", vmin=-delta_vmax, vmax=delta_vmax, aspect="equal"
    )
    axes[3].set_title("Delta (Denoised - Clean)", fontsize=12)
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save the plot
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_denoising_results(
    A_original: np.ndarray,
    A_noisy: np.ndarray,
    A_denoised: np.ndarray,
    noise_type: str = "Unknown",
    eps: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot original, noisy, denoised, and delta adjacency matrices side by side.

    Args:
        A_original: Original clean adjacency matrix
        A_noisy: Noisy adjacency matrix
        A_denoised: Denoised adjacency matrix
        noise_type: Type of noise applied
        eps: Noise level
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Determine a shared color scale for consistency across plots
    vmin = min(A_original.min(), A_noisy.min(), A_denoised.min())
    vmax = max(A_original.max(), A_noisy.max(), A_denoised.max())

    # Original matrix
    im1 = axes[0].imshow(A_original, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Noisy matrix
    im2 = axes[1].imshow(A_noisy, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Noisy ({noise_type}, ε={eps})")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Denoised matrix
    im3 = axes[2].imshow(A_denoised, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title("Denoised")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Delta matrix
    A_delta = A_denoised - A_original
    delta_vmax = np.abs(A_delta).max()
    im4 = axes[3].imshow(A_delta, cmap="bwr", vmin=-delta_vmax, vmax=delta_vmax)
    axes[3].set_title("Delta (Denoised - Original)")
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_noise_level_comparison(
    noise_levels: List[float],
    metrics_dict: Dict[str, List[float]],
    metric_name: str = "MSE",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how model performance varies with noise levels.

    Args:
        noise_levels: List of noise levels
        metrics_dict: Dictionary mapping dataset types to metric values
        metric_name: Name of the metric being plotted
        title: Optional plot title
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for dataset_type, values in metrics_dict.items():
        ax.plot(
            noise_levels, values, "o-", label=dataset_type, linewidth=2, markersize=6
        )

    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_eigenvalue_comparison(
    eigenvals_true: np.ndarray,
    eigenvals_noisy: np.ndarray,
    eigenvals_denoised: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comparison of eigenvalues before and after denoising.

    Args:
        eigenvals_true: True eigenvalues
        eigenvals_noisy: Noisy eigenvalues
        eigenvals_denoised: Denoised eigenvalues
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    indices = range(len(eigenvals_true))
    width = 0.25

    ax.bar(
        [i - width for i in indices],
        eigenvals_true,
        width,
        label="True",
        alpha=0.8,
        color="blue",
    )
    ax.bar(indices, eigenvals_noisy, width, label="Noisy", alpha=0.8, color="red")
    ax.bar(
        [i + width for i in indices],
        eigenvals_denoised,
        width,
        label="Denoised",
        alpha=0.8,
        color="green",
    )

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_multi_noise_visualization(
    A_original: torch.Tensor,
    model: torch.nn.Module,
    noise_function,
    noise_levels: List[float],
    device: str = "cpu",
) -> plt.Figure:
    """
    Create visualization showing denoising results across multiple noise levels.

    This function creates a 4-row visualization for each noise level:
    1. Clean Graph
    2. Noisy Graph
    3. Denoised Graph
    4. Delta (Denoised - Clean)

    Args:
        A_original: Original clean adjacency matrix
        model: Trained denoising model
        noise_function: Function to add noise
        noise_levels: List of noise levels to visualize
        device: Device for computation

    Returns:
        Matplotlib figure
    """
    fig_size = 4
    fig, axes = plt.subplots(
        4, len(noise_levels), figsize=(fig_size * len(noise_levels), fig_size * 4)
    )
    axes = axes.reshape(4, -1)

    model.eval()
    with torch.no_grad():
        for i, eps in enumerate(noise_levels):
            # Add noise
            A_noisy = noise_function(A_original, eps)
            A_noisy = A_noisy.to(device)

            # Get reconstruction
            if A_noisy.ndim == 2:
                A_noisy_input = A_noisy.unsqueeze(0)
            else:
                A_noisy_input = A_noisy

            # Get reconstruction handling different model outputs
            model_output = model(A_noisy_input)
            if isinstance(model_output, tuple):
                A_reconstructed = model_output[0]
            else:
                A_reconstructed = model_output

            A_original_np = A_original.squeeze().cpu().numpy()
            A_noisy_np = A_noisy.squeeze().cpu().numpy()
            A_reconstructed_np = A_reconstructed.squeeze().cpu().numpy()

            # Determine shared vmin/vmax for first 3 rows
            vmin = min(A_original_np.min(), A_noisy_np.min(), A_reconstructed_np.min())
            vmax = max(A_original_np.max(), A_noisy_np.max(), A_reconstructed_np.max())
            plot_params = {"cmap": "viridis", "vmin": vmin, "vmax": vmax}

            # Row 0: Clean
            axes[0, i].imshow(A_original_np, **plot_params)
            axes[0, i].set_title(f"ε={eps:.2f}")
            if i == 0:
                axes[0, i].set_ylabel("Clean", fontsize=12)

            # Row 1: Noisy
            axes[1, i].imshow(A_noisy_np, **plot_params)
            if i == 0:
                axes[1, i].set_ylabel("Noisy", fontsize=12)

            # Row 2: Denoised
            axes[2, i].imshow(A_reconstructed_np, **plot_params)
            if i == 0:
                axes[2, i].set_ylabel("Denoised", fontsize=12)

            # Row 3: Delta
            delta = A_reconstructed_np - A_original_np
            delta_vmax = np.abs(delta).max()
            axes[3, i].imshow(delta, cmap="bwr", vmin=-delta_vmax, vmax=delta_vmax)
            if i == 0:
                axes[3, i].set_ylabel("Delta", fontsize=12)

            for row in range(4):
                axes[row, i].axis("off")

    plt.tight_layout(pad=0.1, h_pad=0.5)
    return fig


def create_graph_denoising_wandb_image(
    A_clean: Union[np.ndarray, torch.Tensor],
    noise_fn: Callable[
        [Union[np.ndarray, torch.Tensor], float],
        Tuple[Union[np.ndarray, torch.Tensor], Any, Any],
    ],
    denoise_fn: Callable[
        [Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]
    ],
    noise_level: float,
    noise_type: str = "Unknown",
    title_prefix: str = "",
    cmap: str = "viridis",
) -> wandb.Image:
    """
    Create a wandb Image for graph denoising visualization.

    This is a wandb-compatible wrapper around plot_graph_denoising_comparison.

    Args:
        A_clean: Clean adjacency matrix
        noise_fn: Function that adds noise
        denoise_fn: Function that denoises
        noise_level: Noise level parameter
        noise_type: Name of the noise type
        title_prefix: Optional prefix for the plot title
        cmap: Colormap to use

    Returns:
        wandb.Image object
    """
    fig = plot_graph_denoising_comparison(
        A_clean=A_clean,
        noise_fn=noise_fn,
        denoise_fn=denoise_fn,
        noise_level=noise_level,
        noise_type=noise_type,
        title_prefix=title_prefix,
        cmap=cmap,
    )

    wandb_image = wandb.Image(fig)
    plt.close(fig)  # Clean up to avoid memory issues

    return wandb_image


def create_graph_denoising_figure(
    A_clean: Union[np.ndarray, torch.Tensor],
    noise_fn: Callable[
        [Union[np.ndarray, torch.Tensor], float],
        Tuple[Union[np.ndarray, torch.Tensor], Any, Any],
    ],
    denoise_fn: Callable[
        [Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]
    ],
    noise_level: float,
    noise_type: str = "Unknown",
    title_prefix: str = "",
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Create a matplotlib figure for graph denoising visualization.

    This is a logger-agnostic version that returns a matplotlib figure.

    Args:
        A_clean: Clean adjacency matrix
        noise_fn: Function that adds noise
        denoise_fn: Function that denoises
        noise_level: Noise level parameter
        noise_type: Name of the noise type
        title_prefix: Optional prefix for the plot title
        cmap: Colormap to use

    Returns:
        matplotlib.figure.Figure object
    """
    return plot_graph_denoising_comparison(
        A_clean=A_clean,
        noise_fn=noise_fn,
        denoise_fn=denoise_fn,
        noise_level=noise_level,
        noise_type=noise_type,
        title_prefix=title_prefix,
        cmap=cmap,
    )


def create_wandb_visualization(
    A_original: torch.Tensor,
    model: torch.nn.Module,
    noise_function,
    noise_levels: List[float],
    dataset_type: str = "test",
    device: str = "cpu",
) -> Dict[str, wandb.Image]:
    """
    Create visualizations for Weights & Biases logging.

    Args:
        A_original: Original clean adjacency matrix
        model: Trained denoising model
        noise_function: Function to add noise
        noise_levels: List of noise levels
        dataset_type: Type of dataset ("train" or "test")
        device: Device for computation

    Returns:
        Dictionary of wandb Images
    """
    fig = create_multi_noise_visualization(
        A_original, model, noise_function, noise_levels, device
    )

    wandb_images = {f"{dataset_type}_reconstruction": wandb.Image(fig)}

    plt.close(fig)  # Clean up to avoid memory issues

    return wandb_images


def plot_eigenvalue_denoising(
    A_original: np.ndarray,
    A_noisy: np.ndarray,
    eigenvals_denoised: np.ndarray,
    eigenvecs_noisy: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot eigenvalue-only denoising visualization.

    This shows the adjacency matrix reconstructed using denoised eigenvalues
    but keeping the noisy eigenvectors, which isolates the effect of
    eigenvalue denoising.

    Args:
        A_original: Original clean adjacency matrix
        A_noisy: Noisy adjacency matrix
        eigenvals_denoised: Denoised eigenvalues
        eigenvecs_noisy: Noisy eigenvectors
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # Reconstruct using denoised eigenvalues and noisy eigenvectors
    A_recon_eigvals_only = (
        eigenvecs_noisy @ np.diag(eigenvals_denoised) @ eigenvecs_noisy.T
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original matrix
    im1 = axes[0].imshow(A_original, cmap="viridis")
    axes[0].set_title("Original")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Noisy matrix
    im2 = axes[1].imshow(A_noisy, cmap="viridis")
    axes[1].set_title("Noisy")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Eigenvalue-only denoised
    im3 = axes[2].imshow(A_recon_eigvals_only, cmap="viridis")
    axes[2].set_title("Denoised (Eigenvalues Only)")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Difference plot
    diff = np.abs(A_original - A_recon_eigvals_only)
    im4 = axes[3].imshow(diff, cmap="hot")
    axes[3].set_title("Absolute Difference")
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_metrics_grid(
    metrics_history: Dict[str, List[float]], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a grid of training metrics over time.

    Args:
        metrics_history: Dictionary mapping metric names to lists of values
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics_history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes)
    else:
        axes = list(axes.flatten())

    for i, (metric_name, values) in enumerate(metrics_history.items()):
        epochs = range(1, len(values) + 1)
        axes[i].plot(epochs, values, "b-", linewidth=2)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f"{metric_name} over Time")
        axes[i].grid(True, alpha=0.3)

        if "loss" in metric_name.lower():
            axes[i].set_yscale("log")

    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
