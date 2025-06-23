"""Plotting utilities for graph denoising experiments."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import wandb


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        save_path: Optional[str] = None,
                        title: str = "Training Curves") -> plt.Figure:
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
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_denoising_results(A_original: np.ndarray,
                          A_noisy: np.ndarray, 
                          A_denoised: np.ndarray,
                          noise_type: str = "Unknown",
                          eps: float = 0.0,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot original, noisy, and denoised adjacency matrices side by side.
    
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original matrix
    im1 = axes[0].imshow(A_original, cmap='viridis')
    axes[0].set_title('Original')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Noisy matrix
    im2 = axes[1].imshow(A_noisy, cmap='viridis')
    axes[1].set_title(f'Noisy ({noise_type}, ε={eps})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Denoised matrix
    im3 = axes[2].imshow(A_denoised, cmap='viridis')
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_noise_level_comparison(noise_levels: List[float],
                               metrics_dict: Dict[str, List[float]],
                               metric_name: str = "MSE",
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
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
        ax.plot(noise_levels, values, 'o-', label=dataset_type, linewidth=2, markersize=6)
    
    ax.set_xlabel('Noise Level (ε)')
    ax.set_ylabel(metric_name)
    ax.set_title(title or f'{metric_name} vs Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_eigenvalue_comparison(eigenvals_true: np.ndarray,
                              eigenvals_noisy: np.ndarray,
                              eigenvals_denoised: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
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
    
    ax.bar([i - width for i in indices], eigenvals_true, width, 
           label='True', alpha=0.8, color='blue')
    ax.bar(indices, eigenvals_noisy, width, 
           label='Noisy', alpha=0.8, color='red')
    ax.bar([i + width for i in indices], eigenvals_denoised, width, 
           label='Denoised', alpha=0.8, color='green')
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_multi_noise_visualization(A_original: torch.Tensor,
                                   model: torch.nn.Module,
                                   noise_function,
                                   noise_levels: List[float],
                                   device: str = 'cpu') -> plt.Figure:
    """
    Create visualization showing denoising results across multiple noise levels.
    
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
    fig, axes = plt.subplots(2, len(noise_levels), figsize=(fig_size*len(noise_levels), fig_size*2))
    
    model.eval()
    with torch.no_grad():
        for i, eps in enumerate(noise_levels):
            # Add noise
            A_noisy, _, _ = noise_function(A_original, eps)
            A_noisy = A_noisy.to(device)
            
            # Get reconstruction
            if A_noisy.ndim == 2:
                A_noisy_input = A_noisy.unsqueeze(0)
            else:
                A_noisy_input = A_noisy
                
            # Get reconstruction handling different model outputs
            model_output = model(A_noisy_input.double())
            
            # Handle different model return types
            if isinstance(model_output, tuple):
                # GNN models return (X, Y) tuple
                X, Y = model_output
                A_recon = torch.sigmoid(torch.bmm(X, Y.transpose(1, 2)))
            else:
                # Direct adjacency matrix output
                A_recon = model_output
            
            # Plot noisy matrix on top row
            # Handle different tensor shapes
            A_noisy_plot = A_noisy
            while A_noisy_plot.dim() > 2:
                A_noisy_plot = A_noisy_plot.squeeze(0)
            im1 = axes[0, i].imshow(A_noisy_plot.cpu().numpy(), cmap='viridis')
            axes[0, i].set_title(f'Noisy (ε={eps})')
            axes[0, i].axis('off')
            
            # Plot reconstructed matrix on bottom row
            im2 = axes[1, i].imshow(A_recon.squeeze().cpu().numpy(), cmap='viridis')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def create_wandb_visualization(A_original: torch.Tensor,
                              model: torch.nn.Module,
                              noise_function,
                              noise_levels: List[float],
                              dataset_type: str = "test",
                              device: str = 'cpu') -> Dict[str, wandb.Image]:
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
    fig = create_multi_noise_visualization(A_original, model, noise_function, 
                                         noise_levels, device)
    
    wandb_images = {
        f"{dataset_type}_reconstruction": wandb.Image(fig)
    }
    
    plt.close(fig)  # Clean up to avoid memory issues
    
    return wandb_images


def plot_eigenvalue_denoising(A_original: np.ndarray,
                              A_noisy: np.ndarray,
                              eigenvals_denoised: np.ndarray,
                              eigenvecs_noisy: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
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
    A_recon_eigvals_only = eigenvecs_noisy @ np.diag(eigenvals_denoised) @ eigenvecs_noisy.T
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original matrix
    im1 = axes[0].imshow(A_original, cmap='viridis')
    axes[0].set_title('Original')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Noisy matrix
    im2 = axes[1].imshow(A_noisy, cmap='viridis')
    axes[1].set_title('Noisy')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Eigenvalue-only denoised
    im3 = axes[2].imshow(A_recon_eigvals_only, cmap='viridis')
    axes[2].set_title('Denoised (Eigenvalues Only)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Difference plot
    diff = np.abs(A_original - A_recon_eigvals_only)
    im4 = axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('Absolute Difference')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_metrics_grid(metrics_history: Dict[str, List[float]], 
                              save_path: Optional[str] = None) -> plt.Figure:
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
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
        axes[i].plot(epochs, values, 'b-', linewidth=2)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'{metric_name} over Time')
        axes[i].grid(True, alpha=0.3)
        
        if 'loss' in metric_name.lower():
            axes[i].set_yscale('log')
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig