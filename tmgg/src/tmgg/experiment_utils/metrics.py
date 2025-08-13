"""Evaluation metrics for graph denoising experiments."""

import torch
import numpy as np
from scipy.sparse.linalg import eigsh
from typing import Dict, Any, Tuple, Union


def compute_eigenvalue_error(A_true: Union[torch.Tensor, np.ndarray], 
                           A_pred: Union[torch.Tensor, np.ndarray], 
                           k: int = 4) -> float:
    """
    Compute normalized eigenvalue error between true and predicted adjacency matrices.
    
    Args:
        A_true: True adjacency matrix
        A_pred: Predicted adjacency matrix
        k: Number of top eigenvalues to compare
        
    Returns:
        Normalized eigenvalue error
    """
    # Convert to numpy if needed
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    
    # Handle batch dimension
    if A_true.ndim == 3:
        A_true = A_true[0]
    if A_pred.ndim == 3:
        A_pred = A_pred[0]
    
    # Compute eigenvalues
    l_true, _ = eigsh(A_true, k=k, which='LM', maxiter=10000)
    l_pred, _ = eigsh(A_pred, k=k, which='LM', maxiter=10000)
    
    # Compute normalized error
    eigval_error = np.linalg.norm(l_pred - l_true) / np.linalg.norm(l_true)
    return float(eigval_error)


def compute_subspace_distance(A_true: Union[torch.Tensor, np.ndarray], 
                             A_pred: Union[torch.Tensor, np.ndarray], 
                             k: int = 4) -> float:
    """
    Compute subspace distance between true and predicted adjacency matrices.
    
    Args:
        A_true: True adjacency matrix
        A_pred: Predicted adjacency matrix
        k: Number of top eigenvectors to compare
        
    Returns:
        Frobenius norm of difference between projection matrices
    """
    # Convert to numpy if needed
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    
    # Handle batch dimension
    if A_true.ndim == 3:
        A_true = A_true[0]
    if A_pred.ndim == 3:
        A_pred = A_pred[0]
    
    # Compute eigenvectors
    _, V_true = eigsh(A_true, k=k, which='LM', maxiter=10000)
    _, V_pred = eigsh(A_pred, k=k, which='LM', maxiter=10000)
    
    # Compute projection matrices
    Proj_true = V_true @ V_true.T
    Proj_pred = V_pred @ V_pred.T
    
    # Compute Frobenius norm of difference
    subspace_distance = np.linalg.norm(Proj_true - Proj_pred, 'fro')
    return float(subspace_distance)


def compute_reconstruction_metrics(A_true: Union[torch.Tensor, np.ndarray], 
                                 A_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction metrics.
    
    Args:
        A_true: True adjacency matrix
        A_pred: Predicted adjacency matrix
        
    Returns:
        Dictionary containing various reconstruction metrics
    """
    # Convert to tensors for metric computation
    if isinstance(A_true, np.ndarray):
        A_true = torch.tensor(A_true, dtype=torch.float32)
    if isinstance(A_pred, np.ndarray):
        A_pred = torch.tensor(A_pred, dtype=torch.float32)
    
    # Handle batch dimension
    if A_true.ndim == 3:
        A_true = A_true.squeeze(0)
    if A_pred.ndim == 3:
        A_pred = A_pred.squeeze(0)
    
    # Mean Squared Error
    mse = torch.mean((A_true - A_pred) ** 2).item()
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(A_true - A_pred)).item()
    
    # Binary Cross Entropy (assuming adjacency matrices are binary)
    A_pred_clipped = torch.clamp(A_pred, min=1e-7, max=1-1e-7)  # Avoid log(0)
    bce = -torch.mean(A_true * torch.log(A_pred_clipped) + 
                     (1 - A_true) * torch.log(1 - A_pred_clipped)).item()
    
    # Frobenius norm error
    frobenius_error = torch.norm(A_true - A_pred, p='fro').item()
    
    # Relative Frobenius error
    relative_frobenius = frobenius_error / torch.norm(A_true, p='fro').item()
    
    # Spectral metrics
    eigenvalue_error = compute_eigenvalue_error(A_true, A_pred)
    subspace_distance = compute_subspace_distance(A_true, A_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'bce': bce,
        'frobenius_error': frobenius_error,
        'relative_frobenius': relative_frobenius,
        'eigenvalue_error': eigenvalue_error,
        'subspace_distance': subspace_distance,
    }


def compute_batch_metrics(A_true_batch: torch.Tensor, 
                         A_pred_batch: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics averaged over a batch.
    
    Args:
        A_true_batch: Batch of true adjacency matrices
        A_pred_batch: Batch of predicted adjacency matrices
        
    Returns:
        Dictionary of averaged metrics
    """
    batch_size = A_true_batch.shape[0]
    
    # Initialize metric accumulators
    metrics_sum = {
        'mse': 0.0,
        'mae': 0.0,
        'bce': 0.0,
        'frobenius_error': 0.0,
        'relative_frobenius': 0.0,
        'eigenvalue_error': 0.0,
        'subspace_distance': 0.0,
    }
    
    # Compute metrics for each sample in batch
    for i in range(batch_size):
        sample_metrics = compute_reconstruction_metrics(A_true_batch[i], A_pred_batch[i])
        for key in metrics_sum:
            metrics_sum[key] += sample_metrics[key]
    
    # Average over batch
    metrics_avg = {key: value / batch_size for key, value in metrics_sum.items()}
    
    return metrics_avg


def evaluate_noise_robustness(model: torch.nn.Module, 
                             A_clean: torch.Tensor,
                             noise_levels: list,
                             noise_function,
                             device: str = 'cpu') -> Dict[str, list]:
    """
    Evaluate model robustness across different noise levels.
    
    Args:
        model: Trained denoising model
        A_clean: Clean adjacency matrix
        noise_levels: List of noise levels to test
        noise_function: Function to add noise (e.g., add_gaussian_noise)
        device: Device for computation
        
    Returns:
        Dictionary mapping metric names to lists of values across noise levels
    """
    model.eval()
    
    metrics_by_noise = {
        'noise_levels': noise_levels,
        'mse': [],
        'mae': [],
        'eigenvalue_error': [],
        'subspace_distance': [],
    }
    
    with torch.no_grad():
        for eps in noise_levels:
            # Add noise
            A_noisy, _, _ = noise_function(A_clean, eps)
            A_noisy = A_noisy.to(device)
            A_clean_tensor = A_clean.to(device)
            
            # Predict denoised matrix
            if A_noisy.ndim == 2:
                A_noisy = A_noisy.unsqueeze(0)
            if A_clean_tensor.ndim == 2:
                A_clean_tensor = A_clean_tensor.unsqueeze(0)
                
            A_pred = model(A_noisy)
            
            # Compute metrics
            sample_metrics = compute_reconstruction_metrics(A_clean_tensor, A_pred)
            
            metrics_by_noise['mse'].append(sample_metrics['mse'])
            metrics_by_noise['mae'].append(sample_metrics['mae'])
            metrics_by_noise['eigenvalue_error'].append(sample_metrics['eigenvalue_error'])
            metrics_by_noise['subspace_distance'].append(sample_metrics['subspace_distance'])
    
    return metrics_by_noise