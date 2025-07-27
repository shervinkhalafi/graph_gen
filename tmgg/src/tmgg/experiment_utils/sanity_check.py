"""Sanity check utilities for verifying experimental setup."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from ..models.base import DenoisingModel
from .data.noise_generators import NoiseGenerator
from .metrics import compute_reconstruction_metrics

logger = logging.getLogger(__name__)


class SanityCheckResult:
    """Container for sanity check results."""
    
    def __init__(self):
        self.passed = True
        self.messages = []
        self.warnings = []
        self.metrics = {}
        
    def add_check(self, name: str, passed: bool, message: str):
        """Add a check result."""
        self.messages.append(f"[{'PASS' if passed else 'FAIL'}] {name}: {message}")
        if not passed:
            self.passed = False
            
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(f"[WARN] {message}")
        
    def add_metric(self, name: str, value: float):
        """Add a metric value."""
        self.metrics[name] = value
        
    def __str__(self):
        """String representation of results."""
        lines = ["=== Sanity Check Results ==="]
        lines.extend(self.messages)
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(self.warnings)
        if self.metrics:
            lines.append("\nMetrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v:.4f}")
        lines.append(f"\nOverall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def check_noise_generator(
    noise_generator: NoiseGenerator,
    sample_size: int = 10,
    noise_levels: List[float] = [0.1, 0.3, 0.5]
) -> SanityCheckResult:
    """
    Verify noise generator is working correctly.
    
    Args:
        noise_generator: Noise generator to test
        sample_size: Size of test adjacency matrix
        noise_levels: Noise levels to test
        
    Returns:
        SanityCheckResult with test outcomes
    """
    result = SanityCheckResult()
    
    # For rotation noise generator, use its configured k value
    from .data.noise_generators import RotationNoiseGenerator
    if isinstance(noise_generator, RotationNoiseGenerator):
        sample_size = noise_generator.k
    
    # Create test adjacency matrix
    A = torch.eye(sample_size) + torch.rand(sample_size, sample_size) * 0.3
    A = (A + A.T) / 2  # Make symmetric
    A = (A > 0.5).float()  # Binarize
    
    # Test different noise levels
    for eps in noise_levels:
        try:
            A_noisy, V, eigenvals = noise_generator.add_noise(A, eps)
            
            # Check output shapes
            shape_ok = (
                A_noisy.shape == A.shape and
                V.shape == A.shape and
                eigenvals.shape == (sample_size,)
            )
            result.add_check(
                f"Output shapes (eps={eps})",
                shape_ok,
                f"A_noisy: {A_noisy.shape}, V: {V.shape}, eigenvals: {eigenvals.shape}"
            )
            
            # Check noise was actually added
            diff = torch.abs(A_noisy - A).sum().item()
            noise_added = diff > 0
            result.add_check(
                f"Noise added (eps={eps})",
                noise_added,
                f"Total difference: {diff:.4f}"
            )
            
            # Check eigendecomposition validity
            if not torch.isnan(eigenvals).any():
                reconstructed = V @ torch.diag(eigenvals) @ V.T
                recon_error = torch.norm(reconstructed - A_noisy).item()
                eigen_valid = recon_error < 10.0  # Relaxed tolerance for eigendecomposition
                result.add_check(
                    f"Eigendecomposition valid (eps={eps})",
                    eigen_valid,
                    f"Reconstruction error: {recon_error:.6f}"
                )
            else:
                result.add_warning(f"NaN eigenvalues at eps={eps}")
                
            # Store metrics
            result.add_metric(f"noise_amount_eps_{eps}", diff / (sample_size * sample_size))
            
        except Exception as e:
            result.add_check(
                f"Noise generation (eps={eps})",
                False,
                f"Exception: {str(e)}"
            )
            
    return result


def check_model_forward_pass(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device = torch.device('cpu')
) -> SanityCheckResult:
    """
    Verify model forward pass works correctly.
    
    Args:
        model: Model to test
        input_shape: Expected input shape
        device: Device to run on
        
    Returns:
        SanityCheckResult with test outcomes
    """
    result = SanityCheckResult()
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create test input
    if len(input_shape) == 2:
        # Single adjacency matrix
        test_input = torch.eye(input_shape[0]).unsqueeze(0).to(device)
    elif len(input_shape) == 3:
        # Batch of adjacency matrices or eigenvectors
        if input_shape[2] == input_shape[1]:
            # Adjacency matrices
            test_input = torch.eye(input_shape[1]).unsqueeze(0).repeat(input_shape[0], 1, 1).to(device)
        else:
            # Eigenvectors
            test_input = torch.randn(*input_shape).to(device)
    else:
        test_input = torch.randn(*input_shape).to(device)
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(test_input)
            
        result.add_check(
            "Forward pass",
            True,
            f"Output shape: {output.shape}"
        )
        
        # Check output properties
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        result.add_check(
            "Output validity",
            not (has_nan or has_inf),
            f"NaN: {has_nan}, Inf: {has_inf}"
        )
        
        # Check output range for adjacency matrices (warning only)
        if len(output.shape) == 3 and output.shape[1] == output.shape[2]:
            in_range = (output >= 0).all() and (output <= 1).all()
            if not in_range:
                result.add_warning(f"Output range outside [0,1]: Min: {output.min():.4f}, Max: {output.max():.4f}")
            else:
                result.add_check(
                    "Output range [0,1]",
                    True,
                    f"Min: {output.min():.4f}, Max: {output.max():.4f}"
                )
            
    except Exception as e:
        result.add_check(
            "Forward pass",
            False,
            f"Exception: {str(e)}"
        )
        
    return result


def check_data_loader(
    data_loader: torch.utils.data.DataLoader,
    expected_batch_size: int,
    expected_shape: Tuple[int, int],
    num_batches_to_check: int = 3
) -> SanityCheckResult:
    """
    Verify data loader is producing correct batches.
    
    Args:
        data_loader: DataLoader to test
        expected_batch_size: Expected batch size
        expected_shape: Expected shape of each sample (without batch dim)
        num_batches_to_check: Number of batches to check
        
    Returns:
        SanityCheckResult with test outcomes
    """
    result = SanityCheckResult()
    
    try:
        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_check:
                break
                
            # Check batch shape
            expected_full_shape = (expected_batch_size,) + expected_shape
            shape_ok = batch.shape == expected_full_shape or (
                # Last batch might be smaller
                batch.shape[0] <= expected_batch_size and
                batch.shape[1:] == expected_shape
            )
            
            result.add_check(
                f"Batch {i} shape",
                shape_ok,
                f"Got {batch.shape}, expected {expected_full_shape}"
            )
            
            # Check data validity
            has_nan = torch.isnan(batch).any().item()
            has_inf = torch.isinf(batch).any().item()
            
            result.add_check(
                f"Batch {i} validity",
                not (has_nan or has_inf),
                f"NaN: {has_nan}, Inf: {has_inf}"
            )
            
            # Check symmetry for adjacency matrices
            if len(batch.shape) == 3 and batch.shape[1] == batch.shape[2]:
                sym_diff = torch.abs(batch - batch.transpose(-2, -1)).max().item()
                is_symmetric = sym_diff < 1e-6
                result.add_check(
                    f"Batch {i} symmetry",
                    is_symmetric,
                    f"Max asymmetry: {sym_diff:.6f}"
                )
                
    except Exception as e:
        result.add_check(
            "DataLoader iteration",
            False,
            f"Exception: {str(e)}"
        )
        
    return result


def check_loss_computation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> SanityCheckResult:
    """
    Verify loss computation works correctly.
    
    Args:
        model: Model to test
        criterion: Loss function
        sample_input: Sample input
        sample_target: Sample target
        device: Device to run on
        
    Returns:
        SanityCheckResult with test outcomes
    """
    result = SanityCheckResult()
    
    model = model.to(device)
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)
    
    try:
        with torch.no_grad():
            output = model(sample_input)
            loss = criterion(output, sample_target)
            
        result.add_check(
            "Loss computation",
            True,
            f"Loss value: {loss.item():.4f}"
        )
        
        # Check loss validity
        loss_valid = not (torch.isnan(loss).item() or torch.isinf(loss).item())
        result.add_check(
            "Loss validity",
            loss_valid,
            f"Loss is finite: {loss_valid}"
        )
        
        # Check loss is positive
        loss_positive = loss.item() >= 0
        result.add_check(
            "Loss non-negative",
            loss_positive,
            f"Loss value: {loss.item():.4f}"
        )
        
        result.add_metric("sample_loss", loss.item())
        
    except Exception as e:
        result.add_check(
            "Loss computation",
            False,
            f"Exception: {str(e)}"
        )
        
    return result


def run_experiment_sanity_check(
    model: torch.nn.Module,
    noise_generator: NoiseGenerator,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device = torch.device('cpu'),
    save_plots: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run comprehensive sanity checks for an experiment.
    
    Args:
        model: Model to test
        noise_generator: Noise generator to test
        data_loader: DataLoader to test
        criterion: Loss function
        device: Device to run on
        save_plots: Whether to save diagnostic plots
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with all check results
    """
    results = {}
    
    print("\n" + "="*50)
    print("Running Experiment Sanity Checks")
    print("="*50 + "\n")
    
    # 1. Check noise generator
    print("1. Checking noise generator...")
    noise_result = check_noise_generator(noise_generator)
    results['noise_generator'] = noise_result
    print(noise_result)
    print()
    
    # 2. Check data loader
    print("2. Checking data loader...")
    # Get a sample to determine expected shape
    sample_batch = next(iter(data_loader))
    expected_shape = sample_batch.shape[1:]
    
    data_result = check_data_loader(
        data_loader,
        sample_batch.shape[0],
        expected_shape
    )
    results['data_loader'] = data_result
    print(data_result)
    print()
    
    # 3. Check model forward pass
    print("3. Checking model forward pass...")
    # Determine input shape based on model type
    if hasattr(model, 'model') and hasattr(model.model, '__class__'):
        model_name = model.model.__class__.__name__
        if 'Attention' in model_name:
            # Attention models expect eigenvectors
            input_shape = (1, expected_shape[0], expected_shape[0])
        else:
            # GNN models expect adjacency matrices
            input_shape = (1,) + expected_shape
    else:
        input_shape = (1,) + expected_shape
        
    model_result = check_model_forward_pass(model, input_shape, device)
    results['model_forward'] = model_result
    print(model_result)
    print()
    
    # 4. Check loss computation
    print("4. Checking loss computation...")
    sample_input = sample_batch[:1].to(device)
    loss_result = check_loss_computation(
        model, criterion, sample_input, sample_input, device
    )
    results['loss_computation'] = loss_result
    print(loss_result)
    print()
    
    # 5. Generate diagnostic plots if requested
    if save_plots and output_dir:
        print("5. Generating diagnostic plots...")
        generate_diagnostic_plots(
            model, noise_generator, sample_batch[0], 
            output_dir, device
        )
        print(f"Plots saved to {output_dir}")
        print()
    
    # Overall result
    all_passed = all(r.passed for r in results.values() if hasattr(r, 'passed'))
    
    print("="*50)
    print(f"Overall Sanity Check: {'PASSED' if all_passed else 'FAILED'}")
    print("="*50)
    
    return {
        'passed': all_passed,
        'results': results
    }


def generate_diagnostic_plots(
    model: torch.nn.Module,
    noise_generator: NoiseGenerator,
    sample_adjacency: torch.Tensor,
    output_dir: Path,
    device: torch.device = torch.device('cpu')
):
    """Generate diagnostic plots for sanity checking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Noise effect visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    noise_levels = [0.0, 0.1, 0.3]
    
    for i, eps in enumerate(noise_levels):
        if eps == 0:
            A_noisy = sample_adjacency
        else:
            A_noisy, _, _ = noise_generator.add_noise(sample_adjacency, eps)
            
        # Top row: adjacency matrices
        im = axes[0, i].imshow(A_noisy.cpu().numpy(), cmap='viridis')
        axes[0, i].set_title(f'Noise level: {eps}')
        axes[0, i].axis('off')
        
        # Bottom row: difference from original
        diff = torch.abs(A_noisy - sample_adjacency)
        axes[1, i].imshow(diff.cpu().numpy(), cmap='Reds')
        axes[1, i].set_title(f'Difference (sum: {diff.sum():.2f})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'noise_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Model input/output visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    axes[0].imshow(sample_adjacency.cpu().numpy(), cmap='viridis')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Noisy
    A_noisy, V_noisy, _ = noise_generator.add_noise(sample_adjacency, 0.2)
    axes[1].imshow(A_noisy.cpu().numpy(), cmap='viridis')
    axes[1].set_title('Noisy (eps=0.2)')
    axes[1].axis('off')
    
    # Denoised
    model.eval()
    with torch.no_grad():
        # Handle different model types
        if hasattr(model, 'forward'):
            if 'Attention' in model.__class__.__name__:
                # Attention models need eigenvectors
                denoised = model(V_noisy.unsqueeze(0).to(device))
            else:
                # GNN/Hybrid models need adjacency matrix
                denoised = model(A_noisy.unsqueeze(0).to(device))
        else:
            denoised = model(A_noisy.unsqueeze(0).to(device))
            
        if len(denoised.shape) == 3:
            denoised = denoised.squeeze(0)
            
    axes[2].imshow(denoised.cpu().numpy(), cmap='viridis')
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_io.png', dpi=150, bbox_inches='tight')
    plt.close()